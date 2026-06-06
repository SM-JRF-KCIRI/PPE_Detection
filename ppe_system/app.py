import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import gradio as gr
from ultralytics import YOLO

from config import (
    BEST_PPE_MODEL_PATH,
    BASE_MODEL_PATH,
    CONFIDENCE_RANGE,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    IOU_RANGE,
    PPE_MODEL_PATH,
    POSE_MODEL_PATH,
    STATUS_COLORS,
)
from detector import PPEDetector
from pose import PoseEstimator
from ppe_rules import evaluate_ppe
from smoother import TemporalSmoother
from tracker import ByteTrackWrapper
from utils import (
    build_report_rows,
    compute_iou,
    draw_box,
    draw_label,
    draw_box as draw_rectangle,
    render_status_overlay,
    safe_int,
)


try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

if (isinstance(PPE_MODEL_PATH, Path) and PPE_MODEL_PATH.name == BASE_MODEL_PATH) or PPE_MODEL_PATH == BASE_MODEL_PATH:
    print(f"WARNING: Custom PPE model not found at {BEST_PPE_MODEL_PATH}. Falling back to YOLOv8m baseline.")
else:
    print(f"Loading custom PPE model from {PPE_MODEL_PATH}")

print(f"Loading pose model from {POSE_MODEL_PATH}")

ppe_detector = PPEDetector(PPE_MODEL_PATH, device=DEVICE)
pose_estimator = PoseEstimator(POSE_MODEL_PATH, device=DEVICE)
tracker = ByteTrackWrapper()
smoother = TemporalSmoother()


def annotate_person(frame, report, detections):
    box = report["person_box"]
    x1, y1, x2, y2 = [safe_int(v) for v in box]
    status_color = STATUS_COLORS[report["status"]]
    draw_box(frame, box, status_color, thickness=2)
    label = f"ID {report['person_id']} | {report['status']}"
    draw_label(frame, label, (x1, y1 - 8), status_color)
    for region_name, region_box in report["regions"].items():
        if region_name == "hands":
            for hand_box in region_box:
                if hand_box is not None:
                    draw_rectangle(frame, hand_box, (255, 255, 0), thickness=1)
        elif region_box is not None:
            draw_rectangle(frame, region_box, (255, 255, 0), thickness=1)
    for detection in detections:
        label_text = f"{detection['label']}:{detection['confidence']:.2f}"
        draw_box(frame, detection["bbox"], (255, 165, 0), thickness=1)
        draw_label(frame, label_text, (detection["bbox"][0], detection["bbox"][1] - 10), (255, 165, 0))


def build_person_reports(frame, persons, ppe_detections, fps: float) -> List[dict]:
    active_ids = []
    reports = []
    track_candidates = tracker.update(
        [{"bbox": person["box"], "confidence": person["confidence"]} for person in persons], frame.shape
    )
    for person in persons:
        assigned_detections = [det for det in ppe_detections if compute_iou(person["box"], det["bbox"]) > 0.05]
        person_id = -1
        best_iou = 0.0
        for track in track_candidates:
            iou = compute_iou(person["box"], track["bbox"])
            if iou > best_iou:
                best_iou = iou
                person_id = int(track["track_id"])
        report = evaluate_ppe(person_id, assigned_detections, person["keypoints"], person["box"])
        report["person_box"] = person["box"]
        active_ids.append(person_id)
        smoothed = smoother.smooth(
            person_id,
            {"helmet": report["helmet"], "vest": report["vest"], "gloves": report["gloves"], "boots": report["boots"]},
        )
        report.update(
            {
                "helmet": smoothed["helmet"],
                "vest": smoothed["vest"],
                "gloves": smoothed["gloves"],
                "boots": smoothed["boots"],
                "status": "GREEN" if all(smoothed.values()) else "RED",
            }
        )
        reports.append(report)
    smoother.prune(active_ids)
    for report in reports:
        annotate_person(frame, report, ppe_detections)
    return reports


def process_image(image, confidence, iou_threshold):
    frame = image.copy()
    pose_results = pose_estimator.detect(frame, confidence, iou_threshold)
    ppe_results = ppe_detector.detect(frame, confidence, iou_threshold)
    start = time.time()
    reports = build_person_reports(frame, pose_results, ppe_results, fps=0.0)
    fps = 1.0 / max(time.time() - start, 1e-6)
    annotated = render_status_overlay(frame, fps, f"Persons: {len(reports)}")
    return annotated, build_report_rows(reports)


def process_video(input_path, confidence, iou_threshold):
    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(cap.get(cv2.CAP_PROP_FPS), 10.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_file = Path(tempfile.gettempdir()) / f"ppe_inspection_{int(time.time())}.mp4"
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    summary_reports = {}
    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose_estimator.detect(frame_rgb, confidence, iou_threshold)
        ppe_results = ppe_detector.detect(frame_rgb, confidence, iou_threshold)
        start = time.time()
        reports = build_person_reports(frame_rgb, pose_results, ppe_results, fps=fps)
        elapsed = max(time.time() - start, 1e-6)
        annotated = render_status_overlay(frame_rgb, fps, f"Frame {frame_idx + 1} / {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        for report in reports:
            summary_reports[report["person_id"]] = report
        frame_idx += 1
    cap.release()
    writer.release()
    final_reports = list(summary_reports.values())
    return str(output_file), build_report_rows(final_reports)


def build_dashboard():
    with gr.Blocks() as demo:
        gr.Markdown("# PPE Inspection Dashboard")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Controls")
                confidence_slider = gr.Slider(minimum=CONFIDENCE_RANGE[0], maximum=CONFIDENCE_RANGE[1], value=DEFAULT_CONFIDENCE, step=0.05, label="Confidence Threshold")
                iou_slider = gr.Slider(minimum=IOU_RANGE[0], maximum=IOU_RANGE[1], value=DEFAULT_IOU, step=0.05, label="NMS IoU Threshold")
                with gr.Tab("Upload Image"):
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    image_btn = gr.Button("Run Image Inspection")
                with gr.Tab("Upload Video"):
                    video_input = gr.Video(label="Upload Video")
                    video_btn = gr.Button("Run Video Inspection")
                with gr.Tab("Webcam Stream"):
                    webcam_input = gr.Video(sources=["webcam"], streaming=True, label="Webcam Stream")
                    webcam_btn = gr.Button("Run Webcam Inspection")
                with gr.Tab("Live Server Camera"):
                    camera_input = gr.Textbox(label="RTSP / HTTP Camera URL", placeholder="rtsp://... or http://...")
                    camera_btn = gr.Button("Run Live Camera")
            with gr.Column(scale=1):
                gr.Markdown("### Inspection Output")
                output_image = gr.Image(label="Annotated Output", type="numpy")
                output_video = gr.Video(label="Annotated Video")
                status_html = gr.Markdown("---")
        report_table = gr.Dataframe(headers=["Person ID", "Helmet", "Vest", "Gloves", "Boots", "Compliance"], label="Compliance Report")

        def run_image(image, confidence, iou):
            annotated, report_rows = process_image(image, confidence, iou)
            return annotated, None, report_rows, "### Result: Image inspection complete"

        def run_video(video, confidence, iou):
            if video is None:
                return None, None, [], "### Result: No video provided"
            output_path, report_rows = process_video(video, confidence, iou)
            return None, output_path, report_rows, "### Result: Video inspection complete"

        def run_camera_stream(url, confidence, iou):
            if not url:
                return None, None, [], "### Result: No camera URL provided"
            return run_video(url, confidence, iou)

        image_btn.click(fn=run_image, inputs=[image_input, confidence_slider, iou_slider], outputs=[output_image, output_video, report_table, status_html])
        video_btn.click(fn=run_video, inputs=[video_input, confidence_slider, iou_slider], outputs=[output_image, output_video, report_table, status_html])
        webcam_btn.click(fn=run_video, inputs=[webcam_input, confidence_slider, iou_slider], outputs=[output_image, output_video, report_table, status_html])
        camera_btn.click(fn=run_camera_stream, inputs=[camera_input, confidence_slider, iou_slider], outputs=[output_image, output_video, report_table, status_html])

    return demo


def find_available_port(start_port: int = 7860, max_port: int = 7899) -> int:
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise OSError(f"Cannot find empty port in range: {start_port}-{max_port}")


if __name__ == "__main__":
    app = build_dashboard()
    requested_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    port = requested_port
    try:
        port = find_available_port(requested_port)
    except OSError as error:
        print(str(error))
    if port != requested_port:
        print(f"Port {requested_port} is busy. Launching on port {port} instead.")
    CSS = ".section-title { font-size: 22px; font-weight: 700; } .panel { background-color: #1e1e1e; border-radius: 12px; padding: 16px; color: white; }"
    app.launch(server_name="0.0.0.0", server_port=port, share=False, css=CSS)
