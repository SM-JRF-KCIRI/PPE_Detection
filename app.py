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

from config import (
    BEST_PPE_MODEL_PATH,
    BASE_MODEL_PATH,
    CONFIDENCE_RANGE,
    DEBUG_MODE,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    HELMET_MODEL_PATH,
    IOU_RANGE,
    PPE_MODEL_PATH,
    POSE_MODEL_PATH,
)
from compliance import evaluate_compliance
from detector import PPEDetector
from ensemble_detector import PPEEnsembleDetector, print_model_comparison_report
from pose_engine import PoseEngine
from ppe_matcher import assign_ppe_to_person, extract_body_regions
from report_generator import build_report_rows
from temporal_smoothing import TemporalSmoother
from tracker import ByteTrackTracker
from visualization import draw_annotations
from utils import compute_iou


try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

print(f"PPE Model Loaded: {PPE_MODEL_PATH}")
print(f"Helmet Model Loaded: {HELMET_MODEL_PATH}")
print(f"Pose Model Loaded: {POSE_MODEL_PATH}")

def health_check() -> dict:
    issues = []
    ppe_loaded = hasattr(ppe_detector, "model") and ppe_detector.model is not None
    pose_loaded = hasattr(pose_engine, "model") and pose_engine.model is not None
    if not ppe_loaded:
        issues.append("PPE detection model failed to load.")
    if not pose_loaded:
        issues.append("Pose model failed to load.")

    return {
        "status": "OK" if ppe_loaded and pose_loaded else "ERROR",
        "device": DEVICE,
        "ppe_model_path": str(PPE_MODEL_PATH),
        "pose_model_path": str(POSE_MODEL_PATH),
        "ppe_model_loaded": ppe_loaded,
        "pose_model_loaded": pose_loaded,
        "issues": issues,
    }

print_model_comparison_report()
ppe_detector = PPEEnsembleDetector(device=DEVICE)
pose_engine = PoseEngine(POSE_MODEL_PATH, device=DEVICE)
tracker = ByteTrackTracker()
smoother = TemporalSmoother()

health_status = health_check()
if health_status["status"] != "OK":
    print("Health check failed:")
    for issue in health_status["issues"]:
        print(f"- {issue}")
else:
    print("Health check passed. System ready.")


def build_person_reports(frame, persons, ppe_detections) -> List[dict]:
    track_candidates = tracker.update([
        {"bbox": person["box"], "confidence": person["confidence"]} for person in persons
    ], frame.shape)

    reports = []
    used_tracks = set()
    for person in persons:
        person_id = -1
        best_iou = 0.0
        for track in track_candidates:
            if track["track_id"] in used_tracks:
                continue
            iou = compute_iou(person["box"], track["bbox"])
            if iou > best_iou and iou > 0.05:
                best_iou = iou
                person_id = int(track["track_id"])
        if person_id != -1:
            used_tracks.add(person_id)

        reports.append(
            {
                "person_id": person_id,
                "box": person["box"],
                "confidence": person["confidence"],
                "keypoints": person["keypoints"],
                "regions": extract_body_regions(person["box"], person["keypoints"]),
                "assigned_ppe": [],
            }
        )

    reports = assign_ppe_to_person(reports, ppe_detections)

    final_reports = []
    active_ids = []
    for report in reports:
        compliance = evaluate_compliance(report["person_id"], report)
        smoothed = smoother.smooth(
            report["person_id"],
            {
                "helmet": compliance.get("helmet"),
                "vest": compliance.get("vest"),
                "hook": compliance.get("hook"),
                "glove": compliance.get("glove"),
                "boot": compliance.get("boot"),
                "goggles": compliance.get("goggles"),
            },
        )
        status = compliance.get("status", "UNKNOWN")
        if status == "COMPLIANT":
            status = "COMPLIANT"
        elif status == "NON-COMPLIANT":
            status = "NON-COMPLIANT"
        else:
            status = status
        final_report = {
            "person_id": report["person_id"],
            "box": report["box"],
            "confidence": report["confidence"],
            "keypoints": report["keypoints"],
            "regions": report["regions"],
            "assigned_ppe": report["assigned_ppe"],
            "helmet": smoothed["helmet"],
            "vest": smoothed["vest"],
            "hook": smoothed.get("hook"),
            "glove": smoothed.get("glove"),
            "boot": smoothed.get("boot"),
            "goggles": smoothed.get("goggles"),
            "status": status,
            "reason": compliance.get("reason", "Matched PPE evaluated."),
        }
        final_reports.append(final_report)
        active_ids.append(report["person_id"])

    smoother.prune(active_ids)
    annotated = draw_annotations(frame, final_reports, ppe_detections, debug=DEBUG_MODE)
    return annotated, final_reports


def process_image(image, confidence, iou_threshold):
    frame = image.copy()
    persons = pose_engine.detect_persons(frame, confidence, iou_threshold)
    ppe_detections = ppe_detector.detect_ppe(frame, confidence, iou_threshold)
    annotated, reports = build_person_reports(frame, persons, ppe_detections)
    return annotated, build_report_rows(reports)


def process_video(input_path, confidence, iou_threshold):
    if isinstance(input_path, str) and input_path.isdigit():
        capture_source = int(input_path)
    elif isinstance(input_path, int):
        capture_source = input_path
    else:
        capture_source = str(input_path)

    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        return None, []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(cap.get(cv2.CAP_PROP_FPS), 10.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_file = Path(tempfile.gettempdir()) / f"ppe_inspection_{int(time.time())}.mp4"
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    summary_reports = {}
    while True:
        success, frame_bgr = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        persons = pose_engine.detect_persons(frame_rgb, confidence, iou_threshold)
        ppe_detections = ppe_detector.detect_ppe(frame_rgb, confidence, iou_threshold)
        annotated, reports = build_person_reports(frame_rgb, persons, ppe_detections)
        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        for report in reports:
            summary_reports[report["person_id"]] = report

    cap.release()
    writer.release()
    return str(output_file), build_report_rows(list(summary_reports.values()))


def process_webcam(device_index: int, confidence, iou_threshold):
    return process_video(device_index, confidence, iou_threshold)


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

        report_table = gr.Dataframe(
            headers=["Person ID", "Helmet", "Vest", "Hook", "Glove", "Boot", "Goggles", "Status", "Reason"],
            datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
            label="Compliance Report",
        )

        def run_image(image, confidence, iou):
            if image is None:
                return None, None, [], "### Result: No image provided"
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
            output_path, report_rows = process_video(url, confidence, iou)
            return None, output_path, report_rows, "### Result: Live camera inspection complete"

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
