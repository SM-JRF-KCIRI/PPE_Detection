"""Debug utility for raw detections, matched PPE, and compliance summaries."""

from pathlib import Path

import cv2

from app import process_image
from compliance import evaluate_compliance
from detector import PPEDetector
from pose_engine import PoseEngine
from ppe_matcher import assign_ppe_to_person, extract_body_regions
from tracker import ByteTrackTracker
from utils import compute_iou


def main(image_path: str = r"D:\newppe\roboflow\test\images\000007_jpg.rf.49b2d719a70bee09ead2b9091ff330c4.jpg") -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    pose_engine = PoseEngine()
    ppe_detector = PPEDetector()
    tracker = ByteTrackTracker()

    persons = pose_engine.detect_persons(image, 0.35, 0.45)
    ppe_detections = ppe_detector.detect_ppe(image, 0.25, 0.45)

    track_candidates = tracker.update([{"bbox": person["box"], "confidence": person["confidence"]} for person in persons], image.shape)
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
        reports.append({"person_id": person_id, "box": person["box"], "keypoints": person["keypoints"], "regions": extract_body_regions(person["box"], person["keypoints"]), "assigned_ppe": []})

    reports = assign_ppe_to_person(reports, ppe_detections)

    print("RAW_DETECTIONS")
    for detection in ppe_detections:
        print("  ", detection)

    print("MATCHED_DETECTIONS")
    for report in reports:
        print("  person_id=", report["person_id"], "matched=", report.get("assigned_ppe"))

    print("COMPLIANCE_RESULTS")
    for report in reports:
        compliance = evaluate_compliance(report["person_id"], report)
        print("  person_id=", report["person_id"], "->", compliance)


if __name__ == "__main__":
    main()
