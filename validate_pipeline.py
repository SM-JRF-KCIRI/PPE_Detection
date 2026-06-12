"""Validation utility for PPE detection, matching, and compliance."""

from pathlib import Path

import cv2

from compliance import evaluate_compliance
from config import HELMET_MODEL_PATH, PPE_MODEL_PATH, POSE_MODEL_PATH
from ensemble_detector import PPEEnsembleDetector, print_model_comparison_report
from pose_engine import PoseEngine
from ppe_matcher import assign_ppe_to_person, extract_body_regions
from tracker import ByteTrackTracker
from utils import compute_iou


DEFAULT_IMAGE = Path("D:/newppe/roboflow/test/images/000007_jpg.rf.49b2d719a70bee09ead2b9091ff330c4.jpg")


def main(image_path: str | Path = DEFAULT_IMAGE) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    print_model_comparison_report()
    print(f"PPE Model Loaded: {PPE_MODEL_PATH}")
    print(f"Helmet Model Loaded: {HELMET_MODEL_PATH}")
    print(f"Pose Model Loaded: {POSE_MODEL_PATH}")

    pose_engine = PoseEngine()
    ppe_detector = PPEEnsembleDetector()
    tracker = ByteTrackTracker()

    persons = pose_engine.detect_persons(image, 0.35, 0.45)
    ppe_detections = ppe_detector.detect_ppe(image, 0.25, 0.45)

    track_candidates = tracker.update([
        {"bbox": person["box"], "confidence": person["confidence"]}
        for person in persons
    ], image.shape)

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
        reports.append({
            "person_id": person_id,
            "box": person["box"],
            "confidence": person["confidence"],
            "keypoints": person["keypoints"],
            "regions": extract_body_regions(person["box"], person["keypoints"]),
            "assigned_ppe": [],
        })

    reports = assign_ppe_to_person(reports, ppe_detections)

    print(f"persons={len(persons)} raw_ppe={len(ppe_detections)} matched_ppe={sum(len(r.get('assigned_ppe', [])) for r in reports)}")
    for report in reports:
        compliance = evaluate_compliance(report["person_id"], report)
        print(
            f"person_id={report['person_id']} "
            f"matched={len(report.get('assigned_ppe', []))} "
            f"status={compliance['status']} "
            f"reason={compliance['reason']}"
        )


if __name__ == "__main__":
    main()
