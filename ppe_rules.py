from typing import Dict, List, Optional, Tuple

import numpy as np

from config import PPE_CLASS_LABELS
from utils import center_of_box, compute_iou

COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def _region_from_points(points: List[Tuple[float, float]], expand: float = 0.25):
    if not points:
        return None
    xs, ys = zip(*points)
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    width = x2 - x1
    height = y2 - y1
    x1 -= width * expand
    y1 -= height * expand
    x2 += width * expand
    y2 += height * expand
    return [x1, y1, x2, y2]


def _keypoint_box(keypoints, names):
    if keypoints is None:
        return None
    keypoints = np.asarray(getattr(keypoints, "data", keypoints))
    if keypoints.ndim == 4 and keypoints.shape[1] == 1:
        keypoints = keypoints[:, 0]
    if keypoints.ndim == 3 and keypoints.shape[0] == 1:
        keypoints = keypoints[0]
    pts = []
    for name in names:
        index = COCO_KEYPOINTS.get(name)
        if index is None or keypoints.shape[0] <= index:
            continue
        x, y, conf = keypoints[index]
        if conf > 0.2:
            pts.append((float(x), float(y)))
    return _region_from_points(pts)


def extract_body_regions(person_box: List[float], keypoints: Optional[np.ndarray]) -> Dict[str, List[float]]:
    head = None
    chest = None
    left_hand = None
    right_hand = None
    leg = None
    foot = None
    belt = None
    if keypoints is not None:
        head = _keypoint_box(keypoints, ["nose", "left_eye", "right_eye", "left_ear", "right_ear"])
        chest = _keypoint_box(keypoints, ["left_shoulder", "right_shoulder", "left_hip", "right_hip"])
        left_hand = _keypoint_box(keypoints, ["left_wrist", "left_elbow"])
        right_hand = _keypoint_box(keypoints, ["right_wrist", "right_elbow"])
        leg = _keypoint_box(keypoints, ["left_hip", "right_hip", "left_knee", "right_knee"])
        foot = _keypoint_box(keypoints, ["left_knee", "right_knee", "left_ankle", "right_ankle"])
        belt = _keypoint_box(keypoints, ["left_hip", "right_hip"])

    x1, y1, x2, y2 = person_box
    width = x2 - x1
    height = y2 - y1
    if head is None:
        head = [x1 + width * 0.15, y1, x2 - width * 0.15, y1 + height * 0.18]
    if chest is None:
        chest = [x1 + width * 0.15, y1 + height * 0.18, x2 - width * 0.15, y1 + height * 0.45]
    if left_hand is None:
        left_hand = [x1, y1 + height * 0.25, x1 + width * 0.28, y1 + height * 0.65]
    if right_hand is None:
        right_hand = [x2 - width * 0.28, y1 + height * 0.25, x2, y1 + height * 0.65]
    if leg is None:
        leg = [x1 + width * 0.12, y1 + height * 0.45, x2 - width * 0.12, y1 + height * 0.78]
    if foot is None:
        foot = [x1 + width * 0.15, y1 + height * 0.78, x2 - width * 0.15, y2]
    if belt is None:
        belt = [x1 + width * 0.18, y1 + height * 0.38, x2 - width * 0.18, y1 + height * 0.55]

    def normalize_box(box):
        if box is None:
            return None
        x1, y1, x2, y2 = box
        x1, x2 = sorted((max(0.0, x1), max(0.0, x2)))
        y1, y2 = sorted((max(0.0, y1), max(0.0, y2)))
        return [float(x1), float(y1), float(x2), float(y2)]

    return {
        "head": normalize_box(head),
        "chest": normalize_box(chest),
        "left_hand": normalize_box(left_hand),
        "right_hand": normalize_box(right_hand),
        "leg": normalize_box(leg),
        "foot": normalize_box(foot),
        "belt": normalize_box(belt),
    }


def _normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized.endswith("s") and normalized not in {"glass", "dress"}:
        normalized = normalized[:-1]
    aliases = {
        "boots": "shoe",
        "boot": "shoe",
        "gloves": "glove",
        "glove": "glove",
        "hardhat": "helmet",
        "safetyhelmet": "helmet",
        "safety helmet": "helmet",
        "safety hook": "hook",
        "hook": "hook",
        "vest": "vest",
        "helmet": "helmet",
        "shoe": "shoe",
    }
    return aliases.get(normalized, normalized)


def _matches_region(ppe_box, region_box):
    if region_box is None:
        return False
    if isinstance(region_box[0], list):
        return any(_matches_region(ppe_box, item) for item in region_box)
    center = center_of_box(ppe_box)
    if region_box[0] <= center[0] <= region_box[2] and region_box[1] <= center[1] <= region_box[3]:
        return True
    return compute_iou(ppe_box, region_box) >= 0.08


def _allowed_regions_for_label(label: str, regions: Dict[str, List[float]]):
    if label == "helmet":
        return [regions.get("head")]
    if label == "vest":
        return [regions.get("chest")]
    if label == "glove":
        return [regions.get("left_hand"), regions.get("right_hand")]
    if label == "hook":
        return [regions.get("belt"), regions.get("chest")]
    if label == "shoe":
        return [regions.get("foot")]
    return []


def assign_ppe_to_person(person_reports: List[Dict], detections: List[Dict]) -> List[Dict]:
    for report in person_reports:
        report["assigned_ppe"] = []
    for detection in detections:
        label = _normalize_label(detection.get("label", ""))
        if label not in PPE_CLASS_LABELS:
            continue
        ppe_center = center_of_box(detection["bbox"])
        best_person = None
        best_distance = float("inf")
        for report in person_reports:
            regions = report.get("regions") or {}
            for region in _allowed_regions_for_label(label, regions):
                if _matches_region(detection["bbox"], region):
                    person_center = center_of_box(report["person_box"])
                    dx = ppe_center[0] - person_center[0]
                    dy = ppe_center[1] - person_center[1]
                    distance = dx * dx + dy * dy
                    if distance < best_distance:
                        best_distance = distance
                        best_person = report
                    break
        if best_person is not None:
            best_person["assigned_ppe"].append({**detection, "label": label})
    return person_reports


def evaluate_compliance(person_id: int, report: Dict) -> Dict:
    assigned = {label: False for label in PPE_CLASS_LABELS}
    assigned_ppe = report.get("assigned_ppe", [])
    regions = report.get("regions", {})
    for detection in assigned_ppe:
        label = _normalize_label(detection.get("label", ""))
        if label not in assigned:
            continue
        if any(_matches_region(detection["bbox"], region) for region in _allowed_regions_for_label(label, regions)):
            assigned[label] = True
    status = "COMPLIANT" if all(assigned.values()) else "NON-COMPLIANT"
    return {
        "person_id": person_id,
        "helmet": assigned["helmet"],
        "vest": assigned["vest"],
        "glove": assigned["glove"],
        "hook": assigned["hook"],
        "shoe": assigned["shoe"],
        "status": status,
        "regions": regions,
        "assigned_ppe": assigned_ppe,
    }
