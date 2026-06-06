from typing import Dict, List, Optional, Tuple

import numpy as np

from config import PPE_CLASS_LABELS, PPE_REGION_MAP
from utils import clamp_box, center_of_box, compute_iou

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
    feet = None
    if keypoints is not None:
        head = _keypoint_box(keypoints, ["nose", "left_eye", "right_eye", "left_ear", "right_ear"])
        chest = _keypoint_box(keypoints, ["left_shoulder", "right_shoulder", "left_hip", "right_hip"])
        left_hand = _keypoint_box(keypoints, ["left_wrist", "left_elbow"])
        right_hand = _keypoint_box(keypoints, ["right_wrist", "right_elbow"])
        feet = _keypoint_box(keypoints, ["left_ankle", "right_ankle"])
    if head is None or chest is None or left_hand is None or right_hand is None or feet is None:
        x1, y1, x2, y2 = person_box
        width = x2 - x1
        height = y2 - y1
        if head is None:
            head = [x1, y1, x2, y1 + height * 0.2]
        if chest is None:
            chest = [x1 + width * 0.15, y1 + height * 0.2, x2 - width * 0.15, y1 + height * 0.5]
        if left_hand is None:
            left_hand = [x1, y1 + height * 0.3, x1 + width * 0.25, y1 + height * 0.7]
        if right_hand is None:
            right_hand = [x2 - width * 0.25, y1 + height * 0.3, x2, y1 + height * 0.7]
        if feet is None:
            feet = [x1 + width * 0.1, y1 + height * 0.7, x2 - width * 0.1, y2]
    def normalize_box(box):
        if box is None:
            return None
        x1, y1, x2, y2 = box
        x1, x2 = sorted((max(0.0, x1), max(0.0, x2)))
        y1, y2 = sorted((max(0.0, y1), max(0.0, y2)))
        return [float(x1), float(y1), float(x2), float(y2)]

    head = normalize_box(head)
    chest = normalize_box(chest)
    left_hand = normalize_box(left_hand)
    right_hand = normalize_box(right_hand)
    feet = normalize_box(feet)
    return {
        "head": head,
        "chest": chest,
        "hands": [left_hand, right_hand],
        "feet": feet,
    }


def _matches_region(ppe_box, region_box):
    if region_box is None:
        return False
    if isinstance(region_box[0], list):
        return any(_matches_region(ppe_box, item) for item in region_box)
    center = center_of_box(ppe_box)
    if region_box[0] <= center[0] <= region_box[2] and region_box[1] <= center[1] <= region_box[3]:
        return True
    return compute_iou(ppe_box, region_box) >= 0.08


def evaluate_ppe(person_id: int, detections: List[Dict], pose_keypoints: Optional[np.ndarray], person_box: List[float]) -> Dict:
    regions = extract_body_regions(person_box, pose_keypoints)
    assigned = {label: False for label in PPE_CLASS_LABELS}
    for detection in detections:
        label = detection["label"].lower()
        if label not in assigned:
            continue
        target_region = PPE_REGION_MAP[label]
        region_box = regions[target_region]
        if _matches_region(detection["bbox"], region_box):
            assigned[label] = True
    return {
        "person_id": person_id,
        "helmet": assigned["helmet"],
        "vest": assigned["vest"],
        "gloves": assigned["gloves"],
        "boots": assigned["boots"],
        "status": "GREEN" if all(assigned.values()) else "RED",
        "regions": regions,
    }
