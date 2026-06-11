from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from config import KEYPOINT_CONFIDENCE_THRESHOLD, PPE_CLASS_LABELS
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

Region = Union[List[Tuple[float, float]], List[float]]


def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside


def _polygon_to_box(polygon: List[Tuple[float, float]]) -> List[float]:
    xs, ys = zip(*polygon)
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _normalize_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [(float(x), float(y)) for x, y in points]


def _polygon_from_keypoints(keypoints: np.ndarray, names: List[str]) -> Optional[List[Tuple[float, float]]]:
    if keypoints is None:
        return None
    points = []
    for name in names:
        index = COCO_KEYPOINTS.get(name)
        if index is None or index >= keypoints.shape[0]:
            continue
        x, y, confidence = keypoints[index]
        if float(confidence) >= KEYPOINT_CONFIDENCE_THRESHOLD:
            points.append((float(x), float(y)))
    if len(points) < 2:
        return None
    return _normalize_points(points)


def _region_from_box(points: List[Tuple[float, float]], expand: float = 0.18) -> List[float]:
    xs, ys = zip(*points)
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    x1 -= width * expand
    y1 -= height * expand
    x2 += width * expand
    y2 += height * expand
    return [float(x1), float(y1), float(x2), float(y2)]


def _fallback_region(box: List[float], x_scale: float, y_scale: float, x_offset: float = 0.0, y_offset: float = 0.0) -> List[float]:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return [
        float(x1 + width * x_offset),
        float(y1 + height * y_offset),
        float(x1 + width * (x_offset + x_scale)),
        float(y1 + height * (y_offset + y_scale)),
    ]


def extract_body_regions(person_box: List[float], keypoints: Optional[np.ndarray]) -> Dict[str, Region]:
    head_region = None
    chest_region = None
    left_hand_region = None
    right_hand_region = None
    leg_region = None
    foot_region = None
    belt_region = None

    if keypoints is not None:
        head_region = _polygon_from_keypoints(keypoints, ["nose", "left_eye", "right_eye", "left_ear", "right_ear"])
        chest_region = _polygon_from_keypoints(keypoints, ["left_shoulder", "right_shoulder", "left_hip", "right_hip"])
        left_hand_region = _polygon_from_keypoints(keypoints, ["left_elbow", "left_wrist"])
        right_hand_region = _polygon_from_keypoints(keypoints, ["right_elbow", "right_wrist"])
        leg_region = _polygon_from_keypoints(keypoints, ["left_hip", "right_hip", "left_knee", "right_knee"])
        foot_region = _polygon_from_keypoints(keypoints, ["left_knee", "right_knee", "left_ankle", "right_ankle"])
        belt_region = _polygon_from_keypoints(keypoints, ["left_hip", "right_hip"])

    if head_region is None:
        head_region = _fallback_region(person_box, x_scale=0.45, y_scale=0.16, x_offset=0.275, y_offset=0.0)
    if chest_region is None:
        chest_region = _fallback_region(person_box, x_scale=0.70, y_scale=0.30, x_offset=0.15, y_offset=0.18)
    if left_hand_region is None:
        left_hand_region = _fallback_region(person_box, x_scale=0.33, y_scale=0.35, x_offset=0.0, y_offset=0.25)
    if right_hand_region is None:
        right_hand_region = _fallback_region(person_box, x_scale=0.33, y_scale=0.35, x_offset=0.67, y_offset=0.25)
    if leg_region is None:
        leg_region = _fallback_region(person_box, x_scale=0.64, y_scale=0.33, x_offset=0.18, y_offset=0.45)
    if foot_region is None:
        foot_region = _fallback_region(person_box, x_scale=0.70, y_scale=0.22, x_offset=0.15, y_offset=0.78)
    if belt_region is None:
        belt_region = _fallback_region(person_box, x_scale=0.38, y_scale=0.14, x_offset=0.31, y_offset=0.42)

    return {
        "head": head_region,
        "chest": chest_region,
        "left_hand": left_hand_region,
        "right_hand": right_hand_region,
        "leg": leg_region,
        "foot": foot_region,
        "belt": belt_region,
    }


def _normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized.endswith("s") and normalized not in {"glass", "dress"}:
        normalized = normalized[:-1]
    aliases = {
        "hardhat": "helmet",
        "safety helmet": "helmet",
        "safetyhelmet": "helmet",
        "safety vest": "vest",
        "safetyvest": "vest",
        "hi vis": "vest",
        "hi-vis": "vest",
        "workboot": "shoe",
        "boot": "shoe",
        "boots": "shoe",
        "gloves": "glove",
        "hook": "hook",
        "safety hook": "hook",
        "person": "person",
    }
    return aliases.get(normalized, normalized)


def _region_contains_center(ppe_box: List[float], region: Region) -> bool:
    center = center_of_box(ppe_box)
    if region is None:
        return False
    if isinstance(region[0], tuple):
        return _point_in_polygon(center, region)
    return region[0] <= center[0] <= region[2] and region[1] <= center[1] <= region[3]


def _region_overlaps_box(ppe_box: List[float], region: Region) -> bool:
    if region is None:
        return False
    if isinstance(region[0], tuple):
        region = _polygon_to_box(region)
    return compute_iou(ppe_box, region) > 0.0


def _allowed_regions(label: str, regions: Dict[str, Region]) -> List[Region]:
    if label == "helmet":
        return [regions.get("head")]
    if label == "vest":
        return [regions.get("chest")]
    if label == "glove":
        return [regions.get("left_hand"), regions.get("right_hand")]
    if label == "hook":
        return [regions.get("belt"), regions.get("chest")]
    if label == "shoe":
        return [regions.get("leg"), regions.get("foot")]
    if label == "goggles":
        return [regions.get("head")]
    return []


def assign_ppe_to_person(person_reports: List[Dict], detections: List[Dict]) -> List[Dict]:
    for report in person_reports:
        report["assigned_ppe"] = []

    for detection in detections:
        normalized_label = detection.get("normalized_name", "")
        if normalized_label not in PPE_CLASS_LABELS:
            continue

        ppe_center = center_of_box(detection["bbox"])
        best_person = None
        best_distance = float("inf")

        for report in person_reports:
            regions = report.get("regions", {})
            for region in _allowed_regions(normalized_label, regions):
                if region is None:
                    continue
                if normalized_label == "vest":
                    if not _region_overlaps_box(detection["bbox"], region):
                        continue
                else:
                    if not _region_contains_center(detection["bbox"], region):
                        continue
                person_center = center_of_box(report["box"])
                dx = ppe_center[0] - person_center[0]
                dy = ppe_center[1] - person_center[1]
                distance = dx * dx + dy * dy
                if distance < best_distance:
                    best_distance = distance
                    best_person = report
                break

        if best_person is not None:
            best_person["assigned_ppe"].append({**detection, "label": normalized_label})
    return person_reports
