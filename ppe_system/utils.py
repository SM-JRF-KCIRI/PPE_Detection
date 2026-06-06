import math
import time
from collections import deque
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np


def is_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter_area = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return inter_area / max(area_a + area_b - inter_area, 1e-8)


def clamp_box(box, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    return [
        float(max(0, min(w - 1, x1))),
        float(max(0, min(h - 1, y1))),
        float(max(0, min(w - 1, x2))),
        float(max(0, min(h - 1, y2))),
    ]


def safe_int(value):
    try:
        return int(round(value))
    except Exception:
        return 0


def center_of_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def draw_label(image, text, tl, color, scale=0.6, thickness=1):
    x, y = safe_int(tl[0]), safe_int(tl[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x, y - h - 6), (x + w + 6, y), color, -1)
    cv2.putText(image, text, (x + 3, y - 5), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_box(image, box, color, label=None, thickness=2):
    x1, y1, x2, y2 = [safe_int(v) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label:
        draw_label(image, label, (x1, y1), color)


def render_status_overlay(image, fps: float, status_text: str):
    overlay = image.copy()
    h, w = image.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, 72), (22, 22, 22), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"PPE Inspection | FPS: {fps:.1f} | {timestamp}"
    cv2.putText(image, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, status_text, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return image


def build_report_rows(person_reports):
    rows = []
    for report in person_reports:
        rows.append(
            {
                "Person ID": report["person_id"],
                "Helmet": "YES" if report["helmet"] else "NO",
                "Vest": "YES" if report["vest"] else "NO",
                "Gloves": "YES" if report["gloves"] else "NO",
                "Boots": "YES" if report["boots"] else "NO",
                "Compliance": report["status"],
            }
        )
    return rows


def timeline_cache(maxlen=5):
    return deque(maxlen=maxlen)
