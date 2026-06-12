from typing import List

import cv2
import numpy as np

from utils import center_of_box, draw_box, draw_label, safe_int

PERSON_COMPLIANT_COLOR = (0, 200, 0)
PERSON_NON_COMPLIANT_COLOR = (0, 0, 220)
PERSON_INTRUDER_COLOR = (0, 0, 255)
PPE_COLOR = (0, 200, 0)

def _format_presence(value) -> str:
    if value is True:
        return "Present"
    if value is False:
        return "Missing"
    return "N/A"


def draw_annotations(frame, person_reports: List[dict], ppe_detections: List[dict], debug: bool = False):
    annotated = frame.copy()
    for report in person_reports:
        status = report.get("status", "UNKNOWN")
        if status == "COMPLIANT":
            color = PERSON_COMPLIANT_COLOR
        elif status == "INTRUDER":
            color = PERSON_INTRUDER_COLOR
        else:
            color = PERSON_NON_COMPLIANT_COLOR
        draw_box(annotated, report["box"], color, thickness=2)
        if debug:
            for region_name, region_box in (report.get("regions") or {}).items():
                if region_box is None:
                    continue
                if isinstance(region_box[0], tuple):
                    xs = [int(v[0]) for v in region_box]
                    ys = [int(v[1]) for v in region_box]
                    pts = np.array([[x, y] for x, y in zip(xs, ys)], dtype=np.int32)
                    cv2.polylines(annotated, [pts], True, (255, 255, 0), 1)
                else:
                    draw_box(annotated, region_box, (255, 255, 0), thickness=1)
            for detection in report.get("assigned_ppe", []):
                ppe_center = center_of_box(detection["bbox"])
                person_center = center_of_box(report["box"])
                cv2.line(annotated, (int(ppe_center[0]), int(ppe_center[1])), (int(person_center[0]), int(person_center[1])), (255, 255, 0), 1)

        x1, y1, _, _ = [safe_int(v) for v in report["box"]]
        label_lines = [
            f"ID: {report.get('person_id', '-1')}",
            f"Helmet: {_format_presence(report.get('helmet'))}",
            f"Vest: {_format_presence(report.get('vest'))}",
            f"Glove: {_format_presence(report.get('glove'))}",
            f"Boot: {_format_presence(report.get('boot'))}",
            f"Goggles: {_format_presence(report.get('goggles'))}",
            f"Status: {report.get('status', 'UNKNOWN')}",
            f"Reason: {report.get('reason', 'Matched PPE evaluated.')}",
        ]
        y_offset = y1 - 8
        for label_text in label_lines:
            draw_label(annotated, label_text, (x1, y_offset), color)
            y_offset -= 18

    for detection in ppe_detections:
        label = detection.get("label", "unknown").title()
        confidence = detection.get("confidence", 0.0)
        draw_box(annotated, detection["bbox"], PPE_COLOR, thickness=2)
        draw_label(annotated, f"{label} {confidence:.2f}", (detection["bbox"][0], detection["bbox"][1] - 10), PPE_COLOR)

    return annotated
