from typing import List

import cv2

from utils import draw_box, draw_label, safe_int

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


def draw_annotations(frame, person_reports: List[dict], ppe_detections: List[dict]):
    annotated = frame
    for report in person_reports:
        status = report.get("status", "VIOLATION")
        if status == "COMPLIANT":
            color = PERSON_COMPLIANT_COLOR
        elif status == "INTRUDER":
            color = PERSON_INTRUDER_COLOR
        else:
            color = PERSON_NON_COMPLIANT_COLOR
        draw_box(annotated, report["box"], color, thickness=2)

        x1, y1, _, _ = [safe_int(v) for v in report["box"]]
        label_lines = [
            f"ID: {report.get('person_id', '-1')}",
            f"Helmet: {_format_presence(report.get('helmet'))}",
            f"Vest: {_format_presence(report.get('vest'))}",
            f"Hook: {_format_presence(report.get('hook'))}",
            f"Glove: {_format_presence(report.get('glove'))}",
            f"Shoe: {_format_presence(report.get('shoe'))}",
            f"Goggles: {_format_presence(report.get('goggles'))}",
            f"Compliance: {report.get('status', 'NON-COMPLIANT')}",
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
