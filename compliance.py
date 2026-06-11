from typing import Dict

from config import OPTIONAL_PPE_LABELS, PPE_CLASS_LABELS, REQUIRED_PPE_LABELS


def evaluate_compliance(person_id: int, report: Dict) -> Dict:
    assigned = {
        label: None if label in OPTIONAL_PPE_LABELS else False for label in PPE_CLASS_LABELS
    }
    for detection in report.get("assigned_ppe", []):
        label = detection.get("label")
        if label in assigned:
            assigned[label] = True

    has_required = any(assigned[label] for label in REQUIRED_PPE_LABELS)
    missing_required = any(not assigned[label] for label in REQUIRED_PPE_LABELS)
    if has_required and not missing_required:
        status = "COMPLIANT"
    elif not has_required:
        status = "INTRUDER"
    else:
        status = "VIOLATION"

    return {
        "person_id": person_id,
        "helmet": assigned["helmet"],
        "vest": assigned["vest"],
        "glove": assigned["glove"],
        "hook": assigned["hook"],
        "shoe": assigned["shoe"],
        "goggles": assigned.get("goggles"),
        "status": status,
    }
