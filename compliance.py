from typing import Dict

from config import CLASS_NORMALIZATION, OPTIONAL_PPE_LABELS, PPE_CLASS_LABELS, REQUIRED_PPE_LABELS


def _canonical_label(label: str) -> str:
    raw = str(label).strip().lower().replace("_", " ").replace("-", " ")
    if raw.endswith("s") and raw not in {"glass", "dress"}:
        raw = raw[:-1]
    return CLASS_NORMALIZATION.get(raw, raw)


def evaluate_compliance(person_id: int, report: Dict) -> Dict:
    assigned = {label: False for label in PPE_CLASS_LABELS}
    matched = []

    for detection in report.get("assigned_ppe", []):
        label = _canonical_label(detection.get("label", ""))
        if label in assigned and not assigned[label]:
            assigned[label] = True
            matched.append(label)

    missing = [label for label in REQUIRED_PPE_LABELS if not assigned.get(label, False)]
    has_required = any(assigned.get(label, False) for label in REQUIRED_PPE_LABELS)

    if not matched:
        status = "UNKNOWN"
        reason = "No PPE matched to this worker."
    elif not missing:
        status = "COMPLIANT"
        reason = "All required PPE items are present."
    elif not has_required:
        status = "INTRUDER"
        reason = "No required PPE matched to this worker."
    else:
        status = "NON-COMPLIANT"
        reason = "Missing: " + ", ".join(label.title() for label in missing)

    if assigned.get("helmet", False) and not assigned.get("vest", False):
        status = "NON-COMPLIANT"
        reason = "Missing Vest"

    return {
        "person_id": person_id,
        "helmet": assigned.get("helmet", False),
        "vest": assigned.get("vest", False),
        "glove": assigned.get("glove", False),
        "boot": assigned.get("boot", False),
        "hook": assigned.get("hook", False),
        "goggles": assigned.get("goggles", False),
        "status": status,
        "reason": reason,
        "matched_ppe": matched,
    }
