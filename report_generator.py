import csv
import json
from pathlib import Path
from typing import List


def _format_presence(value) -> str:
    if value is True:
        return "Present"
    if value is False:
        return "Missing"
    return "N/A"


def build_report_rows(person_reports: List[dict]) -> List[List[str]]:
    rows = []
    for report in person_reports:
        rows.append(
            [
                str(report.get("person_id", "-1")),
                _format_presence(report.get("helmet")),
                _format_presence(report.get("vest")),
                _format_presence(report.get("hook")),
                _format_presence(report.get("glove")),
                _format_presence(report.get("boot")),
                _format_presence(report.get("goggles")),
                report.get("status", "UNKNOWN"),
                report.get("reason", "Matched PPE evaluated."),
            ]
        )
    return rows


def generate_compliance_report(person_reports: List[dict]) -> List[List[str]]:
    return build_report_rows(person_reports)


def export_report_csv(person_reports: List[dict], output_path: str | Path) -> str:
    rows = build_report_rows(person_reports)
    output = Path(output_path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Track ID", "Helmet", "Vest", "Hook", "Glove", "Boot", "Goggles", "Status", "Reason"])
        writer.writerows(rows)
    return str(output)


def export_report_json(person_reports: List[dict], output_path: str | Path) -> str:
    output = Path(output_path)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(person_reports, handle, indent=2)
    return str(output)
