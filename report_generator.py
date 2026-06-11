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
                _format_presence(report.get("shoe")),
                _format_presence(report.get("goggles")),
                report.get("status", "NON-COMPLIANT"),
            ]
        )
    return rows


def generate_compliance_report(person_reports: List[dict]) -> List[List[str]]:
    return build_report_rows(person_reports)
