from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 PPE model and save summary metrics.")
    parser.add_argument("--data", type=Path, default=ROOT / "roboflow" / "data.yaml", help="Path to the dataset YAML file.")
    parser.add_argument("--model", type=Path, default=ROOT / "best" / "best.pt", help="Path to the PPE model to evaluate.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for validation.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", type=str, default=None, help="Device to run validation on (cpu or cuda).")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "eval_reports", help="Directory where evaluation reports will be saved.")
    parser.add_argument("--recall-warning", type=float, default=0.70, help="Threshold to flag low average recall.")
    parser.add_argument("--plots", action="store_true", help="Enable YOLO plot generation during validation.")
    return parser.parse_args()


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _find_latest_validation_run() -> Optional[Path]:
    runs_dir = ROOT / "runs"
    if not runs_dir.exists():
        return None

    val_candidates = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("val")]
    if val_candidates:
        return sorted(val_candidates, key=lambda p: p.stat().st_mtime)[-1]

    fallback_dir = runs_dir / "val"
    if fallback_dir.exists() and fallback_dir.is_dir():
        subdirs = [p for p in fallback_dir.iterdir() if p.is_dir()]
        if subdirs:
            return sorted(subdirs, key=lambda p: p.stat().st_mtime)[-1]
    return None


def _build_confusion_matrix(metrics) -> Optional[Dict[str, Any]]:
    stats = getattr(metrics, "stats", {})
    pred_cls = np.asarray(stats.get("pred_cls", []))
    target_cls = np.asarray(stats.get("target_cls", []))
    if pred_cls.size == 0 or target_cls.size == 0:
        return None

    pred_cls = pred_cls.flatten().astype(int)
    target_cls = target_cls.flatten().astype(int)
    size = max(int(pred_cls.max()), int(target_cls.max())) + 1
    matrix = np.zeros((size, size), dtype=int)
    for truth, pred in zip(target_cls, pred_cls):
        matrix[int(truth), int(pred)] += 1

    names = getattr(metrics, "names", {})
    class_names = [names.get(i, str(i)) for i in range(size)]
    matrix_data = {
        "labels": class_names,
        "matrix": matrix.tolist(),
    }
    return matrix_data


def _save_metrics(output_dir: Path, metrics, conf_matrix: Optional[Dict[str, Any]]) -> None:
    output_dir = _ensure_directory(output_dir)
    summary = metrics.summary()
    results_dict = metrics.results_dict

    metrics_path = output_dir / "metrics_summary.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "results": results_dict,
                "class_summary": summary,
                "confusion_matrix": conf_matrix,
            },
            handle,
            indent=2,
        )

    csv_path = output_dir / "class_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        field_names = ["Class", "Images", "Instances", "Box-P", "Box-R", "Box-F1", "mAP50", "mAP50-95"]
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    if conf_matrix is not None:
        json_path = output_dir / "confusion_matrix.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(conf_matrix, handle, indent=2)

    print(f"Saved metrics summary to: {metrics_path}")
    print(f"Saved per-class summary to: {csv_path}")
    if conf_matrix is not None:
        print(f"Saved confusion matrix JSON to: {json_path}")


def _copy_plot_artifacts(output_dir: Path, validation_run: Optional[Path]) -> None:
    if not validation_run or not validation_run.exists():
        return

    for pattern in ["confusion_matrix.png", "confusion_matrix.jpg", "confusion_matrix.jpeg"]:
        source = validation_run / pattern
        if source.exists():
            destination = output_dir / source.name
            destination.write_bytes(source.read_bytes())
            print(f"Copied plot artifact to {destination}")
            return


def run_evaluation(args: argparse.Namespace) -> int:
    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    model_path = args.model
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return 1

    print(f"Evaluating model: {model_path}")
    model = YOLO(str(model_path))
    model.overrides["device"] = device

    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        save=True,
        plots=args.plots,
    )

    conf_matrix = _build_confusion_matrix(metrics)
    report_directory = _ensure_directory(args.output_dir)
    _save_metrics(report_directory, metrics, conf_matrix)

    validation_run = _find_latest_validation_run()
    _copy_plot_artifacts(report_directory, validation_run)

    recall_value = metrics.results_dict.get("metrics/recall(B)")
    if recall_value is not None and recall_value < args.recall_warning:
        print(f"WARNING: Average recall is below threshold ({recall_value:.3f} < {args.recall_warning})")

    low_recall = [row for row in metrics.summary() if row.get("Box-R", 1.0) < args.recall_warning]
    if low_recall:
        print("Low recall by class:")
        for row in low_recall:
            print(f"  {row['Class']}: recall={row['Box-R']:.3f}")

    if metrics.results_dict.get("fitness") is not None:
        print(f"Validation fitness: {metrics.results_dict['fitness']:.4f}")

    return 0


def main() -> int:
    args = _parse_args()
    return run_evaluation(args)


if __name__ == "__main__":
    raise SystemExit(main())
