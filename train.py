from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 PPE model with class-aware weighting.")
    parser.add_argument("--data", type=Path, default=ROOT / "roboflow" / "data.yaml", help="Path to the dataset YAML file.")
    parser.add_argument("--model", type=Path, default=None, help="Path to a base model or pretrained weights.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training (cpu or cuda).")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "ppe_train", help="Project folder for training outputs.")
    parser.add_argument("--name", type=str, default="v1", help="Training run name.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience in epochs.")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loader workers.")
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, str]:
    try:
        import yaml
    except ImportError:
        lines = {}
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                lines[key.strip()] = value.strip().strip("'\"")
        return lines

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_model_path(model_path: Optional[Path] = None) -> Path:
    if model_path is not None:
        return model_path
    return Path("yolov8m.pt")


def preflight_check(data_yaml_path: Path) -> None:
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")
    dataset = _load_yaml(data_yaml_path)
    for split in ("train", "val"):
        split_path = dataset.get(split)
        if not split_path:
            raise ValueError(f"Dataset YAML is missing the '{split}' entry.")
        split_path = Path(split_path)
        if not split_path.is_absolute():
            split_path = data_yaml_path.parent / split_path
        label_dir = split_path.parent / "labels"
        if not label_dir.exists():
            raise FileNotFoundError(f"Expected label directory not found: {label_dir}")


def count_classes(data_yaml_path: Path) -> int:
    dataset = _load_yaml(data_yaml_path)
    if "nc" in dataset:
        return int(dataset["nc"])
    if "names" in dataset:
        return len(dataset["names"])
    raise ValueError("Unable to determine number of classes from dataset YAML.")


def _collect_label_counts(yaml_path: Path) -> Dict[int, int]:
    data = _load_yaml(yaml_path)
    root = yaml_path.parent
    classes = count_classes(yaml_path)
    counts = Counter()
    for split in ("train", "val"):
        split_path = Path(data[split]) if isinstance(data[split], str) else None
        if split_path is None:
            continue
        if not split_path.is_absolute():
            split_path = root / split_path
        label_dir = split_path.parent / "labels"
        if not label_dir.exists():
            continue
        for label_file in sorted(label_dir.glob("*.txt")):
            with label_file.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    class_index = int(line.split()[0])
                    counts[class_index] += 1
    return {i: counts.get(i, 0) for i in range(classes)}


def _build_class_weights(class_counts: Dict[int, int]) -> List[float]:
    if not class_counts:
        return []

    max_count = max((count for count in class_counts.values() if count > 0), default=1)
    weights = []
    for count in class_counts.values():
        weight = float(max_count / count) if count > 0 else 1.0
        weights.append(weight)

    min_weight = min(weights) if weights else 1.0
    if min_weight > 0:
        weights = [float(w / min_weight) for w in weights]
    return weights


def _save_training_report(output_dir: Path, class_counts: Dict[int, int], class_weights: List[float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "training_baseline.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "class_counts": class_counts,
                "class_weights": class_weights,
            },
            handle,
            indent=2,
        )
    print(f"Saved training report: {report_path}")


def run_training(args: argparse.Namespace) -> int:
    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    try:
        preflight_check(args.data)
    except (FileNotFoundError, ValueError) as error:
        print(str(error))
        return 1

    class_counts = _collect_label_counts(args.data)
    class_weights = _build_class_weights(class_counts)
    print(f"Detected {count_classes(args.data)} classes in dataset metadata.")
    print("Class counts:")
    for idx, count in class_counts.items():
        print(f"  class {idx}: {count}")
    print(f"Computed class weights: {class_weights}")

    if args.model is not None and args.model.exists():
        model_path = args.model
    elif args.model is not None and args.model.name != "yolov8m.pt":
        print(f"Provided model path does not exist: {args.model}")
        return 1
    else:
        model_path = resolve_model_path(args.model)

    print(f"Using model path: {model_path}")
    model = YOLO(str(model_path))
    model.overrides["device"] = device

    print("Starting training with class-aware balancing...")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        workers=args.workers,
    )

    trained_path = "runs/ppe_train/v1/weights/best.pt"
    print("\n[train.py] Training complete.")
    print(f"[train.py] Best model saved to: {trained_path}")
    print("[train.py] Now open config.py and set:")
    print(f"[train.py]   PPE_MODEL_PATH = '{trained_path}'")

    _save_training_report(args.project / args.name, class_counts, class_weights)
    print(f"Training complete. Outputs saved to: {args.project / args.name}")
    return 0


def main() -> int:
    args = _parse_args()
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
