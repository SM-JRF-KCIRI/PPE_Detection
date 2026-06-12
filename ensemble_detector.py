from pathlib import Path
from typing import Dict, List

from ultralytics import YOLO

from config import CLASS_NORMALIZATION, PPE_CLASS_LABELS
from utils import compute_iou


MODEL_SPECS = [
    ("best/best.pt", "helmet_specialist", {"helmet": 1.20, "vest": 0.35, "glove": 0.20, "boot": 0.20, "goggle": 0.25, "person": 0.15}),
    ("best/best_1.pt", "ensemble_a", {"helmet": 1.05, "vest": 1.15, "glove": 1.00, "boot": 0.90, "goggle": 0.95, "person": 0.85}),
    ("best/best_2.pt", "ensemble_b", {"helmet": 1.05, "vest": 1.10, "glove": 0.95, "boot": 0.95, "goggle": 0.90, "person": 0.85}),
    ("runs/ppe_train/v1/weights/best.pt", "trained_ppe", {"helmet": 1.00, "vest": 1.00, "glove": 1.00, "boot": 1.00, "goggle": 1.00, "person": 1.00}),
]

CLASS_THRESHOLDS = {
    "helmet": 0.20,
    "vest": 0.25,
    "glove": 0.20,
    "boot": 0.20,
    "goggle": 0.20,
    "person": 0.15,
}


class PPEEnsembleDetector:
    def __init__(self, device: str = "cpu"):
        self.base_dir = Path(__file__).resolve().parent
        self.models = []
        for rel_path, tag, weights in MODEL_SPECS:
            model_path = (self.base_dir / rel_path).resolve()
            if not model_path.exists():
                print(f"[ensemble] WARNING: model not found -> {model_path}")
                continue
            model = YOLO(str(model_path))
            model.overrides["device"] = device
            self.models.append({"path": model_path, "tag": tag, "weights": weights, "model": model})

        if not self.models:
            raise FileNotFoundError("No PPE ensemble models were found.")

    @staticmethod
    def _normalize_label(label: str) -> str:
        raw = str(label).strip().lower().replace("_", " ").replace("-", " ")
        raw = " ".join(raw.split())
        if raw.endswith("s") and raw not in {"glass", "dress"}:
            raw = raw[:-1]
        if "with helmet" in raw:
            return "helmet"
        if "without helmet" in raw or "no helmet" in raw:
            return "ignore"
        if "helmet" in raw or "hardhat" in raw:
            return "helmet"
        if "vest" in raw:
            return "vest"
        if "glove" in raw or "gloves" in raw:
            return "glove"
        if "boot" in raw or "boots" in raw or "shoe" in raw or "shoes" in raw:
            return "boot"
        if "goggle" in raw or "goggles" in raw:
            return "goggle"
        if "human" in raw or "person" in raw:
            return "person"
        return CLASS_NORMALIZATION.get(raw, raw)

    @staticmethod
    def _box_iou(box_a, box_b) -> float:
        return compute_iou(box_a, box_b)

    def _fuse_detections(self, detections: List[Dict]) -> List[Dict]:
        fused = []
        by_class = {}
        for detection in detections:
            label = detection.get("normalized_name")
            if label not in PPE_CLASS_LABELS:
                continue
            by_class.setdefault(label, []).append(detection)

        for label, items in by_class.items():
            items = sorted(items, key=lambda x: x["confidence"], reverse=True)
            keep = []
            for item in items:
                merged = False
                for candidate in keep:
                    if self._box_iou(item["bbox"], candidate["bbox"]) >= 0.15:
                        candidate["confidence"] = max(candidate["confidence"], item["confidence"])
                        candidate["bbox"] = [
                            (candidate["bbox"][0] * candidate["confidence"] + item["bbox"][0] * item["confidence"]) / max(candidate["confidence"] + item["confidence"], 1e-6),
                            (candidate["bbox"][1] * candidate["confidence"] + item["bbox"][1] * item["confidence"]) / max(candidate["confidence"] + item["confidence"], 1e-6),
                            (candidate["bbox"][2] * candidate["confidence"] + item["bbox"][2] * item["confidence"]) / max(candidate["confidence"] + item["confidence"], 1e-6),
                            (candidate["bbox"][3] * candidate["confidence"] + item["bbox"][3] * item["confidence"]) / max(candidate["confidence"] + item["confidence"], 1e-6),
                        ]
                        candidate["support"] += 1
                        candidate["sources"].add(item.get("source", "model"))
                        merged = True
                        break
                if not merged:
                    keep.append({**item, "support": 1, "sources": {item.get("source", "model")}})
            fused.extend(keep)

        fused.sort(key=lambda x: x["confidence"], reverse=True)
        return fused

    def detect_ppe(self, frame, conf_threshold: float = 0.20, iou_threshold: float = 0.45) -> List[Dict]:
        detections = []
        for entry in self.models:
            model = entry["model"]
            results = model(frame, conf=max(0.05, conf_threshold), iou=iou_threshold, verbose=False)
            if not results or results[0].boxes is None:
                continue
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                raw_name = model.names.get(class_id, str(class_id))
                normalized = self._normalize_label(raw_name)
                if normalized in {"ignore", "person"} and normalized != "person":
                    continue
                if normalized not in PPE_CLASS_LABELS:
                    continue
                class_conf = max(confidence, CLASS_THRESHOLDS.get(normalized, conf_threshold))
                x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy().tolist())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "class_name": raw_name,
                    "label": normalized,
                    "normalized_name": normalized,
                    "confidence": class_conf * entry["weights"].get(normalized, 1.0),
                    "source": entry["tag"],
                    "model_path": str(entry["path"]),
                })

        fused = self._fuse_detections(detections)
        return fused


def print_model_comparison_report() -> None:
    detector = PPEEnsembleDetector(device="cpu")
    print("\n=== PPE MODEL COMPARISON REPORT ===")
    for entry in detector.models:
        model = entry["model"]
        names = list(model.names.values())
        print(f"MODEL {entry['path']} -> {names}")
    print("\nSupported normalized classes: helmet, vest, glove, boot, goggle, person")
    print("Specialization hints: best/best.pt = helmet-focused, best_1/best_2 = helmet/vest, runs/ppe_train/v1/weights/best.pt = PPE classes")
