from pathlib import Path
from typing import Dict, List

import sys
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_NORMALIZATION, PPE_CONF_THRESHOLD, PPE_MODEL_PATH, PPE_CLASS_LABELS


class PPEDetector:
    def __init__(self, model_path: Path = PPE_MODEL_PATH, device: str = "cpu"):
        self.model = YOLO(str(model_path))
        self.model.overrides["device"] = device

        # Build normalized class map from model's actual class names
        self.class_names = {}  # raw_name → normalized_name
        self.ppe_class_ids = []  # model class IDs that are PPE classes

        for class_id, raw_name in self.model.names.items():
            raw_lower = str(raw_name).lower().strip()
            normalized = (
                CLASS_NORMALIZATION.get(raw_name) or
                CLASS_NORMALIZATION.get(raw_lower) or
                None
            )
            if normalized and normalized in PPE_CLASS_LABELS:
                self.class_names[raw_name] = normalized
                self.ppe_class_ids.append(class_id)

        print(f"[PPEDetector] Model: {PPE_MODEL_PATH}")
        print(f"[PPEDetector] Model has {len(self.model.names)} classes")
        print(f"[PPEDetector] Matched PPE classes: {self.class_names}")
        print(f"[PPEDetector] PPE class IDs: {self.ppe_class_ids}")

        if not self.class_names:
            print(f"[PPEDetector] WARNING: No PPE classes matched in this model.")
            print(f"[PPEDetector] Model classes are: {list(self.model.names.values())[:10]}...")
            print(f"[PPEDetector] Expected classes: {list(CLASS_NORMALIZATION.keys())[:10]}...")
            print(f"[PPEDetector] Training has not completed — run train.py first.")

    @staticmethod
    def _normalize_label(label: str) -> str:
        normalized = label.strip().lower()
        if normalized.endswith("s") and normalized not in {"glass", "dress"}:
            normalized = normalized[:-1]
        normalized = CLASS_NORMALIZATION.get(normalized, normalized)
        aliases = {
            "person": "person",
        }
        return aliases.get(normalized, normalized)

    def _load_names(self) -> Dict[int, str]:
        try:
            raw_names = {int(k): str(v) for k, v in self.model.names.items()}
        except Exception:
            raw_names = {0: "person", 1: "helmet", 2: "vest", 3: "glove", 4: "hook", 5: "shoe"}

        normalized_names = {}
        for class_id, label in raw_names.items():
            normalized_names[class_id] = self._normalize_label(label)
        return normalized_names

    def detect_ppe(self, frame, conf_threshold: float, iou_threshold: float) -> List[Dict]:
        results = self.model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        if len(results) == 0:
            return []

        result = results[0]
        detections: List[Dict] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            raw_name = self.model.names[class_id]

            normalized = self.class_names.get(raw_name)
            if normalized is None:
                continue
            if confidence < PPE_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": class_id,
                "class_name": raw_name,
                "label": raw_name,
                "normalized_name": normalized,
                "confidence": confidence,
            })
        return detections
