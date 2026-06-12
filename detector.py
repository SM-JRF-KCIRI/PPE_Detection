from pathlib import Path
from typing import Dict, List

import sys

from ultralytics import YOLO

from utils import compute_iou

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_NORMALIZATION, DEBUG_MODE, HELMET_MODEL_PATH, PPE_CLASS_LABELS, PPE_CONF_THRESHOLD, PPE_MODEL_PATH


class PPEDetector:
    def __init__(self, model_path: Path = PPE_MODEL_PATH, helmet_model_path: Path = HELMET_MODEL_PATH, device: str = "cpu"):
        self.model = YOLO(str(model_path))
        self.model.overrides["device"] = device
        self.helmet_model = YOLO(str(helmet_model_path))
        self.helmet_model.overrides["device"] = device

        self.class_names = {}
        self.ppe_class_ids = []

        for class_id, raw_name in self.model.names.items():
            raw_lower = str(raw_name).lower().strip().replace("_", " ")
            normalized = CLASS_NORMALIZATION.get(raw_name) or CLASS_NORMALIZATION.get(raw_lower)
            if normalized and normalized in PPE_CLASS_LABELS:
                self.class_names[str(raw_name)] = normalized
                self.ppe_class_ids.append(class_id)

        print(f"[PPEDetector] PPE Model: {model_path}")
        print(f"[PPEDetector] Helmet Model: {helmet_model_path}")
        print(f"[PPEDetector] PPE classes: {self.class_names}")
        print(f"[PPEDetector] PPE class IDs: {self.ppe_class_ids}")

        if not self.class_names:
            print("[PPEDetector] WARNING: No PPE classes matched in this model.")

    @staticmethod
    def _normalize_label(label: str) -> str:
        raw = str(label).strip().lower().replace("_", " ").replace("-", " ")
        if raw.endswith("s") and raw not in {"glass", "dress"}:
            raw = raw[:-1]
        raw = " ".join(raw.split())
        if "without helmet" in raw:
            return "ignore"
        if "with helmet" in raw or "helmet" in raw or "hardhat" in raw:
            return "helmet"
        return CLASS_NORMALIZATION.get(raw, raw)

    @staticmethod
    def _merge_helmet_detections(detections: List[Dict]) -> List[Dict]:
        helmets = [d for d in detections if d.get("normalized_name") == "helmet"]
        if not helmets:
            return detections

        keep = []
        for current in helmets:
            duplicate = False
            for candidate in keep:
                if compute_iou(current["bbox"], candidate["bbox"]) > 0.10:
                    if current["confidence"] >= candidate["confidence"]:
                        keep.remove(candidate)
                    else:
                        duplicate = True
                        break
            if not duplicate:
                keep.append(current)

        kept_ids = {id(d) for d in keep}
        return [d for d in detections if d.get("normalized_name") != "helmet" or id(d) in kept_ids]

    def _load_names(self) -> Dict[int, str]:
        try:
            raw_names = {int(k): str(v) for k, v in self.model.names.items()}
        except Exception:
            raw_names = {0: "person", 1: "helmet", 2: "vest", 3: "glove", 4: "hook", 5: "boot"}

        normalized_names = {}
        for class_id, label in raw_names.items():
            normalized_names[class_id] = self._normalize_label(label)
        return normalized_names

    def detect_ppe(self, frame, conf_threshold: float, iou_threshold: float) -> List[Dict]:
        detections: List[Dict] = []

        ppe_results = self.model(frame, conf=max(0.10, conf_threshold), iou=iou_threshold, verbose=False)
        if len(ppe_results) > 0 and ppe_results[0].boxes is not None:
            for box in ppe_results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                raw_name = self.model.names.get(class_id, str(class_id))
                normalized = self._normalize_label(raw_name)
                if normalized == "ignore" or normalized not in PPE_CLASS_LABELS:
                    continue
                if confidence < max(0.10, conf_threshold):
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy().tolist())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "class_name": raw_name,
                    "label": normalized,
                    "normalized_name": normalized,
                    "confidence": confidence,
                    "source": "ppe",
                })

        helmet_results = self.helmet_model(frame, conf=max(0.08, conf_threshold * 0.8), iou=iou_threshold, verbose=False)
        if len(helmet_results) > 0 and helmet_results[0].boxes is not None:
            for box in helmet_results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                raw_name = self.helmet_model.names.get(class_id, str(class_id))
                normalized = self._normalize_label(raw_name)
                if normalized == "ignore" or normalized not in PPE_CLASS_LABELS:
                    continue
                if confidence < max(0.08, conf_threshold * 0.8):
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy().tolist())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "class_name": raw_name,
                    "label": normalized,
                    "normalized_name": normalized,
                    "confidence": confidence,
                    "source": "helmet",
                })

        detections = self._merge_helmet_detections(detections)
        return detections
