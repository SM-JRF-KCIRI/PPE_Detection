from pathlib import Path
from typing import Dict, List

import numpy as np
from ultralytics import YOLO

from config import PPE_MODEL_PATH


class PPEDetector:
    def __init__(self, model_path: Path = PPE_MODEL_PATH, device: str = "cpu"):
        self.model = YOLO(str(model_path))
        self.device = device
        self.model.overrides["device"] = device
        self.names = self._load_names()

    def _load_names(self) -> Dict[int, str]:
        try:
            return {int(k): v for k, v in self.model.names.items()}
        except Exception:
            return {0: "helmet", 1: "vest", 2: "gloves", 3: "boots"}

    def detect(self, frame, conf_threshold: float, iou_threshold: float) -> List[Dict]:
        results = self.model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        result = results[0]
        detections = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls in zip(xyxy, confidences, classes):
            detections.append(
                {
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "confidence": float(score),
                    "class_id": int(cls),
                    "label": self.names.get(int(cls), f"class_{int(cls)}"),
                }
            )
        return detections
