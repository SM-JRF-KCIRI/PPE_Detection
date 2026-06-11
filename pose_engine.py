from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from config import POSE_MODEL_PATH


class PoseEngine:
    def __init__(self, model_path: Path = POSE_MODEL_PATH, device: str = "cpu"):
        self.model = YOLO(str(model_path))
        self.model.overrides["device"] = device

    def detect_persons(self, frame, conf_threshold: float, iou_threshold: float) -> List[Dict]:
        results = self.model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        if len(results) == 0:
            return []

        result = results[0]
        persons = []
        if result.boxes is None or len(result.boxes) == 0:
            return persons

        xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        keypoints = self._extract_keypoints(result)

        for index, box in enumerate(xyxy):
            persons.append(
                {
                    "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "confidence": float(confidences[index]),
                    "keypoints": keypoints[index] if keypoints is not None and index < len(keypoints) else None,
                }
            )
        return persons

    @staticmethod
    def _extract_keypoints(result) -> Optional[np.ndarray]:
        if not hasattr(result, "keypoints") or result.keypoints is None:
            return None
        keypoints = getattr(result.keypoints, "data", result.keypoints)
        if hasattr(keypoints, "cpu"):
            keypoints = keypoints.cpu()
        keypoints_array = np.asarray(keypoints)
        if keypoints_array.ndim == 4 and keypoints_array.shape[1] == 1:
            keypoints_array = keypoints_array[:, 0]
        if keypoints_array.ndim == 4 and keypoints_array.shape[0] == 1 and keypoints_array.shape[1] == 1:
            keypoints_array = keypoints_array[0, 0]
        return keypoints_array.astype(np.float32)
