from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from config import POSE_MODEL_PATH


class PoseEstimator:
    def __init__(self, model_path: Path = POSE_MODEL_PATH, device: str = "cpu"):
        self.model = YOLO(str(model_path))
        self.device = device
        self.model.overrides["device"] = device

    def detect(self, frame, conf_threshold: float, iou_threshold: float) -> List[Dict]:
        results = self.model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        result = results[0]
        persons = []
        if result.boxes is None or len(result.boxes) == 0:
            return persons
        xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        keypoints = self._extract_keypoints(result)
        for idx, box in enumerate(xyxy):
            persons.append(
                {
                    "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "confidence": float(confidences[idx]),
                    "keypoints": keypoints[idx] if keypoints is not None and idx < len(keypoints) else None,
                }
            )
        return persons

    @staticmethod
    def _extract_keypoints(result) -> Optional[np.ndarray]:
        if hasattr(result, "keypoints") and result.keypoints is not None:
            keypoints = result.keypoints
            data = getattr(keypoints, "data", keypoints)
            if hasattr(data, "cpu"):
                data = data.cpu()
            keypoints_array = np.asarray(data)
            if keypoints_array.ndim == 4 and keypoints_array.shape[1] == 1:
                keypoints_array = keypoints_array[:, 0]
            if keypoints_array.ndim == 4 and keypoints_array.shape[0] == 1 and keypoints_array.shape[1] == 1:
                keypoints_array = keypoints_array[0, 0]
            return keypoints_array.astype(np.float32)
        if hasattr(result, "masks"):
            return None
        return None
