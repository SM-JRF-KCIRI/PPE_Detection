from typing import Dict, List

from config import TRACKER_IOU_THRESHOLD, TRACKER_MAX_AGE, TRACKER_MIN_HITS
from utils import center_of_box, compute_iou


class ByteTrackTracker:
    def __init__(self, max_age: int = TRACKER_MAX_AGE, iou_threshold: float = TRACKER_IOU_THRESHOLD, min_hits: int = TRACKER_MIN_HITS):
        self.tracks: List[Dict] = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits

    def update(self, detections, frame_shape=None):
        matched_detections = set()
        frame_h = frame_shape[0] if frame_shape else 1
        frame_w = frame_shape[1] if frame_shape else 1

        for track in self.tracks:
            track["matched"] = False

        for det_idx, det in enumerate(detections):
            best_match = None
            best_score = -1.0
            det_center = center_of_box(det["bbox"])
            for track in self.tracks:
                if track.get("matched", False):
                    continue
                iou = compute_iou(det["bbox"], track["bbox"])
                track_center = center_of_box(track["bbox"])
                distance = ((det_center[0] - track_center[0]) ** 2 + (det_center[1] - track_center[1]) ** 2) ** 0.5
                normalized_distance = distance / max(1.0, (frame_h + frame_w) / 2.0)
                candidate_score = iou * 0.8 + max(0.0, 1.0 - normalized_distance) * 0.2
                if iou < self.iou_threshold and normalized_distance > 0.15:
                    candidate_score = 0.0
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_match = track

            if best_match is not None:
                best_match["matched"] = True
                best_match["bbox"] = det["bbox"]
                best_match["score"] = det["confidence"]
                best_match["hits"] += 1
                best_match["age"] = 0
                best_match["last_seen"] = det_idx
                matched_detections.add(det_idx)

        for track in self.tracks:
            if not track.get("matched", False):
                track["age"] += 1

        for det_idx, det in enumerate(detections):
            if det_idx in matched_detections:
                continue
            self.tracks.append({"track_id": self.next_id, "bbox": det["bbox"], "score": det["confidence"], "age": 0, "hits": 1, "last_seen": det_idx})
            self.next_id += 1

        self.tracks = [track for track in self.tracks if track["age"] <= self.max_age]
        for track in self.tracks:
            track.pop("matched", None)

        return [
            {"track_id": track["track_id"], "bbox": track["bbox"], "score": track["score"]}
            for track in self.tracks
            if track["hits"] >= self.min_hits or track["age"] == 0
        ]
