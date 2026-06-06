import numpy as np

from config import TRACKER_IOU_THRESHOLD, TRACKER_MAX_AGE, TRACKER_MIN_HITS
from utils import compute_iou


class ByteTrackWrapper:
    def __init__(self, max_age: int = TRACKER_MAX_AGE, iou_threshold: float = TRACKER_IOU_THRESHOLD, min_hits: int = TRACKER_MIN_HITS):
        self.trackers = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits

    def _iou_distance(self, bbox, track_bbox):
        return compute_iou(bbox, track_bbox)

    def update(self, detections, frame_shape=None):
        assignments = {}
        used_tracks = set()
        used_detections = set()
        new_tracks = []

        for det_idx, det in enumerate(detections):
            best_track = None
            best_iou = 0.0
            for tr_idx, track in enumerate(self.trackers):
                if tr_idx in used_tracks:
                    continue
                iou = self._iou_distance(det["bbox"], track["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track = tr_idx
            if best_track is not None:
                used_tracks.add(best_track)
                used_detections.add(det_idx)
                self.trackers[best_track]["bbox"] = det["bbox"]
                self.trackers[best_track]["age"] = 0
                self.trackers[best_track]["hits"] += 1
                self.trackers[best_track]["score"] = det["confidence"]
                assignments[det_idx] = self.trackers[best_track]["track_id"]

        for tr_idx, track in enumerate(self.trackers):
            if tr_idx not in used_tracks:
                track["age"] += 1

        for det_idx, det in enumerate(detections):
            if det_idx in used_detections:
                continue
            self.trackers.append(
                {
                    "track_id": self.next_id,
                    "bbox": det["bbox"],
                    "score": det["confidence"],
                    "age": 0,
                    "hits": 1,
                }
            )
            assignments[det_idx] = self.next_id
            self.next_id += 1

        self.trackers = [t for t in self.trackers if t["age"] <= self.max_age]

        tracks = []
        for track in self.trackers:
            if track["hits"] >= self.min_hits or track["age"] == 0:
                tracks.append(
                    {
                        "track_id": track["track_id"],
                        "bbox": track["bbox"],
                        "score": track["score"],
                    }
                )
        return tracks
