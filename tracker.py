from config import TRACKER_IOU_THRESHOLD, TRACKER_MAX_AGE, TRACKER_MIN_HITS
from utils import compute_iou


class ByteTrackTracker:
    def __init__(self, max_age: int = TRACKER_MAX_AGE, iou_threshold: float = TRACKER_IOU_THRESHOLD, min_hits: int = TRACKER_MIN_HITS):
        self.tracks = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits

    def update(self, detections, frame_shape=None):
        matched_detections = set()
        updated_tracks = []

        for track in self.tracks:
            track["matched"] = False

        for det_idx, det in enumerate(detections):
            best_match = None
            best_iou = 0.0
            for track in self.tracks:
                if track["matched"]:
                    continue
                iou = compute_iou(det["bbox"], track["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_match = track
            if best_match is not None:
                best_match["matched"] = True
                best_match["bbox"] = det["bbox"]
                best_match["score"] = det["confidence"]
                best_match["hits"] += 1
                best_match["age"] = 0
                matched_detections.add(det_idx)

        for track in self.tracks:
            if not track.get("matched", False):
                track["age"] += 1

        for det_idx, det in enumerate(detections):
            if det_idx in matched_detections:
                continue
            self.tracks.append(
                {
                    "track_id": self.next_id,
                    "bbox": det["bbox"],
                    "score": det["confidence"],
                    "age": 0,
                    "hits": 1,
                }
            )
            self.next_id += 1

        self.tracks = [track for track in self.tracks if track["age"] <= self.max_age]
        for track in self.tracks:
            if track.get("matched") is not None:
                track.pop("matched", None)

        results = []
        for track in self.tracks:
            if track["hits"] >= self.min_hits or track["age"] == 0:
                results.append({"track_id": track["track_id"], "bbox": track["bbox"], "score": track["score"]})
        return results
