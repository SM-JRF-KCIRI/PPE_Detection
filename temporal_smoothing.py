from collections import deque
from typing import Deque, Dict, Optional

from config import SMOOTHING_WINDOW


class TemporalSmoother:
    def __init__(self, window: int = SMOOTHING_WINDOW):
        self.window = window
        self.history: Dict[int, Deque[Dict[str, Optional[bool]]]] = {}

    def smooth(self, person_id: int, current_status: Dict[str, Optional[bool]]) -> Dict[str, Optional[bool]]:
        history = self.history.setdefault(person_id, deque(maxlen=self.window))
        history.append({key: value for key, value in current_status.items() if value is not None})

        counts = {key: 0 for key in current_status}
        valid_counts = {key: 0 for key in current_status}
        for entry in history:
            for key, value in entry.items():
                if value is True:
                    counts[key] += 1
                if value is not None:
                    valid_counts[key] += 1

        smoothed = {}
        for key in current_status:
            if valid_counts[key] == 0:
                smoothed[key] = None
            else:
                threshold = max(1, (valid_counts[key] + 1) // 2)
                smoothed[key] = counts[key] >= threshold
        return smoothed

    def update(self, person_id: int, current_status: Dict[str, Optional[bool]]) -> Dict[str, Optional[bool]]:
        return self.smooth(person_id, current_status)

    def prune(self, active_ids):
        stale_ids = [person_id for person_id in self.history.keys() if person_id not in active_ids]
        for person_id in stale_ids:
            self.history.pop(person_id, None)
