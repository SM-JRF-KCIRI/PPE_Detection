from collections import deque
from typing import Deque, Dict

from config import SMOOTHING_WINDOW


class TemporalSmoother:
    def __init__(self, window: int = SMOOTHING_WINDOW):
        self.window = window
        self.history: Dict[int, Deque[Dict[str, bool]]] = {}

    def smooth(self, person_id: int, current_status: Dict[str, bool]) -> Dict[str, bool]:
        if person_id not in self.history:
            self.history[person_id] = deque(maxlen=self.window)
        self.history[person_id].append(current_status)
        counts = {key: 0 for key in current_status if isinstance(current_status[key], bool)}
        for item in self.history[person_id]:
            for key, value in item.items():
                if isinstance(value, bool) and value:
                    counts[key] += 1
        stabilised = {
            key: counts[key] >= max(1, len(self.history[person_id]) // 2 + 1)
            for key in counts
        }
        return stabilised

    def prune(self, active_ids):
        stale_ids = [pid for pid in self.history.keys() if pid not in active_ids]
        for pid in stale_ids:
            self.history.pop(pid, None)
