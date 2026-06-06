from pathlib import Path

DATA_ROOT = Path("D:/newppe/roboflow")
BEST_PPE_MODEL_PATH = DATA_ROOT / "best.pt"
BASE_MODEL_PATH = "yolov8m.pt"
PPE_MODEL_PATH = BEST_PPE_MODEL_PATH if BEST_PPE_MODEL_PATH.exists() else Path(BASE_MODEL_PATH)
POSE_MODEL_PATH = "yolov8-pose.pt" if Path("yolov8-pose.pt").exists() else "yolov8n-pose.pt"

DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU = 0.45
CONFIDENCE_RANGE = (0.1, 0.7)
IOU_RANGE = (0.1, 0.7)

TRACKER_MAX_AGE = 30
TRACKER_MIN_HITS = 3
TRACKER_IOU_THRESHOLD = 0.3
SMOOTHING_WINDOW = 5

PPE_CLASS_LABELS = ["helmet", "vest", "gloves", "boots"]
PPE_REGION_MAP = {
    "helmet": "head",
    "vest": "chest",
    "gloves": "hands",
    "boots": "feet",
}

REGION_COLORS = {
    "head": (0, 255, 0),
    "chest": (0, 255, 0),
    "hands": (0, 255, 0),
    "feet": (0, 255, 0),
}

STATUS_COLORS = {
    "GREEN": (0, 230, 0),
    "RED": (0, 0, 230),
}
