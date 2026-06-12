from pathlib import Path

import os as _os

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("D:/newppe/roboflow")
PPE_MODEL_CANDIDATES = [
    BASE_DIR / "runs" / "ppe_train" / "v1" / "weights" / "best.pt",
    BASE_DIR / "best" / "best.pt",
    BASE_DIR / "best" / "best_1.pt",
    BASE_DIR / "best" / "best_2.pt",
]
PPE_MODEL_PATH = next((path for path in PPE_MODEL_CANDIDATES if path.exists()), None)
if PPE_MODEL_PATH is None:
    raise FileNotFoundError("No PPE model file was found under the expected paths.")

HELMET_MODEL_PATH = BASE_DIR / "helmet" / "best.pt"
if not HELMET_MODEL_PATH.exists():
    HELMET_MODEL_PATH = None

POSE_MODEL_PATH = BASE_DIR / "yolov8n-pose.pt"
if not POSE_MODEL_PATH.exists():
    POSE_MODEL_PATH = BASE_DIR / "yolov8-pose.pt"

BEST_PPE_MODEL_PATH = PPE_MODEL_PATH
BASE_MODEL_PATH = "yolov8m.pt"

DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU = 0.45
PPE_CONFIDENCE = float(_os.getenv("PPE_CONFIDENCE", "0.25"))
DEBUG_MODE = _os.getenv("PPE_DEBUG", "0") == "1"
POSE_CONFIDENCE = float(_os.getenv("POSE_CONFIDENCE", "0.35"))
PERSON_CONFIDENCE = float(_os.getenv("PERSON_CONFIDENCE", "0.35"))
CONFIDENCE_RANGE = (0.1, 0.7)
IOU_RANGE = (0.1, 0.7)

DATASET_CLASS_NAMES = ["Boots", "Gloves", "Goggles", "Helmet", "Person", "Vest"]

# Canonical PPE labels used throughout the pipeline.
CLASS_NORMALIZATION = {
    "boots": "boot",
    "boot": "boot",
    "safety_boots": "boot",
    "safety boots": "boot",
    "workboot": "boot",
    "shoe": "boot",
    "shoes": "boot",
    "gloves": "glove",
    "glove": "glove",
    "goggles": "goggle",
    "goggle": "goggle",
    "safety_goggles": "goggle",
    "safety goggles": "goggle",
    "helmet": "helmet",
    "hardhat": "helmet",
    "safety helmet": "helmet",
    "safety_helmet": "helmet",
    "safetyhelmet": "helmet",
    "vest": "vest",
    "safety vest": "vest",
    "safety_vest": "vest",
    "hi-vis": "vest",
    "hi vis": "vest",
    "hook": "hook",
    "safety hook": "hook",
    "person": "person",
}
KEYPOINT_CONFIDENCE_THRESHOLD = 0.3
PPE_CONF_THRESHOLD = PPE_CONFIDENCE

PPE_CLASS_LABELS = ["helmet", "vest", "glove", "boot", "goggle", "hook"]
OPTIONAL_PPE_LABELS = {"goggle", "hook"}
REQUIRED_PPE_LABELS = ["helmet", "vest", "glove", "boot"]

TRACKER_MAX_AGE = 30
TRACKER_MIN_HITS = 3
TRACKER_IOU_THRESHOLD = 0.3
SMOOTHING_WINDOW = 5

PPE_REGION_MAP = {
    "helmet": "head",
    "vest": "chest",
    "glove": "hands",
    "boot": "foot",
    "hook": "belt",
    "goggles": "head",
}

REGION_COLORS = {
    "head": (0, 255, 0),
    "chest": (0, 255, 0),
    "left_hand": (0, 255, 0),
    "right_hand": (0, 255, 0),
    "leg": (0, 255, 0),
    "foot": (0, 255, 0),
    "belt": (0, 255, 0),
}

STATUS_COLORS = {
    "COMPLIANT": (0, 230, 0),
    "NON-COMPLIANT": (0, 0, 230),
}
