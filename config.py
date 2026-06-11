from pathlib import Path

import os as _os

DATA_ROOT = Path("D:/newppe/roboflow")
BEST_PPE_MODEL_PATH = DATA_ROOT / "best.pt"
BASE_MODEL_PATH = "yolov8m.pt"
_TRAINED_MODEL = "runs/ppe_train/v1/weights/best.pt"
_FALLBACK_MODEL = BASE_MODEL_PATH
if _os.path.isfile(_TRAINED_MODEL):
    PPE_MODEL_PATH = _TRAINED_MODEL
    print(f"[config] Using trained PPE model: {PPE_MODEL_PATH}")
else:
    PPE_MODEL_PATH = _FALLBACK_MODEL
    print(f"[config] WARNING: Trained model not found at {_TRAINED_MODEL}")
    print(f"[config] WARNING: Falling back to {_FALLBACK_MODEL} (COCO model)")
    print(f"[config] WARNING: PPE detection will NOT work until training completes.")
    print(f"[config] Run: python train.py --data roboflow/data.yaml --epochs 100 --batch 16")
POSE_MODEL_PATH = "yolov8-pose.pt" if Path("yolov8-pose.pt").exists() else "yolov8n-pose.pt"

DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU = 0.45
CONFIDENCE_RANGE = (0.1, 0.7)
IOU_RANGE = (0.1, 0.7)

DATASET_CLASS_NAMES = ["Boots", "Gloves", "Goggles", "Helmet", "Person", "Vest"]

# These mappings normalize detector labels to canonical PPE categories.
CLASS_NORMALIZATION = {
    "boots": "shoe",
    "boot": "shoe",
    "workboot": "shoe",
    "gloves": "glove",
    "glove": "glove",
    "goggles": "goggles",
    "goggle": "goggles",
    "safety goggles": "goggles",
    "safety goggle": "goggles",
    "helmet": "helmet",
    "hardhat": "helmet",
    "safety helmet": "helmet",
    "safetyhelmet": "helmet",
    "vest": "vest",
    "safety vest": "vest",
    "hi-vis": "vest",
    "hi vis": "vest",
    "hook": "hook",
    "safety hook": "hook",
}
KEYPOINT_CONFIDENCE_THRESHOLD = 0.3
PPE_CONF_THRESHOLD = 0.25

PPE_CLASS_LABELS = ["helmet", "vest", "glove", "hook", "shoe", "goggles"]
OPTIONAL_PPE_LABELS = {"hook", "goggles"}
REQUIRED_PPE_LABELS = [label for label in PPE_CLASS_LABELS if label not in OPTIONAL_PPE_LABELS]

TRACKER_MAX_AGE = 30
TRACKER_MIN_HITS = 3
TRACKER_IOU_THRESHOLD = 0.3
SMOOTHING_WINDOW = 5

PPE_REGION_MAP = {
    "helmet": "head",
    "vest": "chest",
    "glove": "hands",
    "hook": "belt",
    "shoe": "foot",
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
