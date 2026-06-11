# PPE Detection System Accuracy and Architecture Report

## Dataset Health Check

Dataset source: `D:/newppe/roboflow`
Dataset export: Roboflow YOLOv8 format

### Dataset split sizes

- Train images: 2437
- Validation images: 348
- Test images: 696
- Total images: 3481

### Annotation counts

- Train annotations: 6508
- Validation annotations: 906
- Test annotations: 1801
- Total annotations: 8215

### Class distribution

| Class   | Train | Valid | Test | Total |
|--------:|------:|------:|-----:|------:|
| Boots   | 335   | 57    | 94   | 486   |
| Gloves  | 495   | 73    | 91   | 659   |
| Goggles | 77    | 11    | 25   | 113   |
| Helmet  | 1418  | 215   | 417  | 2050  |
| Person  | 234   | 41    | 80   | 355   |
| Vest    | 3949  | 509   | 1094 | 5552  |

### Observations

- The dataset is unbalanced: `Vest` and `Helmet` are the most frequent PPE classes.
- `Goggles` has the smallest representation in the dataset.
- Person annotations are sparse relative to PPE objects, indicating many images contain multiple PPE items per person.

### Evaluation metrics status

No quantitative recall/precision/mAP evaluation results are available in the current repository files.

To generate metrics, perform evaluation with YOLOv8 using the available `best.pt` model and the dataset triplet:

- `train` images/labels
- `valid` images/labels
- `test` images/labels

A recommended evaluation command would be:

```bash
yolo val model=best.pt data=roboflow/data.yaml
```

This will produce mAP50-95, precision, recall, and F1 values for the current dataset.

## Project Architecture

### High-level system pipeline

1. Person detection
2. ByteTrack-style tracking
3. Pose estimation
4. Body region generation
5. PPE detection
6. PPE-to-person assignment
7. Compliance evaluation
8. Temporal smoothing
9. Gradio dashboard visualization
10. Inspection reporting

### Main application entry point

- `app.py`
  - Launches the Gradio PPE Inspection Dashboard
  - Handles image, video, webcam, and RTSP camera inspection
  - Coordinates model inference, tracking, smoothing, and reporting

### Core modules and roles

- `config.py`
  - Stores constants, paths, model locations, thresholds, and class labels

- `detector.py`
  - Implements `PPEDetector`
  - Loads the PPE detection model from `best.pt` or fallback `yolov8m.pt`
  - Normalizes class names and filters PPE detections

- `pose_engine.py`
  - Implements `PoseEngine`
  - Runs YOLOv8 Pose model inference
  - Returns person bounding boxes and pose keypoints

- `tracker.py`
  - Implements `ByteTrackTracker`
  - Manages track lifecycle, detection-to-track association, aging, and pruning

- `ppe_matcher.py`
  - Generates body regions from pose keypoints (`head`, `chest`, `left_hand`, `right_hand`, `belt`, `foot`)
  - Applies PPE assignment rules by checking PPE center placement or overlap with regions
  - Ensures PPE items are assigned to the nearest valid person

- `compliance.py`
  - Implements compliance logic
  - Determines present/missing status for `helmet`, `vest`, `hook`, `glove`, and `shoe`
  - Classifies persons as `COMPLIANT` only when all PPE are present

- `temporal_smoothing.py`
  - Implements majority voting over the last 5 frames
  - Reduces flicker and false positives/negatives in temporal video streams

- `visualization.py`
  - Draws person boxes, PPE boxes, labels, and compliance status on frames
  - Removes debugging overlays and auxiliary region drawings

- `report_generator.py`
  - Builds compliance report rows for Gradio dataframes
  - Produces final inspection table output for each tracked person

- `utils.py`
  - Provides helper functions for box math, IoU, centroids, drawing, and safe casting

## Algorithm details

### Person-centric compliance logic

- Persons are detected using pose model output.
- Each person is assigned a track ID via tracking.
- PPE detections are filtered by normalized labels and restricted to supported PPE classes.
- Body regions are generated from pose keypoints; fallback boxes are used when keypoints are incomplete.
- PPE assignment rules:
  - Helmet: PPE center must lie inside head region
  - Vest: PPE bbox must overlap the chest region
  - Glove: PPE center must lie inside either left or right hand region
  - Hook: PPE center must lie inside the belt or chest region
  - Shoe: PPE center must lie inside foot region
- Only the closest valid person receives a PPE assignment.
- Compliance is determined per person, not per frame.

### Temporal smoothing

- Uses a sliding window of 5 frames per tracked person
- Applies majority voting for each PPE status
- Final detection is `Present` if the majority of recent frames indicate presence

### Visualization and reporting

- Only draws:
  - Person bounding boxes
  - PPE bounding boxes
  - Person compliance labels
- Generates a Gradio table with columns:
  - `Person ID`, `Helmet`, `Vest`, `Hook`, `Glove`, `Shoe`, `Compliance`

## Module architecture summary

- `app.py` → orchestrates inference, dashboard UI, and output generation
- `detector.py` → PPE detection model and label normalization
- `pose_engine.py` → pose detection model and keypoint extraction
- `tracker.py` → object tracking across frames
- `ppe_matcher.py` → PPE assignment and region-based matching
- `compliance.py` → compliance evaluation logic
- `temporal_smoothing.py` → frame history smoothing
- `report_generator.py` → structured compliance reporting
- `visualization.py` → clean annotation rendering
- `config.py` → system constants and model paths
- `utils.py` → shared mathematical and drawing utilities

## Notes

- The dataset mapping in `roboflow/data.yaml` uses class names:
  - `Boots`, `Gloves`, `Goggles`, `Helmet`, `Person`, `Vest`
- Project mapping normalizes these to the expected PPE classes:
  - `helmet`, `vest`, `glove`, `hook`, `shoe`
- `hook` is handled as a supported PPE class even when the dataset may not explicitly include a dedicated `hook` label.

If you want, I can also add an evaluation section that includes a model performance matrix once the YOLOv8 validation metrics are available.