# Project Details

## Root folder

The repository root contains:

- `PPE_Detection/` — active application, training, detection, tracking, and reporting code.
- `roboflow/` — dataset assets in YOLOv8 format, including `data.yaml` and split folders for train / valid / test.
- `yolov8m.pt` and `yolov8n-pose.pt` — model weight files present in the workspace.

## Folder-by-folder breakdown

### `PPE_Detection/`

This is the main application package.

- `app.py`  
  Launches the Gradio dashboard and coordinates image/video/camera inspection.

- `config.py`  
  Stores model paths, thresholds, label mappings, and PPE category settings.

- `detector.py`  
  Implements PPE detection with YOLOv8 and label normalization.

- `pose_engine.py`  
  Implements the pose inference pipeline that returns person boxes and keypoints.

- `tracker.py`  
  Applies a lightweight tracking loop for person IDs across frames.

- `ppe_rules.py`  
  Defines body-region extraction and PPE-to-person matching rules.

- `ppe_matcher.py`  
  Provides a second region-matching implementation used by the inspection pipeline.

- `compliance.py`  
  Evaluates whether PPE items are present and determines compliance status.

- `temporal_smoothing.py`  
  Smooths compliance decisions over recent frame history.

- `smoother.py`  
  Contains a smoothing helper used by the application flow.

- `visualization.py`  
  Draws PPE and person annotations on output frames.

- `report_generator.py`  
  Formats report rows for the dashboard table.

- `train.py`  
  Trains a YOLOv8 PPE model from the Roboflow dataset YAML.

- `evaluate.py`  
  Provides evaluation-oriented utilities for the current project.

- `inspect_pose.py`  
  A utility script related to pose inspection and debugging.

- `debug_run_image.py`  
  Supports image-based debugging runs.

- `utils.py`  
  Contains shared helpers for box math, IoU, and point calculations.

- `requirements.txt`  
  Lists the Python dependencies used by the application.

- `package.json`  
  Declares an npm dependency but does not define runnable scripts in the checked-in file.

- `runs/`  
  Stores training artifacts and outputs created during model experiments.

### `roboflow/`

This folder contains the training dataset in YOLOv8 format.

- `data.yaml`  
  Dataset configuration for training and validation.

- `README.dataset.txt` and `README.roboflow.txt`  
  Dataset and export notes from the source dataset.

- `train/`, `valid/`, `test/`  
  Image and label folders used for model training and evaluation.

## Entry points

- Application UI: `app.py`
- Training entry point: `train.py`

## Architecture flow

1. The Gradio app in `app.py` loads the PPE detector and pose model.
2. `pose_engine.py` extracts person boxes and keypoints from frames.
3. `tracker.py` assigns stable track IDs to persons across frames.
4. `ppe_rules.py` and `ppe_matcher.py` generate body regions and match PPE detections to persons.
5. `compliance.py` decides whether each person is compliant, missing PPE, or an intruder.
6. `temporal_smoothing.py` reduces frame-to-frame flicker.
7. `visualization.py` and `report_generator.py` produce the annotated output and table display.

## Technical notes

- The code uses Ultralytics YOLOv8 for both pose detection and PPE object detection.
- The system supports image, video, webcam, and RTSP or HTTP camera input.
- Model path selection is controlled in `config.py` and can fall back to a base YOLOv8 model if a trained model is not present.
- The workspace currently contains training outputs under `runs/ppe_train/v1/`.
