# PPE Inspection System

A production-style PPE compliance inspection system built with Ultralytics YOLOv8, pose estimation, tracking, temporal smoothing, and a Gradio dashboard.

## Features

- Person detection with YOLOv8-pose
- PPE detection with YOLOv8m / custom `best.pt`
- ByteTrack-style tracking for stable person IDs
- Temporal smoothing over the last 5 frames
- Region-based assignment for helmet, vest, gloves, and boots
- Industrial-style Gradio dashboard with image/video/webcam/live camera support

## Project Structure

```
ppe_system/
│
├── app.py
├── config.py
├── detector.py
├── pose.py
├── tracker.py
├── ppe_rules.py
├── smoother.py
├── utils.py
├── requirements.txt
└── README.md
```

## Setup

1. Create a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r ppe_system/requirements.txt
```

3. Place your custom PPE model at:

- `D:/newppe/roboflow/best.pt`

4. Optionally provide YOLOv8 model weights locally or allow Ultralytics to download them:

- `yolov8-pose.pt`
- `yolov8m.pt`

## Run

From the workspace root:

```bash
cd d:\newppe\ppe_system
python app.py
```

Then open the Gradio UI on the displayed port.

## Notes

- The tracker wrapper is designed to integrate with ByteTrack-style tracking and can be upgraded when `yolox` / `byte-track` packages are available.
- The system uses pose keypoints to build body regions and evaluates PPE compliance per person.
- The Gradio UI supports image upload, video upload, webcam stream, and live camera URL.
