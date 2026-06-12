# PPE Inspection System

## Overview

This repository contains a PPE compliance inspection workflow built around YOLOv8 pose detection, PPE object detection, tracking, region-based assignment, temporal smoothing, and a Gradio dashboard.

The application entry point is in `app.py`, with supporting logic under the main `PPE_Detection/` project folder and dataset assets under `roboflow/`.

## What the project does

- Detects people from video or image input using a YOLOv8 pose model.
- Detects PPE items such as helmet, vest, gloves, hook, boots, and goggles.
- Assigns PPE detections to the most likely person using body-region logic.
- Evaluates compliance status per person and smooths results across frames.
- Displays the inspection result in a Gradio interface for image, video, webcam, or live camera input.

## Features visible in the code

- Gradio dashboard with image / video / webcam / camera URL modes.
- Optional `GRADIO_SERVER_PORT` environment variable for the launch port.
- Training script for a YOLOv8 PPE model using `roboflow/data.yaml`.
- Compliance reporting rows for the UI table output.

## Installation

From the workspace root:

```bash
python -m pip install -r requirements.txt
```

The legacy `ppe_system/` sources were moved into `legacy_ppe_system/` for archival reference; the active project now lives entirely under `PPE_Detection/`.

## Usage

### Run the dashboard

```bash
cd d:\newppe\PPE_Detection
python app.py
```

### Train a PPE model

```bash
cd d:\newppe
python train.py --data roboflow/data.yaml --epochs 100 --batch 16
```

The training flow is implemented in `train.py` and writes outputs under `runs/ppe_train/`.

## Environment variables

The current code uses one optional runtime environment variable:

- `GRADIO_SERVER_PORT` — override the default Gradio launch port (`7860`).

No other environment variables are referenced in the checked-in code.

## Folder reference

See [details.md](details.md) for the full folder and file breakdown.
