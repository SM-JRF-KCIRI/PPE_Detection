# Autonomous PTZ Patrol & PPE Violation Tracking System

An end-to-end computer vision and PTZ camera control system designed for Nvidia Jetson and edge GPU platforms. It performs autonomous multi-stage PTZ waypoint patrolling, real-time person detection, visual-servo lock-on tracking, and Personal Protective Equipment (PPE) compliance inspection with automatic Excel reporting and live HTTP MJPEG streaming.

---

## 🌟 Key Features

- **Autonomous Multi-Stage PTZ Patrol**:
  - Sweeps configurable pan/tilt/zoom waypoints (HOME -> LEFT -> HOME -> RIGHT -> HOME).
  - Implements 4 distinct sweep stages with varying tilt angles, zoom factors, and motor speeds.
  - Dedicated slow tilt transitions between wide and close inspection stages.

- **Visual-Servo Target Tracking**:
  - Automatically interrupts patrol mode when a person is detected.
  - Locks onto the target using ByteTrack and drives ONVIF ContinuousMove velocity commands (proportional pan/tilt visual-servo with exponential smoothing and deadband stabilization).
  - Configurable tracking duration with a smart tracking cooldown manager to avoid endlessly following the same person.

- **Real-Time PPE Compliance Analysis**:
  - Checks for 4 critical PPE items: **Hardhat**, **Safety Vest**, **Gloves**, and **Boots**.
  - Distinguishes compliant items vs. non-compliant violations (no_hardhat, no_vest, etc.).
  - Calculates bounding-box overlap between tracked persons and detected PPE items.
  - Generates automated visual annotations and on-screen HUD alerts.

- **Automated Excel Compliance Logger (ppe_log.xlsx)**:
  - Automatically records timestamped session logs per tracked person.
  - Logs track ID, active patrol stage, individual status for each PPE category (OK vs VIOLATION), overall compliance, and violation details.

- **Integrated HTTP MJPEG Feed Server**:
  - Embedded Flask video server operating on port 8080 (reusing the existing RTSP grabber thread without opening redundant camera connections).
  - Endpoints for raw camera stream and annotated AI detection stream.

- **Jetson & TensorRT Optimized**:
  - Native support for TensorRT .engine models and FP16 half-precision inference.
  - Hardware-accelerated GStreamer video decoding fallback to software decode.

---

## 📁 Repository Structure

```
├── patrol_track_main.py      # Main application (Patrol state machine, YOLO inference, tracking loop)
├── ptz_shared.py             # ONVIF PTZ controller, FrameGrabber, tracker, Excel logger, HTTP server
├── requirements.txt          # Python project dependencies
├── .gitignore                # Git ignore configuration
├── extra/
│   ├── jetson_feed_server.py # Standalone minimal MJPEG feed test server
│   └── patrol_test.py        # Standalone PTZ patrol calibration & waypoint test script
└── README.md                 # Project documentation
```

---

## 🏷️ Model Classes & PPE Mapping

The system supports the following 9 detection classes:

| Class ID | Label | Category | Compliance Status |
|---|---|---|---|
| 0 | boots | Boots | **OK** |
| 1 | gloves | Gloves | **OK** |
| 2 | hardhat | Hardhat | **OK** |
| 3 | no_boots | Boots | **VIOLATION** |
| 4 | no_gloves | Gloves | **VIOLATION** |
| 5 | no_hardhat | Hardhat | **VIOLATION** |
| 6 | no_vest | Vest | **VIOLATION** |
| 7 | person | Person | Target for PTZ Tracking |
| 8 | vest | Vest | **OK** |

---

## 📡 Live HTTP Streaming Endpoints

Once the application is running, the embedded web server provides live MJPEG streams accessible from any browser or dashboard on the network:

| Endpoint | Description |
|---|---|
| http://<device-ip>:8080/ | Web landing page with stream links and status |
| http://<device-ip>:8080/feed | Live Raw Camera Feed (MJPEG) |
| http://<device-ip>:8080/feed/0 | Camera Channel 0 Raw Feed |
| http://<device-ip>:8080/annotated_feed | Live AI Detection Feed with bounding boxes & HUD |

---

## ⚙️ Configuration

Key settings can be adjusted in ptz_shared.py:

- **Camera & ONVIF Connection**:
  ```python
  CAMERA_IP = "192.168.1.126"
  USERNAME = "admin"
  PASSWORD = "Admin@123"
  RTSP_URL = "rtsp://admin:Admin@123@192.168.1.126:554/unicaststream/1"
  ```

- **Patrol Stages (Zoom, Tilt, Speed)**:
  ```python
  ZOOM_STAGES = [0.00, 0.00, 0.38, 0.75]
  TILT_STAGES = [0.25, -0.75, -0.75, -0.75]
  SPEED_STAGES = [0.20, 0.16, 0.09, 0.038]
  ```

- **Tracking & Visual Servo Parameters**:
  ```python
  PAN_KP = 0.6
  TILT_KP = 0.6
  MAX_PAN_SPEED = 0.1
  MAX_TILT_SPEED = 0.1
  TRACK_DURATION = 5.0
  ```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/SM-JRF-KCIRI/PPE_Detection.git
cd PPE_Detection
git checkout ppe-final

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Exporting TensorRT Engine (Jetson Optional / Recommended)

To build a hardware-optimized TensorRT engine on your Jetson:

```bash
yolo export model=best.pt format=engine device=0 half=True imgsz=640
```

Update YOLO_MODEL_PATH = "best_native.engine" in ptz_shared.py.

### 3. Running the Pipeline

```bash
python patrol_track_main.py
```

Press q in the OpenCV video window to stop execution and save the Excel log.
