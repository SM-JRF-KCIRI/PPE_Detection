import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from ultralytics import YOLO
import cv2
import threading
import time

# ==========================
# CONFIG
# ==========================

MODEL_PATH = "vel best.pt"

# Replace @ in password with %40
RTSP_URL      = "rtsp://admin:Admin@123@192.168.1.126:554/unicaststream/1"

# ==========================
# LOAD MODEL
# ==========================

model = YOLO(MODEL_PATH)

# ==========================
# SHARED VARIABLES
# ==========================

latest_frame = None
latest_result = None

frame_lock = threading.Lock()

running = True

# ==========================
# CAMERA THREAD
# ==========================

def camera_thread():
    global latest_frame, running

    while running:

        print("Connecting to camera...")

        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print("Failed to connect. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        print("Camera connected.")

        while running:

            ret, frame = cap.read()

            if not ret:
                print("Camera stream lost. Reconnecting...")
                break

            with frame_lock:
                latest_frame = frame

        cap.release()
        time.sleep(2)

# ==========================
# INFERENCE THREAD
# ==========================

def inference_thread():
    global latest_frame, latest_result, running

    while running:

        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        try:
            results = model(
                frame,
                imgsz=320,
                device=0,
                verbose=False
            )

            annotated = results[0].plot()

            with frame_lock:
                latest_result = annotated

        except Exception as e:
            print("Inference error:", e)
            time.sleep(1)

# ==========================
# START THREADS
# ==========================

threading.Thread(
    target=camera_thread,
    daemon=True
).start()

threading.Thread(
    target=inference_thread,
    daemon=True
).start()

# ==========================
# DISPLAY LOOP
# ==========================

prev_time = time.time()

while True:

    with frame_lock:

        if latest_result is not None:
            display = latest_result.copy()

        elif latest_frame is not None:
            display = latest_frame.copy()

        else:
            display = None

    if display is None:
        time.sleep(0.01)
        continue

    current_time = time.time()

    fps = 1.0 / max(current_time - prev_time, 1e-6)

    prev_time = current_time

    cv2.putText(
        display,
        f"FPS: {fps:.1f}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO PPE Detection", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        running = False
        break

cv2.destroyAllWindows()
