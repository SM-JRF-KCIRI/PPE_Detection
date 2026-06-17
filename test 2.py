from ultralytics import YOLO
import cv2
import threading
import time

model = YOLO("best.pt")

RTSP_URL = "rtsp://admin:Admin@123@192.168.1.126:554/unicaststream/1"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # ✅ 1080p width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # ✅ 1080p height
cap.set(cv2.CAP_PROP_FPS, 24)             # ✅ 24 FPS
cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)

if not cap.isOpened():
    print("❌ Failed to connect. Check URL or credentials.")
    exit()

# Confirm actual values camera accepted
print(f"✅ Camera connected!")
print(f"Resolution : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"FPS        : {cap.get(cv2.CAP_PROP_FPS)}")

latest_frame = None
latest_result = None
lock = threading.Lock()

def inference_thread():
    global latest_result
    while True:
        with lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.001)
            continue

        try:
            results = model(frame, imgsz=640, device=0, verbose=False)  # ✅ 640 for 1080p input
            annotated = results[0].plot()
        except Exception as e:
            print(f"Inference error: {e}")
            continue

        with lock:
            latest_result = annotated

thread = threading.Thread(target=inference_thread, daemon=True)
thread.start()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Stream lost. Reconnecting...")
        cap.open(RTSP_URL, cv2.CAP_FFMPEG)
        continue

    with lock:
        latest_frame = frame.copy()
        display = latest_result.copy() if latest_result is not None else frame.copy()

    now = time.time()
    fps = 1 / (now - prev_time + 1e-9)
    prev_time = now

    cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO - SATATYA PTZ 1080p", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()