from ultralytics import YOLO
import cv2
import threading
import time

model = YOLO("best.pt")

# ✅ Change this URL to match your camera credentials
RTSP_URL = "rtsp://admin:Admin@123@192.168.1.126:554/unicaststream/1"

cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
            results = model(frame, imgsz=320, device=0, verbose=False)
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
        print("Camera read failed / stream disconnected.")
        break

    with lock:
        latest_frame = frame.copy()
        display = latest_result.copy() if latest_result is not None else frame.copy()

    now = time.time()
    fps = 1 / (now - prev_time + 1e-9)
    prev_time = now

    cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO - SATATYA PTZ", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()