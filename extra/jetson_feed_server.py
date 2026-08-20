"""
jetson_feed_server.py
Bare-minimum feed test. NO PTZ, NO patrol, NO YOLO.
"""

from flask import Flask, Response
import cv2
import threading
import time

RTSP_URL = "rtsp://admin:Admin@123@192.168.1.126:554/unicaststream/1"
HTTP_PORT = 8080

app = Flask(__name__)

lock = threading.Lock()
latest_frame = None
running = True


def capture_loop():
    global latest_frame
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("RTSP opened:", cap.isOpened())

    while running:
        ok, frame = cap.read()
        if not ok:
            print("frame read failed, retrying...")
            time.sleep(0.5)
            continue
        with lock:
            latest_frame = frame


def mjpeg_stream():
    while True:
        with lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )


@app.route("/feed/0")
def feed():
    return Response(
        mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/")
def index():
    return "Jetson feed server is running. Go to /feed to view the stream."


if __name__ == "__main__":
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    print(f"Serving feed at http://0.0.0.0:{HTTP_PORT}/feed")
    app.run(host="0.0.0.0", port=HTTP_PORT, threaded=True)