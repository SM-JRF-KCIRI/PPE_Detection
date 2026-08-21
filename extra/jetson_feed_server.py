"""
jetson_feed_server.py
Bare-minimum low-latency feed server for Jetson Orin Nano. NO PTZ, NO patrol, NO YOLO.
"""

import os
import time
import threading
import cv2
from flask import Flask, Response

RTSP_URL = "rtsp://admin:Admin@123@192.168.1.126:554/unicaststream/1"
HTTP_PORT = 8080

app = Flask(__name__)

# Low-latency GStreamer pipeline for Jetson Orin Nano
def _get_video_capture(rtsp_url):
    gst_pipeline = (
        f"rtspsrc location=\"{rtsp_url}\" protocols=tcp latency=0 drop-on-latency=true ! "
        "rtph264depay ! h264parse ! "
        "nvv4l2decoder enable-max-performance=1 drop-frame-interval=0 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("Using Jetson Hardware GStreamer Decoder (nvv4l2decoder).")
        return cap
    
    cap.release()
    print("GStreamer hardware decode unavailable, using optimized FFMPEG TCP.")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000|reorder_queue_size;0"
    )
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class Broadcaster:
    def __init__(self, quality=80):
        self.params = [int(cv2.IMWRITE_JPEG_QUALITY), quality, int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.jpeg_bytes = None
        self.seq = 0

    def update(self, frame):
        if frame is None:
            return
        ok, buf = cv2.imencode(".jpg", frame, self.params)
        if ok:
            with self.cond:
                self.jpeg_bytes = buf.tobytes()
                self.seq += 1
                self.cond.notify_all()

    def generate(self):
        last_seq = 0
        while True:
            with self.cond:
                while self.seq == last_seq or self.jpeg_bytes is None:
                    if not self.cond.wait(timeout=0.1):
                        break
                data = self.jpeg_bytes
                last_seq = self.seq
            if data is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                    + data + b"\r\n"
                )


broadcaster = Broadcaster(quality=80)
running = True


def capture_loop():
    cap = _get_video_capture(RTSP_URL)
    print("RTSP opened:", cap.isOpened())

    while running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        broadcaster.update(frame)
    cap.release()


@app.route("/feed")
@app.route("/feed/0")
@app.route("/feed/<path:camera_id>")
def feed(camera_id=None):
    res = Response(
        broadcaster.generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    res.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    res.headers["Pragma"] = "no-cache"
    res.headers["Expires"] = "0"
    return res


@app.route("/")
def index():
    return "Jetson feed server is running. Go to <a href='/feed'>/feed</a> to view the stream."


if __name__ == "__main__":
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    print(f"Serving feed at http://0.0.0.0:{HTTP_PORT}/feed")
    app.run(host="0.0.0.0", port=HTTP_PORT, threaded=True, use_reloader=False)