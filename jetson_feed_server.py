"""
jetson_feed_server.py
Bare-minimum low-latency feed server for Jetson Orin Nano. NO PTZ, NO patrol, NO YOLO.

Routes:
  /feed              -> raw RTSP camera frame (unchanged from original)
  /feed/0            -> raw camera frame (alias)
  /feed/<camera_id>  -> raw camera frame (alias)
  /annotated_feed    -> NEW: YOLO-annotated frame when pushed by patrol_track_main.py;
                        falls back to raw frame in standalone mode (no YOLO running).
  /feed/annotated    -> same as /annotated_feed

Standalone use (no YOLO):
  python3 jetson_feed_server.py
  -> Opens the RTSP stream directly and serves raw frames on both /feed and
     /annotated_feed (annotated_feed = raw in this mode since there is no
     YOLO inference running here).

Integrated use (with patrol_track_main.py):
  patrol_track_main.py now uses the FeedServer class from ptz_shared.py,
  which runs its own Flask instance on the same port with the YOLO-annotated
  frames pushed from the detection loop.  This file is kept as a standalone
  fallback / alternative launch option.
"""

import os
import time
import threading

import cv2
from flask import Flask, Response

RTSP_URL  = "rtsp://admin:Admin@123@10.1.68.45:554/unicaststream/1"
HTTP_PORT = 8080

app = Flask(__name__)

# =====================================================
# LOW-LATENCY GSTREAMER / FFMPEG CAPTURE
# =====================================================
# Tries Jetson hardware H.264 decode (nvv4l2decoder) first;
# falls back to FFMPEG TCP if GStreamer is unavailable.

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


# =====================================================
# MJPEG BROADCASTER
# =====================================================
# Thread-safe, condition-variable-driven broadcaster.
# update() encodes frame to JPEG and wakes all connected clients.
# generate() is a streaming generator used by Flask Response.

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


# =====================================================
# BROADCASTERS
# =====================================================
# broadcaster         -> raw RTSP frames  (/feed)
# annotated_broadcaster -> YOLO-annotated frames (/annotated_feed)
#
# In standalone mode annotated_broadcaster receives raw frames too
# (capture_loop pushes the same frame to both), so /annotated_feed
# is always functional even without a YOLO process running.
#
# When patrol_track_main.py is running it uses the FeedServer class
# from ptz_shared.py instead, which pushes actual YOLO-annotated
# frames to its own annotated broadcaster.

broadcaster           = Broadcaster(quality=75)   # raw   - slightly lower quality for bandwidth
annotated_broadcaster = Broadcaster(quality=80)   # annotated / YOLO bounding-box frames

running = True


# =====================================================
# NEW: update_annotated_frame() - called by external code
# =====================================================
# If patrol_track_main.py is imported alongside this module in a
# shared-process setup, it can push YOLO-annotated frames here.
# In standalone mode this function is never called.

def update_annotated_frame(frame):
    """Push a YOLO-annotated frame to the annotated MJPEG broadcaster."""
    annotated_broadcaster.update(frame)


# =====================================================
# CAPTURE LOOP (standalone mode)
# =====================================================

def capture_loop():
    cap = _get_video_capture(RTSP_URL)
    print("RTSP opened:", cap.isOpened())

    while running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        # Push raw frame to both broadcasters.
        # annotated_broadcaster gets raw as fallback; when YOLO is running
        # in patrol_track_main.py, it uses FeedServer instead of this file.
        broadcaster.update(frame)
        annotated_broadcaster.update(frame)

    cap.release()


# =====================================================
# FLASK ROUTES
# =====================================================

def _mjpeg_response(bc):
    res = Response(
        bc.generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    res.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    res.headers["Pragma"] = "no-cache"
    res.headers["Expires"] = "0"
    return res


# --- Raw feed (unchanged from original) ---

@app.route("/feed")
@app.route("/feed/0")
@app.route("/feed/<path:camera_id>")
def feed(camera_id=None):
    return _mjpeg_response(broadcaster)


# --- NEW: Annotated feed ---
# In standalone mode: mirrors raw frames (no YOLO).
# When used with patrol_track_main.py via FeedServer class: receives
# actual YOLO-annotated frames pushed by the detection loop.

@app.route("/annotated_feed")
@app.route("/feed/annotated")
def annotated_feed():
    return _mjpeg_response(annotated_broadcaster)


# --- Index ---

@app.route("/")
def index():
    return (
        "Jetson feed server is running.<br>"
        "<a href='/feed'>/feed</a> &mdash; raw camera stream<br>"
        "<a href='/annotated_feed'>/annotated_feed</a> "
        "&mdash; YOLO annotated stream (raw fallback in standalone mode)"
    )


# =====================================================
# ENTRY POINT (standalone mode)
# =====================================================

if __name__ == "__main__":
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)   # suppress per-request Flask logs

    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    print(f"Serving raw feed at        http://0.0.0.0:{HTTP_PORT}/feed")
    print(f"Serving annotated feed at  http://0.0.0.0:{HTTP_PORT}/annotated_feed")
    print("(annotated_feed = raw frame in standalone mode - no YOLO running)")
    app.run(host="0.0.0.0", port=HTTP_PORT, threaded=True, use_reloader=False)