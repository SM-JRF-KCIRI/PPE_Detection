import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;102400"

import requests
import xml.etree.ElementTree as ET
import threading
import time

import cv2
from ultralytics import YOLO

from requests.auth import HTTPDigestAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =====================================================
# CAMERA SETTINGS
# =====================================================

CAMERA_IP = "192.168.1.126"
USERNAME = "admin"
PASSWORD = "Admin@123"

PROFILE_TOKEN = "profiletoken01"

PTZ_URL = f"http://{CAMERA_IP}/onvif/ptz_service"
MEDIA_URL = f"http://{CAMERA_IP}/onvif/media_service"

# Known RTSP stream URL (credentials already embedded).
# Using this directly skips the ONVIF GetStreamUri call.
RTSP_URL = "rtsp://admin:Admin@123@192.168.1.126:554/unicaststream/1"

# =====================================================
# PATROL SETTINGS
# =====================================================

HOME_PAN = -0.899
LEFT_PAN = -0.473
RIGHT_PAN = 0.539

TILT = -0.75

PATROL_SPEED = 0.10
DWELL_TIME = 2.0

POSITION_TOLERANCE = 0.015

# =====================================================
# ZOOM STAGE SETTINGS
# =====================================================
# The camera has 38x optical zoom. ONVIF zoom is normalized
# between 0.0 (widest, 1x) and 1.0 (full 38x). We split that
# range into 5 even stages:
#
#   Stage 1 -> 0.00  (~1x)
#   Stage 2 -> 0.25  (~10.5x)
#   Stage 3 -> 0.50  (~19.5x)
#   Stage 4 -> 0.75  (~28.5x)
#   Stage 5 -> 1.00  (~38x)

ZOOM_STAGES = [0.00, 0.25, 0.50, 0.75, 1.00]

# Speed used for the zoom-only move. Independent of pan/tilt speed.
ZOOM_SPEED = 0.5

# How long (seconds) the camera stays parked at HOME while the
# zoom motor physically moves before any pan/tilt is allowed.
ZOOM_HOLD_TIME = 10.0

# Pan/tilt speed is reduced by 25% at every new zoom stage.
SPEED_REDUCTION_FACTOR = 0.75

SPEED_STAGES = [
    PATROL_SPEED * (SPEED_REDUCTION_FACTOR ** i)
    for i in range(len(ZOOM_STAGES))
]

# =====================================================
# YOLO / PPE DETECTION SETTINGS
# =====================================================

YOLO_MODEL_PATH = "vel best.pt"      # your trained YOLOv8m PPE model
YOLO_CONF_THRESHOLD = 0.5
YOLO_IMG_SIZE = 480                   # lower = faster inference

# Resize incoming frames before inference/display. Set to None
# to keep the camera's native resolution.
PROCESS_WIDTH = 960
PROCESS_HEIGHT = 540

# Laptop RTX 3050 -> use GPU 0. Falls back to CPU automatically
# inside run_detection() if CUDA isn't available.
YOLO_DEVICE = 0

WINDOW_NAME = "PPE Detection - Live Feed"

# =====================================================
# THREADED FRAME GRABBER
# =====================================================
# Reads frames from the RTSP stream as fast as the camera
# sends them, in its own thread, and always keeps only the
# MOST RECENT frame. This is what stops the feed from
# "freezing" - without it, if YOLO inference is slower than
# the camera's frame rate, OpenCV's internal buffer fills up
# and you end up watching older and older frames until it
# looks stuck.

class FrameGrabber:

    def __init__(self, rtsp_url):

        self.rtsp_url = rtsp_url

        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        # Ask OpenCV to keep as small a buffer as possible.
        # Not all backends honor this, which is why the
        # background thread below is the real fix.

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():

            raise RuntimeError(
                "Could not open RTSP stream. Check the URI, "
                "credentials, and network access to the camera."
            )

        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = False
        self.thread = None

    def start(self):

        self.running = True

        def loop():

            while self.running:

                ret, frame = self.cap.read()

                if not ret:

                    time.sleep(0.05)
                    continue

                with self.lock:

                    self.latest_frame = frame

        self.thread = threading.Thread(
            target=loop,
            daemon=True
        )

        self.thread.start()

        return self

    def read(self):

        with self.lock:

            if self.latest_frame is None:
                return False, None

            return True, self.latest_frame.copy()

    def stop(self):

        self.running = False

        if self.thread:
            self.thread.join(timeout=1)

        self.cap.release()

# =====================================================
# PTZ CONTROLLER
# =====================================================

class PTZController:

    def __init__(self):

        self.session = requests.Session()

        retry = Retry(
            total=3,
            backoff_factor=0.5
        )

        adapter = HTTPAdapter(max_retries=retry)

        self.session.mount("http://", adapter)

        self.auth = HTTPDigestAuth(
            USERNAME,
            PASSWORD
        )

        self.pan = None
        self.tilt = None

        self.running = False
        self.poll_thread = None

    # =================================================
    # SOAP
    # =================================================

    def send_soap(self, url, body):

        soap = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
<s:Body>
{body}
</s:Body>
</s:Envelope>"""

        return self.session.post(
            url,
            data=soap,
            headers={
                "Content-Type": "application/soap+xml"
            },
            auth=self.auth,
            timeout=10
        )

    # =================================================
    # STATUS
    # =================================================

    def get_status(self):

        body = f"""
<tptz:GetStatus
xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
<tptz:ProfileToken>{PROFILE_TOKEN}</tptz:ProfileToken>
</tptz:GetStatus>
"""

        try:

            response = self.send_soap(PTZ_URL, body)

            root = ET.fromstring(response.text)

            for elem in root.iter():

                if elem.tag.endswith("PanTilt"):

                    self.pan = float(elem.attrib["x"])
                    self.tilt = float(elem.attrib["y"])

                    return

        except Exception:
            pass

    # =================================================
    # POLLING THREAD
    # =================================================

    def start_polling(self):

        self.running = True

        def loop():

            while self.running:

                self.get_status()

                time.sleep(0.2)

        self.poll_thread = threading.Thread(
            target=loop,
            daemon=True
        )

        self.poll_thread.start()

    def stop_polling(self):

        self.running = False

        if self.poll_thread:
            self.poll_thread.join(timeout=1)

    # =================================================
    # ABSOLUTE MOVE (PAN / TILT)
    # =================================================
    # Used ONLY for pan/tilt sweeps. Zoom is held fixed here -
    # it must NOT change during a pan/tilt move.

    def absolute_move(
        self,
        pan,
        tilt,
        zoom,
        speed=PATROL_SPEED
    ):

        body = f"""
<tptz:AbsoluteMove
xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
xmlns:tt="http://www.onvif.org/ver10/schema">

<tptz:ProfileToken>{PROFILE_TOKEN}</tptz:ProfileToken>

<tptz:Position>
<tt:PanTilt x="{pan}" y="{tilt}" />
<tt:Zoom x="{zoom}" />
</tptz:Position>

<tptz:Speed>
<tt:PanTilt x="{speed}" y="{speed}" />
<tt:Zoom x="0.0" />
</tptz:Speed>

</tptz:AbsoluteMove>
"""

        self.send_soap(PTZ_URL, body)

    # =================================================
    # ZOOM-ONLY MOVE
    # =================================================
    # Used ONLY while parked at HOME. Pan/tilt speed is 0 so
    # only the zoom motor moves.

    def move_zoom(
        self,
        pan,
        tilt,
        zoom,
        zoom_speed=ZOOM_SPEED
    ):

        body = f"""
<tptz:AbsoluteMove
xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
xmlns:tt="http://www.onvif.org/ver10/schema">

<tptz:ProfileToken>{PROFILE_TOKEN}</tptz:ProfileToken>

<tptz:Position>
<tt:PanTilt x="{pan}" y="{tilt}" />
<tt:Zoom x="{zoom}" />
</tptz:Position>

<tptz:Speed>
<tt:PanTilt x="0.0" y="0.0" />
<tt:Zoom x="{zoom_speed}" />
</tptz:Speed>

</tptz:AbsoluteMove>
"""

        self.send_soap(PTZ_URL, body)

    # =================================================
    # WAIT FOR ARRIVAL
    # =================================================

    def wait_until_reached(
        self,
        target_pan,
        tolerance=POSITION_TOLERANCE
    ):

        stall_count = 0
        prev_pan = None

        while True:

            current_pan = self.pan

            if current_pan is None:

                time.sleep(0.2)
                continue

            error_direct = abs(
                current_pan - target_pan
            )

            error_wrap = 2.0 - error_direct

            error = min(
                error_direct,
                error_wrap
            )

            if error < tolerance:

                return True

            # Stall detection

            if prev_pan is not None:

                if abs(current_pan - prev_pan) < 0.001:

                    stall_count += 1

                    if stall_count >= 15:

                        print(
                            f"STALL detected near "
                            f"{current_pan:.4f}"
                        )

                        return False

                else:

                    stall_count = 0

            prev_pan = current_pan

            time.sleep(0.2)

    # =================================================
    # GET RTSP STREAM URI (ONVIF MEDIA SERVICE)
    # =================================================

    def get_stream_uri(self):

        body = f"""
<trt:GetStreamUri
xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
xmlns:tt="http://www.onvif.org/ver10/schema">

<trt:StreamSetup>
<tt:Stream>RTP-Unicast</tt:Stream>
<tt:Transport>
<tt:Protocol>RTSP</tt:Protocol>
</tt:Transport>
</trt:StreamSetup>

<trt:ProfileToken>{PROFILE_TOKEN}</trt:ProfileToken>

</trt:GetStreamUri>
"""

        response = self.send_soap(MEDIA_URL, body)

        root = ET.fromstring(response.text)

        for elem in root.iter():

            if elem.tag.endswith("Uri"):

                return elem.text.strip()

        raise RuntimeError(
            "Could not retrieve RTSP stream URI "
            "from camera. Check MEDIA_URL / "
            "PROFILE_TOKEN."
        )

# =====================================================
# MOVE HELPER (PAN/TILT ONLY, ZOOM FIXED)
# =====================================================

def move_and_wait(
    ptz,
    pan,
    zoom,
    speed,
    label
):

    print(
        f"\nMoving to {label} "
        f"(speed={speed:.4f}, zoom={zoom:.2f})"
    )

    ptz.absolute_move(
        pan,
        TILT,
        zoom,
        speed
    )

    ptz.wait_until_reached(pan)

    print(
        f"Reached {label} "
        f"(pan={pan}, zoom={zoom:.2f})"
    )

# =====================================================
# APPLY ZOOM WHILE PARKED AT HOME
# =====================================================

def apply_zoom_at_home(ptz, zoom, stage_number):

    print(
        f"\nHolding at HOME to apply "
        f"zoom stage {stage_number} "
        f"(zoom={zoom:.2f})"
    )

    ptz.move_zoom(
        HOME_PAN,
        TILT,
        zoom
    )

    time.sleep(ZOOM_HOLD_TIME)

    print(f"Zoom stage {stage_number} applied.")

# =====================================================
# ONE FULL LEFT-HOME-RIGHT-HOME CYCLE
# (zoom fixed for the whole cycle)
# =====================================================

def run_cycle(ptz, zoom, speed, cycle_number, stage_number):

    print(
        f"\n========== "
        f"CYCLE {cycle_number} "
        f"(STAGE {stage_number}, "
        f"zoom={zoom:.2f}, speed={speed:.4f}) "
        f"=========="
    )

    move_and_wait(ptz, LEFT_PAN, zoom, speed, "LEFT")
    time.sleep(DWELL_TIME)

    move_and_wait(ptz, HOME_PAN, zoom, speed, "HOME")
    time.sleep(DWELL_TIME)

    move_and_wait(ptz, RIGHT_PAN, zoom, speed, "RIGHT")
    time.sleep(DWELL_TIME)

    move_and_wait(ptz, HOME_PAN, zoom, speed, "HOME")
    time.sleep(DWELL_TIME)

# =====================================================
# PATROL (5-STAGE ZOOM PATROL) - RUNS IN BACKGROUND THREAD
# =====================================================

def patrol(ptz):

    ptz.start_polling()

    try:

        print("Moving to HOME")

        move_and_wait(
            ptz,
            HOME_PAN,
            ZOOM_STAGES[0],
            SPEED_STAGES[0],
            "HOME"
        )

        time.sleep(DWELL_TIME)

        last_zoom = ZOOM_STAGES[0]

        cycle = 1

        while True:

            for stage_index, zoom in enumerate(ZOOM_STAGES):

                stage_number = stage_index + 1
                speed = SPEED_STAGES[stage_index]

                if zoom != last_zoom:

                    apply_zoom_at_home(
                        ptz,
                        zoom,
                        stage_number
                    )

                    last_zoom = zoom

                run_cycle(
                    ptz,
                    zoom,
                    speed,
                    cycle,
                    stage_number
                )

                cycle += 1

            print(
                "\nCompleted all 5 zoom stages. "
                "Resetting zoom to stage 1 at HOME."
            )

            apply_zoom_at_home(
                ptz,
                ZOOM_STAGES[0],
                1
            )

            last_zoom = ZOOM_STAGES[0]

    except Exception as exc:

        print(f"\nPatrol stopped: {exc}")

    finally:

        ptz.stop_polling()

# =====================================================
# YOLO PPE DETECTION - LIVE VIDEO FEED
# =====================================================

def build_authenticated_rtsp_url(raw_uri):

    # Many ONVIF cameras return an RTSP URI without
    # credentials embedded. Insert them if missing.

    if "@" in raw_uri:
        return raw_uri

    if raw_uri.startswith("rtsp://"):
        return raw_uri.replace(
            "rtsp://",
            f"rtsp://{USERNAME}:{PASSWORD}@",
            1
        )

    return raw_uri

def run_detection(rtsp_url):

    print(f"\nLoading YOLO model: {YOLO_MODEL_PATH}")

    model = YOLO(YOLO_MODEL_PATH)

    # Try GPU (RTX 3050) first, fall back to CPU if CUDA
    # isn't available in this environment.

    device = YOLO_DEVICE

    try:

        import torch

        if not torch.cuda.is_available():

            print("CUDA not available, falling back to CPU.")
            device = "cpu"

        else:

            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    except Exception:

        device = "cpu"

    print(f"\nOpening camera stream: {rtsp_url}")

    grabber = FrameGrabber(rtsp_url).start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Give the grabber a moment to receive the first frame.

    wait_start = time.time()

    while grabber.latest_frame is None:

        if time.time() - wait_start > 10:

            grabber.stop()

            raise RuntimeError(
                "No frames received from the RTSP stream "
                "within 10 seconds."
            )

        time.sleep(0.1)

    fps_timer = time.time()
    fps_counter = 0
    display_fps = 0.0

    try:

        while True:

            ret, frame = grabber.read()

            if not ret:

                time.sleep(0.01)
                continue

            if PROCESS_WIDTH and PROCESS_HEIGHT:

                frame = cv2.resize(
                    frame,
                    (PROCESS_WIDTH, PROCESS_HEIGHT)
                )

            results = model.predict(
                frame,
                device=device,
                conf=YOLO_CONF_THRESHOLD,
                imgsz=YOLO_IMG_SIZE,
                verbose=False
            )

            annotated_frame = results[0].plot()

            # Simple on-screen FPS counter, useful for tuning
            # PROCESS_WIDTH / PROCESS_HEIGHT / YOLO_IMG_SIZE.

            fps_counter += 1

            if time.time() - fps_timer >= 1.0:

                display_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            cv2.putText(
                annotated_frame,
                f"FPS: {display_fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow(WINDOW_NAME, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):

                print("\n'q' pressed, stopping detection feed.")
                break

    finally:

        grabber.stop()
        cv2.destroyAllWindows()

# =====================================================
# MAIN
# =====================================================

def main():

    # Using the known RTSP URL directly (credentials already
    # embedded), so no ONVIF GetStreamUri lookup is needed.

    stream_uri = RTSP_URL

    print(f"RTSP stream URI: {stream_uri}")

    # PTZ patrol runs in the background.

    patrol_ptz = PTZController()

    patrol_thread = threading.Thread(
        target=patrol,
        args=(patrol_ptz,),
        daemon=True
    )

    patrol_thread.start()

    # YOLO detection + display runs on the main thread
    # (cv2.imshow needs to run on the main thread).

    try:

        run_detection(stream_uri)

    except KeyboardInterrupt:

        print("\nStopping...")

if __name__ == "__main__":

    main()
