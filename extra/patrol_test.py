"""
patrol_test.py
=====================================================
STANDALONE PATROL-ONLY TEST SCRIPT (with live camera feed)
=====================================================
This file contains ONLY the patrol (PTZ sweep) logic, extracted
from patrol_track_main.py + ptz_shared.py, plus a live RTSP
preview window so you can watch the patrol happen. There is:
  - NO YOLO / model loading
  - NO person detection
  - NO ByteTrack / tracking

Run this file on its own to verify the camera patrol sweep
(4 zoom/tilt stages x LEFT/HOME/RIGHT/HOME) works correctly over
ONVIF, independent of any detection code.

All speeds you'll want to tune are grouped at the top under
"STAGE SPEEDS - EDIT ME".

Press 'q' in the video window (or Ctrl+C in the terminal) to stop.
"""

import time
import threading

import cv2
import requests
import xml.etree.ElementTree as ET

from requests.auth import HTTPDigestAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =====================================================
# CAMERA SETTINGS
# =====================================================

CAMERA_IP = "10.1.68.45"
USERNAME = "admin"
PASSWORD = "Admin@123"

PROFILE_TOKEN = "profiletoken01"

PTZ_URL = f"http://{CAMERA_IP}/onvif/ptz_service"
MEDIA_URL = f"http://{CAMERA_IP}/onvif/media_service"

# Known RTSP stream URL (credentials already embedded) - used only
# for the live preview window, has nothing to do with PTZ control.
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/unicaststream/1"

WINDOW_NAME = "Patrol Test - Live Feed"

# Resize the incoming stream before display (set both to None to
# show the stream at its native resolution).
PROCESS_WIDTH = 960
PROCESS_HEIGHT = 540

# =====================================================
# PATROL SETTINGS (absolute-move sweep)
# =====================================================

HOME_PAN = -0.899
LEFT_PAN = -0.600
RIGHT_PAN = 0.770

# Exact stop time at each waypoint (LEFT/HOME/RIGHT) ONCE the
# camera has actually arrived there. This is a clean, deterministic
# pause - it does not include however long the physical move itself
# took to get there.
HOLD_TIME = 2.0

# How close (in normalized pan units) counts as "arrived". Loosened
# from 0.015 -> 0.03 to match real PTZ head precision/backlash -
# too tight a tolerance means the camera can be sitting still,
# physically arrived, and STILL never read as "REACHED".
POSITION_TOLERANCE = 0.03

# =====================================================
# ZOOM / TILT STAGE SETTINGS
# =====================================================
# Stage 0 -> zoom 0.00, tilt  0.25
# Stage 1 -> zoom 0.00, tilt -0.75
# Stage 2 -> zoom 0.38, tilt -0.75
# Stage 3 -> zoom 0.75, tilt -0.75

ZOOM_STAGES = [0.00, 0.00, 0.38, 0.75]
TILT_STAGES = [0.25, -0.75, -0.75, -0.75]

ZOOM_SPEED = 0.5

# Bumped up a bit from 10 -> 12s: since TILT_ZOOM_SPEED below now
# slows the tilt jump down, it needs a little more time to finish
# before the next leg starts. Raise further if you still see the
# first LEFT/RIGHT move of a new stage start before tilt settles.
ZOOM_HOLD_TIME = 12.0

# Pan/tilt speed used ONLY while applying a new zoom/tilt stage at
# HOME (i.e. the tilt jump between stages, e.g. 0.25 -> -0.75).
# Previously this was left at 0.0, which most ONVIF cameras treat
# as "move at full/default speed" rather than "don't move" - that
# is why the tilt snapped so fast between stages (most noticeable
# going from Stage 3 back to Stage 0). Lower this to slow it down.
TILT_ZOOM_SPEED = 0.08

# =====================================================
# STAGE SPEEDS - EDIT ME
# =====================================================
# One pan/tilt speed per stage (same indexing as ZOOM_STAGES /
# TILT_STAGES above: index 0 = Stage 0, index 1 = Stage 1, etc).
#
# If a stage is taking too long, just raise its number below.
# Valid range is roughly 0.0 (won't move) to 1.0 (fastest).
#
#              Stage 0   Stage 1   Stage 2   Stage 3
SPEED_STAGES = [0.20,     0.11,     0.07,     0.03]

LOOP_POLL_INTERVAL = 0.1

# Hard safety-net ceiling on the MOVE phase only (in seconds) - if
# the camera genuinely never reaches the target (mechanical fault,
# blocked head, etc.) the wait gives up after this long and the
# patrol moves on rather than hanging forever. This is NOT the
# hold time - it is decoupled from HOLD_TIME above so a slow/failed
# arrival can never masquerade as "holding at the point".
ARRIVAL_MAX_WAIT = 10.0

# Number of consecutive polls (at LOOP_POLL_INTERVAL spacing) that
# must read within POSITION_TOLERANCE before arrival is confirmed.
# This debounce avoids false "REACHED" on a single noisy/lucky
# reading while the camera is still actually moving.
ARRIVAL_DEBOUNCE_POLLS = 3


# =====================================================
# MINIMAL SHARED STATE (status text only - no
# detection/tracking events since there is no detection
# thread in this test script)
# =====================================================

class SharedState:

    def __init__(self):
        self._lock = threading.Lock()
        self.status_text = "Patrol - Stage 0"

        # Kept so PTZController's interruptible wait/dwell code is
        # unchanged from the original - but nothing ever sets this
        # event in this test script, so patrol never gets
        # interrupted and just runs forever.
        self.person_trigger_event = threading.Event()

    def set_status(self, text):
        with self._lock:
            self.status_text = text

    def get_status(self):
        with self._lock:
            return self.status_text


# =====================================================
# THREADED FRAME GRABBER (live preview only - not used for
# any patrol logic)
# =====================================================

class FrameGrabber:

    def __init__(self, rtsp_url):

        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
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

        self.thread = threading.Thread(target=loop, daemon=True)
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
# PTZ CONTROLLER (ONVIF SOAP - AbsoluteMove + polling)
# =====================================================

class PTZController:

    def __init__(self):
        self.session = requests.Session()

        retry = Retry(total=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)

        self.auth = HTTPDigestAuth(USERNAME, PASSWORD)

        self.pan = None
        self.tilt = None

        self.running = False
        self.poll_thread = None

    def send_soap(self, url, body, quiet=False):

        soap = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
<s:Body>
{body}
</s:Body>
</s:Envelope>"""

        try:
            response = self.session.post(
                url,
                data=soap,
                headers={"Content-Type": "application/soap+xml"},
                auth=self.auth,
                timeout=10
            )
        except requests.RequestException as exc:
            if not quiet:
                print(f"ONVIF request to {url} failed: {exc}")
            return None

        if not quiet:
            if response.status_code >= 300:
                print(
                    f"ONVIF command to {url} failed: "
                    f"HTTP {response.status_code} - {response.text[:300]}"
                )
            elif "Fault" in response.text:
                print(
                    f"ONVIF command to {url} returned a SOAP "
                    f"Fault: {response.text[:300]}"
                )

        return response

    def get_status(self):

        body = f"""
<tptz:GetStatus
xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
<tptz:ProfileToken>{PROFILE_TOKEN}</tptz:ProfileToken>
</tptz:GetStatus>
"""
        try:
            response = self.send_soap(PTZ_URL, body, quiet=True)
            if response is None:
                return

            root = ET.fromstring(response.text)
            for elem in root.iter():
                if elem.tag.endswith("PanTilt"):
                    self.pan = float(elem.attrib["x"])
                    self.tilt = float(elem.attrib["y"])
                    return
        except Exception:
            pass

    def start_polling(self):
        self.running = True

        def loop():
            while self.running:
                self.get_status()
                time.sleep(0.2)

        self.poll_thread = threading.Thread(target=loop, daemon=True)
        self.poll_thread.start()

    def stop_polling(self):
        self.running = False
        if self.poll_thread:
            self.poll_thread.join(timeout=1)

    def absolute_move(self, pan, tilt, zoom, speed=SPEED_STAGES[0]):

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

    def move_zoom(self, pan, tilt, zoom, zoom_speed=ZOOM_SPEED, tilt_speed=TILT_ZOOM_SPEED):

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
<tt:PanTilt x="{tilt_speed}" y="{tilt_speed}" />
<tt:Zoom x="{zoom_speed}" />
</tptz:Speed>

</tptz:AbsoluteMove>
"""
        self.send_soap(PTZ_URL, body)

    def stop_move(self):

        body = f"""
<tptz:Stop
xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
<tptz:ProfileToken>{PROFILE_TOKEN}</tptz:ProfileToken>
<tptz:PanTilt>true</tptz:PanTilt>
<tptz:Zoom>true</tptz:Zoom>
</tptz:Stop>
"""
        self.send_soap(PTZ_URL, body)

    def wait_until_reached_interruptible(
        self, shared_state, target_pan, tilt, zoom, speed,
        tolerance=POSITION_TOLERANCE
    ):

        start_time = time.time()
        in_tolerance_streak = 0

        while True:

            if shared_state.person_trigger_event.is_set():
                self.stop_move()
                return "INTERRUPTED"

            if time.time() - start_time >= ARRIVAL_MAX_WAIT:
                print(f"Arrival timeout ({ARRIVAL_MAX_WAIT:.0f}s) - moving on")
                return "TIMEOUT"

            current_pan = self.pan

            if current_pan is None:
                time.sleep(LOOP_POLL_INTERVAL)
                continue

            error_direct = abs(current_pan - target_pan)
            error_wrap = 2.0 - error_direct
            error = min(error_direct, error_wrap)

            if error < tolerance:
                in_tolerance_streak += 1
                if in_tolerance_streak >= ARRIVAL_DEBOUNCE_POLLS:
                    return "REACHED"
            else:
                in_tolerance_streak = 0

            time.sleep(LOOP_POLL_INTERVAL)


# =====================================================
# PATROL LOGIC (same state machine as patrol_track_main.py,
# minus the person-interruption handoff to tracking - that
# branch can never trigger here since nothing ever sets
# person_trigger_event)
# =====================================================

def move_and_wait(ptz, shared_state, pan, tilt, zoom, speed, label, stage_number):

    print(f"\nMoving to {label} (stage {stage_number}, speed={speed:.4f}, tilt={tilt:.2f}, zoom={zoom:.2f})")
    shared_state.set_status(f"Patrol - Stage {stage_number} - Moving to {label}")

    ptz.absolute_move(pan, tilt, zoom, speed)

    result = ptz.wait_until_reached_interruptible(shared_state, pan, tilt, zoom, speed)

    if result == "REACHED":
        print(f"Reached {label} (pan={pan}, tilt={tilt:.2f}, zoom={zoom:.2f})")
    else:
        print(f"Did not confirm arrival at {label} ({result}) - continuing patrol anyway")


def hold_at_point(shared_state, seconds, stage_number):
    print(f"Holding for {seconds:.1f}s (stage {stage_number})")
    time.sleep(seconds)


def apply_zoom_at_home(ptz, shared_state, tilt, zoom, speed, stage_number):

    print(f"\nHolding at HOME to apply stage {stage_number} (tilt={tilt:.2f}, zoom={zoom:.2f})")
    shared_state.set_status(f"Patrol - applying stage {stage_number}")

    ptz.move_zoom(HOME_PAN, tilt, zoom)
    time.sleep(ZOOM_HOLD_TIME)

    print(f"Stage {stage_number} (tilt/zoom) applied.")


def run_cycle(ptz, shared_state, tilt, zoom, speed, cycle_number, stage_number):

    print(
        f"\n========== CYCLE {cycle_number} "
        f"(STAGE {stage_number}, tilt={tilt:.2f}, zoom={zoom:.2f}, speed={speed:.4f}) =========="
    )

    move_and_wait(ptz, shared_state, LEFT_PAN, tilt, zoom, speed, "LEFT", stage_number)
    hold_at_point(shared_state, HOLD_TIME, stage_number)

    move_and_wait(ptz, shared_state, HOME_PAN, tilt, zoom, speed, "HOME", stage_number)
    hold_at_point(shared_state, HOLD_TIME, stage_number)

    move_and_wait(ptz, shared_state, RIGHT_PAN, tilt, zoom, speed, "RIGHT", stage_number)
    hold_at_point(shared_state, HOLD_TIME, stage_number)

    move_and_wait(ptz, shared_state, HOME_PAN, tilt, zoom, speed, "HOME", stage_number)
    hold_at_point(shared_state, HOLD_TIME, stage_number)


def patrol(ptz, shared_state):

    ptz.start_polling()

    try:
        print("Moving to HOME")

        move_and_wait(
            ptz, shared_state, HOME_PAN, TILT_STAGES[0], ZOOM_STAGES[0], SPEED_STAGES[0], "HOME", 0
        )
        hold_at_point(shared_state, HOLD_TIME, 0)

        last_tilt = TILT_STAGES[0]
        last_zoom = ZOOM_STAGES[0]
        cycle = 1

        while True:

            for stage_index, zoom in enumerate(ZOOM_STAGES):

                stage_number = stage_index
                tilt = TILT_STAGES[stage_index]
                speed = SPEED_STAGES[stage_index]

                if zoom != last_zoom or tilt != last_tilt:
                    apply_zoom_at_home(ptz, shared_state, tilt, zoom, speed, stage_number)
                    last_zoom = zoom
                    last_tilt = tilt

                run_cycle(ptz, shared_state, tilt, zoom, speed, cycle, stage_number)
                cycle += 1

            print(f"\nCompleted all {len(ZOOM_STAGES)} stages. Resetting to stage 0 at HOME.")
            apply_zoom_at_home(ptz, shared_state, TILT_STAGES[0], ZOOM_STAGES[0], SPEED_STAGES[0], 0)
            last_tilt = TILT_STAGES[0]
            last_zoom = ZOOM_STAGES[0]

    except Exception as exc:
        print(f"\nPatrol stopped: {exc}")

    finally:
        ptz.stop_polling()


# =====================================================
# LIVE CAMERA FEED (main thread - just a preview window,
# has no effect on patrol logic)
# =====================================================

def show_camera_feed(shared_state):

    print(f"\nOpening camera stream: {RTSP_URL}")
    grabber = FrameGrabber(RTSP_URL).start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:

            ret, frame = grabber.read()
            if not ret:
                time.sleep(0.05)
                continue

            if PROCESS_WIDTH and PROCESS_HEIGHT:
                frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

            cv2.putText(
                frame, shared_state.get_status(),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n'q' pressed, stopping.")
                break

    finally:
        grabber.stop()
        cv2.destroyAllWindows()


# =====================================================
# MAIN
# =====================================================

def main():

    shared_state = SharedState()
    ptz = PTZController()

    # Patrol runs in the background; the live preview window
    # needs to run on the main thread.
    patrol_thread = threading.Thread(
        target=patrol, args=(ptz, shared_state), daemon=True
    )
    patrol_thread.start()

    try:
        show_camera_feed(shared_state)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ptz.stop_move()
        ptz.stop_polling()


if __name__ == "__main__":
    main()