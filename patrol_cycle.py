import requests
import xml.etree.ElementTree as ET
import threading
import time

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

# Speed used for the zoom-only move. This is independent of
# the pan/tilt speed.
ZOOM_SPEED = 0.5

# How long (seconds) the camera stays parked at HOME while
# the zoom motor physically moves to the new position before
# any pan/tilt movement is allowed to start. This is what
# prevents "turning and zooming at the same time".
ZOOM_HOLD_TIME = 10.0

# Pan/tilt speed is reduced by 25% at every new zoom stage
# (higher zoom -> slower pan/tilt, since the field of view is
# narrower and fast pans look jumpy).
SPEED_REDUCTION_FACTOR = 0.75

SPEED_STAGES = [
    PATROL_SPEED * (SPEED_REDUCTION_FACTOR ** i)
    for i in range(len(ZOOM_STAGES))
]

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

    def send_soap(self, body):

        soap = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
<s:Body>
{body}
</s:Body>
</s:Envelope>"""

        return self.session.post(
            PTZ_URL,
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

            response = self.send_soap(body)

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
    # Used ONLY for pan/tilt sweeps. Zoom is intentionally
    # held fixed at whatever it currently is - it must NOT
    # be changed here, otherwise the camera pans/tilts and
    # zooms at the same time.

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

        self.send_soap(body)

    # =================================================
    # ZOOM-ONLY MOVE
    # =================================================
    # Used ONLY while the camera is parked at HOME. Pan/tilt
    # speed is set to 0 so only the zoom motor moves.

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

        self.send_soap(body)

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

    # Fixed hold time so the zoom motor finishes moving
    # before any pan/tilt motion starts.

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

    move_and_wait(
        ptz,
        LEFT_PAN,
        zoom,
        speed,
        "LEFT"
    )

    time.sleep(DWELL_TIME)

    move_and_wait(
        ptz,
        HOME_PAN,
        zoom,
        speed,
        "HOME"
    )

    time.sleep(DWELL_TIME)

    move_and_wait(
        ptz,
        RIGHT_PAN,
        zoom,
        speed,
        "RIGHT"
    )

    time.sleep(DWELL_TIME)

    move_and_wait(
        ptz,
        HOME_PAN,
        zoom,
        speed,
        "HOME"
    )

    time.sleep(DWELL_TIME)

# =====================================================
# PATROL (5-STAGE ZOOM PATROL)
# =====================================================

def patrol():

    ptz = PTZController()

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

                # Only change zoom (and hold) if it actually
                # differs from the current zoom level. The
                # camera must be sitting still at HOME while
                # this happens.

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

            # Completed all 5 zoom stages.
            # Camera is currently at HOME with stage-5 zoom.
            # Reset zoom back to stage 1 (holding at HOME
            # while it happens) before starting next round.

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

    except KeyboardInterrupt:

        print("\nStopping patrol...")

    finally:

        ptz.stop_polling()

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    patrol()
