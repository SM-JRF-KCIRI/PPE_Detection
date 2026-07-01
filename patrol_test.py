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

TILT = -0.656
ZOOM = 0.0

PATROL_SPEED = 0.10
DWELL_TIME = 2.0

POSITION_TOLERANCE = 0.015

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
    # ABSOLUTE MOVE
    # =================================================

    def absolute_move(
        self,
        pan,
        tilt,
        zoom=0.0,
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
# MOVE HELPER
# =====================================================

def move_and_wait(
    ptz,
    pan,
    label
):

    print(f"\nMoving to {label}")

    ptz.absolute_move(
        pan,
        TILT,
        ZOOM,
        PATROL_SPEED
    )

    ptz.wait_until_reached(pan)

    print(
        f"Reached {label} "
        f"(pan={pan})"
    )

# =====================================================
# PATROL
# =====================================================

def patrol():

    ptz = PTZController()

    ptz.start_polling()

    try:

        print("Moving to HOME")

        move_and_wait(
            ptz,
            HOME_PAN,
            "HOME"
        )

        time.sleep(DWELL_TIME)

        cycle = 1

        while True:

            print(
                f"\n========== "
                f"CYCLE {cycle} "
                f"=========="
            )

            move_and_wait(
                ptz,
                LEFT_PAN,
                "LEFT"
            )

            time.sleep(DWELL_TIME)

            move_and_wait(
                ptz,
                HOME_PAN,
                "HOME"
            )

            time.sleep(DWELL_TIME)

            move_and_wait(
                ptz,
                RIGHT_PAN,
                "RIGHT"
            )

            time.sleep(DWELL_TIME)

            move_and_wait(
                ptz,
                HOME_PAN,
                "HOME"
            )

            time.sleep(DWELL_TIME)

            cycle += 1

    except KeyboardInterrupt:

        print("\nStopping patrol...")

    finally:

        ptz.stop_polling()

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    patrol()
