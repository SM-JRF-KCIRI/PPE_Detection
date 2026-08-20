"""
ptz_shared.py
=====================================================
Shared infrastructure used by patrol_track_main.py:

  - camera / ONVIF / patrol / zoom-stage / YOLO / tracking config
  - FrameGrabber          (threaded "always latest frame" RTSP reader)
  - PTZController         (ONVIF SOAP: AbsoluteMove for patrol,
                            ContinuousMove for visual-servo tracking)
  - TargetTracker         (ByteTrack ID lock-on logic)
  - SharedState           (mode + PTZ-velocity handoff between threads)
  - ptz_command_loop()    (background thread that actually sends
                            ContinuousMove/Stop while tracking)
  - small helpers (box overlap, PPE alert throttling)

This file has no "patrol logic" and no YOLO model loading in it -
that all lives in patrol_track_main.py. This file is just the
plumbing both the patrol state machine and the tracking state
machine share.
"""

import os
import time
import datetime
import threading

import cv2
import requests
import xml.etree.ElementTree as ET

from requests.auth import HTTPDigestAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from openpyxl import Workbook, load_workbook
from flask import Flask, Response
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
RTSP_URL = "rtsp://admin:Admin@123@192.168.1.126:554/unicaststream/1"

# =====================================================
# PATROL SETTINGS (absolute-move sweep)
# =====================================================

HOME_PAN = -0.899
LEFT_PAN = -0.600
RIGHT_PAN = 0.770

# Default/fallback tilt. Individual patrol stages now use
# TILT_STAGES below (indexed the same way as ZOOM_STAGES /
# SPEED_STAGES) so tilt can vary per stage. This constant is kept
# only as a fallback for any code path that doesn't have a stage
# index handy.
TILT = -0.75

PATROL_SPEED = 0.10
DWELL_TIME = 0.01

POSITION_TOLERANCE = 0.015

# =====================================================
# ZOOM / TILT / SPEED STAGE SETTINGS
# =====================================================
# Stage 0 is a special "wide-tilt" pass added in front of the
# original 3-stage sweep: same pan waypoints and same zoom as the
# old Stage 1 (0.00), but with its own tilt (0.25) instead of the
# standard patrol tilt (-0.75).
#
#   Stage 0 -> zoom 0.00, tilt  0.25   (new)
#   Stage 1 -> zoom 0.00, tilt -0.75   (was "Stage 1" before)
#   Stage 2 -> zoom 0.38, tilt -0.75   (was "Stage 2" before)
#   Stage 3 -> zoom 0.75, tilt -0.75   (was "Stage 3" before)
#
# ZOOM_STAGES, TILT_STAGES, and SPEED_STAGES must all stay the
# same length and are indexed together (index i -> that stage's
# zoom + tilt + speed).

ZOOM_STAGES = [0.00, 0.00, 0.38, 0.75]
TILT_STAGES = [0.25, -0.75, -0.75, -0.75]

# Explicit per-stage patrol pan/tilt speed. Previously this was
# derived from PATROL_SPEED * (SPEED_REDUCTION_FACTOR ** i); now
# each stage's speed is set directly so it can be tuned
# independently instead of through a single global reduction
# factor. (Values below match what the old 0.60 reduction factor
# produced, as a starting point.)
SPEED_STAGES = [
    0.20,    # Stage 0 -> zoom 0.00, tilt  0.25
    0.16,    # Stage 1 -> zoom 0.00, tilt -0.75
    0.09,    # Stage 2 -> zoom 0.38, tilt -0.75
    0.038,   # Stage 3 -> zoom 0.75, tilt -0.75
]

assert len(SPEED_STAGES) == len(ZOOM_STAGES) == len(TILT_STAGES), (
    "ZOOM_STAGES, TILT_STAGES, and SPEED_STAGES must all be the same length"
)

ZOOM_SPEED = 0.5

# How long (seconds) the camera stays parked at HOME while the
# zoom motor physically moves before any pan/tilt is allowed.
ZOOM_HOLD_TIME = 3.0

# Dedicated (slow) pan/tilt speed used ONLY for the tilt-only move
# between Stage 0 and Stage 1 (both directions: Stage 0 -> Stage 1
# during the forward sweep, and Stage 1 -> Stage 0 during the
# end-of-sweep reset). Previously this tilt move had no explicit
# speed (0.0), so it ran at the camera's default/fastest rate.
STAGE0_1_TILT_SPEED = 0.3

# How long (seconds) the camera dwells at Stage 1's position during
# the end-of-sweep reset, after zooming back to Stage 1's level and
# before tilting on up to Stage 0.
STAGE1_RESET_DWELL_TIME = 2.0

# =====================================================
# YOLO / DETECTION SETTINGS (shared by patrol-trigger AND tracking)
# =====================================================

# JETSON OPTIMIZATION (biggest speed win, manual one-time step - not
# applied automatically here since it must be done ON the Jetson itself):
# export your best.pt to a TensorRT engine on the Jetson:
#   yolo export model=best.pt format=engine device=0 half=True imgsz=480
# then point this at the resulting "best.engine" file. ultralytics'
# model.track()/predict() call signature and every line of this project's
# logic stays 100% identical either way - only the loaded weights file
# changes, since YOLO(...) auto-detects .pt vs .engine.
YOLO_MODEL_PATH = "best_native.engine"
YOLO_CONF_THRESHOLD = 0.6
YOLO_IMG_SIZE = 640

# ByteTrack config shipped with ultralytics.
TRACKER_CONFIG = "bytetrack.yaml"

PERSON_CLASS_NAME = "person"

PROCESS_WIDTH = 960
PROCESS_HEIGHT = 540

YOLO_DEVICE = 0  # falls back to CPU automatically if CUDA isn't available

# JETSON OPTIMIZATION: run inference in FP16 (half precision) instead of
# FP32. This is a pure inference-backend speed optimization - it does not
# change any detection/tracking/PPE logic, only how the numbers are
# computed on the GPU (Jetson's Tensor Cores are much faster at FP16).
# Only takes effect when running on CUDA (device != "cpu") - on CPU this
# flag is ignored by ultralytics. Set to False to match the original
# laptop FP32 behavior exactly if you ever need to A/B compare.
YOLO_HALF = True

WINDOW_NAME = "Patrol + PPE Tracking - Live Feed"

# =====================================================
# PPE COMPLIANCE SETTINGS (used only while actively tracking someone)
# =====================================================

PPE_OVERLAP_THRESHOLD = 0.3
VIOLATION_ALERT_COOLDOWN = 5.0

# -----------------------------------------------------
# PPE ITEM MAP (used by the Excel compliance log)
# -----------------------------------------------------
# Maps every PPE-related class name your model can output to
# (item_column_name, status). status is "OK" for the "wearing it"
# class and "VIOLATION" for the "not wearing it" class.
#
# Matched against your model's actual class list:
#   {0: 'boots', 1: 'gloves', 2: 'hardhat', 3: 'no_boots',
#    4: 'no_gloves', 5: 'no_hardhat', 6: 'no_vest',
#    7: 'person', 8: 'vest'}
#
# The 4 item names on the right ("Hardhat", "Vest", "Gloves",
# "Boots") become the 4 status columns in the Excel log - keep
# those consistent with PPE_ITEMS below.

PPE_CLASS_STATUS_MAP = {
    "hardhat":    ("Hardhat", "OK"),
    "no_hardhat": ("Hardhat", "VIOLATION"),

    "vest":       ("Vest", "OK"),
    "no_vest":    ("Vest", "VIOLATION"),

    "gloves":     ("Gloves", "OK"),
    "no_gloves":  ("Gloves", "VIOLATION"),

    "boots":      ("Boots", "OK"),
    "no_boots":   ("Boots", "VIOLATION"),
}

# Column order for the PPE status columns in the Excel sheet.
PPE_ITEMS = ["Hardhat", "Vest", "Gloves", "Boots"]

# Lowercase-keyed lookup so class-name matching is case-insensitive
# (mirrors how PPE_VIOLATION_CLASSES matching worked previously).
PPE_CLASS_STATUS_MAP_LOWER = {
    k.lower(): v for k, v in PPE_CLASS_STATUS_MAP.items()
}

# -----------------------------------------------------
# EXCEL LOG SETTINGS
# -----------------------------------------------------

EXCEL_LOG_PATH = "ppe_log.xlsx"

# Minimum seconds between actual disk saves. The in-memory sheet
# is updated every frame a tracked person's PPE is (re)checked,
# but the file itself is only written to disk at most this often,
# so we don't hammer the disk at 20-30fps. Open the file in Excel
# and re-open / refresh it every few seconds to see it "live".
EXCEL_SAVE_INTERVAL = 1.0

# =====================================================
# VISUAL SERVO (PAN/TILT FOLLOW) SETTINGS - tracking mode only
# =====================================================

PAN_KP = 0.6
TILT_KP = 0.6

# Hard ceiling on tracking speed. The proportional control below
# still tapers speed down as the person approaches center (so it
# eases in/out near the deadband instead of a hard on/off), but it
# can never command faster than this - 0.1 keeps PTZ motion slow
# and smooth even when the person first appears far off-center.
MAX_PAN_SPEED = 0.1
MAX_TILT_SPEED = 0.1

TILT_SIGN = -1.0

DEADBAND = 0.06

LOST_TARGET_GRACE_FRAMES = 10

COMMAND_RATE_HZ = 10.0

# =====================================================
# MODE-SWITCH TIMING
# =====================================================
# How long (seconds) the camera stays in tracking mode once a
# person is locked on, before handing control back to patrol.

TRACK_DURATION = 5.0

# =====================================================
# TRACKING SELECTION / RE-TRACK COOLDOWN
# =====================================================
# Once a given ByteTrack ID's TRACK_DURATION window ends, that ID
# must not be picked again as the active PTZ-follow target until
# this many seconds have passed - even if they're still standing
# right in front of the camera. This is separate from
# PPE_LOG_COOLDOWN_SECONDS below (which only gates Excel rows):
# this one gates who the camera physically follows next.

TRACKING_COOLDOWN_SECONDS = 120.0  # 2 minutes

# =====================================================
# PPE LOG COOLDOWN / SCREENSHOT SETTINGS
# =====================================================
# Once a given ByteTrack ID has been logged to the PPE Excel
# sheet, it must not be logged again until this many seconds have
# passed - even though tracking/PTZ-following of that person
# keeps working normally during the cooldown window. This stops
# the same person being written to the sheet on every single
# tracking session if they walk past the camera repeatedly.

PPE_LOG_COOLDOWN_SECONDS = 300.0

# Folder (relative to the working directory) where a snapshot
# image is saved each time a PPE log row is written, so the row's
# "Screenshot Path" column has something to point to.

SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# How often (seconds) blocking wait-loops re-check for an
# interrupt / re-check camera position. Smaller = camera stops
# "immediately" when a person is seen, at the cost of slightly
# more chatter.
LOOP_POLL_INTERVAL = 0.1

# Number of consecutive unchanged-position polls (at
# LOOP_POLL_INTERVAL spacing) before a pan/tilt move is
# considered stalled.
STALL_COUNT_THRESHOLD = 30


# =====================================================
# SHARED STATE BETWEEN PATROL THREAD, DETECTION/TRACKING
# LOOP (main thread), AND THE PTZ COMMAND THREAD
# =====================================================
# mode:
#   "PATROL"   -> patrol thread is free to drive absolute moves
#   "TRACKING" -> patrol thread is parked, ptz_command_loop is
#                 free to drive continuous-move visual servo
#
# person_trigger_event:
#   SET by the main thread the instant a person is seen while
#   mode == "PATROL". The patrol thread polls this during every
#   move / dwell / zoom-hold and, when it sees it set, stops the
#   camera immediately and waits for tracking to finish.
#
# tracking_done_event:
#   SET by the main thread once TRACK_DURATION has elapsed and
#   continuous-move has been stopped. Tells the patrol thread it
#   may now issue the "return HOME at current zoom stage" move
#   and continue patrolling.

class SharedState:

    def __init__(self):

        self._lock = threading.Lock()
        self._mode = "PATROL"

        self.person_trigger_event = threading.Event()
        self.tracking_done_event = threading.Event()

        self._pan_speed = 0.0
        self._tilt_speed = 0.0
        self._locked = False

        # Purely informational, for the on-screen overlay.
        self.status_text = "Patrol - Stage 0"

        # Which patrol stage (0-3) was interrupted to start the
        # current tracking session. Set by the patrol thread right
        # before it hands off to tracking; read by the main
        # detection/tracking loop so PPE log rows can record which
        # stage the person was spotted during.
        self._patrol_stage = 0

    def get_mode(self):
        with self._lock:
            return self._mode

    def set_mode(self, mode):
        with self._lock:
            self._mode = mode

    def set_desired_velocity(self, pan_speed, tilt_speed, locked):
        with self._lock:
            self._pan_speed = pan_speed
            self._tilt_speed = tilt_speed
            self._locked = locked

    def get_desired_velocity(self):
        with self._lock:
            return self._pan_speed, self._tilt_speed, self._locked

    def set_status(self, text):
        with self._lock:
            self.status_text = text

    def get_status(self):
        with self._lock:
            return self.status_text

    def set_patrol_stage(self, stage_number):
        with self._lock:
            self._patrol_stage = stage_number

    def get_patrol_stage(self):
        with self._lock:
            return self._patrol_stage


# =====================================================
# PPE VIOLATION ALERT THROTTLING
# =====================================================

_violation_alert_lock = threading.Lock()
_last_alert_time = {}  # (track_id, violation_name) -> timestamp


def maybe_alert_violation(track_id, violation_name):
    key = (track_id, violation_name)
    now = time.time()

    with _violation_alert_lock:
        last = _last_alert_time.get(key, 0.0)
        if now - last < VIOLATION_ALERT_COOLDOWN:
            return
        _last_alert_time[key] = now

    print(f"\n[PPE VIOLATION] Person {track_id}: {violation_name}")


# =====================================================
# COOLDOWN MANAGER (generic per-ID "not too soon again" gate)
# =====================================================
# Reused for two independent purposes, each with its own instance
# and its own cooldown_seconds:
#
#   1. PPE Excel log dedup (PPE_LOG_COOLDOWN_SECONDS / 300s) -
#      gates whether a Track ID's PPE check produces a new
#      spreadsheet row. Uses should_log(), an atomic check+mark.
#
#   2. Tracking re-selection (TRACKING_COOLDOWN_SECONDS / 120s) -
#      gates whether a Track ID may be picked again as the active
#      PTZ-follow target. Uses is_on_cooldown() (read-only check,
#      can be called many times per frame without side effects)
#      plus mark_now() (called once, exactly when a person's
#      TRACK_DURATION window ends, to start their cooldown).
#
# These are separate CooldownManager instances - a person going on
# the 2-minute tracking cooldown has no effect on the 5-minute
# Excel-log cooldown or vice versa.

class CooldownManager:

    def __init__(self, cooldown_seconds=PPE_LOG_COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._last_used = {}  # track_id -> last-used timestamp

    def is_on_cooldown(self, track_id):
        """Read-only check - does NOT start/reset the cooldown.
        Safe to call repeatedly (e.g. once per visible person, every
        frame) while deciding who's eligible to track next."""

        with self._lock:
            last = self._last_used.get(track_id)

        if last is None:
            return False

        return (time.time() - last) < self.cooldown_seconds

    def mark_now(self, track_id):
        """Explicitly starts/resets this track_id's cooldown window
        from right now."""

        with self._lock:
            self._last_used[track_id] = time.time()

    def should_log(self, track_id):
        """Atomic check+mark: returns True (and starts a fresh
        cooldown window) if this track_id is not currently on
        cooldown, False otherwise. Used by the PPE Excel logger,
        where every eligible check should also count as the use."""

        now = time.time()

        with self._lock:
            last = self._last_used.get(track_id, 0.0)

            if now - last >= self.cooldown_seconds:
                self._last_used[track_id] = now
                return True

            return False

    def seconds_remaining(self, track_id):
        """Informational only (e.g. for on-screen display)."""

        with self._lock:
            last = self._last_used.get(track_id)

        if last is None:
            return 0.0

        return max(0.0, self.cooldown_seconds - (time.time() - last))


# =====================================================
# PPE EXCEL LOGGER
# =====================================================
# APPENDS ONE ROW PER LOG EVENT (not one row per person). A "log
# event" is: a tracked person's PPE was checked, and that Track ID
# is not currently on cooldown (see CooldownManager above) - so in
# practice a given person produces a new row at most once every
# PPE_LOG_COOLDOWN_SECONDS, no matter how many tracking sessions
# they trigger in that window.
#
# Columns:
#   Timestamp | Track ID | Hardhat | Vest | Gloves | Boots |
#   PPE Status | Screenshot Path
#
# Each PPE item gets its OWN column so it's immediately visible
# which specific item a given Track ID was violated on (e.g. the
# "Hardhat" column shows "VIOLATION" on that person's row if they
# weren't wearing one, "OK" if they were, blank if that item
# wasn't seen at all for this event).
#
# The workbook is kept open in memory and only saved to disk at
# most once every EXCEL_SAVE_INTERVAL seconds (see above), so you
# can leave the file open in Excel and refresh periodically to
# watch it update without generating a disk write on every frame.

EXCEL_HEADERS = (
    ["Timestamp", "Track ID"] + PPE_ITEMS + ["PPE Status", "Screenshot Path"]
)


class PPEExcelLogger:

    def __init__(self, path=EXCEL_LOG_PATH):

        self.path = path
        self._lock = threading.Lock()
        self._last_save = 0.0

        self._load_or_create()

    def _load_or_create(self):

        if os.path.exists(self.path):
            self.wb = load_workbook(self.path)
            self.ws = self.wb.active
            print(f"PPE log: appending to existing file '{self.path}'.")

        else:
            self.wb = Workbook()
            self.ws = self.wb.active
            self.ws.title = "PPE Log"

            self.ws.append(EXCEL_HEADERS)

            for col in range(1, len(EXCEL_HEADERS) + 1):
                self.ws.column_dimensions[
                    self.ws.cell(row=1, column=col).column_letter
                ].width = 16

            self.wb.save(self.path)
            print(f"PPE log: created new file '{self.path}'.")

    def log(self, track_id, item_status, screenshot_path=None):
        """Appends ONE new row for this log event.

        item_status: dict of {item_name: "OK"/"VIOLATION"} for
        whichever PPE items were detected against this person at
        the moment of logging (item_name is one of PPE_ITEMS -
        "Hardhat", "Vest", "Gloves", "Boots"). Each item gets its
        own column in the row, so it's obvious at a glance exactly
        which item(s) this Track ID was violated on. Items not
        detected at all for this event are left blank. Overall
        "PPE Status" is "Non-Compliant" if any item is a
        VIOLATION, else "Compliant".
        """

        if not item_status:
            return

        violations = [
            item for item, status in item_status.items()
            if status == "VIOLATION"
        ]
        overall_status = "Non-Compliant" if violations else "Compliant"

        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._lock:

            item_columns = [item_status.get(item, "") for item in PPE_ITEMS]

            self.ws.append(
                [timestamp_str, track_id]
                + item_columns
                + [overall_status, screenshot_path or ""]
            )

            now_t = time.time()
            if now_t - self._last_save >= EXCEL_SAVE_INTERVAL:
                self._save_locked()
                self._last_save = now_t

    def _save_locked(self):
        try:
            self.wb.save(self.path)
        except Exception as exc:
            print(f"PPE log: save failed ({exc}). Is the file open/locked in Excel?")

    def close(self):
        """Force a final save - call this on shutdown so the last
        few updates (which may be sitting inside the throttle
        window) aren't lost."""
        with self._lock:
            self._save_locked()


# =====================================================
# BOX OVERLAP HELPER (attributing a PPE box to a tracked person)
# =====================================================

def intersection_over_box_area(inner_box, outer_box):
    """Fraction of inner_box's area that overlaps outer_box."""

    ix1 = max(inner_box[0], outer_box[0])
    iy1 = max(inner_box[1], outer_box[1])
    ix2 = min(inner_box[2], outer_box[2])
    iy2 = min(inner_box[3], outer_box[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    inner_area = max(
        1e-6,
        (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
    )

    return intersection / inner_area


# =====================================================
# TARGET LOCK-ON LOGIC (ByteTrack ID persistence)
# =====================================================
# TargetTracker's only job is: "is the ID I'm currently locked onto
# still visible this frame?" - it does NOT decide which ID to lock
# onto in the first place. That decision (which person to track
# next, respecting the 2-minute tracking cooldown) is made
# explicitly by the caller via select_next_target() below, and then
# handed to this tracker by setting .target_id directly. Keeping
# these two responsibilities separate is what makes "only track
# using Track ID, never by re-detection/proximity" hold everywhere.

class TargetTracker:

    def __init__(self):
        self.target_id = None
        self.missing_frames = 0

    def update(self, persons):
        """persons: list of dicts with keys id, cx, cy, box - cx/cy are
        normalized center-x/y in [-1, 1] relative to frame center.

        Looks for self.target_id among this frame's persons. If
        found, returns it. If not found, allows up to
        LOST_TARGET_GRACE_FRAMES consecutive missed frames (e.g. a
        brief occlusion) before giving up and clearing target_id.
        Returns None whenever nothing is currently locked on, or the
        locked target isn't visible this frame."""

        if self.target_id is None:
            return None

        for p in persons:
            if p["id"] == self.target_id:
                self.missing_frames = 0
                return p

        self.missing_frames += 1
        if self.missing_frames > LOST_TARGET_GRACE_FRAMES:
            self.target_id = None
            self.missing_frames = 0

        return None


def select_next_target(persons, tracking_cooldown_mgr):
    """Picks who the camera should track next, from the people
    visible THIS frame - purely by Track ID, never by re-detection
    or proximity to center.

    Eligible = not currently on the 2-minute tracking cooldown
    (i.e. wasn't tracked in the last TRACKING_COOLDOWN_SECONDS).
    Among the eligible people, picks the SMALLEST Track ID: since
    ByteTrack assigns IDs in the order tracks are first created,
    the smallest ID still visible is the earliest-detected person
    still in frame - so this naturally tracks people in
    first-detected order, then moves on to the next one.

    Returns the person dict to track next, or None if nobody
    currently visible is eligible (everyone in frame was already
    tracked within the last 2 minutes, or the frame is empty) - the
    caller should resume/continue patrol in that case.
    """

    eligible = [
        p for p in persons
        if not tracking_cooldown_mgr.is_on_cooldown(p["id"])
    ]

    if not eligible:
        return None

    return min(eligible, key=lambda p: p["id"])


# =====================================================
# JETSON OPTIMIZATION: HARDWARE-ACCELERATED RTSP DECODE
# =====================================================
# On the laptop, cv2.VideoCapture(..., cv2.CAP_FFMPEG) decodes H.264/H.265
# in software on the CPU. The Jetson Orin Nano has a dedicated hardware
# video decoder (NVDEC) that JetPack's OpenCV/GStreamer build can use via
# the "nvv4l2decoder" element, which decodes RTSP frames almost for free
# and leaves the CPU free for the ONVIF/patrol threads and Python/YOLO
# glue code. This does NOT change what a "frame" is or how it's read
# (FrameGrabber still just exposes the latest BGR frame the same way) -
# it only changes which piece of hardware does the decoding.
#
# USE_GSTREAMER_HW_DECODE = True tries the hardware pipeline first and
# transparently falls back to the original CAP_FFMPEG path if the
# GStreamer pipeline can't be opened (e.g. running this same file on a
# laptop without the Jetson multimedia API / nvv4l2decoder available) -
# so the exact same script still runs unmodified on a laptop too.
USE_GSTREAMER_HW_DECODE = True
GSTREAMER_LATENCY_MS = 100


def _build_gstreamer_pipeline(rtsp_url, latency_ms=GSTREAMER_LATENCY_MS):
    return (
        f"rtspsrc location=\"{rtsp_url}\" latency={latency_ms} ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


# =====================================================
# THREADED FRAME GRABBER
# =====================================================

class FrameGrabber:

    def __init__(self, rtsp_url):

        self.rtsp_url = rtsp_url
        self.cap = None

        if USE_GSTREAMER_HW_DECODE:
            pipeline = _build_gstreamer_pipeline(rtsp_url)
            gst_cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if gst_cap.isOpened():
                print("FrameGrabber: using Jetson hardware decode (nvv4l2decoder).")
                self.cap = gst_cap
            else:
                gst_cap.release()
                print(
                    "FrameGrabber: hardware-decode GStreamer pipeline "
                    "unavailable, falling back to software (CAP_FFMPEG) decode."
                )

        if self.cap is None:
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
# PTZ CONTROLLER
# =====================================================
# Supports BOTH:
#   - AbsoluteMove + polling  -> used by the PATROL state machine
#   - ContinuousMove          -> used by the TRACKING visual servo
# on the same ONVIF session, since only one of the two is ever
# actively commanding the camera at a given moment (see SharedState
# mode + the coordination in patrol_track_main.py).

class PTZController:

    def __init__(self):

        self.session = requests.Session()

        retry = Retry(total=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)

        self.auth = HTTPDigestAuth(USERNAME, PASSWORD)

        self.pan = None
        self.tilt = None
        self.zoom = None

        self.running = False
        self.poll_thread = None

    # =================================================
    # SOAP
    # =================================================

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
                    f"HTTP {response.status_code} - "
                    f"{response.text[:300]}"
                )

            elif "Fault" in response.text:
                print(
                    f"ONVIF command to {url} returned a SOAP "
                    f"Fault: {response.text[:300]}"
                )

        return response

    # =================================================
    # STATUS / POLLING (used by the patrol state machine)
    # =================================================

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
                elif elem.tag.endswith("Zoom") and "x" in elem.attrib:
                    self.zoom = float(elem.attrib["x"])

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

    # =================================================
    # ABSOLUTE MOVE (PAN / TILT) - patrol only
    # =================================================

    def absolute_move(self, pan, tilt, zoom, speed=PATROL_SPEED):

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
    # ZOOM-ONLY MOVE - patrol only, used while parked at HOME
    # =================================================

    def move_zoom(self, pan, tilt, zoom, zoom_speed=ZOOM_SPEED, tilt_speed=0.0):

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

    # =================================================
    # CONTINUOUS MOVE - tracking only (visual servo)
    # =================================================

    def continuous_move(self, pan_speed, tilt_speed, zoom_speed=0.0):

        body = f"""
<tptz:ContinuousMove
xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
xmlns:tt="http://www.onvif.org/ver10/schema">

<tptz:ProfileToken>{PROFILE_TOKEN}</tptz:ProfileToken>

<tptz:Velocity>
<tt:PanTilt x="{pan_speed}" y="{tilt_speed}" />
<tt:Zoom x="{zoom_speed}" />
</tptz:Velocity>

</tptz:ContinuousMove>
"""
        self.send_soap(PTZ_URL, body)

    # =================================================
    # STOP (halts either AbsoluteMove or ContinuousMove)
    # =================================================

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

    # =================================================
    # WAIT FOR ARRIVAL - interrupt-aware (patrol only)
    # =================================================
    # Waits for the camera to reach target_pan. Unlike a plain
    # "pause and resume the same move" loop, this STOPS the
    # camera immediately and returns "INTERRUPTED" the moment a
    # person is detected (shared_state.person_trigger_event set).
    # The caller (patrol_track_main.py) is responsible for
    # handling that by waiting out the tracking session and then
    # re-issuing a fresh move.

    def wait_until_reached_interruptible(
        self,
        shared_state,
        target_pan,
        tilt,
        zoom,
        speed,
        tolerance=POSITION_TOLERANCE
    ):

        stall_count = 0
        prev_pan = None

        while True:

            if shared_state.person_trigger_event.is_set():
                self.stop_move()
                return "INTERRUPTED"

            current_pan = self.pan

            if current_pan is None:
                time.sleep(LOOP_POLL_INTERVAL)
                continue

            error_direct = abs(current_pan - target_pan)
            error_wrap = 2.0 - error_direct
            error = min(error_direct, error_wrap)

            if error < tolerance:
                return "REACHED"

            if prev_pan is not None:
                if abs(current_pan - prev_pan) < 0.001:
                    stall_count += 1
                    if stall_count >= STALL_COUNT_THRESHOLD:
                        print(f"STALL detected near {current_pan:.4f}")
                        return "STALLED"
                else:
                    stall_count = 0

            prev_pan = current_pan
            time.sleep(LOOP_POLL_INTERVAL)

    # =================================================
    # GET RTSP STREAM URI (ONVIF MEDIA SERVICE) - optional,
    # not used since RTSP_URL is already known, kept for
    # completeness / fallback.
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
            "Could not retrieve RTSP stream URI from camera. "
            "Check MEDIA_URL / PROFILE_TOKEN."
        )


# =====================================================
# PTZ COMMAND THREAD - tracking only
# =====================================================
# Sends ContinuousMove / Stop at a fixed rate based on whatever
# velocity the main detection/tracking loop most recently
# computed. Only ever actually moves the camera while
# shared_state's velocity is "locked" (i.e. mode == TRACKING and
# a target is currently held) - during PATROL it stays silent so
# it never fights with the patrol thread's AbsoluteMove calls.

def ptz_command_loop(ptz, shared_state):

    period = 1.0 / COMMAND_RATE_HZ
    was_moving = False

    while True:

        pan_speed, tilt_speed, locked = shared_state.get_desired_velocity()

        moving = locked and (
            abs(pan_speed) > 0.0 or abs(tilt_speed) > 0.0
        )

        if moving:
            ptz.continuous_move(pan_speed, tilt_speed, 0.0)
            was_moving = True
        elif was_moving:
            ptz.stop_move()
            was_moving = False

        time.sleep(period)


# =====================================================
# RAW & ANNOTATED FEED HTTP SERVER (for dashboard / clients)
# =====================================================

FEED_SERVER_PORT = 8080

_feed_app = Flask(__name__)
_feed_grabber_ref = {
    "grabber": None,
    "annotated_frame": None,
    "lock": threading.Lock()
}


def update_annotated_frame(frame):
    """Update the latest YOLO-annotated frame for HTTP stream clients."""
    with _feed_grabber_ref["lock"]:
        _feed_grabber_ref["annotated_frame"] = frame.copy() if frame is not None else None


def _raw_mjpeg_stream():
    while True:
        grabber = _feed_grabber_ref["grabber"]
        if grabber is None:
            time.sleep(0.1)
            continue
        ok, frame = grabber.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        ok2, buf = cv2.imencode(".jpg", frame)
        if not ok2:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )


def _annotated_mjpeg_stream():
    while True:
        with _feed_grabber_ref["lock"]:
            frame = _feed_grabber_ref["annotated_frame"]
        if frame is None:
            time.sleep(0.05)
            continue
        ok2, buf = cv2.imencode(".jpg", frame)
        if not ok2:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )


@_feed_app.route("/")
def _feed_index():
    return (
        "Jetson feed server is running.<br>"
        "Raw Feed: <a href='/feed'>/feed</a> or <a href='/feed/0'>/feed/0</a><br>"
        "Annotated Feed: <a href='/annotated_feed'>/annotated_feed</a>"
    )


@_feed_app.route("/feed")
@_feed_app.route("/feed/0")
@_feed_app.route("/feed/<path:camera_id>")
def _feed_route(camera_id=None):
    return Response(
        _raw_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@_feed_app.route("/annotated_feed")
@_feed_app.route("/feed/annotated")
def _annotated_feed_route():
    return Response(
        _annotated_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def start_raw_feed_server(grabber, port=FEED_SERVER_PORT):
    """Starts a Flask HTTP server (background thread) serving the camera
    frames as MJPEG. Reuses the SAME FrameGrabber the detection loop
    already has open, so this does not open a second connection to the camera."""

    _feed_grabber_ref["grabber"] = grabber

    def _run():
        _feed_app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print(f"[feed_server] Jetson feed server running at http://0.0.0.0:{port}/ (endpoints: /feed, /feed/0, /annotated_feed)")