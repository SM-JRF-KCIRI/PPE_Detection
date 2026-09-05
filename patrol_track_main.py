"""
patrol_track_main.py
=====================================================
Merged patrol + person-tracking behavior:

  1. Camera patrols (4 zoom/tilt stages x LEFT/HOME/RIGHT/HOME
     sweeps: Stage 0 is a wide-tilt pass at tilt=0.25, then Stages
     1-3 are the original zoom sweep at the standard tilt).
  2. The instant a person is detected, patrol STOPS immediately
     (wherever it currently is) and hands control to a ByteTrack
     visual-servo tracker, which follows the person and also
     reports PPE violations, exactly like the original ByteTrack
     script.
  3. After TRACK_DURATION (5 sec) of tracking, the camera returns
     to HOME at whatever tilt/zoom stage patrol was on when it got
     interrupted (pan is the same at every stage - only tilt/zoom
     differ), then patrol resumes from the NEXT leg in its sweep
     (the interrupted leg is not repeated).
  4. After the last stage (Stage 3) completes, the camera does NOT
     jump straight back to Stage 0 in one move. It eases back in
     steps: first zoom-only back down to Stage 1's zoom level
     (Stage 3 and Stage 1 share the same tilt, so this step moves
     only the zoom motor), then it DWELLS at Stage 1 for a bit,
     then tilts (slowly - see below) from there up to Stage 0's
     tilt (zoom is already at Stage 0's level by then, so this
     step moves only pan/tilt). This keeps the reset to one axis
     of motion at a time instead of a single fast double-axis jump.
  5. The tilt-only move between Stage 0 and Stage 1 - in EITHER
     direction (Stage 0 -> Stage 1 during the forward sweep, and
     Stage 1 -> Stage 0 during the reset above) - runs at a slower,
     dedicated tilt speed (STAGE0_1_TILT_SPEED in ptz_shared.py)
     instead of the camera's default/fastest rate.
  6. Loops forever.

Run this file - it imports everything it needs from ptz_shared.py
(which must be in the same folder).
"""

import os
import time
import threading
import subprocess
import json
import atexit

import cv2
from ultralytics import YOLO

from ptz_shared import (
    RTSP_URL,
    HOME_PAN, LEFT_PAN, RIGHT_PAN,
    DWELL_TIME,
    ZOOM_STAGES, TILT_STAGES, ZOOM_HOLD_TIME, SPEED_STAGES,
    STAGE0_1_TILT_SPEED, STAGE1_RESET_DWELL_TIME,
    YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD, YOLO_IMG_SIZE, YOLO_DEVICE,
    TRACKER_CONFIG, PERSON_CLASS_NAME,
    PROCESS_WIDTH, PROCESS_HEIGHT, WINDOW_NAME,
    PPE_OVERLAP_THRESHOLD, PPE_CLASS_STATUS_MAP, PPE_CLASS_STATUS_MAP_LOWER,
    PAN_KP, TILT_KP, MAX_PAN_SPEED, MAX_TILT_SPEED, TILT_SIGN, DEADBAND,
    TRACK_DURATION, LOOP_POLL_INTERVAL, SCREENSHOT_DIR,
    TRACKING_COOLDOWN_SECONDS,
    SharedState, PTZController, FrameGrabber, TargetTracker, PPEExcelLogger,
    CooldownManager, select_next_target, ptz_command_loop,
    maybe_alert_violation, intersection_over_box_area,
    # NEW: backend alert + feed server
    PPEAlertManager, FeedServer,
    CAMERA_ID, FEED_SERVER_PORT,
)

# =====================================================
# PATROL STATE MACHINE (runs in its own background thread)
# =====================================================
# NOTE ON THE INTERRUPT DESIGN:
# Every blocking wait in here (move / dwell / zoom-hold) is
# "interruptible": it polls shared_state.person_trigger_event and,
# the moment it's set, stops the camera and calls
# handle_person_interruption(), which blocks until tracking mode
# has finished (tracking_done_event), then issues the mandatory
# "return HOME at the current tilt/zoom stage" move. After that,
# the calling function (run_cycle / patrol) simply falls through to
# whatever line of code comes next - which is the NEXT leg of the
# sweep. The interrupted leg itself is never retried, matching
# "skip to the next leg" behavior.
#
# NOTE ON TILT:
# Tilt used to be a single global constant (TILT). It's now
# per-stage (TILT_STAGES, indexed the same way as ZOOM_STAGES /
# SPEED_STAGES) so Stage 0 can hold a different tilt (0.25) than
# the rest of the sweep (-0.75). Every function below therefore
# takes `tilt` as an explicit argument instead of reading a global.


def handle_person_interruption(ptz, shared_state, current_tilt, current_zoom, current_speed, stage_number):

    print("\n>>> Person detected - patrol stopped, handing off to tracking mode.")
    shared_state.set_status(f"Tracking (was Stage {stage_number})")

    # Remember which patrol stage was interrupted so the main
    # thread's PPE log rows can record it (Patrol Stage column).
    shared_state.set_patrol_stage(stage_number)

    shared_state.person_trigger_event.clear()

    # Block here for the whole tracking session. The main thread
    # (run_detection_and_tracking) is driving the ContinuousMove
    # visual servo during this time.
    shared_state.tracking_done_event.wait()
    shared_state.tracking_done_event.clear()

    print(
        f">>> Tracking finished - returning HOME "
        f"(stage {stage_number}, tilt={current_tilt:.2f}, zoom={current_zoom:.2f})"
    )
    shared_state.set_status(f"Returning HOME - Stage {stage_number}")

    ptz.absolute_move(HOME_PAN, current_tilt, current_zoom, current_speed)

    result = ptz.wait_until_reached_interruptible(
        shared_state, HOME_PAN, current_tilt, current_zoom, current_speed
    )

    if result == "INTERRUPTED":
        # A person was seen again immediately during the
        # return-home move - handle that too before continuing.
        handle_person_interruption(
            ptz, shared_state, current_tilt, current_zoom, current_speed, stage_number
        )
        return

    print(">>> Resuming patrol.\n")
    shared_state.set_status(f"Patrol - Stage {stage_number}")


def move_and_wait_interruptible(ptz, shared_state, pan, tilt, zoom, speed, label, stage_number):

    print(f"\nMoving to {label} (stage {stage_number}, speed={speed:.4f}, tilt={tilt:.2f}, zoom={zoom:.2f})")
    shared_state.set_status(f"Patrol - Stage {stage_number} - Moving to {label}")

    ptz.absolute_move(pan, tilt, zoom, speed)

    result = ptz.wait_until_reached_interruptible(shared_state, pan, tilt, zoom, speed)

    if result == "INTERRUPTED":
        handle_person_interruption(ptz, shared_state, tilt, zoom, speed, stage_number)
    elif result == "REACHED":
        print(f"Reached {label} (pan={pan}, tilt={tilt:.2f}, zoom={zoom:.2f})")
    else:
        print(f"Did not confirm arrival at {label} - continuing patrol anyway")


def interruptible_dwell(ptz, shared_state, seconds, current_tilt, current_zoom, current_speed, stage_number):

    end_time = time.time() + seconds

    while time.time() < end_time:

        if shared_state.person_trigger_event.is_set():
            handle_person_interruption(ptz, shared_state, current_tilt, current_zoom, current_speed, stage_number)
            return

        time.sleep(LOOP_POLL_INTERVAL)


def apply_zoom_at_home(ptz, shared_state, tilt, zoom, speed, stage_number, status_label=None, tilt_speed=0.0):

    label = status_label if status_label is not None else f"stage {stage_number}"

    print(f"\nHolding at HOME to apply {label} (tilt={tilt:.2f}, zoom={zoom:.2f}, tilt_speed={tilt_speed:.3f})")
    shared_state.set_status(f"Patrol - applying {label}")

    ptz.move_zoom(HOME_PAN, tilt, zoom, tilt_speed=tilt_speed)

    end_time = time.time() + ZOOM_HOLD_TIME

    while time.time() < end_time:

        if shared_state.person_trigger_event.is_set():
            handle_person_interruption(ptz, shared_state, tilt, zoom, speed, stage_number)
            return

        time.sleep(LOOP_POLL_INTERVAL)

    print(f"{label} (tilt/zoom) applied.")


def reset_to_stage_zero(ptz, shared_state):
    """Eases the camera from the end of the sweep (Stage 3) back to
    Stage 0, in two slow single-axis steps instead of one fast
    double-axis jump:

      Step 1: Stage 3 -> Stage 1's zoom/tilt.
              Stage 3 and Stage 1 share the same tilt (-0.75), so
              this step only moves the zoom motor (0.75 -> 0.00).

      Step 2: Stage 1's position -> Stage 0.
              Zoom is already at Stage 0's level (0.00) after step
              1, so this step only moves pan/tilt (-0.75 -> 0.25).

    Each step is a normal interruptible apply_zoom_at_home() call,
    so if a person is detected mid-reset, patrol stops immediately,
    hands off to tracking, and - once tracking finishes - resumes
    the reset seamlessly from where it left off (the next step is
    simply the next line of code that runs).
    """

    print("\nCompleted all stages. Easing back to Stage 0 (via Stage 1 zoom level).")

    # Step 1 - zoom-only: Stage 3's zoom level -> Stage 1's zoom level.
    apply_zoom_at_home(
        ptz, shared_state,
        TILT_STAGES[1], ZOOM_STAGES[1], SPEED_STAGES[1],
        stage_number=0,
        status_label="Stage 0 reset - step 1/2 (zoom out)",
    )

    # Dwell at Stage 1's position before continuing on to Stage 0.
    print(f"Dwelling at Stage 1 for {STAGE1_RESET_DWELL_TIME:.1f}s before moving to Stage 0.")
    shared_state.set_status("Stage 0 reset - dwelling at Stage 1")
    interruptible_dwell(
        ptz, shared_state, STAGE1_RESET_DWELL_TIME,
        TILT_STAGES[1], ZOOM_STAGES[1], SPEED_STAGES[1],
        stage_number=0,
    )

    # Step 2 - tilt-only: Stage 1's position -> Stage 0, at the
    # dedicated slow tilt speed (same speed used for the Stage 0 ->
    # Stage 1 tilt move in the forward sweep).
    apply_zoom_at_home(
        ptz, shared_state,
        TILT_STAGES[0], ZOOM_STAGES[0], SPEED_STAGES[0],
        stage_number=0,
        status_label="Stage 0 reset - step 2/2 (tilt to Stage 0)",
        tilt_speed=STAGE0_1_TILT_SPEED,
    )

    print("Reset to Stage 0 complete.\n")


def run_cycle(ptz, shared_state, tilt, zoom, speed, cycle_number, stage_number):

    print(
        f"\n========== CYCLE {cycle_number} "
        f"(STAGE {stage_number}, tilt={tilt:.2f}, zoom={zoom:.2f}, speed={speed:.4f}) =========="
    )

    move_and_wait_interruptible(ptz, shared_state, LEFT_PAN, tilt, zoom, speed, "LEFT", stage_number)
    interruptible_dwell(ptz, shared_state, DWELL_TIME, tilt, zoom, speed, stage_number)

    move_and_wait_interruptible(ptz, shared_state, HOME_PAN, tilt, zoom, speed, "HOME", stage_number)
    interruptible_dwell(ptz, shared_state, DWELL_TIME, tilt, zoom, speed, stage_number)

    move_and_wait_interruptible(ptz, shared_state, RIGHT_PAN, tilt, zoom, speed, "RIGHT", stage_number)
    interruptible_dwell(ptz, shared_state, DWELL_TIME, tilt, zoom, speed, stage_number)

    move_and_wait_interruptible(ptz, shared_state, HOME_PAN, tilt, zoom, speed, "HOME", stage_number)
    interruptible_dwell(ptz, shared_state, DWELL_TIME, tilt, zoom, speed, stage_number)


def patrol(ptz, shared_state):

    ptz.start_polling()

    try:
        print("Moving to HOME")

        move_and_wait_interruptible(
            ptz, shared_state, HOME_PAN, TILT_STAGES[0], ZOOM_STAGES[0], SPEED_STAGES[0], "HOME", 0
        )
        interruptible_dwell(ptz, shared_state, DWELL_TIME, TILT_STAGES[0], ZOOM_STAGES[0], SPEED_STAGES[0], 0)

        last_tilt = TILT_STAGES[0]
        last_zoom = ZOOM_STAGES[0]
        cycle = 1

        while True:

            for stage_index, zoom in enumerate(ZOOM_STAGES):

                stage_number = stage_index
                tilt = TILT_STAGES[stage_index]
                speed = SPEED_STAGES[stage_index]

                if zoom != last_zoom or tilt != last_tilt:
                    # The Stage 0 -> Stage 1 leg is a tilt-only move
                    # (both stages share zoom 0.00) - use the slow,
                    # dedicated tilt speed for it specifically.
                    tilt_speed = STAGE0_1_TILT_SPEED if stage_index == 1 else 0.0
                    apply_zoom_at_home(
                        ptz, shared_state, tilt, zoom, speed, stage_number,
                        tilt_speed=tilt_speed,
                    )
                    last_zoom = zoom
                    last_tilt = tilt

                run_cycle(ptz, shared_state, tilt, zoom, speed, cycle, stage_number)
                cycle += 1

            print(f"\nCompleted all {len(ZOOM_STAGES)} stages.")
            reset_to_stage_zero(ptz, shared_state)
            last_tilt = TILT_STAGES[0]
            last_zoom = ZOOM_STAGES[0]

    except Exception as exc:
        print(f"\nPatrol stopped: {exc}")

    finally:
        ptz.stop_polling()


# =====================================================
# YOLO DETECTION + TRACKING - MAIN THREAD (cv2.imshow lives here)
# =====================================================

def pick_device():

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

    return device


def find_person_class_id(model):

    for class_id, name in model.names.items():
        if name.lower() == PERSON_CLASS_NAME:
            return class_id

    print(
        f"Warning: class '{PERSON_CLASS_NAME}' not found in "
        f"model.names ({model.names}). Patrol will never switch to "
        f"tracking mode - set PERSON_CLASS_NAME to match your "
        f"model's person label."
    )
    return None


def extract_detections(results, model, person_class_id, frame_w, frame_h):
    """Returns (persons, ppe_boxes).

    persons: list of dicts {id, cx, cy, box} for the "person" class,
             cx/cy normalized to [-1, 1] relative to frame center.
    ppe_boxes: list of (class_name, (x1,y1,x2,y2)) for EVERY class
             listed in PPE_CLASS_STATUS_MAP - both the "wearing it"
             classes (Hardhat, Safety Vest, Gloves, Boots) and the
             "not wearing it" classes (no_hardhat, etc). Every
             class is collected here (not just violations) so the
             Excel logger can record a compliant status too, not
             just violations.
    """

    persons = []
    ppe_boxes = []

    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None:

        xyxy = boxes.xyxy.tolist()
        ids = boxes.id.tolist()
        cls_ids = boxes.cls.tolist()

        for (x1, y1, x2, y2), track_id, cls_id in zip(xyxy, ids, cls_ids):

            cls_id = int(cls_id)
            class_name = model.names.get(cls_id, "")

            if class_name.lower() in PPE_CLASS_STATUS_MAP_LOWER:
                ppe_boxes.append((class_name, (x1, y1, x2, y2)))

            if person_class_id is not None and cls_id != person_class_id:
                continue

            box_cx = (x1 + x2) / 2.0
            box_cy = (y1 + y2) / 2.0

            norm_cx = (box_cx - frame_w / 2.0) / (frame_w / 2.0)
            norm_cy = (box_cy - frame_h / 2.0) / (frame_h / 2.0)

            persons.append({
                "id": int(track_id),
                "cx": norm_cx,
                "cy": norm_cy,
                "box": (x1, y1, x2, y2),
            })

    return persons, ppe_boxes


def wait_for_first_frame(grabber):

    wait_start = time.time()

    while grabber.latest_frame is None:

        if time.time() - wait_start > 10:
            grabber.stop()
            raise RuntimeError(
                "No frames received from the RTSP stream within 10 seconds."
            )

        time.sleep(0.1)


def run_detection_and_tracking(rtsp_url, ptz, shared_state):

    print(f"\nLoading YOLO model: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
    print(f"Model classes: {model.names}")

    device = pick_device()
    person_class_id = find_person_class_id(model)

    print(f"\nOpening camera stream: {rtsp_url}")
    grabber = FrameGrabber(rtsp_url).start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    wait_for_first_frame(grabber)

    tracker = TargetTracker()
    tracking_start_time = 0.0

    # Exponential smoothing state for the visual-servo pan/tilt
    # speeds (see the TRACKING branch below). This is on top of the
    # existing DEADBAND (dead zone) and the proportional PAN_KP /
    # TILT_KP control (which already tapers speed down as the
    # target approaches center) - smoothing here removes the
    # remaining frame-to-frame jitter so the commanded speed eases
    # in/out instead of jumping, and stops it oscillating back and
    # forth around the center.
    SMOOTHING_ALPHA = 0.4
    smoothed_pan_speed = 0.0
    smoothed_tilt_speed = 0.0

    ppe_logger = PPEExcelLogger()

    # Gates PPE Excel rows: a given Track ID can only produce a new
    # row once every PPE_LOG_COOLDOWN_SECONDS (default 300s / 5min),
    # independently per ID. This does NOT affect who gets tracked -
    # only whether a log row gets written.
    ppe_log_cooldown = CooldownManager()

    # Gates who the camera is allowed to physically follow next: a
    # given Track ID can't be re-selected as the active PTZ target
    # again until TRACKING_COOLDOWN_SECONDS (2 min) have passed
    # since its last 5s tracking window ended. Independent from
    # ppe_log_cooldown above - a person can be "off" the tracking
    # cooldown/on the log cooldown or vice versa.
    tracking_cooldown = CooldownManager(cooldown_seconds=TRACKING_COOLDOWN_SECONDS)

    # =====================================================
    # NEW: Backend PPE alert manager + integrated feed server
    # =====================================================
    # PPEAlertManager: async worker thread that sends HTTP POST alerts
    # to the backend.  Should_send_alert() / queue_alert() are called
    # from the detection loop below and are completely non-blocking.
    alert_manager = PPEAlertManager()

    # FeedServer: Flask MJPEG server running in a daemon thread.
    # Serves /feed (raw) and /annotated_feed (YOLO bounding-box).
    # update_raw() / update_annotated() are called every frame.
    feed_server = FeedServer(port=FEED_SERVER_PORT)
    feed_server.start()

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
                frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

            frame_h, frame_w = frame.shape[:2]

            # Always run the tracker (persist=True) regardless of mode,
            # so ByteTrack IDs stay continuous across the PATROL <->
            # TRACKING transition instead of resetting.
            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CONFIG,
                device=device,
                conf=YOLO_CONF_THRESHOLD,
                imgsz=YOLO_IMG_SIZE,
                verbose=False,
            )

            persons, ppe_boxes = extract_detections(
                results, model, person_class_id, frame_w, frame_h
            )

            mode = shared_state.get_mode()
            target = None
            active_violations = []
            # NEW: set once per frame when a backend alert is due
            _pending_ppe_alert = None

            if mode == "PATROL":

                if persons:
                    next_target = select_next_target(persons, tracking_cooldown)

                    if next_target is not None:
                        # Hand off to tracking: lock onto the
                        # earliest-detected eligible person (lowest
                        # Track ID that isn't on the 2-minute
                        # tracking cooldown), start the 5s timer,
                        # and signal the patrol thread to stop
                        # immediately.
                        shared_state.set_mode("TRACKING")
                        shared_state.set_status("TRACKING")

                        tracker.target_id = next_target["id"]
                        tracker.missing_frames = 0
                        target = next_target

                        tracking_start_time = time.time()
                        shared_state.person_trigger_event.set()

                        mode = "TRACKING"

                    # else: people are visible, but every one of
                    # them was already tracked within the last 2
                    # minutes - stay in PATROL, don't interrupt the
                    # sweep for someone we just finished following.

            else:  # mode == "TRACKING"
                target = tracker.update(persons)

            if mode == "TRACKING":

                if target is None:
                    shared_state.set_desired_velocity(0.0, 0.0, locked=False)
                    smoothed_pan_speed = 0.0
                    smoothed_tilt_speed = 0.0
                else:
                    err_x = target["cx"]
                    err_y = target["cy"]

                    if abs(err_x) < DEADBAND:
                        pan_speed = 0.0
                    else:
                        pan_speed = max(-MAX_PAN_SPEED, min(MAX_PAN_SPEED, PAN_KP * err_x))

                    if abs(err_y) < DEADBAND:
                        tilt_speed = 0.0
                    else:
                        tilt_speed = max(
                            -MAX_TILT_SPEED, min(MAX_TILT_SPEED, TILT_SIGN * TILT_KP * err_y)
                        )

                    # Ease the commanded speed toward the new
                    # proportional-control target instead of jumping
                    # straight to it - removes jitter and prevents
                    # oscillation around the center.
                    smoothed_pan_speed += SMOOTHING_ALPHA * (pan_speed - smoothed_pan_speed)
                    smoothed_tilt_speed += SMOOTHING_ALPHA * (tilt_speed - smoothed_tilt_speed)

                    shared_state.set_desired_velocity(
                        smoothed_pan_speed, smoothed_tilt_speed, locked=True
                    )

                    # For every PPE-related box (both "wearing it"
                    # and "not wearing it" classes) that overlaps
                    # the locked target enough to belong to them,
                    # record that item's status. If the same item
                    # somehow fires both ways in one frame, the
                    # violation wins (fail-safe: don't mark someone
                    # compliant on an ambiguous frame).
                    item_status_this_frame = {}

                    for class_name, ppe_box in ppe_boxes:
                        overlap = intersection_over_box_area(ppe_box, target["box"])
                        if overlap < PPE_OVERLAP_THRESHOLD:
                            continue

                        item, status = PPE_CLASS_STATUS_MAP_LOWER[class_name.lower()]

                        if status == "VIOLATION":
                            active_violations.append(class_name)
                            maybe_alert_violation(target["id"], class_name)

                        if item_status_this_frame.get(item) != "VIOLATION":
                            item_status_this_frame[item] = status

                    # NEW: Check global 15-second backend alert cooldown.
                    # If violations exist and cooldown has expired, stage
                    # an alert. The actual queue_alert() call is deferred
                    # until annotated_frame (with bounding boxes) is ready.
                    if active_violations and alert_manager.should_send_alert():
                        _pending_ppe_alert = (
                            target["id"],
                            list(active_violations),
                            dict(item_status_this_frame),
                        )

                    # Only write a new Excel row if this Track ID
                    # isn't currently on cooldown (see spec: each
                    # person may be logged at most once every 5
                    # minutes, independent per ID). Tracking and the
                    # on-screen PPE overlay above are unaffected -
                    # this only gates the spreadsheet write.
                    if item_status_this_frame and ppe_log_cooldown.should_log(target["id"]):

                        screenshot_path = None
                        try:
                            screenshot_name = (
                                f"track_{target['id']}_"
                                f"{int(time.time() * 1000)}.jpg"
                            )
                            screenshot_path = os.path.join(SCREENSHOT_DIR, screenshot_name)
                            cv2.imwrite(screenshot_path, frame)
                        except Exception as exc:
                            print(f"PPE log: screenshot save failed ({exc}).")
                            screenshot_path = None

                        ppe_logger.log(
                            target["id"],
                            item_status_this_frame,
                            screenshot_path=screenshot_path,
                        )

                elapsed = time.time() - tracking_start_time

                if elapsed >= TRACK_DURATION:

                    # This Track ID's 5s window is over - it goes on
                    # the 2-minute tracking cooldown so it can't be
                    # re-selected as the active target immediately,
                    # even if it's still standing right there.
                    if tracker.target_id is not None:
                        tracking_cooldown.mark_now(tracker.target_id)

                    # If another eligible person (not on cooldown)
                    # is STILL visible this frame, hand off straight
                    # to them - patrol stays paused and the camera
                    # does NOT return home in between different
                    # people, only after everyone currently visible
                    # has been covered.
                    next_target = select_next_target(persons, tracking_cooldown)

                    if next_target is not None:
                        print(
                            f"\n>>> {TRACK_DURATION:.0f}s window elapsed for ID "
                            f"{tracker.target_id} - handing off to ID {next_target['id']}."
                        )

                        tracker.target_id = next_target["id"]
                        tracker.missing_frames = 0
                        tracking_start_time = time.time()
                        # mode stays "TRACKING" - patrol remains
                        # paused, no return-HOME move is issued.

                    else:
                        # Nobody currently visible is eligible
                        # (everyone in frame is on cooldown, or the
                        # frame's empty) - finish the tracking
                        # session so the patrol thread returns HOME
                        # and resumes its sweep.
                        print(
                            f"\n>>> {TRACK_DURATION:.0f}s tracking window elapsed - "
                            f"no other eligible person in frame, resuming patrol."
                        )

                        shared_state.set_desired_velocity(0.0, 0.0, locked=False)
                        ptz.stop_move()

                        shared_state.set_mode("PATROL")
                        shared_state.tracking_done_event.set()

            # ---- display ----

            annotated_frame = results[0].plot()

            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                display_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            cv2.putText(
                annotated_frame, f"FPS: {display_fps:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            status_text = shared_state.get_status()
            status_color = (0, 165, 255) if mode == "TRACKING" else (255, 255, 0)

            cv2.putText(
                annotated_frame, status_text,
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
            )

            if mode == "TRACKING":

                remaining = max(0.0, TRACK_DURATION - (time.time() - tracking_start_time))
                cv2.putText(
                    annotated_frame, f"Tracking ID {tracker.target_id} - {remaining:.1f}s left",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )

                if target is not None:
                    if active_violations:
                        ppe_text = "VIOLATION: " + ", ".join(active_violations)
                        ppe_color = (0, 0, 255)
                    else:
                        ppe_text = "PPE OK"
                        ppe_color = (0, 255, 0)

                    cv2.putText(
                        annotated_frame, ppe_text,
                        (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ppe_color, 2
                    )

            # NEW: Push frames to MJPEG feed server (non-blocking)
            feed_server.update_raw(frame)
            feed_server.update_annotated(annotated_frame)

            # NEW: Queue backend PPE alert if one was staged in the
            # tracking block above. Deferred to here so the snapshot
            # captures the fully-annotated frame (bounding boxes +
            # status overlays).  alert_manager.queue_alert() is
            # non-blocking - it saves the JPEG and drops the job on
            # the worker queue; the HTTP POST happens in the background.
            if _pending_ppe_alert is not None:
                _pid, _viols, _istat = _pending_ppe_alert
                alert_manager.queue_alert(
                    person_id      = _pid,
                    violations     = _viols,
                    item_status    = _istat,
                    snapshot_frame = annotated_frame.copy(),
                )
                _pending_ppe_alert = None

            cv2.imshow(WINDOW_NAME, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n'q' pressed, stopping.")
                break

    finally:
        shared_state.set_desired_velocity(0.0, 0.0, locked=False)
        ptz.stop_move()
        grabber.stop()
        cv2.destroyAllWindows()
        ppe_logger.close()
        print(f"PPE log saved to: {ppe_logger.path}")
        # NEW: gracefully shut down the background alert worker
        alert_manager.stop()



# =====================================================
# NGROK AUTO-TUNNEL (Fixed Static Domain)
# =====================================================
# Automatically starts an ngrok tunnel on FEED_SERVER_PORT so the
# MJPEG feed is reachable from anywhere on the internet.
# Using fixed static domain: flatfoot-coat-rosy.ngrok-free.dev
# The dashboard / developer can hardcode this domain permanently.

NGROK_DOMAIN = "flatfoot-coat-rosy.ngrok-free.dev"

def start_ngrok_tunnel(port, domain=NGROK_DOMAIN):
    """
    Starts ngrok in the background and returns the public HTTPS URL.
    Returns None if ngrok is not installed or fails to start.
    """
    try:
        # Kill any lingering ngrok processes to avoid "session limit" conflicts
        try:
            subprocess.run(["pkill", "-x", "ngrok"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
        except Exception:
            pass

        cmd = ["ngrok", "http", str(port), "--log=stdout", "--log-format=json"]
        if domain:
            cmd.extend(["--url", domain])

        # Launch ngrok as a background subprocess
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(lambda: proc.terminate())

        # Give ngrok up to 6 seconds to establish the tunnel
        deadline = time.time() + 6.0
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            try:
                entry = json.loads(line.decode("utf-8", errors="ignore"))
                if entry.get("msg") == "started tunnel":
                    url = entry.get("url", "")
                    if url.startswith("https"):
                        return url
            except Exception:
                pass

        # Fallback: query ngrok local API for tunnel URL
        try:
            import urllib.request
            with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=3) as r:
                tunnels = json.loads(r.read())
                for t in tunnels.get("tunnels", []):
                    if t.get("proto") == "https":
                        return t["public_url"]
        except Exception:
            pass

        # Fallback: if domain is set and process is alive, return the fixed domain URL
        if domain and proc.poll() is None:
            return f"https://{domain}"

    except FileNotFoundError:
        print("[NGROK] ngrok not found. Install with:")
        print("  wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.tgz")
        print("  tar -xzf ngrok-v3-stable-linux-arm64.tgz && sudo mv ngrok /usr/local/bin/")
        print("  ngrok config add-authtoken YOUR_TOKEN")
    except Exception as exc:
        print(f"[NGROK] Failed to start tunnel: {exc}")

    return None


# =====================================================
# MAIN
# =====================================================

def main():

    print(f"RTSP stream URI: {RTSP_URL}")

    # ---- Auto-start ngrok tunnel for public feed access ----
    print(f"\n[NGROK] Starting permanent public tunnel for feed server (port {FEED_SERVER_PORT})...")
    ngrok_url = start_ngrok_tunnel(FEED_SERVER_PORT, domain=NGROK_DOMAIN)

    if ngrok_url:
        print("\n" + "=" * 65)
        print("  PERMANENT PUBLIC FEED URLs (hardcode once in backend / dashboard):")
        print(f"  Raw feed       : {ngrok_url}/feed")
        print(f"  Annotated feed : {ngrok_url}/annotated_feed")
        print("=" * 65 + "\n")
    else:
        print("[NGROK] No public URL — feed available on local network only.")
        print(f"  Local feed: http://10.1.68.42:{FEED_SERVER_PORT}/annotated_feed\n")

    shared_state = SharedState()
    ptz = PTZController()

    # Sends ContinuousMove/Stop for the tracking visual servo.
    # Only actually moves the camera while shared_state's velocity
    # is "locked" (mode == TRACKING) - stays silent during patrol.
    cmd_thread = threading.Thread(
        target=ptz_command_loop, args=(ptz, shared_state), daemon=True
    )
    cmd_thread.start()

    # Runs the 4-stage tilt/zoom patrol sweep in the background.
    # Parks itself (via handle_person_interruption) whenever a
    # person is detected, and resumes automatically once tracking
    # hands back.
    patrol_thread = threading.Thread(
        target=patrol, args=(ptz, shared_state), daemon=True
    )
    patrol_thread.start()

    # YOLO detection + tracking + display run on the main thread
    # (cv2.imshow needs to run on the main thread).
    try:
        run_detection_and_tracking(RTSP_URL, ptz, shared_state)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ptz.stop_move()


if __name__ == "__main__":
    main()