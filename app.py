"""
Flask Backend for OralScan Dashboard
=====================================
Connects strip_analysis.py to the HTML dashboard.
Also drives PiCamera2 + ULN2003 stepper motor for Raspberry Pi 5 deployment.

Install dependencies (Pi):
    pip install flask flask-cors picamera2 RPi.GPIO opencv-python-headless numpy
    pip install edge_impulse_linux   # Edge Impulse Linux SDK

Run:
    python app.py

Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import tempfile
import base64
import threading
import time
import json
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
#  STEPPER MOTOR CONFIG
# ─────────────────────────────────────────────────────────────────────────────
USE_STEPPER   = True
MOTOR_PINS    = [17, 18, 27, 22]   # IN1..IN4 → BCM GPIO
STEP_DELAY    = 0.003              # seconds between half-steps (medium speed)

HALF_STEP_SEQ = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
]

# ─────────────────────────────────────────────────────────────────────────────
#  GPIO SETUP (safe-fail — skips on Windows / if not wired)
# ─────────────────────────────────────────────────────────────────────────────
_gpio_ok = False
_motor_devices = []

if USE_STEPPER:
    try:
        from gpiozero import DigitalOutputDevice
        for pin in MOTOR_PINS:
            _motor_devices.append(DigitalOutputDevice(pin, initial_value=False))
        _gpio_ok = True
        print("[GPIO] Stepper motor pins initialised via gpiozero.")
    except Exception as e:
        print(f"[WARN] GPIO init failed ({e}) — motor disabled (OK on non-Pi).")

# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA SETUP (safe-fail)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
except ImportError:
    USE_PICAMERA = False
    print("[WARN] picamera2 not found — camera features unavailable (OK on non-Pi).")

import cv2
import numpy as np

PREVIEW_W, PREVIEW_H = 1640, 1232   # Half-sensor resolution — faster streaming

OUTPUT_DIR = os.path.expanduser("~/smd_captures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  EDGE IMPULSE MODEL SETUP
# ─────────────────────────────────────────────────────────────────────────────
EIM_MODEL_PATH = '/home/pi/model.eim'

EI_CONFIDENCE_THRESHOLD = 0.4   # minimum confidence to accept a detection

# Labels your Edge Impulse model was trained with (edit to match your project):
# These correspond to the analytes detected by the saliva strip.
EI_LABELS = ["Cysteine", "Glutathione", "Sialic Acid"]

_ei_runner = None
_ei_model_params = None

def _load_ei_model():
    """Load the Edge Impulse .eim model. Called once at startup."""
    global _ei_runner, _ei_model_params
    try:
        from edge_impulse_linux.image import ImageImpulseRunner
        _ei_runner = ImageImpulseRunner(EIM_MODEL_PATH)
        _ei_model_params = _ei_runner.init()
        print(f"[EI] Model loaded: {_ei_model_params['project']['name']}")
        print(f"[EI] Labels: {_ei_model_params['model_parameters']['labels']}")
        print(f"[EI] Input: {_ei_model_params['model_parameters']['image_input_width']}"
              f"x{_ei_model_params['model_parameters']['image_input_height']}")
    except Exception as e:
        print(f"[ERROR] Failed to load Edge Impulse model: {e}")
        _ei_runner = None


def run_ei_inference(image_path):
    """
    Run Edge Impulse inference on an image file.

    Returns detections or None on failure.
    """
    global _ei_runner, _ei_model_params

    if _ei_runner is None:
        print("[EI] Runner not initialised — cannot run inference.")
        return None

    try:
        # Read image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"[EI] Cannot read image: {image_path}")
            return None

        # ✅ FIX: Convert to GRAYSCALE (your model expects 1 channel)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Resize to model input size
        w = _ei_model_params['model_parameters']['image_input_width']
        h = _ei_model_params['model_parameters']['image_input_height']

        img_resized = cv2.resize(img_gray, (w, h))

        # Flatten → correct size now
        features = img_resized.flatten().tolist()

        # 🔍 Debug (optional)
        # print("Expected:", len(_ei_runner._input_shm['array']))
        # print("Got:", len(features))

        # Run inference
        result = _ei_runner.classify(features)

        detections = []

        # Object detection
        if "bounding_boxes" in result["result"]:
            for bb in result["result"]["bounding_boxes"]:
                if bb["value"] >= EI_CONFIDENCE_THRESHOLD:
                    detections.append({
                        "label":  bb["label"],
                        "value":  bb["value"],
                        "x":      bb["x"],
                        "y":      bb["y"],
                        "width":  bb["width"],
                        "height": bb["height"],
                    })

        # Classification fallback
        elif "classification" in result["result"]:
            for label, confidence in result["result"]["classification"].items():
                if confidence >= EI_CONFIDENCE_THRESHOLD:
                    detections.append({
                        "label":  label,
                        "value":  confidence,
                        "x":      0,
                        "y":      0,
                        "width":  w,
                        "height": h,
                    })

        return detections

    except Exception as e:
        print(f"[EI] Inference error: {e}")
        import traceback
        traceback.print_exc()
        return None

def ei_detections_to_results(detections, image_path):
    """
    Convert Edge Impulse detections into the same results dict
    that _format_results() expects.

    Maps detected labels to analyte slots (Cysteine, Glutathione, Sialic Acid).
    Reads average RGB from the detected bounding-box region of the source image.
    """
    img_bgr = cv2.imread(image_path)
    ih, iw = img_bgr.shape[:2] if img_bgr is not None else (1, 1)

    w_in = _ei_model_params['model_parameters']['image_input_width']  if _ei_model_params else iw
    h_in = _ei_model_params['model_parameters']['image_input_height'] if _ei_model_params else ih

    def get_avg_rgb(x, y, bw, bh):
        """Return mean BGR→RGB of the detection region, scaled to source image."""
        if img_bgr is None:
            return [128, 128, 128]
        sx = int(x  * iw / w_in)
        sy = int(y  * ih / h_in)
        ex = int((x + bw) * iw / w_in)
        ey = int((y + bh) * ih / h_in)
        sx, sy = max(0, sx), max(0, sy)
        ex, ey = min(iw, ex), min(ih, ey)
        if ex <= sx or ey <= sy:
            return [128, 128, 128]
        roi = img_bgr[sy:ey, sx:ex]
        mean = roi.mean(axis=(0, 1))   # BGR order
        return [int(mean[2]), int(mean[1]), int(mean[0])]   # RGB

    # Index detections by label (use highest-confidence per label)
    det_by_label = {}
    for d in detections:
        lbl = d["label"]
        if lbl not in det_by_label or d["value"] > det_by_label[lbl]["value"]:
            det_by_label[lbl] = d

    def make_analyte(label):
        d = det_by_label.get(label)
        if d is None:
            return None
        rgb = get_avg_rgb(d["x"], d["y"], d["width"], d["height"])
        # Derive a pseudo concentration_level (0–19) from confidence
        concentration_level = int(d["value"] * 19)
        concentration_pct   = d["value"] * 100.0
        return {
            "analyte":            label,
            "confidence":         d["value"],
            "rgb":                rgb,
            "concentration_level": concentration_level,
            "concentration_pct":   concentration_pct,
            "bbox":               {
                "x": d["x"], "y": d["y"],
                "width": d["width"], "height": d["height"],
            },
        }

    analytes_found = [make_analyte(lbl) for lbl in EI_LABELS]
    analytes_found = [a for a in analytes_found if a is not None]

    cys  = det_by_label.get("Cysteine")
    glut = det_by_label.get("Glutathione")
    sial = det_by_label.get("Sialic Acid")

    cys_risk  = cys["value"]  if cys  else 0.0
    glut_risk = glut["value"] if glut else 0.0
    sial_risk = sial["value"] if sial else 0.0

    # Simple weighted average cancer risk
    weights   = [0.4, 0.3, 0.3]
    risks     = [cys_risk, glut_risk, sial_risk]
    pct       = sum(w * r for w, r in zip(weights, risks)) * 100.0

    if pct < 30:
        category, emoji = "Low Risk", "🟢"
    elif pct < 60:
        category, emoji = "Moderate Risk", "🟡"
    else:
        category, emoji = "High Risk", "🔴"

    return {
        "detections": analytes_found,
        "cancer_risk": {
            "percentage": pct,
            "category":   category,
            "emoji":      emoji,
            "cys_risk":   cys_risk,
            "glut_risk":  glut_risk,
            "sial_risk":  sial_risk,
        },
        "biomarkers": {
            "cysteine_detected":    cys  is not None,
            "glutathione_detected": glut is not None,
            "sialic_acid_detected": sial is not None,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MOTOR DRIVER (background thread)  — v4: variable-speed via delay control
# ─────────────────────────────────────────────────────────────────────────────
STEP_FAST  = 0.005   # fastest — searching
STEP_SLOW  = 0.015   # medium  — approaching
STEP_CRAWL = 0.030   # slowest — about to capture

_motor_running = False
_motor_delay   = STEP_FAST
_motor_lock    = threading.Lock()
_motor_thread  = None


def _step_loop():
    seq_index = 0
    while True:
        with _motor_lock:
            should_run = _motor_running
            delay      = _motor_delay
        if not should_run:
            if _gpio_ok:
                for dev in _motor_devices:
                    dev.off()
            time.sleep(0.02)
            continue
        if _gpio_ok:
            step = HALF_STEP_SEQ[seq_index % len(HALF_STEP_SEQ)]
            for pin_idx, dev in enumerate(_motor_devices):
                if step[pin_idx]:
                    dev.on()
                else:
                    dev.off()
        seq_index += 1
        time.sleep(delay)


def _ensure_motor_thread():
    global _motor_thread
    if _motor_thread is None or not _motor_thread.is_alive():
        _motor_thread = threading.Thread(target=_step_loop, daemon=True)
        _motor_thread.start()


def motor_go(d=STEP_FAST):
    global _motor_running, _motor_delay
    _ensure_motor_thread()
    with _motor_lock:
        _motor_running = True
        _motor_delay   = d
    print(f"[MOTOR] ON  {d*1000:.0f}ms/half-step")


def motor_speed(d):
    """Adjust step delay without stopping the motor."""
    global _motor_delay
    with _motor_lock:
        _motor_delay = d


def motor_stop():
    global _motor_running
    with _motor_lock:
        _motor_running = False
    print("[MOTOR] OFF")


# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_cam = None
_cam_lock = threading.Lock()
_latest_frame = None
_cam_thread_running = False

def _cam_worker():
    global _latest_frame
    while _cam_thread_running:
        with _cam_lock:
            if _cam is None:
                time.sleep(0.1)
                continue
            try:
                if USE_PICAMERA:
                    arr = _cam.capture_array()
                else:
                    ok, f = _cam.read()
                    if not ok: f = None
            except Exception:
                f = None
        
        if f is not None:
            _latest_frame = f
        else:
            time.sleep(0.01)

def open_camera():
    global _cam, _cam_thread_running
    with _cam_lock:
        if _cam is not None:
            return _cam
        
        if not USE_PICAMERA:
            _cam = cv2.VideoCapture(0)
            _cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            _cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            _cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        else:
            try:
                import subprocess
                for pattern in ["libcamera", "rpicam", "picamera2", "picamera"]:
                    subprocess.run(["sudo", "pkill", "-f", pattern],
                                   capture_output=True, timeout=2)
                time.sleep(0.5)
                cam = Picamera2()
                cfg = cam.create_video_configuration(
                    main={"size": (PREVIEW_W, PREVIEW_H), "format": "RGB888"},
                    controls={"FrameRate": 15}
                )
                cam.configure(cfg)
                cam.start()
                time.sleep(1.5)
                _cam = cam
            except Exception as e:
                print(f"[WARN] Camera open failed: {e}")
                return None

    if not _cam_thread_running:
        _cam_thread_running = True
        threading.Thread(target=_cam_worker, daemon=True).start()
    return _cam


def grab_frame():
    """Returns the latest frame from the fast background camera thread."""
    if _latest_frame is None:
        return None
    return _latest_frame.copy()


def release_camera():
    global _cam, _cam_thread_running
    _cam_thread_running = False
    with _cam_lock:
        if _cam is not None:
            try:
                if USE_PICAMERA:
                    _cam.stop()
                else:
                    _cam.release()
            except Exception:
                pass
            _cam = None


# ─────────────────────────────────────────────────────────────────────────────
#  SMD DETECTION HELPERS  — v4: shape-based, colour-agnostic
# ─────────────────────────────────────────────────────────────────────────────
MAX_CAPTURES = 10

TARGET_ASPECT    = 2.8 / 3.0

MIN_FILL = 0.005     # allow smaller objects
MAX_FILL = 0.95

MIN_RECT_SCORE = 0.25   # was too strict

ASPECT_TOL = 1.0     # allow more shapes (critical)
CONFIDENCE_THRESHOLD = 0.30
CONFIRM_FRAMES       = 3
SETTLE_TIME          = 0.35
SHOW_CAPTURED_SEC    = 1.2
COOLDOWN_SEC         = 1.0
SMOOTH_ALPHA         = 0.40
EDGE_FRAC            = 0.20
CENTRE_MARGIN        = 25


def find_component(frame):
    """
    Shape-based detection — colour agnostic.
    Returns (bbox, score) or (None, 0).
    """
    fh, fw = frame.shape[:2]
    small  = cv2.resize(frame, (fw // 2, fh // 2))
    sh, sw = small.shape[:2]
    s_area = sh * sw

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    blurred5 = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred7 = cv2.GaussianBlur(gray, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        blurred7, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        31, 4
    )
    canny = cv2.Canny(blurred5, 15, 60)
    combined = cv2.bitwise_or(thresh, canny)

    k_close  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,  k_close,  iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_DILATE, k_dilate, iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best_bbox, best_score = None, 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        fill = area / s_area
        if fill < MIN_FILL or fill > MAX_FILL:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw == 0 or bh == 0:
            continue

        aspect = bw / bh
        if abs(aspect - TARGET_ASPECT) > ASPECT_TOL:
            continue

        rect_fill = area / (bw * bh)

        asp_score = max(0, 1.0 - abs(aspect - TARGET_ASPECT) / ASPECT_TOL)
        score = 0.6 * asp_score + 0.4 * rect_fill

        print(f"  [DBG] fill={fill:.4f} asp={aspect:.3f} "
              f"rect={rect_fill:.2f} score={score:.3f}")

        if score > best_score:
            best_score = score
            best_bbox  = (x * 2, y * 2, bw * 2, bh * 2)
            
        # Fallback: if nothing found, pick largest contour
        if best_bbox is None and contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(cnt)
            best_bbox = (x * 2, y * 2, bw * 2, bh * 2)
            best_score = 0.3   # force minimal confidence

    return best_bbox, best_score


def smooth_bbox(prev, curr, alpha=SMOOTH_ALPHA):
    """Exponential smoothing to suppress bbox jitter."""
    if prev is None:
        return curr
    return tuple(int(alpha*c + (1-alpha)*p) for p, c in zip(prev, curr))


def in_edge_zone(bbox, fshape):
    """True if bbox centre is in the outer EDGE_FRAC of the frame."""
    x, y, bw, bh = bbox
    fh, fw = fshape[:2]
    cx, cy = x + bw // 2, y + bh // 2
    ez = EDGE_FRAC
    return cx < fw*ez or cx > fw*(1-ez) or cy < fh*ez or cy > fh*(1-ez)


def fully_centred(bbox, fshape):
    """True if bounding box is fully inside the centre margin."""
    x, y, bw, bh = bbox
    fh, fw = fshape[:2]
    m = CENTRE_MARGIN
    return x > m and y > m and x+bw < fw-m and y+bh < fh-m


# ─────────────────────────────────────────────────────────────────────────────
#  SCAN STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────
scan_state = {
    "running":          False,
    "motor":            "STOPPED",
    "camera":           "OFF",
    "phase":            "IDLE",
    "captures":         0,
    "last_score":       0.0,
    "last_capture_path": None,
    "last_capture_url":  None,
    "error":            None,
    "confirm_count":    0,
}
_scan_lock = threading.Lock()
_scan_thread = None
session_captures = []


def _scan_loop():
    """
    Background thread: motor + detection loop.
    - Searches for object, calculates score average over 5 seconds
    - If average score >= 0.5, captures
    - Otherwise, continues rotating
    - Motor decelerates as object enters frame
    """
    global scan_state, _cam

    with _scan_lock:
        scan_state["phase"] = "SCANNING"
        scan_state["motor"] = "SPINNING"

    motor_go(STEP_FAST)
    smooth_box      = None
    wait_for_gone   = False
    score_buffer    = []  # Track scores over 5 seconds
    buffer_start    = time.time()
    BUFFER_WINDOW   = 5.0  # seconds

    while True:
        with _scan_lock:
            if not scan_state["running"]:
                break
            total_captures = scan_state["captures"]

        if total_captures >= MAX_CAPTURES:
            with _scan_lock:
                scan_state["phase"] = "DONE"
                scan_state["motor"] = "STOPPED"
            motor_stop()
            print(f"[SCAN] Session complete — {total_captures}/{MAX_CAPTURES} images.")
            break

        frame = grab_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        bbox, score = find_component(frame)
        now = time.time()

        # Clear old scores outside the 5-second window
        score_buffer = [s for s in score_buffer if now - s['time'] < BUFFER_WINDOW]

        with _scan_lock:
            scan_state["last_score"] = round(score, 3)

        if wait_for_gone:
            if bbox is None or score < CONFIDENCE_THRESHOLD:
                wait_for_gone = False
                score_buffer = []
                buffer_start = time.time()
                smooth_box   = None
                motor_go(STEP_FAST)
                with _scan_lock:
                    scan_state["phase"] = "SCANNING"
                    scan_state["motor"] = "SPINNING"
                print("[INFO] Object cleared — re-armed")
            else:
                with _scan_lock:
                    scan_state["phase"] = "WAIT_GONE"
            time.sleep(0.02)
            continue

        if bbox is not None and score >= CONFIDENCE_THRESHOLD:
            # Add score to buffer
            score_buffer.append({'score': score, 'time': now, 'bbox': bbox})

            # Calculate average score
            if score_buffer:
                avg_score = sum(s['score'] for s in score_buffer) / len(score_buffer)
                frame_count = len(score_buffer)
            else:
                avg_score = 0.0
                frame_count = 0

            smooth_box = smooth_bbox(smooth_box, bbox)
            centred    = fully_centred(smooth_box, frame.shape)
            edge       = in_edge_zone(smooth_box, frame.shape)

            # Update phase with average score info
            with _scan_lock:
                scan_state["phase"]         = f"ANALYZING ({frame_count} frames)"
                scan_state["last_score"]    = round(avg_score, 3)
                scan_state["confirm_count"] = frame_count

            # Adjust motor speed based on position
            if not centred:
                motor_speed(STEP_SLOW if edge else STEP_CRAWL)
            else:
                motor_speed(STEP_CRAWL)

            # Check if we've been analyzing for 5 seconds
            elapsed = now - buffer_start
            if elapsed >= BUFFER_WINDOW and score_buffer:
                avg_score = sum(s['score'] for s in score_buffer) / len(score_buffer)
                
                print(f"[ANALYSIS] 5-second window complete:")
                print(f"  Frames: {len(score_buffer)}")
                print(f"  Average score: {avg_score:.3f}")
                print(f"  Threshold: {0.5}")

                if avg_score >= 0.5:
                    # CAPTURE
                    motor_stop()
                    with _scan_lock:
                        scan_state["phase"] = "CAPTURING"
                        scan_state["motor"] = "STOPPED"

                    time.sleep(SETTLE_TIME)

                    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    photo_name = f"smd_{timestamp}.jpg"
                    photo_path = os.path.join(OUTPUT_DIR, photo_name)

                    frame2 = grab_frame()
                    if frame2 is not None:
                        x, y, bw, bh = [int(v) for v in smooth_box]
                        fh, fw = frame2.shape[:2]
                        pad_x = int(bw * 0.10)
                        pad_y = int(bh * 0.10)
                        x1 = max(0,  x - pad_x)
                        y1 = max(0,  y - pad_y)
                        x2 = min(fw, x + bw + pad_x)
                        y2 = min(fh, y + bh + pad_y)
                        crop = frame2[y1:y2, x1:x2]
                        if crop.size > 0:
                            cv2.imwrite(photo_path, crop)
                        else:
                            cv2.imwrite(photo_path, frame2)
                            crop = frame2

                        annotated_name = photo_name.replace('.jpg', '_annotated.jpg')
                        annotated_path = os.path.join(OUTPUT_DIR, annotated_name)
                        ann = crop.copy()
                        ax  = x - x1
                        ay  = y - y1
                        abw, abh = bw, bh
                        col = (0, 220, 80)
                        dk  = (20, 20, 20)
                        L   = min(abw, abh) // 5
                        FONT_ANN = cv2.FONT_HERSHEY_SIMPLEX
                        for pts in [
                            ((ax,       ay+L),     (ax,       ay),     (ax+L,     ay)),
                            ((ax+abw-L, ay),       (ax+abw,   ay),     (ax+abw,   ay+L)),
                            ((ax+abw,   ay+abh-L), (ax+abw,   ay+abh), (ax+abw-L, ay+abh)),
                            ((ax+L,     ay+abh),   (ax,       ay+abh), (ax,       ay+abh-L)),
                        ]:
                            cv2.polylines(ann, [np.array(pts, np.int32)], False, col, 3, cv2.LINE_AA)
                        lbl = f"3.0x2.8cm  score:{avg_score:.2f}"
                        (tw, th), _ = cv2.getTextSize(lbl, FONT_ANN, 0.55, 2)
                        cv2.rectangle(ann, (ax, ay-th-14), (ax+tw+10, ay), col, -1)
                        cv2.putText(ann, lbl, (ax+5, ay-6), FONT_ANN, 0.55, dk, 2, cv2.LINE_AA)
                        cv2.imwrite(annotated_path, ann)

                        with _scan_lock:
                            scan_state["captures"]          += 1
                            scan_state["last_capture_path"]  = photo_path
                            scan_state["last_capture_url"]   = f"/capture-image/{photo_name}"
                            total_captures                   = scan_state["captures"]
                            session_captures.append({
                                "name":           photo_name,
                                "annotated_name": annotated_name,
                            })

                        print(f"[✓] {photo_path}  ({total_captures}/{MAX_CAPTURES})")
                        print(f"[✓] Annotated: {annotated_path}")

                    time.sleep(COOLDOWN_SEC)
                    motor_go(STEP_FAST)
                    wait_for_gone = True
                    score_buffer = []
                    buffer_start = time.time()
                    smooth_box   = None
                    with _scan_lock:
                        scan_state["confirm_count"] = 0
                        scan_state["phase"]         = "WAIT_GONE"
                        scan_state["motor"]         = "SPINNING"

                else:
                    # Average score too low — continue rotating
                    print(f"[INFO] Average score {avg_score:.3f} below threshold 0.5 — continuing rotation")
                    score_buffer = []
                    buffer_start = time.time()
                    smooth_box   = None
                    motor_go(STEP_FAST)
                    with _scan_lock:
                        scan_state["phase"]         = "SEARCHING"
                        scan_state["confirm_count"] = 0

        else:
            # No object detected — reset buffer
            score_buffer = []
            buffer_start = time.time()
            smooth_box   = None
            motor_speed(STEP_FAST)
            with _scan_lock:
                scan_state["phase"]         = "SEARCHING"
                scan_state["confirm_count"] = 0

        time.sleep(0.02)

    motor_stop()
    with _scan_lock:
        scan_state["phase"] = "IDLE"
        scan_state["motor"] = "STOPPED"
        scan_state["confirm_count"] = 0
    print("[SCAN] Loop exited.")

# ─────────────────────────────────────────────────────────────────────────────
#  MJPEG STREAM GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def _gen_mjpeg():
    """Yields MJPEG frames for the /stream endpoint."""
    while True:
        frame = grab_frame()
        if frame is None:
            placeholder = np.full((240, 320, 3), 40, dtype=np.uint8)
            cv2.putText(placeholder, "Camera not available",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            _, buf = cv2.imencode('.jpg', placeholder)
        else:
            with _scan_lock:
                phase = scan_state["phase"]
                score = scan_state["last_score"]
            display = cv2.resize(frame, (640, 480))
            color = (0, 220, 80) if phase == "CAPTURING" else (0, 165, 255)
            label = f"{phase}  score:{score:.2f}"
            cv2.putText(display, label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            _, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 70])

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.05)


# ─────────────────────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.')
CORS(app)

# ── Load Edge Impulse model once at startup ──
print("Loading Edge Impulse model...")
_load_ei_model()
if _ei_runner is not None:
    print("✓ Edge Impulse model ready!\n")
else:
    print("⚠ Edge Impulse model failed to load — analysis endpoints will return errors.\n")


# ── Serve dashboard ──
@app.route('/')
def index():
    return send_from_directory('.', 'saliva_dashboard.html')


# ── Serve captured images ──
@app.route('/capture-image/<filename>')
def serve_capture(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS ENDPOINT  — now uses Edge Impulse model
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'images' not in request.files and 'image' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400

    files = request.files.getlist('images')
    if not files:
        files = request.files.getlist('image')
    if not files or files[0].filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if _ei_runner is None:
        return jsonify({'error': 'Edge Impulse model not loaded'}), 500

    all_results = []
    try:
        for file in files:
            if file.filename == '':
                continue
            suffix = os.path.splitext(file.filename)[1] or '.png'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            try:
                detections = run_ei_inference(tmp_path)
                if detections is None:
                    continue
                results = ei_detections_to_results(detections, tmp_path)
                all_results.append(_format_results(file.filename, results))
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if not all_results:
            return jsonify({'error': 'Could not process images'}), 500
        return jsonify({'success': True, 'results': all_results})

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  HARDWARE ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/calibrate', methods=['POST'])
def calibrate():
    global _cam
    errors = []

    cam_ok = False
    try:
        if _cam is None:
            _cam = open_camera()
        cam_ok = (_cam is not None)
    except Exception:
        pass

    if not cam_ok:
        errors.append("Failed to open Camera")

    with _scan_lock:
        scan_state["camera"] = "LIVE" if cam_ok else "OFF"

    motor_ok = _gpio_ok
    if not motor_ok:
        errors.append("GPIO not available on this platform")
    else:
        try:
            motor_go()
            time.sleep(0.5)
            motor_stop()
        except Exception as e:
            errors.append(f"Motor error: {e}")
            motor_ok = False

    success = cam_ok and motor_ok
    return jsonify({
        "success":   success,
        "camera":    "LIVE" if cam_ok else "OFF",
        "motor":     "OK"   if motor_ok else "UNAVAILABLE",
        "errors":    errors,
    })


@app.route('/start-scan', methods=['POST'])
def start_scan():
    """Start the SMD detection loop in a background thread."""
    global _scan_thread, _cam

    with _scan_lock:
        if scan_state["running"]:
            return jsonify({"success": False, "error": "Already scanning"}), 400

    if _cam is None:
        _cam = open_camera()
    with _scan_lock:
        scan_state["camera"] = "LIVE" if _cam is not None else "OFF"

    with _scan_lock:
        scan_state["running"]       = True
        scan_state["phase"]         = "SCANNING"
        scan_state["error"]         = None
        scan_state["confirm_count"] = 0
        scan_state["captures"]      = 0
        global session_captures
        session_captures = []

    _scan_thread = threading.Thread(target=_scan_loop, daemon=True)
    _scan_thread.start()

    return jsonify({"success": True, "message": "Scan started"})


@app.route('/stop-scan', methods=['POST'])
def stop_scan():
    """Stop the SMD detection loop."""
    with _scan_lock:
        scan_state["running"] = False
        scan_state["phase"]   = "IDLE"
    motor_stop()
    return jsonify({"success": True, "message": "Scan stopped"})


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Cleanly stop everything and terminate the Flask process."""
    def _do_exit():
        time.sleep(0.4)
        motor_stop()
        release_camera()
        if _ei_runner is not None:
            try:
                _ei_runner.stop()
            except Exception:
                pass
        print("[APP] Shutdown requested — bye.")
        os._exit(0)

    with _scan_lock:
        scan_state["running"] = False
        scan_state["phase"]   = "IDLE"
    motor_stop()
    threading.Thread(target=_do_exit, daemon=True).start()
    return jsonify({"success": True, "message": "Shutting down…"})


@app.route('/scan-status')
def scan_status():
    """Polling endpoint — returns current scan_state as JSON."""
    with _scan_lock:
        return jsonify(dict(scan_state))


@app.route('/stream')
def stream():
    """MJPEG live stream from PiCamera2."""
    return Response(
        _gen_mjpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/captures')
def captures():
    """Returns list of captured image filenames from the current session."""
    try:
        global session_captures
        files = list(reversed(session_captures))
        return jsonify({
            "success": True,
            "count":   len(files),
            "files":   [{
                "name":           f["name"] if isinstance(f, dict) else f,
                "url":            f"/capture-image/{f['name'] if isinstance(f, dict) else f}",
                "annotated_url": f"/capture-image/{f['annotated_name']}" if isinstance(f, dict) else None,
            } for f in files]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/analyze-captures', methods=['POST'])
def analyze_captures():
    """Run Edge Impulse analysis on multiple captured images by filename."""
    data = request.get_json(silent=True) or {}
    filenames = data.get('filenames', [])
    ann_map   = data.get('annotated_map', {})
    if not filenames:
        return jsonify({'error': 'No filenames provided'}), 400

    if _ei_runner is None:
        return jsonify({'error': 'Edge Impulse model not loaded'}), 500

    all_results = []
    for filename in filenames:
        path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
        if os.path.exists(path):
            try:
                detections = run_ei_inference(path)
                if detections is not None:
                    results  = ei_detections_to_results(detections, path)
                    response = _format_results(filename, results)
                    ann_name = ann_map.get(filename, filename.replace('.jpg', '_annotated.jpg'))
                    if os.path.exists(os.path.join(OUTPUT_DIR, ann_name)):
                        response['annotated_url'] = f'/capture-image/{ann_name}'
                    all_results.append(response)
            except Exception as e:
                print(f"[WARN] Analysis failed for {filename}: {e}")

    if not all_results:
        return jsonify({'error': 'Could not analyze any images'}), 500

    return jsonify({'success': True, 'results': all_results})


@app.route('/analyze-capture', methods=['POST'])
def analyze_capture():
    """Run Edge Impulse analysis on a single captured image by filename."""
    data = request.get_json(silent=True) or {}
    filename       = data.get('filename', '')
    annotated_name = data.get('annotated_name', '')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        return jsonify({'error': f'File not found: {filename}'}), 404

    if _ei_runner is None:
        return jsonify({'error': 'Edge Impulse model not loaded'}), 500

    try:
        detections = run_ei_inference(path)
        if detections is None:
            return jsonify({'error': 'Could not analyze image'}), 500
        results  = ei_detections_to_results(detections, path)
        response = _format_results(filename, results)
        if annotated_name:
            response['annotated_url'] = f'/capture-image/{os.path.basename(annotated_name)}'
        elif os.path.exists(os.path.join(OUTPUT_DIR, filename.replace('.jpg', '_annotated.jpg'))):
            response['annotated_url'] = f'/capture-image/{filename.replace(".jpg", "_annotated.jpg")}'
        return jsonify({'success': True, 'results': [response]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Health check ──
@app.route('/health')
def health():
    return jsonify({
        'status':       'ok',
        'model':        'edge_impulse' if _ei_runner is not None else 'not_loaded',
        'model_path':   EIM_MODEL_PATH,
        'gpio':         _gpio_ok,
        'camera':       USE_PICAMERA,
        'captures_dir': OUTPUT_DIR,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED RESULT FORMATTER  (unchanged — same output shape as before)
# ─────────────────────────────────────────────────────────────────────────────
def _format_results(filename, results):
    cancer     = results['cancer_risk']
    biomarkers = results['biomarkers']
    detections = results['detections']

    def get_det(name):
        for d in detections:
            if d['analyte'] == name:
                return d
        return None

    cys  = get_det('Cysteine')
    glut = get_det('Glutathione')
    sial = get_det('Sialic Acid')

    cys_conc  = (cys['concentration_level']  / 19) * 130 if cys  else 0
    glut_conc = (glut['concentration_level'] / 19) * 150 if glut else 0
    sial_conc = (sial['concentration_level'] / 19) * 6.0 if sial else 0

    pct = cancer['percentage']
    if pct < 30:
        risk_level = 'low'
    elif pct < 60:
        risk_level = 'moderate'
    else:
        risk_level = 'high'

    return {
        'filename':  filename,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'cysteine': {
            'val':   round(cys_conc, 1),
            'pct':   round(cys['concentration_pct'], 1)  if cys  else 0,
            'rgb':   [int(cys['rgb'][0]),  int(cys['rgb'][1]),  int(cys['rgb'][2])]  if cys  else [0,0,0],
            'conf':  round(cys['confidence']  * 100, 1) if cys  else 0,
            'risk':  round(cancer['cys_risk']  * 100, 1),
            'level': cys['concentration_level']  if cys  else 0,
        },
        'glutathione': {
            'val':   round(glut_conc, 1),
            'pct':   round(glut['concentration_pct'], 1) if glut else 0,
            'rgb':   [int(glut['rgb'][0]), int(glut['rgb'][1]), int(glut['rgb'][2])] if glut else [0,0,0],
            'conf':  round(glut['confidence'] * 100, 1) if glut else 0,
            'risk':  round(cancer['glut_risk'] * 100, 1),
            'level': glut['concentration_level'] if glut else 0,
        },
        'sialic': {
            'val':   round(sial_conc, 3),
            'pct':   round(sial['concentration_pct'], 1) if sial else 0,
            'rgb':   [int(sial['rgb'][0]), int(sial['rgb'][1]), int(sial['rgb'][2])] if sial else [0,0,0],
            'conf':  round(sial['confidence'] * 100, 1) if sial else 0,
            'risk':  round(cancer['sial_risk'] * 100, 1),
            'level': sial['concentration_level'] if sial else 0,
        },
        'overallRisk':   round(cancer['percentage'], 2),
        'riskLevel':     risk_level,
        'riskCategory':  cancer['category'],
        'riskEmoji':     cancer['emoji'],
        'spotsDetected': len(detections),
    }


if __name__ == '__main__':
    print("=" * 52)
    print("  OralScan + SMD Scanner Backend running!")
    print("  Open: http://localhost:5000")
    print("  GPIO  available:", _gpio_ok)
    print("  Camera available:", USE_PICAMERA)
    print("  EI model loaded:", _ei_runner is not None)
    print("=" * 52 + "\n")
    app.run(debug=False, port=5000, threaded=True)
