"""
Flask Backend for OralScan Dashboard
leoo
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
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
#  STEPPER MOTOR CONFIGgggggg
# ─────────────────────────────────────────────────────────────────────────────
USE_STEPPER   = True
MOTOR_PINS    = [17, 18, 27, 22]   # IN1..IN4 → BCM GPIO
MOTOR_DELAY   = 0.008              # slower stepping for smoother, low-vibration spin

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
        pass  # GPIO ready
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
PICAMERA_OUTPUT_IS_BGR = os.getenv("PICAMERA_OUTPUT_IS_BGR", "1").lower() in ("1", "true", "yes", "on")

OUTPUT_DIR = os.path.expanduser("~/smd_captures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MOTOR DRIVER (background thread)  — SIMPLIFIED: single slow speed
# ─────────────────────────────────────────────────────────────────────────────
_motor_running = False
_motor_lock    = threading.Lock()
_motor_thread  = None


def _step_loop():
    seq_index = 0
    while True:
        with _motor_lock:
            should_run = _motor_running
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
        time.sleep(MOTOR_DELAY)


def _ensure_motor_thread():
    global _motor_thread
    if _motor_thread is None or not _motor_thread.is_alive():
        _motor_thread = threading.Thread(target=_step_loop, daemon=True)
        _motor_thread.start()


def motor_go():
    global _motor_running
    _ensure_motor_thread()
    with _motor_lock:
        _motor_running = True
    pass  # motor on


def motor_stop():
    global _motor_running
    with _motor_lock:
        _motor_running = False
    # De-energise all coils immediately
    if _gpio_ok:
        for dev in _motor_devices:
            dev.off()
    pass  # motor off


# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA HELPERS — *** ALL FRAMES ARE RGB ***
# ─────────────────────────────────────────────────────────────────────────────
_cam = None
_cam_lock = threading.Lock()
_latest_frame = None          # always RGB, never BGR
_cam_thread_running = False


def _cam_worker():
    """Background thread that continuously grabs RGB frames."""
    global _latest_frame
    while _cam_thread_running:
        with _cam_lock:
            if _cam is None:
                time.sleep(0.1)
                continue
            try:
                if USE_PICAMERA:
                    # Some PiCamera setups still deliver BGR even in RGB888 mode.
                    # Normalize once here so downstream always sees RGB.
                    f = _cam.capture_array()
                    if (
                        f is not None
                        and len(f.shape) == 3
                        and f.shape[2] == 3
                        and PICAMERA_OUTPUT_IS_BGR
                    ):
                        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                else:
                    # OpenCV webcam gives BGR → convert to RGB
                    ok, bgr = _cam.read()
                    if ok:
                        f = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    else:
                        f = None
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
    """Returns the latest RGB frame (never BGR)."""
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
#  STRIP DETECTION — WHITE-MASK on RED BACKGROUND (colour-based, RGB input)
# ─────────────────────────────────────────────────────────────────────────────
MAX_CAPTURES = 10

# Aspect ratio: width / height  (3.0 cm wide, 3.0 cm tall → 1.0)
TARGET_ASPECT    = 1.0
ASPECT_TOL       = 0.25      # accepts 0.75 … 1.25 (selective for square shape)

MIN_FILL = 0.10    # raised from 0.015 — 3cm x 3cm should fill at least 10% of frame
MAX_FILL = 0.95    # card can fill >85% at 3-4 cm distance

MIN_RECT_SCORE       = 0.65  # ensure it's fairly rectangular
CONFIDENCE_THRESHOLD = 0.30
CAPTURE_TRIGGER_SCORE = 0.85
TRIGGER_FRAMES_REQUIRED = 1
VERIFY_WAIT_SEC      = 1.0
SETTLE_TIME          = 2    # wait 1.5s after motor stop before capture
COOLDOWN_SEC         = 0.5    # short pause after capture before resuming motor
POST_CAPTURE_LOCKOUT_SEC = 3.0
CENTER_MARGIN_FRAC   = 0.16

# Shared detection bbox for MJPEG overlay (written by scan loop, read by stream)
_det_bbox  = None
_det_lock  = threading.Lock()


def find_component(frame_rgb):
    """
    Detect the strip card by finding the NON-RED region on a red background.

    Instead of detecting white (which fragments because the strip has colored
    biomarker circles), we detect the red platform and invert the mask.
    The largest non-red blob is the full strip card.

    Input : RGB frame
    Output: (bbox, score, contour) or (None, 0, None)
    """
    global _det_bbox
    fh, fw = frame_rgb.shape[:2]
    small = cv2.resize(frame_rgb, (fw // 2, fh // 2))
    sh, sw = small.shape[:2]

    # RGB → HSV
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)

    # ── Step 1: Detect the RED background ──
    # Red hue wraps around 0/180 in OpenCV HSV
    red_lo1 = cv2.inRange(hsv, np.array([0,   50, 50]),  np.array([10,  255, 255]))
    red_hi1 = cv2.inRange(hsv, np.array([160, 50, 50]),  np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_lo1, red_hi1)

    # ── Step 2: Invert → everything that is NOT red = candidate strip ──
    not_red = cv2.bitwise_not(red_mask)

    # ── Step 3: Also add a white/bright mask to catch overexposed strip areas ──
    white_mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 80, 255]))
    # Combine: strip = (not red) OR (white/bright)
    strip_mask = cv2.bitwise_or(not_red, white_mask)

    # ── Step 4: Clean up with aggressive morphology ──
    # Large kernel fills gaps from channels/circles within the strip
    k_big = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    strip_mask = cv2.morphologyEx(strip_mask, cv2.MORPH_CLOSE, k_big, iterations=4)
    # Erode slightly to remove edge noise from the red background
    k_sm = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    strip_mask = cv2.morphologyEx(strip_mask, cv2.MORPH_OPEN, k_sm, iterations=2)

    # ── Step 5: Find contours ──
    contours, _ = cv2.findContours(strip_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        with _det_lock:
            _det_bbox = None
        return None, 0.0, None

    # Pick the largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < (sh * sw * MIN_FILL):
        with _det_lock:
            _det_bbox = None
        return None, 0.0, None

    # Use convex hull to fill any remaining concavities (biomarker circles)
    hull = cv2.convexHull(largest)
    x, y, bw, bh = cv2.boundingRect(hull)

    # ── VALIDATION: Ensure it's roughly 3cm x 3cm (square) ──
    aspect = bw / bh
    if abs(aspect - TARGET_ASPECT) > ASPECT_TOL:
        with _det_lock:
            _det_bbox = None
        return None, 0.0, None

    # Ensure it's rectangular enough (not a circle or random blob)
    rect_fill = area / (bw * bh)
    if rect_fill < MIN_RECT_SCORE:
        with _det_lock:
            _det_bbox = None
        return None, 0.0, None

    # Small padding (5%) to ensure the full edge is captured
    pad_x = int(bw * 0.05)
    pad_y = int(bh * 0.05)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    bw = min(sw - x, bw + 2 * pad_x)
    bh = min(sh - y, bh + 2 * pad_y)

    score = 0.85  # Confidence increased for square-validated detection

    # Scale back to full resolution
    bbox = (x * 2, y * 2, bw * 2, bh * 2)
    contour_full = hull * 2

    # Share with MJPEG stream for live overlay
    with _det_lock:
        _det_bbox = bbox

    print(f"  [DET] bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}) "
          f"fill={area/(sh*sw):.3f} asp={aspect:.2f} score={score:.2f}")

    return bbox, score, contour_full

def crop_and_align(frame_rgb, contour):
    """
    Perspective-correct and tightly crop the detected strip.

    Uses cv2.minAreaRect → getPerspectiveTransform to straighten
    the strip edges, then crops to just the object.

    Input : RGB frame, contour (at full resolution)
    Output: cropped RGB numpy array (just the strip, no background)
    """
    if contour is None or len(contour) < 4:
        return None

    # Get the minimum-area rotated rectangle
    rect = cv2.minAreaRect(contour)
    box_pts = cv2.boxPoints(rect).astype(np.float32)

    # Order points: top-left, top-right, bottom-right, bottom-left
    box_pts = _order_points(box_pts)

    # Compute output dimensions (preserve the strip's actual w/h ratio)
    w = int(max(
        np.linalg.norm(box_pts[0] - box_pts[1]),
        np.linalg.norm(box_pts[2] - box_pts[3])
    ))
    h = int(max(
        np.linalg.norm(box_pts[0] - box_pts[3]),
        np.linalg.norm(box_pts[1] - box_pts[2])
    ))

    if w < 20 or h < 20:
        return None

    # Ensure width < height (strip is taller than wide: 3.0 cm tall, 2.8 cm wide)
    if w > h:
        w, h = h, w
        # Rotate points to match
        box_pts = np.array([box_pts[1], box_pts[2], box_pts[3], box_pts[0]], dtype=np.float32)

    dst_pts = np.array([
        [0,     0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0,     h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(box_pts, dst_pts)
    warped = cv2.warpPerspective(frame_rgb, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    return warped


def _order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    # Sort by Y first to get top pair and bottom pair
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top_pair = sorted_by_y[:2]
    bot_pair = sorted_by_y[2:]

    # Within each pair, sort by X
    top_pair = top_pair[np.argsort(top_pair[:, 0])]
    bot_pair = bot_pair[np.argsort(bot_pair[:, 0])]

    return np.array([top_pair[0], top_pair[1], bot_pair[1], bot_pair[0]], dtype=np.float32)


def save_rgb_image(rgb_array, path):
    """Save an RGB numpy array as a JPEG using PIL — guaranteed RGB output."""
    img = Image.fromarray(rgb_array.astype(np.uint8), 'RGB')
    img.save(path, 'JPEG', quality=95)
    pass  # saved


def smooth_bbox(prev, curr, alpha=0.50):
    """Exponential smoothing to suppress bbox jitter."""
    if prev is None:
        return curr
    return tuple(int(alpha*c + (1-alpha)*p) for p, c in zip(prev, curr))


def is_center_aligned(bbox, frame_shape):
    """Require bbox center to stay near frame center."""
    if bbox is None:
        return False
    x, y, bw, bh = [int(v) for v in bbox]
    fh, fw = frame_shape[:2]
    if bw <= 0 or bh <= 0 or fw <= 0 or fh <= 0:
        return False

    cx = x + bw / 2.0
    cy = y + bh / 2.0
    min_x = fw * CENTER_MARGIN_FRAC
    max_x = fw * (1.0 - CENTER_MARGIN_FRAC)
    min_y = fh * CENTER_MARGIN_FRAC
    max_y = fh * (1.0 - CENTER_MARGIN_FRAC)
    return (min_x <= cx <= max_x) and (min_y <= cy <= max_y)


# ─────────────────────────────────────────────────────────────────────────────
#  SCAN STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────
scan_state = {
    "running":          False,
    "motor":            "STOPPED",   # "SPINNING" | "STOPPED"
    "camera":           "OFF",       # "LIVE" | "OFF"
    "phase":            "IDLE",      # "IDLE" | "SCANNING" | "CAPTURING" | etc.
    "captures":         0,
    "last_score":       0.0,
    "last_capture_path": None,
    "last_capture_url":  None,
    "error":            None,
    "confirm_count":    0,
}
_scan_lock = threading.Lock()
_scan_thread = None
session_captures = []   # list of dicts: {name, annotated_name}


def _scan_loop():
    """
    Background thread: SIMPLIFIED motor + detection loop.

    1. Motor rotates slowly (constant speed)
    2. Detect white strip via colour mask
    3. When confirmed (3 frames) → STOP motor
    4. Wait 1.5 seconds for settle
    5. Grab fresh frame → crop + align → save as RGB
    6. Resume motor
    7. Wait for strip to leave frame before re-arming
    """
    global scan_state, _cam

    with _scan_lock:
        scan_state["phase"] = "SCANNING"
        scan_state["motor"] = "SPINNING"

    motor_go()
    trigger_count  = 0
    smooth_box     = None
    last_contour   = None
    lockout_until = 0.0

    while True:
        with _scan_lock:
            if not scan_state["running"]:
                break
            total_captures = scan_state["captures"]

        # ── Session limit ────────────────────────────────────────────────────
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

        bbox, score, contour = find_component(frame)
        now = time.time()

        with _scan_lock:
            scan_state["last_score"] = round(score, 3)

        # ── Post-capture lockout (slow disk) ─────────────────────────────────
        if now < lockout_until:
            with _scan_lock:
                scan_state["phase"] = "LOCKOUT"
                scan_state["last_score"] = 0.0
            time.sleep(0.02)
            continue

        # ── Simple detection rule: Immediate capture on high score ────────────────
        if bbox is not None and score >= CAPTURE_TRIGGER_SCORE and is_center_aligned(bbox, frame.shape):
            motor_stop()
            with _scan_lock:
                scan_state["phase"] = "CAPTURING"
                scan_state["motor"] = "STOPPED"
                scan_state["last_score"] = round(score, 3)

            print(f"[CAPTURE] Score {score:.2f} >= {CAPTURE_TRIGGER_SCORE:.2f}. Stopping and settling...")
            time.sleep(SETTLE_TIME)

            # Grab fresh high-quality frame after motor settle
            final_frame = grab_frame()
            if final_frame is None: final_frame = frame
            
            # Re-detect to get final accurate bbox for cropping
            final_bbox, final_score, final_contour = find_component(final_frame)
            if final_bbox is None:
                final_bbox = bbox
                final_score = score
            
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            photo_name = f"smd_{timestamp}.jpg"
            photo_path = os.path.join(OUTPUT_DIR, photo_name)
            
            # ── CROP TO BBOX ──
            bx, by, bw, bh = [int(v) for v in final_bbox]
            # Add small 5% buffer to crop
            ih, iw = final_frame.shape[:2]
            buff_w, buff_h = int(bw * 0.05), int(bh * 0.05)
            x1, y1 = max(0, bx - buff_w), max(0, by - buff_h)
            x2, y2 = min(iw, bx + bw + buff_w), min(ih, by + bh + buff_h)
            
            cropped_raw = final_frame[y1:y2, x1:x2]
            save_rgb_image(cropped_raw, photo_path)

            # Create annotated version on the cropped frame
            annotated_name = photo_name.replace('.jpg', '_annotated.jpg')
            annotated_path = os.path.join(OUTPUT_DIR, annotated_name)
            
            ann = cropped_raw.copy()
            # Draw bbox (relative to the crop)
            # Since it's cropped TO the bbox, we just draw at the edge
            cv2.rectangle(ann, (buff_w, buff_h), (buff_w + bw, buff_h + bh), (0, 220, 80), 3)
            
            FONT_ANN = cv2.FONT_HERSHEY_SIMPLEX
            lbl = f"Score:{final_score:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, FONT_ANN, 0.7, 2)
            cv2.rectangle(ann, (buff_w, buff_h - th - 5), (buff_w + tw + 5, buff_h), (0, 220, 80), -1)
            cv2.putText(ann, lbl, (buff_w + 2, buff_h - 5),
                        FONT_ANN, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
            
            save_rgb_image(ann, annotated_path)

            with _scan_lock:
                scan_state["captures"]          += 1
                scan_state["last_capture_path"]  = photo_path
                scan_state["last_capture_url"]   = f"/capture-image/{photo_name}"
                total_captures                   = scan_state["captures"]
                session_captures.append({
                    "name":           photo_name,
                    "annotated_name": annotated_name,
                })

            print(f"[✓] Saved cropped capture: {photo_name}")

            time.sleep(COOLDOWN_SEC)
            motor_go()
            lockout_until = time.time() + POST_CAPTURE_LOCKOUT_SEC
            trigger_count = 0
            smooth_box    = None
            last_contour  = None
            with _scan_lock:
                scan_state["confirm_count"] = 0
                scan_state["last_score"]    = 0.0
                scan_state["phase"]         = "LOCKOUT"
                scan_state["motor"]         = "SPINNING"
        else:
            trigger_count = 0
            with _scan_lock:
                scan_state["phase"] = "SEARCHING"
                scan_state["confirm_count"] = 0

        time.sleep(0.02)

    # Cleanup on exit
    motor_stop()
    with _scan_lock:
        scan_state["phase"] = "IDLE"
        scan_state["motor"] = "STOPPED"
        scan_state["confirm_count"] = 0
    pass  # scan loop exited


# ─────────────────────────────────────────────────────────────────────────────
#  MJPEG STREAM GENERATOR  — frames are RGB, encode needs BGR
# ─────────────────────────────────────────────────────────────────────────────
def _gen_mjpeg():
    """Yields MJPEG frames for the /stream endpoint."""
    while True:
        frame = grab_frame()
        if frame is None:
            # Send a placeholder grey frame
            placeholder = np.full((240, 320, 3), 40, dtype=np.uint8)
            cv2.putText(placeholder, "Camera not available",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            _, buf = cv2.imencode('.jpg', placeholder)
        else:
            # Draw detection overlay
            with _scan_lock:
                phase = scan_state["phase"]
                score = scan_state["last_score"]
            display = cv2.resize(frame, (640, 480))
            # Convert RGB → BGR for the overlay text colours and JPEG encoding
            display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

            # Draw detection bounding box (scaled from full res to 640x480)
            with _det_lock:
                det_box = _det_bbox
            if det_box is not None:
                fh, fw = frame.shape[:2]
                sx = 640.0 / fw
                sy = 480.0 / fh
                bx = int(det_box[0] * sx)
                by = int(det_box[1] * sy)
                bw = int(det_box[2] * sx)
                bh = int(det_box[3] * sy)
                box_color = (0, 220, 80) if phase == "CAPTURING" else (0, 255, 0)
                cv2.rectangle(display_bgr, (bx, by), (bx + bw, by + bh), box_color, 2)

            color = (0, 220, 80) if phase == "CAPTURING" else (0, 165, 255)
            label = f"{phase}  score:{score:.2f}"
            cv2.putText(display_bgr, label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            _, buf = cv2.imencode('.jpg', display_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.05)   # ~20 fps cap


# ─────────────────────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
from strip_analysis_simple import SimpleSalivaStripAnalyzer

app = Flask(__name__, static_folder='.')
CORS(app)

# ── Load Edge Impulse model once ──
analyzer = SimpleSalivaStripAnalyzer(
    model_path='modalv2.eim',
    confidence=0.5
)


# ── Serve dashboard ──
@app.route('/')
def index():
    return send_from_directory('.', 'saliva_dashboard.html')


# ── Serve captured images ──
@app.route('/capture-image/<filename>')
def serve_capture(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# ─────────────────────────────────────────────────────────────────────────────
#  EXISTING ANALYSIS ENDPOINT (unchanged)
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
                results = analyzer.analyze(tmp_path)
                if results is None:
                    continue
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
    """
    1. Open camera (if not already open)
    2. Spin motor for 0.5 s then stop
    3. Return OK or error
    """
    global _cam

    errors = []

    # Camera
    cam_ok = False
    try:
        if _cam is None:
            _cam = open_camera()   # blocking setup
        cam_ok = (_cam is not None)
    except Exception:
        pass

    if not cam_ok:
        errors.append("Failed to open Camera")

    with _scan_lock:
        scan_state["camera"] = "LIVE" if cam_ok else "OFF"

    # Motor calibration spin
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
    """Start the strip detection loop in a background thread."""
    global _scan_thread, _cam

    with _scan_lock:
        if scan_state["running"]:
            return jsonify({"success": False, "error": "Already scanning"}), 400

    # Ensure camera is open
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
        session_captures = []   # reset to empty list of dicts

    _scan_thread = threading.Thread(target=_scan_loop, daemon=True)
    _scan_thread.start()

    return jsonify({"success": True, "message": "Scan started"})


@app.route('/stop-scan', methods=['POST'])
def stop_scan():
    """Stop the strip detection loop."""
    with _scan_lock:
        scan_state["running"] = False
        scan_state["phase"]   = "IDLE"
    motor_stop()
    return jsonify({"success": True, "message": "Scan stopped"})


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Cleanly stop everything and terminate the Flask process."""
    def _do_exit():
        time.sleep(0.4)   # allow the HTTP response to be sent first
        motor_stop()
        release_camera()
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
        files = list(reversed(session_captures))  # Newest first
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
    filenames = data.get('filenames', [])        # plain names
    ann_map   = data.get('annotated_map', {})   # plain_name -> annotated_name
    if not filenames:
        return jsonify({'error': 'No filenames provided'}), 400

    all_results = []
    for filename in filenames:
        path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
        if os.path.exists(path):
            try:
                # Determine annotated path (already has the green box)
                ann_name = ann_map.get(filename, filename.replace('.jpg', '_annotated.jpg'))
                ann_path = os.path.join(OUTPUT_DIR, ann_name)
                
                # Analyze raw image and DRAW on the annotated one
                results = analyzer.analyze(path, annotated_path=ann_path)
                
                if results is not None:
                    response = _format_results(filename, results)
                    if os.path.exists(ann_path):
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
    filename         = data.get('filename', '')
    annotated_name   = data.get('annotated_name', '')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        return jsonify({'error': f'File not found: {filename}'}), 404

    try:
        ann_name = annotated_name or filename.replace('.jpg', '_annotated.jpg')
        ann_path = os.path.join(OUTPUT_DIR, os.path.basename(ann_name))
        
        # Analyze raw image and DRAW on the annotated one
        results = analyzer.analyze(path, annotated_path=ann_path)
        
        if results is None:
            return jsonify({'error': 'Could not analyze image'}), 500
        
        response = _format_results(filename, results)
        if os.path.exists(ann_path):
            response['annotated_url'] = f'/capture-image/{os.path.basename(ann_name)}'
        
        return jsonify({'success': True, 'results': [response]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Health check ──
@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model':  'loaded',
        'gpio':   _gpio_ok,
        'camera': USE_PICAMERA,
        'captures_dir': OUTPUT_DIR,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED RESULT FORMATTER
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
    print(f"[APP] OralScan running → http://localhost:5000  |  GPIO={_gpio_ok}  camera={USE_PICAMERA}")
    app.run(debug=False, port=5000, threaded=True)
