"""
Flask Backend for OralScan Dashboard
=====================================
Connects strip_analysis.py to the HTML dashboard.
Also drives PiCamera2 + ULN2003 stepper motor for Raspberry Pi 5 deployment.

*** ALL FRAMES ARE RGB — NO BGR ANYWHERE IN THE PIPELINE ***

Install dependencies (Pi):
    pip install flask flask-cors picamera2 RPi.GPIO opencv-python-headless numpy ultralytics Pillow

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
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
#  STEPPER MOTOR CONFIG
# ─────────────────────────────────────────────────────────────────────────────
USE_STEPPER   = True
MOTOR_PINS    = [17, 18, 27, 22]   # IN1..IN4 → BCM GPIO
MOTOR_DELAY   = 0.008              # single slow speed (seconds per half-step)

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
    print(f"[MOTOR] ON  (delay={MOTOR_DELAY*1000:.0f}ms/half-step)")


def motor_stop():
    global _motor_running
    with _motor_lock:
        _motor_running = False
    # De-energise all coils immediately
    if _gpio_ok:
        for dev in _motor_devices:
            dev.off()
    print("[MOTOR] OFF")


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
                    # PiCamera2 with RGB888 → already RGB, NO conversion
                    f = _cam.capture_array()
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

# Aspect ratio: width / height  (2.8 cm wide, 3.0 cm tall → 0.933)
TARGET_ASPECT    = 2.8 / 3.0
ASPECT_TOL       = 0.40      # accepts 0.53 … 1.33 (generous for tilt/perspective)

MIN_FILL = 0.015   # ~1.5%
MAX_FILL = 0.85

MIN_RECT_SCORE       = 0.50
CONFIDENCE_THRESHOLD = 0.30
CONFIRM_FRAMES       = 3
SETTLE_TIME          = 1.5    # wait 1.5s after motor stop before capture
COOLDOWN_SEC         = 0.5    # short pause after capture before resuming motor


def find_component(frame_rgb):
    """
    Detect the white strip on a red background using colour masking.

    Input : RGB frame (numpy array)
    Output: (bbox, score, contour) or (None, 0, None)

    Pipeline:
    1. Downscale 2× for speed
    2. RGB → HSV
    3. White mask: low Saturation (0-60), high Value (160-255)
    4. Morphological close + dilate to fill holes
    5. Find contours → filter by fill, aspect ratio, rectangularity
    6. Return best match with its contour (needed for perspective crop)
    """
    fh, fw = frame_rgb.shape[:2]
    small = cv2.resize(frame_rgb, (fw // 2, fh // 2))
    sh, sw = small.shape[:2]
    s_area = sh * sw

    # RGB → HSV (OpenCV's cvtColor with COLOR_RGB2HSV works on RGB input)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)

    # White mask: any hue, low saturation, high value
    # White objects have S < 60 and V > 160 typically
    lower_white = np.array([0,   0, 160], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Clean up the mask
    k_close  = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE,  k_close,  iterations=3)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_DILATE, k_dilate, iterations=1)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best_bbox, best_score, best_contour = None, 0.0, None

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
        if rect_fill < MIN_RECT_SCORE:
            continue

        asp_penalty = 1.0 - abs(aspect - TARGET_ASPECT) / ASPECT_TOL
        score = rect_fill * asp_penalty

        print(f"  [DET] fill={fill:.4f} asp={aspect:.3f} "
              f"rect={rect_fill:.2f} score={score:.3f}")

        if score > best_score:
            best_score   = score
            best_bbox    = (x * 2, y * 2, bw * 2, bh * 2)   # scale back to full res
            # Scale contour back to full resolution
            best_contour = cnt * 2

    return best_bbox, best_score, best_contour


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
    print(f"  [SAVE] RGB image → {path}")


def smooth_bbox(prev, curr, alpha=0.50):
    """Exponential smoothing to suppress bbox jitter."""
    if prev is None:
        return curr
    return tuple(int(alpha*c + (1-alpha)*p) for p, c in zip(prev, curr))


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
    confirm_count  = 0
    smooth_box     = None
    last_contour   = None
    wait_for_gone  = False

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

        with _scan_lock:
            scan_state["last_score"] = round(score, 3)

        # ── Wait for object to leave after a capture ─────────────────────────
        if wait_for_gone:
            if bbox is None or score < CONFIDENCE_THRESHOLD:
                wait_for_gone = False
                confirm_count = 0
                smooth_box    = None
                last_contour  = None
                motor_go()
                with _scan_lock:
                    scan_state["phase"] = "SCANNING"
                    scan_state["motor"] = "SPINNING"
                print("[INFO] Object cleared — re-armed")
            else:
                with _scan_lock:
                    scan_state["phase"] = "WAIT_GONE"
            time.sleep(0.02)
            continue

        # ── Detection ────────────────────────────────────────────────────────
        if bbox is not None and score >= CONFIDENCE_THRESHOLD:
            smooth_box = smooth_bbox(smooth_box, bbox)
            last_contour = contour
            confirm_count += 1

            with _scan_lock:
                scan_state["phase"]         = "LOCKED"
                scan_state["confirm_count"] = confirm_count

            if confirm_count >= CONFIRM_FRAMES:
                # ══════════════════════════════════════════════════════════════
                #  STOP → WAIT → CAPTURE → RESUME
                # ══════════════════════════════════════════════════════════════
                motor_stop()
                with _scan_lock:
                    scan_state["phase"] = "CAPTURING"
                    scan_state["motor"] = "STOPPED"

                print(f"[CAPTURE] Motor stopped. Waiting {SETTLE_TIME}s ...")
                time.sleep(SETTLE_TIME)

                # Grab a fresh, settled frame
                frame2 = grab_frame()
                if frame2 is None:
                    frame2 = frame  # fallback

                # Re-detect on the settled frame for best alignment
                bbox2, score2, contour2 = find_component(frame2)
                if contour2 is not None:
                    use_contour = contour2
                    use_frame   = frame2
                else:
                    use_contour = last_contour
                    use_frame   = frame2

                # Perspective-correct crop
                cropped = crop_and_align(use_frame, use_contour)

                timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                photo_name = f"smd_{timestamp}.jpg"
                photo_path = os.path.join(OUTPUT_DIR, photo_name)

                if cropped is not None and cropped.size > 0:
                    # Save the clean, perspective-corrected RGB crop
                    save_rgb_image(cropped, photo_path)
                else:
                    # Fallback: save the bbox crop (still RGB)
                    x, y, bw, bh = [int(v) for v in smooth_box]
                    fh, fw = use_frame.shape[:2]
                    pad_x = int(bw * 0.05)
                    pad_y = int(bh * 0.05)
                    x1 = max(0,  x - pad_x)
                    y1 = max(0,  y - pad_y)
                    x2 = min(fw, x + bw + pad_x)
                    y2 = min(fh, y + bh + pad_y)
                    fallback_crop = use_frame[y1:y2, x1:x2]
                    if fallback_crop.size > 0:
                        save_rgb_image(fallback_crop, photo_path)
                    else:
                        save_rgb_image(use_frame, photo_path)

                # ── Annotated copy with bounding box overlay ──────────────
                annotated_name = photo_name.replace('.jpg', '_annotated.jpg')
                annotated_path = os.path.join(OUTPUT_DIR, annotated_name)
                # Load the saved crop for annotation (already RGB)
                ann_source = cropped if (cropped is not None and cropped.size > 0) else use_frame
                ann = ann_source.copy()
                ah, aw = ann.shape[:2]
                # Draw a green border to show detection
                border = 4
                cv2.rectangle(ann, (border, border), (aw - border, ah - border),
                              (0, 220, 80), 3)
                # Label
                FONT_ANN = cv2.FONT_HERSHEY_SIMPLEX
                lbl = f"3.0x2.8cm  score:{score:.2f}"
                (tw, th), _ = cv2.getTextSize(lbl, FONT_ANN, 0.55, 2)
                cv2.rectangle(ann, (border, border), (border + tw + 10, border + th + 14),
                              (0, 220, 80), -1)
                cv2.putText(ann, lbl, (border + 5, border + th + 6),
                            FONT_ANN, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
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

                print(f"[✓] {photo_path}  ({total_captures}/{MAX_CAPTURES})")

                # Resume motor and wait for object to clear
                time.sleep(COOLDOWN_SEC)
                motor_go()
                wait_for_gone = True
                confirm_count = 0
                smooth_box    = None
                last_contour  = None
                with _scan_lock:
                    scan_state["confirm_count"] = 0
                    scan_state["phase"]         = "WAIT_GONE"
                    scan_state["motor"]         = "SPINNING"

        else:
            confirm_count = max(0, confirm_count - 1)
            if confirm_count == 0:
                smooth_box   = None
                last_contour = None
            with _scan_lock:
                scan_state["phase"]         = "SEARCHING"
                scan_state["confirm_count"] = confirm_count

        time.sleep(0.02)

    # Cleanup on exit
    motor_stop()
    with _scan_lock:
        scan_state["phase"] = "IDLE"
        scan_state["motor"] = "STOPPED"
        scan_state["confirm_count"] = 0
    print("[SCAN] Loop exited.")


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

# ── Load YOLO model once ──
print("Loading YOLOv8 model...")
analyzer = SimpleSalivaStripAnalyzer(
    model_path='runs/detect/train2/weights/best.pt',
    confidence=0.5
)
print("✓ Model ready!\n")


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
    """Run YOLOv8 analysis on multiple captured images by filename."""
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
                results = analyzer.analyze(path)
                if results is not None:
                    response = _format_results(filename, results)
                    # Attach annotated image URL
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
    """Run YOLOv8 analysis on a single captured image by filename."""
    data = request.get_json(silent=True) or {}
    filename         = data.get('filename', '')
    annotated_name   = data.get('annotated_name', '')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        return jsonify({'error': f'File not found: {filename}'}), 404

    try:
        results = analyzer.analyze(path)
        if results is None:
            return jsonify({'error': 'Could not analyze image'}), 500
        response = _format_results(filename, results)
        # Attach annotated image URL when available
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
    print("=" * 52)
    print("  OralScan + Strip Scanner Backend running!")
    print("  *** ALL FRAMES ARE RGB — NO BGR ***")
    print("  Open: http://localhost:5000")
    print("  GPIO  available:", _gpio_ok)
    print("  Camera available:", USE_PICAMERA)
    print("=" * 52 + "\n")
    app.run(debug=False, port=5000, threaded=True)