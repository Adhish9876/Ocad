#!/usr/bin/env python3
"""
SMD Component Detector v5 — WHITE-STRIP DETECTION (BGR native)
===============================================================
Complete rewrite:
  1. ALL frames are BGR — OpenCV native format throughout
  2. White-mask colour detection (HSV) for white strip on red background
  3. Perspective correction via minAreaRect + getPerspectiveTransform
  4. Simplified motor: single slow speed, clean stop/capture/resume
  5. PIL used for image saving — converts BGR→RGB before save
  6. cv2.imshow uses BGR directly — no conversion needed

Target   : ~3.0 cm (H) × 2.8 cm (W) white microfluidic strip
Strategy : Colour-based (white mask on red background) + perspective correction

10-image session → motor stops.
Motor rotates slowly → detects strip → stops → waits 1.5s → captures → resumes.

WIRING (BCM):  STEP → GPIO 17   DIR → GPIO 27
"""

import cv2
import numpy as np
import time
import os
import threading
import subprocess
from datetime import datetime
from PIL import Image

try:
    from gpiozero import OutputDevice
    MOTOR_AVAILABLE = True
except (ImportError, Exception):
    MOTOR_AVAILABLE = False
    print("[WARN] gpiozero not available — motor disabled (test mode)")

# ─────────────────────────────────────────────────────────────────────────────
#  MOTOR — SIMPLIFIED: single slow speed
# ─────────────────────────────────────────────────────────────────────────────
if MOTOR_AVAILABLE:
    STEP_PIN = OutputDevice(17)
    DIR_PIN  = OutputDevice(27)
    DIR_PIN.on()

MOTOR_DELAY = 0.008   # single slow speed

_run    = False
_lock   = threading.Lock()
_thread = None


def _loop():
    while True:
        with _lock:
            r = _run
        if not r:
            time.sleep(0.02)
            continue
        if MOTOR_AVAILABLE:
            STEP_PIN.on();  time.sleep(MOTOR_DELAY)
            STEP_PIN.off(); time.sleep(MOTOR_DELAY)
        else:
            time.sleep(MOTOR_DELAY * 2)


def _start():
    global _thread
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=_loop, daemon=True)
        _thread.start()


def motor_go():
    global _run
    _start()
    with _lock:
        _run = True
    print(f"[MOTOR] ON  (delay={MOTOR_DELAY*1000:.0f}ms/half-step)")


def motor_stop():
    global _run
    with _lock:
        _run = False
    print("[MOTOR] OFF")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = os.path.expanduser("~/smd_captures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_CAPTURES = 10

# Aspect ratio: width / height of the card
# 2.8 cm wide ÷ 3.0 cm tall = 0.933
TARGET_ASPECT = 2.8 / 3.0   # 0.933
ASPECT_TOL    = 0.40         # accepts 0.53 … 1.33  (generous for tilt/perspective)

MIN_FILL = 0.015   # ~1.5%
MAX_FILL = 0.85

MIN_RECT_SCORE = 0.50

CONFIRM_FRAMES       = 3
SMOOTH_ALPHA         = 0.50
CONFIDENCE_THRESHOLD = 0.30

SETTLE_TIME   = 1.5    # wait 1.5s after motor stop before capture
COOLDOWN_SEC  = 0.5

# Show debug mask window so you can tune while live
SHOW_DEBUG_MASK = True   # set False in production

# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA — ALL FRAMES ARE BGR (OpenCV native)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
except ImportError:
    USE_PICAMERA = False
    print("[WARN] picamera2 not found — webcam fallback")

PREVIEW_W, PREVIEW_H = 3280, 2464   # Pi HQ cam full sensor
# For Pi Camera Module 3 use: PREVIEW_W, PREVIEW_H = 4608, 2592


def release_stuck():
    for p in ["libcamera", "rpicam", "picamera2", "picamera"]:
        subprocess.run(["sudo", "pkill", "-f", p], capture_output=True)
    time.sleep(0.8)


def open_camera():
    if USE_PICAMERA:
        release_stuck()
        cam = Picamera2()
        cfg = cam.create_video_configuration(
            main={"size": (PREVIEW_W, PREVIEW_H), "format": "BGR888"},
            controls={"FrameRate": 15}
        )
        cam.configure(cfg)
        cam.start()
        time.sleep(2.0)   # longer warmup for AEC to settle
        return cam
    # Webcam fallback
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_AUTOFOCUS,    1)
    return cam


def grab(cam):
    """Returns a BGR frame (always), or None on failure."""
    if USE_PICAMERA:
        arr = cam.capture_array()   # arrives as BGR888 — already OpenCV native
        return arr
    # OpenCV webcam gives BGR natively
    ok, bgr = cam.read()
    if ok:
        return bgr
    return None


def save_bgr_image(bgr_array, path):
    """Save a BGR numpy array as a JPEG using PIL — converts BGR→RGB before save."""
    rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb.astype(np.uint8), 'RGB')
    img.save(path, 'JPEG', quality=95)
    print(f"  [SAVE] image → {path}")


def release_camera(cam):
    if USE_PICAMERA:
        cam.stop()
    else:
        cam.release()


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTION  — WHITE MASK on RED BACKGROUND (BGR input)
# ─────────────────────────────────────────────────────────────────────────────

_last_mask = None   # global for debug display


def find_component(frame_bgr):
    """
    Detect the white strip on a red background using colour masking.

    Input : BGR frame (numpy array)
    Output: (bbox, score, contour) or (None, 0, None)

    Pipeline:
    1. Downscale 2× for speed
    2. BGR → HSV
    3. White mask: low Saturation (0-60), high Value (160-255)
    4. Morphological close + dilate to fill holes
    5. Find contours → filter by fill, aspect ratio, rectangularity
    6. Return best match with its contour (needed for perspective crop)
    """
    global _last_mask

    fh, fw = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (fw // 2, fh // 2))
    sh, sw = small.shape[:2]
    s_area = sh * sw

    # BGR → HSV
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # White mask: any hue, low saturation, high value
    lower_white = np.array([0,   0, 160], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Clean up the mask
    k_close  = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE,  k_close,  iterations=3)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_DILATE, k_dilate, iterations=1)

    _last_mask = white_mask

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
            best_bbox    = (x * 2, y * 2, bw * 2, bh * 2)
            best_contour = cnt * 2   # scale back to full resolution

    return best_bbox, best_score, best_contour


def crop_and_align(frame_bgr, contour):
    """
    Perspective-correct and tightly crop the detected strip.

    Uses cv2.minAreaRect → getPerspectiveTransform to straighten
    the strip edges, then crops to just the object.

    Input : BGR frame, contour (at full resolution)
    Output: cropped BGR numpy array (just the strip, no background)
    """
    if contour is None or len(contour) < 4:
        return None

    rect = cv2.minAreaRect(contour)
    box_pts = cv2.boxPoints(rect).astype(np.float32)

    # Order points: top-left, top-right, bottom-right, bottom-left
    box_pts = _order_points(box_pts)

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
        box_pts = np.array([box_pts[1], box_pts[2], box_pts[3], box_pts[0]], dtype=np.float32)

    dst_pts = np.array([
        [0,     0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0,     h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(box_pts, dst_pts)
    warped = cv2.warpPerspective(frame_bgr, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    return warped


def _order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top_pair = sorted_by_y[:2]
    bot_pair = sorted_by_y[2:]

    top_pair = top_pair[np.argsort(top_pair[:, 0])]
    bot_pair = bot_pair[np.argsort(bot_pair[:, 0])]

    return np.array([top_pair[0], top_pair[1], bot_pair[1], bot_pair[0]], dtype=np.float32)


def smooth_bbox(prev, curr, alpha=SMOOTH_ALPHA):
    if prev is None:
        return curr
    return tuple(int(alpha * c + (1 - alpha) * p) for p, c in zip(prev, curr))


# ─────────────────────────────────────────────────────────────────────────────
#  DRAW — colours in BGR for cv2.imshow display
# ─────────────────────────────────────────────────────────────────────────────
FONT   = cv2.FONT_HERSHEY_SIMPLEX
# These are BGR for cv2.imshow
GREEN  = (0, 220, 80)
ORANGE = (0, 165, 255)
CYAN   = (255, 200, 0)
RED    = (60,  60, 255)
WHITE  = (240, 240, 240)
DARK   = (20,  20,  20)


def draw_hud(frame_bgr, state, score, confirm, total):
    """Draw HUD overlay on a BGR display frame."""
    h, w = frame_bgr.shape[:2]
    ov = frame_bgr.copy()
    cv2.rectangle(ov, (0, 0), (w, 60), DARK, -1)
    cv2.addWeighted(ov, 0.55, frame_bgr, 0.45, 0, frame_bgr)
    tbl = {
        "SEARCHING":  (ORANGE, f"SEARCHING  score:{score:.2f}  [{total}/{MAX_CAPTURES}]"),
        "LOCKED":     (GREEN,  f"LOCKED  score:{score:.2f}  confirm:{confirm}/{CONFIRM_FRAMES}"),
        "CAPTURING":  (WHITE,  "CAPTURING — motor STOPPED"),
        "WAIT_GONE":  (CYAN,   f"Waiting for object to clear … [{total}/{MAX_CAPTURES}]"),
        "DONE":       (RED,    f"DONE — {total}/{MAX_CAPTURES} images saved"),
    }
    col, txt = tbl.get(state, (WHITE, state))
    cv2.putText(frame_bgr, txt, (16, 40), FONT, 0.80, col, 2, cv2.LINE_AA)
    ov2 = frame_bgr.copy()
    cv2.rectangle(ov2, (0, h - 46), (w, h), DARK, -1)
    cv2.addWeighted(ov2, 0.55, frame_bgr, 0.45, 0, frame_bgr)
    cv2.putText(frame_bgr,
        f"Captures:{total}/{MAX_CAPTURES}  "
        f"MIN_FILL:{MIN_FILL}  ASP:{TARGET_ASPECT:.2f}±{ASPECT_TOL}  Q:quit",
        (16, h - 14), FONT, 0.55, WHITE, 1, cv2.LINE_AA)


def draw_box(frame_bgr, bbox, state):
    """Draw bounding box overlay on a BGR display frame."""
    x, y, bw, bh = [int(v) for v in bbox]
    col = GREEN if state == "LOCKED" else ORANGE
    L = min(bw, bh) // 5
    for pts in [
        ((x,       y + L),    (x,    y),    (x + L,    y)),
        ((x+bw-L,  y),        (x+bw, y),    (x+bw,     y + L)),
        ((x+bw,    y+bh-L),   (x+bw, y+bh), (x+bw-L,   y+bh)),
        ((x+L,     y+bh),     (x,    y+bh), (x,        y+bh-L)),
    ]:
        cv2.polylines(frame_bgr, [np.array(pts, np.int32)], False, col, 3, cv2.LINE_AA)
    lbl = f"3.0x2.8cm  asp:{bw/bh:.2f}"
    (tw, th), _ = cv2.getTextSize(lbl, FONT, 0.55, 2)
    cv2.rectangle(frame_bgr, (x, y - th - 14), (x + tw + 10, y), col, -1)
    cv2.putText(frame_bgr, lbl, (x + 5, y - 6), FONT, 0.55, DARK, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print(f"  Strip Detector v5 — BGR native | {MAX_CAPTURES}-image session")
    print(f"  MIN_FILL={MIN_FILL}  ASP={TARGET_ASPECT:.3f}±{ASPECT_TOL}")
    print(f"  CONF_THRESH={CONFIDENCE_THRESHOLD}  WHITE MASK: ON")
    print(f"  Motor delay: {MOTOR_DELAY*1000:.0f}ms/half-step")
    print("=" * 64)

    cam = open_camera()
    cv2.namedWindow("Strip Detector",  cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Strip Detector", 1280, 720)

    if SHOW_DEBUG_MASK:
        cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("White Mask", 640, 480)

    confirm_count  = 0
    total_captures = 0
    smooth_box     = None
    last_contour   = None
    wait_for_gone  = False

    motor_go()

    try:
        while total_captures < MAX_CAPTURES:
            frame_bgr = grab(cam)
            if frame_bgr is None:
                time.sleep(0.02)
                continue

            bbox, score, contour = find_component(frame_bgr)

            # Already BGR — display directly
            disp = frame_bgr.copy()

            # ── Show debug mask ──────────────────────────────────────────────
            if SHOW_DEBUG_MASK and _last_mask is not None:
                cv2.imshow("White Mask",
                           cv2.resize(_last_mask, (640, 480)))

            # ── Wait for object to leave after a capture ─────────────────────
            if wait_for_gone:
                if bbox is None or score < CONFIDENCE_THRESHOLD:
                    wait_for_gone = False
                    confirm_count = 0
                    smooth_box    = None
                    last_contour  = None
                    motor_go()
                    print("[INFO] Object cleared — re-armed")
                else:
                    draw_hud(disp, "WAIT_GONE", score, 0, total_captures)
                    cv2.imshow("Strip Detector", cv2.resize(disp, (1280, 720)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

            # ── Detection ────────────────────────────────────────────────────
            if bbox is not None and score >= CONFIDENCE_THRESHOLD:
                smooth_box   = smooth_bbox(smooth_box, bbox)
                last_contour = contour
                confirm_count += 1

                draw_box(disp, smooth_box, "LOCKED" if confirm_count >= CONFIRM_FRAMES - 1 else "SEARCHING")
                draw_hud(disp, "LOCKED", score, confirm_count, total_captures)

                if confirm_count >= CONFIRM_FRAMES:
                    # ════════════════════════════════════════════════════════
                    #  STOP → WAIT 1.5s → CAPTURE → RESUME
                    # ════════════════════════════════════════════════════════
                    motor_stop()
                    draw_hud(disp, "CAPTURING", score, confirm_count, total_captures)
                    cv2.imshow("Strip Detector", cv2.resize(disp, (1280, 720)))
                    cv2.waitKey(1)

                    print(f"[CAPTURE] Motor stopped. Waiting {SETTLE_TIME}s ...")
                    time.sleep(SETTLE_TIME)

                    # Grab a fresh, settled frame (BGR)
                    capture_frame = grab(cam) or frame_bgr

                    # Re-detect on the settled frame for best alignment
                    bbox2, score2, contour2 = find_component(capture_frame)
                    use_contour = contour2 if contour2 is not None else last_contour

                    # Perspective-correct crop
                    cropped = crop_and_align(capture_frame, use_contour)

                    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    path = os.path.join(OUTPUT_DIR, f"smd_{ts}.jpg")

                    if cropped is not None and cropped.size > 0:
                        save_bgr_image(cropped, path)
                    else:
                        # Fallback: bbox crop
                        x, y, bw, bh = [int(v) for v in smooth_box]
                        fh, fw = capture_frame.shape[:2]
                        px, py = max(int(bw * 0.05), 5), max(int(bh * 0.05), 5)
                        fc = capture_frame[max(0, y-py):min(fh, y+bh+py),
                                           max(0, x-px):min(fw, x+bw+px)]
                        if fc.size > 0:
                            save_bgr_image(fc, path)
                        else:
                            save_bgr_image(capture_frame, path)

                    total_captures += 1
                    print(f"[✓] Saved: {path}  ({total_captures}/{MAX_CAPTURES})")

                    # Banner (already BGR — display directly)
                    cb_bgr = capture_frame.copy()
                    cv2.putText(cb_bgr,
                        f"CAPTURED!  {total_captures}/{MAX_CAPTURES}",
                        (cb_bgr.shape[1] // 2 - 220, cb_bgr.shape[0] // 2),
                        FONT, 2.5, GREEN, 5, cv2.LINE_AA)
                    cv2.imshow("Strip Detector", cv2.resize(cb_bgr, (1280, 720)))
                    cv2.waitKey(1200)

                    if total_captures >= MAX_CAPTURES:
                        break

                    motor_go()
                    wait_for_gone = True
                    confirm_count = 0
                    smooth_box    = None
                    last_contour  = None

                    cl_end = time.time() + COOLDOWN_SEC
                    while time.time() < cl_end:
                        f = grab(cam)
                        if f is not None:
                            draw_hud(f, "WAIT_GONE", 0, 0, total_captures)
                            cv2.imshow("Strip Detector", cv2.resize(f, (1280, 720)))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            total_captures = MAX_CAPTURES
                            break
                    continue

            else:
                confirm_count = max(0, confirm_count - 1)
                if confirm_count == 0:
                    smooth_box   = None
                    last_contour = None
                draw_hud(disp, "SEARCHING", score, 0, total_captures)

            cv2.imshow("Strip Detector", cv2.resize(disp, (1280, 720)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ── Done ─────────────────────────────────────────────────────────────
        motor_stop()
        print(f"\n  SESSION DONE — {total_captures} images in {OUTPUT_DIR}\n")
        deadline = time.time() + 5.0
        while time.time() < deadline:
            f = grab(cam)
            if f is not None:
                draw_hud(f, "DONE", 0, 0, total_captures)
                cv2.imshow("Strip Detector", cv2.resize(f, (1280, 720)))
                if SHOW_DEBUG_MASK and _last_mask is not None:
                    cv2.imshow("White Mask",
                               cv2.resize(_last_mask, (640, 480)))
            if cv2.waitKey(200) & 0xFF == ord('q'):
                break

    finally:
        motor_stop()
        release_camera(cam)
        cv2.destroyAllWindows()
        print("[GPIO] Done.")


if __name__ == "__main__":
    main()
