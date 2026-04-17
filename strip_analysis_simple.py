"""
Simplified Saliva Strip Biomarker Analysis
==========================================
Uses fixed-position sampling of the 3 biomarker circles within the cropped
strip image — avoids contour-merging problems caused by the microfluidic
channels connecting the circles.

Calibration model:
  Dull dark red  → low concentration floor (5%)
  Dull dark blue → high concentration ceiling (95%)
  Intermediate shades interpolate smoothly between 5-95%.

Strip circle layout (relative to cropped strip image):
  ┌─────────────┐
  │    [ top ]  │  ← Cysteine    @ (row 20%, col 50%)
  │ [L]     [R] │  ← Glutathione @ (row 35%, col 20%)
  │             │    Sialic Acid  @ (row 35%, col 80%)
  │  [sample]   │
  └─────────────┘

---das
"""

import cv2
import numpy as np
import argparse
from datetime import datetime
import os
import sys

try:
    from edge_impulse_linux.image import ImageImpulseRunner
    _HAS_EI = True
except ImportError:
    _HAS_EI = False
    print("[WARN] edge_impulse_linux not installed — model detection disabled")


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION  (low-light red↔blue interpolation)
# ─────────────────────────────────────────────────────────────────────────────
# Red should represent very low concentration and blue very high concentration.
# Red-dominant shades are compressed into ~0-5% to match expected baseline.
MIN_CONC_PCT = 0.0
MAX_CONC_PCT = 95.0

# Green suppress factor improves robustness under mixed/low lighting so that
# muddy shades are still projected along the red↔blue axis.
GREEN_SUPPRESS = 0.35
RED_DOMINANCE_RATIO = 1.20

# Relative (row, col) positions of the 3 circles within the cropped strip
# Adjust if your strip layout differs
CIRCLE_POSITIONS = {
    'Cysteine':    (0.20, 0.50),   # top circle
    'Glutathione': (0.35, 0.20),   # left circle
    'Sialic Acid': (0.35, 0.80),   # right circle
}
SAMPLE_RADIUS_FRAC = 0.07   # larger window (7%) for better color stability


def blueness_to_pct(r, g, b):
    """
    Map RGB → concentration (1-95%) using red/blue dominance.
    Includes a 'Purity Check': if Red is still 'spotable' (high R value),
    we cap the result below 95%.
    """
    r, g, b = float(r), float(g), float(b)

    # Use a direct ratio of Blue to Red
    ratio = b / (r + 1.0)

    if ratio > 1.0:
        # Blue dominant logic
        # Purity: If R is low (<40), it's a 'Pure' blue. If R is high (>100), it's 'Spotable'.
        # We calculate a purity factor (1.0 for pure, 0.6 for dirty purple)
        purity = np.clip(1.0 - (max(0, r - 45) / 120.0), 0.6, 1.0)
        
        # Base value: ratio 1.1 hits the cap
        val = 50.0 + (min(1.0, (ratio - 1.0) / 0.10) * 45.0)
        
        # Apply purity to ensure 'spotable' red doesn't hit 95%
        final_val = 50.0 + (val - 50.0) * purity
        return float(np.clip(final_val, 1.0, 95.0))
    else:
        # Red dominant logic: quadratic curve for the red background
        val = 1.0 + (ratio ** 2) * 49.0
        return float(np.clip(val, 1.0, 50.0))


class SimpleSalivaStripAnalyzer:
    """
    Saliva strip analyzer.

    Primary path: fixed-position sampling of the 3 known circle locations.
    Fallback:     HSV contour detection (for images where the strip is
                  full-frame and circles can be isolated).
    """

    def __init__(self, model_path='modalv2.eim', confidence=0.5):
        self.confidence = confidence
        self.model_path = model_path
        self.runner = None

        # Try to load Edge Impulse model
        if _HAS_EI and os.path.isfile(model_path):
            try:
                self.runner = ImageImpulseRunner(model_path)
                self.runner.init()
                print(f"[OK] Edge Impulse model loaded: {model_path}")
            except Exception as e:
                print(f"[WARN] Could not load EI model ({e}) — falling back to fixed-position")
                self.runner = None
        else:
            reason = 'SDK missing' if not _HAS_EI else 'file not found'
            print(f"[WARN] EI model not loaded ({reason}) — using fixed-position sampling")

        mode = 'model + colour-based' if self.runner else 'fixed-position + colour-based'
        print(f"[OK] Analyzer initialized ({mode})")
        print(f"[OK] Confidence threshold: {self.confidence * 100:.1f}%")
        print(f"[OK] Calibration RED  -> {MIN_CONC_PCT:.0f}% (low, dull red)")
        print(f"[OK] Calibration BLUE -> {MAX_CONC_PCT:.0f}% (high, dull blue)")

    # ─────────────────────────────────────────────────────────────────────────
    #  COLOUR EXTRACTION  (core primitive)
    # ─────────────────────────────────────────────────────────────────────────

    def _sample_patch(self, image_bgr, rel_row, rel_col, radius_px):
        """
        Sample a square patch centred at (rel_row, rel_col) with radius_px.
        Returns mean (R, G, B).
        """
        ih, iw = image_bgr.shape[:2]
        cy = int(rel_row * ih)
        cx = int(rel_col * iw)
        r  = max(5, int(radius_px))
        y1, y2 = max(0, cy - r), min(ih, cy + r)
        x1, x2 = max(0, cx - r), min(iw, cx + r)
        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return (128, 128, 128)
        m = cv2.mean(roi)[:3]
        return (int(m[2]), int(m[1]), int(m[0]))   # BGR → RGB

    def extract_rgb(self, image_bgr, bbox):
        """
        Smart Extraction: Samples the provided bounding box, but actively
        ignores white/beige background pixels, only averaging the dark dye.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        ih, iw = image_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        
        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return (128, 128, 128)
            
        # Create a mask to ignore the white/beige strip background.
        # The strip is bright, the dye (whether red or blue) is dark.
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Keep pixels that are darker than 150 (ignores white/beige)
        _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        
        # If the box somehow completely missed the dye (all white)
        if cv2.countNonZero(mask) == 0:
            # Fallback to standard center crop if no dark pixels found
            dx = int((x2 - x1) * 0.20)
            dy = int((y2 - y1) * 0.20)
            roi = image_bgr[y1+dy:y2-dy, x1+dx:x2-dx]
            if roi.size == 0: return (128, 128, 128)
            m = cv2.mean(roi)[:3]
            return (int(m[2]), int(m[1]), int(m[0]))
            
        # Calculate mean using the dark-pixel mask
        m = cv2.mean(roi, mask=mask)[:3]
        return (int(m[2]), int(m[1]), int(m[0])) # Return RGB

    # ─────────────────────────────────────────────────────────────────────────
    #  EDGE IMPULSE MODEL DETECTION  (primary when model available)
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_with_model(self, image_bgr):
        """
        Run Edge Impulse object detection model on the image.
        Returns list of detection dicts with bbox, center, confidence, label.
        Note: Edge Impulse expects RGB input, so we convert BGR→RGB.
        """
        if self.runner is None:
            return []

        try:
            # Edge Impulse SDK expects RGB, but cv2.imread gives BGR
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            features, _ = self.runner.get_features_from_image(image_rgb)
            res = self.runner.classify(features)
            bboxes = res.get('result', {}).get('bounding_boxes', [])

            detections = []
            for bb in bboxes:
                if bb.get('confidence', 0) < self.confidence:
                    continue
                x, y, w, h = bb['x'], bb['y'], bb['width'], bb['height']
                cx, cy = x + w / 2, y + h / 2
                detections.append({
                    'bbox':       (x, y, x + w, y + h),
                    'center':     (cx, cy),
                    'confidence': bb['confidence'],
                    'label':      bb.get('label', 'unknown'),
                    'area':       w * h,
                })
                print(f"  [EI] Detected '{bb.get('label', '?')}' "
                      f"conf={bb['confidence']:.2f} "
                      f"@ ({cx:.0f},{cy:.0f}) size={w:.0f}x{h:.0f}")

            if not detections:
                print("  [EI] No objects detected above confidence threshold")
            return detections

        except Exception as e:
            print(f"  [EI ERROR] Model inference failed: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    #  FIXED-POSITION DETECTION  (fallback)
    # ─────────────────────────────────────────────────────────────────────────

    def _sample_fixed_positions(self, image_bgr):
        """
        Sample the 3 known circle positions directly.
        Returns a list of detection dicts compatible with the rest of the pipeline.
        """
        ih, iw = image_bgr.shape[:2]
        radius_px = min(ih, iw) * SAMPLE_RADIUS_FRAC

        detections = []
        for analyte, (ry, rx) in CIRCLE_POSITIONS.items():
            rgb = self._sample_patch(image_bgr, ry, rx, radius_px)
            r, g, b = rgb
            pct = blueness_to_pct(r, g, b)

            cy = int(ry * ih)
            cx = int(rx * iw)
            rp = int(radius_px)

            detections.append({
                'analyte':    analyte,
                'symbol':     '*',
                'bbox':       (cx - rp, cy - rp, cx + rp, cy + rp),
                'center':     (cx, cy),
                'confidence': 0.90,   # high confidence — we know exactly where circles are
                'area':       (2 * rp) ** 2,
                'rgb':        rgb,
                'concentration_pct':   pct,
                'concentration_level': int(np.clip(round(pct / 100.0 * 19), 0, 19)),
            })
            print(f"  [POS] {analyte:15} "
                  f"@ ({ry*100:.0f}%,{rx*100:.0f}%)  "
                  f"RGB=({r:3d},{g:3d},{b:3d})  "
                  f"B-R={b-r:+4d}  → {pct:.1f}%")

        return detections

    # ─────────────────────────────────────────────────────────────────────────
    #  FALLBACK CONTOUR DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    def detect_spots(self, image_bgr):
        """Contour-based detection — used only as a fallback."""
        detections = self._detect_coloured_regions(image_bgr)
        if not detections:
            print("  [WARN] Contour detection: no spots found")
        else:
            print(f"  [OK] Contour detection: {len(detections)} spot(s)")
        return sorted(detections, key=lambda d: d['center'][0])

    def _detect_circles_cv(self, image_bgr):
        """
        Multi-strategy circle detection to guarantee finding all 3 biomarker
        circles, even when they are connected by microfluidic channels.
        
        Strategy 1: HoughCircles (finds circles even when connected)
        Strategy 2: HSV blue-mask isolation (targets the dye color directly)
        Strategy 3: Heavy erosion + contour (breaks channel connections)
        """
        ih, iw = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        detections = []
        
        # ═══════════════════════════════════════════════════════════════════
        # Strategy 1: HoughCircles — best for connected shapes
        # ═══════════════════════════════════════════════════════════════════
        min_r = int(min(ih, iw) * 0.04)
        max_r = int(min(ih, iw) * 0.15)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=int(min(ih, iw) * 0.12),
            param1=80, param2=35,
            minRadius=min_r, maxRadius=max_r
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for cx, cy, r in circles[0]:
                detections.append({
                    'bbox': (int(cx-r), int(cy-r), int(cx+r), int(cy+r)),
                    'center': (float(cx), float(cy)),
                    'confidence': 0.9,
                    'area': float(np.pi * r * r),
                    'label': 'hough',
                    'radius': int(r)
                })
            print(f"  [HOUGH] Found {len(detections)} circle(s)")
        
        if len(detections) >= 3:
            return self._best_three(detections)
        
        # ═══════════════════════════════════════════════════════════════════
        # Strategy 2: HSV blue-mask — targets the dye color directly
        # ═══════════════════════════════════════════════════════════════════
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # Broad blue range to catch dark blue, purple-blue, etc.
        blue_mask = cv2.inRange(hsv, np.array([90, 40, 30]), np.array([150, 255, 255]))
        
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # Heavy erosion breaks channel connections between circles
        blue_mask = cv2.erode(blue_mask, k, iterations=3)
        blue_mask = cv2.dilate(blue_mask, k, iterations=2)
        
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (ih * iw) * 0.002
        max_area = (ih * iw) * 0.10
        
        blue_dets = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area < area < max_area):
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            blue_dets.append({
                'bbox': (x, y, x + w, y + h),
                'center': (float(cx), float(cy)),
                'confidence': 0.85,
                'area': area,
                'label': 'hsv_blue',
                'radius': int(radius)
            })
        
        if blue_dets:
            print(f"  [HSV-BLUE] Found {len(blue_dets)} blue region(s)")
            # Merge with existing detections (avoid duplicates)
            detections = self._merge_detections(detections, blue_dets)
        
        if len(detections) >= 3:
            return self._best_three(detections)
        
        # ═══════════════════════════════════════════════════════════════════
        # Strategy 3: Heavy erosion on adaptive threshold
        # ═══════════════════════════════════════════════════════════════════
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 12
        )
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # Aggressively erode to break thin channel connections
        eroded = cv2.erode(thresh, k2, iterations=4)
        eroded = cv2.dilate(eroded, k2, iterations=2)
        
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        erode_dets = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area < area < max_area):
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            if aspect > 0.4:
                erode_dets.append({
                    'bbox': (x, y, x + w, y + h),
                    'center': (float(cx), float(cy)),
                    'confidence': 0.7,
                    'area': area,
                    'label': 'eroded',
                    'radius': int(radius)
                })
        
        if erode_dets:
            print(f"  [ERODE] Found {len(erode_dets)} region(s)")
            detections = self._merge_detections(detections, erode_dets)
        
        print(f"  [TOTAL] {len(detections)} unique circle(s) detected across all strategies")
        return self._best_three(detections)
    
    def _merge_detections(self, existing, new_dets):
        """Merge new detections into existing, skipping duplicates within 40px."""
        merged = list(existing)
        for nd in new_dets:
            nx, ny = nd['center']
            is_dup = False
            for ed in merged:
                ex, ey = ed['center']
                if np.sqrt((nx-ex)**2 + (ny-ey)**2) < 40:
                    is_dup = True
                    break
            if not is_dup:
                merged.append(nd)
        return merged
    
    def _best_three(self, detections):
        """Return the 3 best detections sorted by confidence."""
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections[:3]

    def _refine_detections(self, model_dets, cv_dets):
        """
        Snap model 'hints' to the nearest precisely-located CV circle.
        If model missed one, use a remaining CV circle.
        """
        final_dets = []
        used_cv_idx = set()
        
        # 1. Try to snap each model detection to a CV circle
        for mdet in model_dets:
            mx, my = mdet['center']
            best_cv = None
            best_dist = 1000000
            best_idx = -1
            
            for i, cdet in enumerate(cv_dets):
                cx, cy = cdet['center']
                dist = np.sqrt((mx - cx)**2 + (my - cy)**2)
                if dist < best_dist and dist < 60: # Within 60px
                    best_dist = dist
                    best_cv = cdet
                    best_idx = i
            
            if best_cv:
                # Snap! Keep the model's label but use the CV box
                mdet['bbox']   = best_cv['bbox']
                mdet['center'] = best_cv['center']
                used_cv_idx.add(best_idx)
            
            final_dets.append(mdet)
            
        # 2. If we have fewer than 3 detections, take the best remaining CV circles
        if len(final_dets) < 3:
            remaining_cv = [c for i, c in enumerate(cv_dets) if i not in used_cv_idx]
            remaining_cv.sort(key=lambda x: x['area'], reverse=True)
            
            for rcv in remaining_cv:
                if len(final_dets) >= 3: break
                final_dets.append(rcv)
                
        return final_dets[:3]

    def classify_by_position(self, detections):
        """
        Assign analytes to detections based on model labels or strip geometry.
        Strip Layout:
             [Top]    -> Cysteine
          [L]     [R] -> Glutathione, Sialic Acid
        """
        analytes = ['Cysteine', 'Glutathione', 'Sialic Acid']
        
        # 1. Try to use labels from the model first
        labeled_count = 0
        for det in detections:
            lbl = det.get('label', '').strip().lower()
            found = False
            for a in analytes:
                if a.lower() in lbl:
                    det['analyte'] = a
                    det['symbol']  = '*'
                    labeled_count += 1
                    found = True
                    break
            if not found:
                det['analyte'] = None # Mark for geometric assignment

        # If all 3 are labeled correctly by the model, we are done
        if labeled_count == 3:
            return detections

        # 2. Geometric fallback: Top=Cys, Bottom-Left=Glut, Bottom-Right=Sial
        # Sort all detections by Y-coordinate to find the top one
        sorted_by_y = sorted(detections, key=lambda d: d['center'][1])
        if not sorted_by_y:
            return detections

        # Top-most is Cysteine
        sorted_by_y[0]['analyte'] = 'Cysteine'
        sorted_by_y[0]['symbol']  = '*'
        
        # Remaining ones sorted by X-coordinate
        if len(sorted_by_y) > 1:
            remaining = sorted(sorted_by_y[1:], key=lambda d: d['center'][0])
            if len(remaining) >= 1:
                remaining[0]['analyte'] = 'Glutathione'
                remaining[0]['symbol']  = '*'
            if len(remaining) >= 2:
                remaining[1]['analyte'] = 'Sialic Acid'
                remaining[1]['symbol']  = '*'
                
        return detections

    # ─────────────────────────────────────────────────────────────────────────
    #  CONCENTRATION MAPPING
    # ─────────────────────────────────────────────────────────────────────────

    def find_concentration(self, extracted_rgb, analyte_name):
        r, g, b = extracted_rgb
        pct     = blueness_to_pct(r, g, b)
        return int(np.clip(round(pct / 100.0 * 19), 0, 19))

    # ─────────────────────────────────────────────────────────────────────────
    #  CANCER RISK
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_cancer_risk(self, biomarkers):
        def p(key):
            return biomarkers.get(key, {}).get('concentration_pct', 0)

        cr = float(np.clip(p('Cysteine')   / 100, 0, 1))
        gr = float(np.clip(p('Glutathione') / 100, 0, 1))
        sr = float(np.clip(p('Sialic Acid') / 100, 0, 1))

        overall = float(np.clip(cr * 0.35 + gr * 0.35 + sr * 0.30, 0, 1))
        pct     = overall * 100

        if pct < 15:   cat, emoji = "VERY LOW RISK (< 15%)",  "[OK]"
        elif pct < 30: cat, emoji = "LOW RISK (15-30%)",       "[OK]"
        elif pct < 50: cat, emoji = "MODERATE RISK (30-50%)",  "[!]"
        elif pct < 70: cat, emoji = "HIGH RISK (50-70%)",      "[!!]"
        else:          cat, emoji = "VERY HIGH RISK (70%+)",   "[ALERT]"

        return {'score': overall, 'percentage': pct,
                'category': cat,  'emoji': emoji,
                'cys_risk': cr,   'glut_risk': gr, 'sial_risk': sr}

    # ─────────────────────────────────────────────────────────────────────────
    #  MAIN PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(self, image_path, annotated_path=None):
        """
        Robust Hybrid Pipeline: uses CV to find circles, falls back to fixed
        positions ONLY for missing ones.
        """
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"[ERROR] Could not load: {image_path}")
            return None

        ih, iw = image_bgr.shape[:2]
        print(f"\n{'='*80}\nHYBRID CV ANALYSIS: {image_path}\n{'='*80}\n")

        # ── Step 1: Detect circles using Computer Vision ──
        cv_dets = self._detect_circles_cv(image_bgr)
        print(f"  [CV] Found {len(cv_dets)} circle(s) via adaptive thresholding.")

        # Expected relative positions (y, x)
        EXPECTED = {
            'Cysteine':    (0.20, 0.50),
            'Glutathione': (0.43, 0.24),
            'Sialic Acid': (0.43, 0.76),
        }
        
        detections = []
        assigned_cv_indices = set()
        
        # ── Step 2: Match CV detections to expected biomarkers ──
        for name, (ey, ex) in EXPECTED.items():
            target_y, target_x = int(ey * ih), int(ex * iw)
            best_cv = None
            best_dist = 1000000
            best_idx = -1
            
            for i, cvd in enumerate(cv_dets):
                if i in assigned_cv_indices: continue
                cx, cy = cvd['center']
                dist = np.sqrt((cx - target_x)**2 + (cy - target_y)**2)
                # If it's within a reasonable distance of the expected spot
                if dist < (min(ih, iw) * 0.25) and dist < best_dist:
                    best_dist = dist
                    best_cv = cvd
                    best_idx = i
            
            if best_cv:
                # We found a real circle for this biomarker!
                best_cv['analyte'] = name
                detections.append(best_cv)
                assigned_cv_indices.add(best_idx)
                print(f"  [CV-MATCH] {name:12} @ ({best_cv['center'][0]:.0f}, {best_cv['center'][1]:.0f})")
            else:
                # Fallback to fixed position for this specific biomarker
                print(f"  [FALLBACK] {name:12} @ {target_x},{target_y}")
                r = 55 # Larger fallback box (55x55) to capture more area
                detections.append({
                    'analyte': name,
                    'center':  (target_x, target_y),
                    'bbox':    (target_x - r, target_y - r, target_x + r, target_y + r),
                    'confidence': 0.5,
                    'label':   'fallback'
                })

        # ── Step 3: Sample Colors ──
        for det in detections:
            # We sample the center 70% of the detected box to avoid background bleed
            x1, y1, x2, y2 = det['bbox']
            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Sub-region (center 70%)
            r_w, r_h = int(bw * 0.7), int(bh * 0.7)
            sample_bbox = (int(cx - r_w/2), int(cy - r_h/2), int(cx + r_w/2), int(cy + r_h/2))
            
            det['rgb'] = self.extract_rgb(image_bgr, sample_bbox)
            r, g, b = det['rgb']
            pct = blueness_to_pct(r, g, b)
            det['concentration_pct'] = pct
            det['concentration_level'] = int(np.clip(round(pct / 100.0 * 19), 0, 19))

        # ── Step 4: Draw Annotations ──
        if annotated_path and os.path.exists(annotated_path):
            self._draw_analysis_annotations(annotated_path, detections)

        # ── Step 5: Format Results ──
        results = {
            'timestamp':  datetime.now().isoformat(),
            'image_path': image_path,
            'detections': [],
            'biomarkers': {},
        }

        print("\nStep 3: Results...")
        for det in detections:
            analyte = det.get('analyte') or "Unknown"
            r, g, b  = det['rgb']
            cpct     = det['concentration_pct']
            print(f"  ✓ {analyte:15} RGB=({int(r):3d},{int(g):3d},{int(b):3d}) → {cpct:5.1f}%")

            results['biomarkers'][analyte] = {
                'concentration_level': det['concentration_level'],
                'concentration_pct':   cpct,
            }
            results['detections'].append({
                'analyte':             analyte,
                'rgb':                 det['rgb'],
                'concentration_pct':   cpct,
                'concentration_level': det['concentration_level'],
                'confidence':          det.get('confidence', 0.95)
            })

        results['cancer_risk'] = self.calculate_cancer_risk(results['biomarkers'])
        return results

    def _draw_analysis_annotations(self, path, detections):
        """Draw biomarker colored rectangles and labels on the image."""
        ann = cv2.imread(path)
        if ann is None: return

        # Color mapping for analytes (BGR)
        COLORS = {
            'Cysteine':    (0, 255, 255),    # Yellow
            'Glutathione': (203, 192, 255),  # Pinkish
            'Sialic Acid': (128, 128, 128),  # Gray
            'Unknown':     (255, 255, 255)   # White
        }
        
        for det in detections:
            analyte = det.get('analyte') or "Unknown"
            color = COLORS.get(analyte, COLORS['Unknown'])
            
            # Use detection bbox if available, otherwise construct from center
            if 'bbox' in det:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            else:
                cx, cy = [int(v) for v in det['center']]
                r = 25
                x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
            
            # Draw thick colored rectangle
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, 4)
            
            # Draw label with background for readability
            lbl = f"{analyte}: {det['concentration_pct']:.0f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 0.6
            (tw, th), _ = cv2.getTextSize(lbl, font, f_scale, 2)
            
            # Label background (above the box)
            cv2.rectangle(ann, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
            # Label text
            cv2.putText(ann, lbl, (x1 + 5, y1 - 8),
                        font, f_scale, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imwrite(path, ann)
        print(f"  [OK] Updated annotations with colored biomarker boxes: {path}")

    # ─────────────────────────────────────────────────────────────────────────
    #  REPORT
    # ─────────────────────────────────────────────────────────────────────────

    def print_report(self, results):
        print(f"\n{'='*80}")
        print("SALIVA STRIP BIOMARKER ANALYSIS REPORT")
        print(f"{'='*80}\n")
        print(f"Timestamp : {results['timestamp']}")
        print(f"Image     : {results['image_path']}\n")

        print(f"{'-'*80}")
        print(f"DETECTED BIOMARKERS ({len(results['detections'])} circles sampled)")
        print(f"{'-'*80}\n")

        for i, det in enumerate(results['detections'], 1):
            rv, gv, bv = [int(v) for v in det['rgb']]
            print(f"{i}. {det['analyte'].upper()}")
            print(f"   RGB:           R={rv} G={gv} B={bv}  (B-R={bv-rv:+d})")
            print(f"   Concentration: {det['concentration_pct']:.1f}%"
                  f"  (level {det['concentration_level']}/19)\n")

        print(f"{'-'*80}\nCONCENTRATION SUMMARY\n{'-'*80}\n")
        for analyte, data in results['biomarkers'].items():
            pv  = data['concentration_pct']
            bar = '█' * int(pv / 5) + '░' * (20 - int(pv / 5))
            print(f"  {analyte:15} |{bar}| {pv:5.1f}%")

        c = results['cancer_risk']
        print(f"\n{'-'*80}\nORAL CANCER RISK ASSESSMENT\n{'-'*80}\n")
        print(f"  {c['emoji']}  {c['category']}")
        print(f"  Overall Risk: {c['percentage']:.2f}%\n")
        print(f"  Cysteine    (35%): {c['cys_risk']*100:.1f}%")
        print(f"  Glutathione (35%): {c['glut_risk']*100:.1f}%")
        print(f"  Sialic Acid (30%): {c['sial_risk']*100:.1f}%")
        print(f"\n{'='*80}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image',      type=str,   required=True)
    p.add_argument('--model',      type=str,   default='modalv2.eim')
    p.add_argument('--confidence', type=float, default=0.5)
    a = p.parse_args()
    SimpleSalivaStripAnalyzer(a.model, a.confidence).analyze(a.image)

if __name__ == "__main__":
    main()