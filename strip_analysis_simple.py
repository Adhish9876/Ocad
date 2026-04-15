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

---
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
SAMPLE_RADIUS_FRAC = 0.04   # sample window = ±4% of image dimension


def blueness_to_pct(r, g, b):
    """
    Map RGB → concentration (1-95%) using red/blue dominance.

    - Clearly blueish shades (b > r) map towards 95%
    - Red-dominant shades (r > b) map towards 1-5%
    """
    r = float(r)
    g = float(g)
    b = float(b)

    # Calculate effective blue and red by removing common brightness (green)
    # This helps isolate the actual dye color from ambient white light.
    r_eff = max(0.0, r - GREEN_SUPPRESS * g)
    b_eff = max(0.0, b - GREEN_SUPPRESS * g)

    if b_eff > r_eff:
        # Blue dominance: very steep curve.
        # Even a 15% lead in blue (lead = 1.15) will hit 95%.
        lead = b_eff / (r_eff + 1.0)
        pct = 50.0 + (min(1.0, (lead - 1.0) / 0.15) * 45.0)
    else:
        # Red dominance: quadratic curve to keep it very low (1-5%) 
        # unless it's nearly neutral.
        ratio = b_eff / (r_eff + 1.0)
        pct = 1.0 + (ratio ** 2) * 49.0

    return float(np.clip(pct, 1.0, 95.0))


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
        """Kept for backward-compat with app.py. Uses inner-60% mean."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        ih, iw = image_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        dx = int((x2 - x1) * 0.20)
        dy = int((y2 - y1) * 0.20)
        roi = image_bgr[y1+dy:y2-dy, x1+dx:x2-dx]
        if roi.size == 0:
            roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return (128, 128, 128)
        m = cv2.mean(roi)[:3]
        return (int(m[2]), int(m[1]), int(m[0]))

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

    def _detect_coloured_regions(self, image_bgr):
        ih, iw = image_bgr.shape[:2]
        total  = ih * iw
        hsv    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Red mask
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,   60, 50]), np.array([12,  255, 210])),
            cv2.inRange(hsv, np.array([165, 60, 50]), np.array([180, 255, 210]))
        )
        # Blue/violet mask (circles: H≈100-148, S≥80)
        blue_mask = cv2.inRange(hsv,
                                np.array([100, 80,  50]),
                                np.array([148, 255, 200]))

        combined = cv2.bitwise_or(red_mask, blue_mask)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k, iterations=1)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        min_area = max(50,  total * 0.0005)
        max_area = total * 0.05        # tighter cap to avoid merged blobs

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            perim = cv2.arcLength(cnt, True)
            circ  = (4 * np.pi * area / perim ** 2) if perim > 0 else 0
            asp   = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
            conf  = min(1.0, circ / 0.65) * 0.6 + min(1.0, asp / 0.70) * 0.4
            if conf >= self.confidence * 0.5:
                detections.append({
                    'bbox':       (x, y, x + bw, y + bh),
                    'center':     (x + bw / 2, y + bh / 2),
                    'confidence': conf,
                    'area':       area,
                })

        if len(detections) > 3:
            detections = sorted(detections,
                                key=lambda d: d['confidence'],
                                reverse=True)[:3]
        return detections

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
        Main pipeline: loads image, detects circles (model or fallback),
        samples colors, and optionally draws circles on an annotated image.
        """
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"[ERROR] Could not load: {image_path}")
            return None

        print(f"\n{'='*80}\nProcessing: {image_path}\n{'='*80}\n")

        # ── Step 1: Detect circles (Model primary, Fixed-position fallback) ──
        detections = []
        if self.runner is not None:
            print("Step 1: Running Edge Impulse model detection...")
            detections = self._detect_with_model(image_bgr)

        if detections:
            print(f"Step 2: Sampling RGB at {len(detections)} model-detected positions...")
            for det in detections:
                det['rgb'] = self.extract_rgb(image_bgr, det['bbox'])
                r, g, b = det['rgb']
                pct = blueness_to_pct(r, g, b)
                det['concentration_pct'] = pct
                det['concentration_level'] = int(np.clip(round(pct / 100.0 * 19), 0, 19))
            
            # Use geometric layout to assign 'Cysteine', etc.
            self.classify_by_position(detections)
        else:
            print("Step 1-2: Fallback — sampling biomarker circles at fixed positions...")
            detections = self._sample_fixed_positions(image_bgr)

        # ── Step 3: Draw circles on annotated image if requested ──
        if annotated_path and os.path.exists(annotated_path):
            self._draw_analysis_annotations(annotated_path, detections)

        results = {
            'timestamp':  datetime.now().isoformat(),
            'image_path': image_path,
            'detections': [],
            'biomarkers': {},
        }

        print("\nStep 3: Concentrations...")
        for det in detections:
            analyte = det.get('analyte') or "Unknown"
            r, g, b  = det['rgb']
            cpct     = det['concentration_pct']
            level    = det['concentration_level']
            print(f"  ✓ {analyte:15} RGB=({r:3d},{g:3d},{b:3d}) → {cpct:5.1f}%")

            results['detections'].append({
                'analyte':             analyte,
                'symbol':              '*',
                'confidence':          det.get('confidence', 0.9),
                'rgb':                 det['rgb'],
                'concentration_level': level,
                'concentration_pct':   cpct,
            })
            results['biomarkers'][analyte] = {
                'concentration_level': level,
                'concentration_pct':   cpct,
            }

        print("\nStep 4: Cancer risk...")
        cancer = self.calculate_cancer_risk(results['biomarkers'])
        print(f"  ✓ {cancer['emoji']} {cancer['category']}  ({cancer['percentage']:.1f}%)\n")
        results['cancer_risk'] = cancer

        self.print_report(results)
        return results

    def _draw_analysis_annotations(self, path, detections):
        """Draw biomarker circles and labels on the annotated image."""
        ann = cv2.imread(path)
        if ann is None: return
        
        for det in detections:
            analyte = det.get('analyte') or "Circle"
            cx, cy = [int(v) for v in det['center']]
            # Draw circle
            cv2.circle(ann, (cx, cy), 15, (255, 255, 255), 2)
            cv2.circle(ann, (cx, cy), 5,  (0, 255, 255), -1)
            
            # Label
            lbl = f"{analyte}: {det['concentration_pct']:.0f}%"
            cv2.putText(ann, lbl, (cx + 20, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(ann, lbl, (cx + 20, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        cv2.imwrite(path, ann)
        print(f"  [OK] Updated annotations with biomarker circles: {path}")

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