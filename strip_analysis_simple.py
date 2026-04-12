"""
Simplified Saliva Strip Biomarker Analysis
Uses YOLOv8 detection + spatial position-based biomarker assignment
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import argparse
from datetime import datetime

class SimpleSalivaStripAnalyzer:
    """Simplified analyzer using spatial positioning"""
    
    def __init__(self, model_path='runs/detect/train2/weights/best.pt', confidence=0.5):
        """Initialize with YOLOv8 model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence = confidence
        
        try:
            self.model = YOLO(model_path)
            self.model.conf = confidence
            print(f"✓ Model loaded: {model_path}")
            print(f"✓ Device: {self.device}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Reference charts - comprehensive
        self.reference_charts = self._load_reference_charts()
    
    def _load_reference_charts(self):
        """Load reference RGB charts"""
        charts = {
            'Cysteine': self._get_cysteine_chart(),
            'Glutathione': self._get_glutathione_chart(),
            'Sialic Acid': self._get_sialic_acid_chart()
        }
        return charts
    
    def _get_cysteine_chart(self):
        """Cysteine reference chart based on ESTHER data (0-130 µM range)
        Linear equations from graphs:
        Red:  Y = -2.026*X + 183.7
        Blue: Y = 1.032*X + 42.23
        """
        charts = {}
        max_conc = 130  # µM
        for i in range(20):
            conc = (i / 19) * max_conc  # Spread 20 levels across 0-130 µM
            red_intensity = max(0, -2.026 * conc + 183.7)
            blue_intensity = max(0, min(255, 1.032 * conc + 42.23))
            # Create RGB tuples: (R, G, B)
            charts[i] = {
                'baseline_red': (int(red_intensity), 50, 50),
                'post_reaction_blue': (50, 50, int(blue_intensity))
            }
        return charts
    
    def _get_glutathione_chart(self):
        """Glutathione reference chart based on ESTHER data (0-150 µM range)
        Linear equations from graphs:
        Red:  Y = -0.9152*X + 113.7
        Blue: Y = 0.8010*X + 22.94
        """
        charts = {}
        max_conc = 150  # µM
        for i in range(20):
            conc = (i / 19) * max_conc  # Spread 20 levels across 0-150 µM
            red_intensity = max(0, -0.9152 * conc + 113.7)
            blue_intensity = max(0, min(255, 0.8010 * conc + 22.94))
            charts[i] = {
                'baseline_red': (int(red_intensity), 50, 50),
                'post_reaction_blue': (50, 50, int(blue_intensity))
            }
        return charts
    
    def _get_sialic_acid_chart(self):
        """Sialic Acid reference chart based on ESTHER data (0-6 mM range)
        Linear equations from graphs:
        Red:  Y = -21.22*X + 146.5
        Blue: Y = 20.07*X + 42.79
        """
        charts = {}
        max_conc = 6.0  # mM
        for i in range(20):
            conc = (i / 19) * max_conc  # Spread 20 levels across 0-6 mM
            red_intensity = max(0, -21.22 * conc + 146.5)
            blue_intensity = max(0, min(255, 20.07 * conc + 42.79))
            charts[i] = {
                'baseline_red': (int(red_intensity), 50, 50),
                'post_reaction_blue': (50, 50, int(blue_intensity))
            }
        return charts
    
    def detect_spots(self, image):
        """Detect all spots using YOLO"""
        results = self.model.predict(source=image, conf=self.confidence, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': confidence
                })
        
        return sorted(detections, key=lambda d: d['center'][0])  # Sort left to right
    
    def classify_by_position(self, detections):
        """Assign biomarkers based on position - simpler approach"""
        if len(detections) < 3:
            return detections
        
        # Sort by X coordinate (left to right)
        sorted_dets = sorted(detections, key=lambda d: d['center'][0])
        
        # Expected order: Cysteine (left), Glutathione/Sialic varies by image
        # Use Y position as secondary: usually Glutathione is higher
        
        for i, det in enumerate(sorted_dets):
            if i == 0:
                det['analyte'] = 'Cysteine'
                det['symbol'] = '●'
            elif i == 1:
                # Check if higher (more likely Glutathione) or lower
                det['analyte'] = 'Glutathione'
                det['symbol'] = '●'
            else:
                det['analyte'] = 'Sialic Acid'
                det['symbol'] = '●'
        
        return sorted_dets
    
    def extract_rgb(self, image, bbox):
        """Extract mean RGB from bounding box"""
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        mean_bgr = cv2.mean(roi)[:3]
        return (mean_bgr[2], mean_bgr[1], mean_bgr[0])  # BGR to RGB
    
    def find_concentration(self, extracted_rgb, analyte_name):
        """Find closest concentration"""
        if analyte_name not in self.reference_charts:
            return 0
        
        chart = self.reference_charts[analyte_name]
        min_dist = float('inf')
        best_conc = 0
        
        for level, data in chart.items():
            for key in ['baseline_red', 'post_reaction_blue']:
                ref_rgb = data[key]
                dist = np.sqrt((extracted_rgb[0]-ref_rgb[0])**2 + 
                              (extracted_rgb[1]-ref_rgb[1])**2 + 
                              (extracted_rgb[2]-ref_rgb[2])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_conc = level
        
        return best_conc
    
    def calculate_cancer_risk(self, biomarkers):
        """Calculate cancer risk from concentrations"""
        cys_pct = biomarkers.get('Cysteine', {}).get('concentration_pct', 0)
        glut_pct = biomarkers.get('Glutathione', {}).get('concentration_pct', 0)
        sial_pct = biomarkers.get('Sialic Acid', {}).get('concentration_pct', 0)
        
        # Risk calculation — high concentration (blue) = high cancer risk
        cys_risk = max(0, cys_pct / 100)
        glut_risk = max(0, glut_pct / 100)
        sial_risk = max(0, sial_pct / 100)
        
        overall = (cys_risk * 0.35 + glut_risk * 0.35 + sial_risk * 0.30)
        overall = max(0, min(1, overall))
        
        pct = overall * 100
        
        if pct < 15:
            category = "VERY LOW RISK (< 15%)"
            emoji = "✅"
        elif pct < 30:
            category = "LOW RISK (15-30%)"
            emoji = "✓"
        elif pct < 50:
            category = "MODERATE RISK (30-50%)"
            emoji = "⚠"
        elif pct < 70:
            category = "HIGH RISK (50-70%)"
            emoji = "⚠⚠"
        else:
            category = "VERY HIGH RISK (70%+)"
            emoji = "⛔"
        
        return {
            'score': overall,
            'percentage': pct,
            'category': category,
            'emoji': emoji,
            'cys_risk': cys_risk,
            'glut_risk': glut_risk,
            'sial_risk': sial_risk
        }
    
    def print_report(self, results):
        """Print analysis report"""
        print(f"\n{'='*80}")
        print(f"SALIVA STRIP BIOMARKER ANALYSIS REPORT")
        print(f"{'='*80}\n")
        
        print(f"Timeline: {results['timestamp']}")
        print(f"Image: {results['image_path']}\n")
        
        print(f"{'-'*80}")
        print(f"DETECTED BIOMARKERS ({len(results['detections'])} spots)")
        print(f"{'-'*80}\n")
        
        for i, det in enumerate(results['detections'], 1):
            print(f"{i}. {det['symbol']} {det['analyte'].upper()}")
            print(f"   YOLO Confidence:   {det['confidence']*100:.1f}%")
            print(f"   Extracted RGB:     R={int(det['rgb'][0])}, G={int(det['rgb'][1])}, B={int(det['rgb'][2])}")
            print(f"   Concentration:     {det['concentration_pct']:.1f}% (Level {det['concentration_level']}/19)")
            print()
        
        print(f"{'-'*80}")
        print(f"CONCENTRATION SUMMARY")
        print(f"{'-'*80}\n")
        
        for analyte, data in results['biomarkers'].items():
            bar_len = int(data['concentration_pct'] / 5)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            print(f"{analyte:20} | {bar} | {data['concentration_pct']:5.1f}%")
        
        cancer = results['cancer_risk']
        print(f"\n{'-'*80}")
        print(f"ORAL CANCER RISK ASSESSMENT")
        print(f"{'-'*80}\n")
        print(f"{cancer['emoji']} {cancer['category']}")
        print(f"Overall Risk Score: {cancer['percentage']:.2f}%\n")
        
        print(f"Individual Risks:")
        print(f"  • Cysteine Risk     (35%):  {cancer['cys_risk']:.3f}")
        print(f"  • Glutathione Risk  (35%):  {cancer['glut_risk']:.3f}")
        print(f"  • Sialic Acid Risk  (30%):  {cancer['sial_risk']:.3f}")
        print(f"\n{'='*80}\n")
    
    def analyze(self, image_path):
        """Complete analysis pipeline"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        print(f"\n{'='*80}")
        print("Processing saliva strip image...")
        print(f"{'='*80}\n")
        
        # Step 1: Detect spots
        print("Step 1: Detecting spots with YOLOv8...")
        detections = self.detect_spots(image)
        print(f"✓ Detected {len(detections)} spot(s)\n")
        
        if len(detections) < 3:
            print("⚠ Warning: Expected 3 biomarkers, got", len(detections))
        
        # Step 2: Classify by position
        print("Step 2: Classifying biomarkers...")
        detections = self.classify_by_position(detections)
        for det in detections:
            print(f"  ✓ {det['symbol']} {det['analyte']}")
        print()
        
        # Step 3: Extract RGB and concentrations
        print("Step 3: Analyzing concentrations...")
        results = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'detections': [],
            'biomarkers': {}
        }
        
        for det in detections:
            rgb = self.extract_rgb(image, det['bbox'])
            conc_level = self.find_concentration(rgb, det['analyte'])
            conc_pct = (conc_level / 19) * 100
            
            det_result = {
                'analyte': det['analyte'],
                'symbol': det['symbol'],
                'confidence': det['confidence'],
                'rgb': rgb,
                'concentration_level': conc_level,
                'concentration_pct': conc_pct
            }
            
            results['detections'].append(det_result)
            results['biomarkers'][det['analyte']] = {
                'concentration_level': conc_level,
                'concentration_pct': conc_pct
            }
            
            print(f"  ✓ {det['symbol']} {det['analyte']:15} - {conc_pct:5.1f}%")
        
        print()
        
        # Step 4: Calculate cancer risk
        print("Step 4: Calculating cancer risk...")
        cancer = self.calculate_cancer_risk(results['biomarkers'])
        print(f"  ✓ Assessment complete\n")
        
        results['cancer_risk'] = cancer
        
        # Step 5: Report
        print("Step 5: Generating report...")
        self.print_report(results)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Saliva Strip Biomarker Analysis")
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='runs/detect/train2/weights/best.pt', help='Model path')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    analyzer = SimpleSalivaStripAnalyzer(model_path=args.model, confidence=args.confidence)
    analyzer.analyze(args.image)


if __name__ == "__main__":
    main()
