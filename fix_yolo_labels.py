"""
Fix YOLO label format to properly distinguish between the three biomarkers.
Converts JSON annotations to correct YOLO .txt format with proper class IDs.

Class mapping:
- Class 0: Cysteine (Circle)
- Class 1: Glutathione (Circle)
- Class 2: Sialic Acid (Circle)
"""

import json
import os
from pathlib import Path

# Define class mapping
CLASS_MAP = {
    'Cysteine': 0,
    'cysteine': 0,
    'Glutathione': 1,
    'glutathione': 1,
    'Sialic Acid': 2,
    'sialic acid': 2,
    'Sialic_Acid': 2,
}

def json_to_yolo_format(json_file, image_width, image_height):
    """
    Convert JSON bounding boxes to YOLO normalized format.
    YOLO format: class_id center_x center_y width height (all normalized 0-1)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    yolo_lines = []
    
    for shape in data.get('shapes', []):
        label = shape.get('label', '').strip()
        points = shape.get('points', [])
        
        # Get class ID
        class_id = CLASS_MAP.get(label)
        if class_id is None:
            print(f"⚠ Warning: Unknown label '{label}' in {json_file}")
            continue
        
        if len(points) >= 2:
            # Points are [x1, y1], [x2, y2]
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # Ensure x1 < x2 and y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Calculate center and dimensions
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Normalize to 0-1
            norm_center_x = center_x / image_width
            norm_center_y = center_y / image_height
            norm_width = width / image_width
            norm_height = height / image_height
            
            # Create YOLO line
            yolo_line = f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
            yolo_lines.append(yolo_line)
    
    return yolo_lines

def process_directory(labels_dir, images_dir):
    """
    Process all JSON files in a directory and create/update YOLO .txt files.
    """
    # Get the split name (train, val, test)
    split_name = Path(labels_dir).name
    json_files = sorted(Path(labels_dir).glob("*.json"))
    
    print(f"\nProcessing {len(json_files)} files in {labels_dir}...")
    
    for json_file in json_files:
        # Find corresponding image
        image_name = json_file.stem + ".png"
        image_path = Path(images_dir) / split_name / image_name
        
        if not image_path.exists():
            print(f"⚠ Image not found: {image_path}")
            continue
        
        # Get image dimensions
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"⚠ Could not read image: {image_path}")
                continue
            height, width = img.shape[:2]
        except Exception as e:
            print(f"⚠ Error reading image {image_path}: {e}")
            continue
        
        # Convert to YOLO format
        yolo_lines = json_to_yolo_format(str(json_file), width, height)
        
        # Write to .txt file
        txt_file = json_file.with_suffix('.txt')
        with open(txt_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        print(f"✓ {json_file.name:20} → {txt_file.name:20} ({len(yolo_lines)} objects)")
    
    print(f"✓ Completed processing {labels_dir}\n")

def main():
    """
    Fix all label files in train, val, and test directories.
    """
    base_dir = Path("c:\\Users\\adhis\\OneDrive\\Documents\\react-js\\Strip Deep.v1i.yolov5pytorch")
    labels_dir = base_dir / "labels"
    images_dir = base_dir / "images"
    
    # Process each dataset split
    for split in ['train', 'val', 'test']:
        split_labels_dir = labels_dir / split
        if split_labels_dir.exists():
            process_directory(str(split_labels_dir), str(images_dir))
    
    print("=" * 80)
    print("✓ YOLO label files have been fixed!")
    print("\nClass ID mapping:")
    print("  Class 0: Cysteine (Circle)")
    print("  Class 1: Glutathione (Circle)")
    print("  Class 2: Sialic Acid (Circle)")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Retrain your YOLOv8 model with these fixed labels")
    print("2. The model will now learn to distinguish between the three biomarkers")
    print("=" * 80)

if __name__ == "__main__":
    main()
