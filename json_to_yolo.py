"""
Convert labelme JSON annotations to YOLO .txt format
Run this from your project folder.
"""
import json
import os
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Map your class names to IDs (add more if you have multiple classes)
CLASS_NAMES = {"spot": 0}

SPLITS = ["train", "val", "test"]
BASE    = "yolo_dataset"
# ─────────────────────────────────────────────────────────────────────────────

def convert_json_to_yolo(json_path, label_path):
    with open(json_path) as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    lines = []

    for shape in data["shapes"]:
        label = shape["label"].lower().strip()
        class_id = CLASS_NAMES.get(label, 0)

        pts = shape["points"]

        if shape["shape_type"] == "rectangle":
            x1, y1 = pts[0]
            x2, y2 = pts[1]
        else:
            # polygon or other — use bounding box of all points
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        # Normalise to 0–1
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w  = abs(x2 - x1) / img_w
        h  = abs(y2 - y1) / img_h

        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    return len(lines)


total = 0
for split in SPLITS:
    lbl_dir = Path("labels") / split

    if not lbl_dir.exists():
        print(f"  [{split}] Folder not found — skipping")
        continue

    json_files = list(lbl_dir.glob("*.json"))
    if not json_files:
        print(f"  [{split}] No JSON files found — skipping")
        continue

    print(f"\n[{split}] Converting {len(json_files)} files...")
    for json_file in sorted(json_files):
        label_file = lbl_dir / (json_file.stem + ".txt")
        count = convert_json_to_yolo(json_file, label_file)
        print(f"  {json_file.name} → {label_file.name}  ({count} boxes)")
        total += count

print(f"\n✅ Done! {total} total bounding boxes converted to YOLO format.")
print("Your labels are saved in labels/train, labels/val, labels/test")