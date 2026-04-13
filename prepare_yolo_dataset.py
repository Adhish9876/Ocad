"""
YOLO Dataset Preparation Script
================================
This script:
1. Extracts spot images from the DOCX file
2. Organizes them into YOLO train/val/test splits (70/20/10)
3. Creates the YOLO folder structure and dataset YAML
4. Provides instructions for annotation

Usage:
    python prepare_yolo_dataset.py --docx your_file.docx --output yolo_dataset

Dependencies:
    pip install pillow python-docx scikit-learn pyyaml
"""

import os
import shutil
import zipfile
import random
import argparse
import yaml
from pathlib import Path
from PIL import Image


# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70   # 70% training
VAL_RATIO   = 0.20   # 20% validation
TEST_RATIO  = 0.10   # 10% test
RANDOM_SEED = 42

# Image indices that are "spot" images (not graphs).
# From your DOCX: image1–image20 are spot images, image21–image32 are graphs.
SPOT_IMAGE_RANGE = range(1, 21)   # images 1 to 20 inclusive


# ── STEP 1: EXTRACT IMAGES FROM DOCX ─────────────────────────────────────────
def extract_images_from_docx(docx_path: str, output_dir: str) -> list:
    """
    Extracts spot images from the DOCX's media folder.
    Returns a sorted list of extracted image paths.
    """
    extract_dir = Path(output_dir) / "_raw_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    spot_images = []

    with zipfile.ZipFile(docx_path, 'r') as z:
        media_files = [f for f in z.namelist() if f.startswith("word/media/")]
        for media_file in sorted(media_files):
            filename = Path(media_file).name
            # Extract only spot images (image1.png to image20.png)
            for i in SPOT_IMAGE_RANGE:
                if filename == f"image{i}.png":
                    dest = extract_dir / f"spot_{i:03d}.png"
                    with z.open(media_file) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                    # Convert RGBA → RGB (YOLO models prefer RGB)
                    img = Image.open(dest).convert("RGB")
                    img.save(dest)
                    spot_images.append(str(dest))
                    print(f"  Extracted: {dest.name} ({img.size})")

    spot_images.sort()
    print(f"\n✅ Extracted {len(spot_images)} spot images.\n")
    return spot_images


# ── STEP 2: SPLIT INTO TRAIN / VAL / TEST ─────────────────────────────────────
def split_dataset(image_paths: list, seed: int = RANDOM_SEED) -> dict:
    """
    Randomly splits image paths into train/val/test.
    Returns a dict with keys 'train', 'val', 'test'.
    """
    random.seed(seed)
    shuffled = image_paths.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }

    print("📊 Dataset split:")
    for split, paths in splits.items():
        print(f"   {split:5s}: {len(paths)} images")
    print()
    return splits


# ── STEP 3: BUILD YOLO FOLDER STRUCTURE ──────────────────────────────────────
def build_yolo_structure(splits: dict, output_dir: str) -> None:
    """
    Creates the standard YOLO folder layout:

        yolo_dataset/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/    ← you fill these with annotation .txt files
            ├── val/
            └── test/
    """
    base = Path(output_dir)

    for split, paths in splits.items():
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for src_path in paths:
            src = Path(src_path)
            dst = img_dir / src.name
            shutil.copy2(src, dst)

            # Create a placeholder empty label file (to be filled after annotation)
            label_file = lbl_dir / (src.stem + ".txt")
            if not label_file.exists():
                label_file.write_text(
                    "# Annotate this file using LabelImg or Roboflow\n"
                    "# Format per line: <class_id> <x_center> <y_center> <width> <height>\n"
                    "# All values normalised 0–1 relative to image size\n"
                    "# Example: 0 0.512 0.498 0.134 0.206\n"
                )

    # Clean up raw extraction folder
    raw = Path(output_dir) / "_raw_extracted"
    if raw.exists():
        shutil.rmtree(raw)

    print(f"✅ YOLO folder structure created at: {output_dir}/\n")
    _print_tree(base)


# ── STEP 4: GENERATE dataset.yaml ────────────────────────────────────────────
def create_yaml(output_dir: str, class_names: list) -> None:
    """
    Creates the YOLO dataset.yaml config file.
    Edit 'nc' and 'names' to match your actual annotation classes.
    """
    base = Path(output_dir).resolve()
    config = {
        "path":  str(base),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(class_names),
        "names": class_names,
    }
    yaml_path = base / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ dataset.yaml written → {yaml_path}\n")
    print("   ⚠️  Update 'names' list in dataset.yaml to match your annotation classes!\n")


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _print_tree(base: Path, prefix: str = "") -> None:
    print(f"\n📁 Folder structure:\n{base}/")
    for item in sorted(base.rglob("*")):
        rel = item.relative_to(base)
        depth = len(rel.parts) - 1
        connector = "└── " if depth == len(rel.parts) - 1 else "├── "
        print("    " * depth + connector + item.name)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from DOCX spot images")
    parser.add_argument("--docx",    required=True,               help="Path to your .docx file")
    parser.add_argument("--output",  default="yolo_dataset",      help="Output folder name")
    parser.add_argument("--classes", nargs="+",
                        default=["spot"],
                        help="Class names for YOLO (e.g. --classes spot background)")
    args = parser.parse_args()

    print("=" * 60)
    print("  YOLO Dataset Preparation")
    print("=" * 60 + "\n")

    # Step 1 – Extract
    images = extract_images_from_docx(args.docx, args.output)

    # Step 2 – Split
    splits = split_dataset(images)

    # Step 3 – Build structure
    build_yolo_structure(splits, args.output)

    # Step 4 – YAML
    create_yaml(args.output, args.classes)

    print("=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    print("""
1. ANNOTATE your images:
   - Tool (recommended): LabelImg  →  pip install labelImg && labelImg
   - Or online:           Roboflow  →  https://roboflow.com
   - Format: YOLO (.txt) — one file per image in labels/train|val|test/

2. YOLO label format (one row per object):
   <class_id> <x_center> <y_center> <width> <height>
   (all values 0–1, relative to image dimensions)

3. TRAIN with YOLOv8 (ultralytics):
   pip install ultralytics
   yolo detect train data=yolo_dataset/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640

4. VALIDATE:
   yolo detect val data=yolo_dataset/dataset.yaml model=runs/detect/train/weights/best.pt

5. PREDICT on new images:
   yolo detect predict model=runs/detect/train/weights/best.pt source=new_images/
""")