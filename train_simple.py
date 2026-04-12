"""
Simple YOLOv8 training script for saliva strip biomarkers.
"""

from ultralytics import YOLO
import torch

print("="*80)
print("YOLOV8 SALIVA STRIP BIOMARKER RETRAINING")
print("="*80)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Device: {device}")

# Load model
print("\nLoading YOLOv8 Nano model...")
model = YOLO('yolov8n.pt')

# Train
print("\nStarting training...")
print("Config: 100 epochs, batch=16, imgsz=640")

results = model.train(
    data='saliva_data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,                  # Reduced batch size
    patience=20,
    device=device,
    project='runs/detect',
    name='saliva_biomarkers',
    save=True,
    verbose=True,
    workers=0,                # Disable multiprocessing workers
)

print("\n" + "="*80)
if results:
    print("✓ TRAINING COMPLETED!")
    print("Best model: runs/detect/saliva_biomarkers/weights/best.pt")
print("="*80)
