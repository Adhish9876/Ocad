"""
Retrain YOLOv8 model with corrected biomarker labels.

This script trains a YOLOv8 model to detect and classify three types of saliva strip biomarkers:
- Class 0: Cysteine (Circle shape)
- Class 1: Glutathione (Circle shape)
- Class 2: Sialic Acid (Circle shape)
"""

import os
import torch
from ultralytics import YOLO
from pathlib import Path

def retrain_model():
    """
    Retrain YOLOv8 model with the corrected dataset labels.
    """
    
    print("="*80)
    print("YOLOV8 SALIVA STRIP BIOMARKER DETECTION - MODEL RETRAINING")
    print("="*80)
    
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Path to dataset configuration
    data_yaml = current_dir / "saliva_data.yaml"
    
    # Check if data.yaml exists
    if not data_yaml.exists():
        print(f"❌ Error: {data_yaml} not found")
        return False
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Load model
    print(f"\n{'─'*80}")
    print("Step 1: Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # Load small pretrained model (yolov8n)
    print("✓ Model loaded: YOLOv8 Nano")
    
    # Training parameters
    print(f"\n{'─'*80}")
    print("Step 2: Starting training with corrected labels...")
    print(f"\nTraining Configuration:")
    print(f"  - Dataset: Saliva Strip Biomarkers (3 classes)")
    print(f"  - Model: YOLOv8 Nano")
    print(f"  - Classes: Cysteine, Glutathione, Sialic Acid")
    print(f"  - Device: {device}")
    print(f"  - Data YAML: {data_yaml}")
    
    # Train the model
    results = model.train(
        data=str(data_yaml),
        epochs=100,                    # Train for 100 epochs
        imgsz=640,                     # Image size
        batch=16,                      # Batch size (adjust based on GPU memory)
        patience=20,                   # Early stopping patience
        device=device,
        project="runs/detect",
        name="saliva_biomarkers",      # Experiment name
        save=True,
        save_period=10,                # Save every 10 epochs
        verbose=True,
        pretrained=True,               # Use pretrained weights
        optimizer='SGD',               # Optimizer
        lr0=0.01,                      # Initial learning rate
        lrf=0.01,                      # Final learning rate
        momentum=0.937,                # Momentum
        weight_decay=0.0005,           # Weight decay
        warmup_epochs=3,               # Warmup epochs
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,                       # Box loss weight
        cls=0.5,                       # Class loss weight
        dfl=1.5,                       # Focal loss weight
        hsv_h=0.015,                   # HSV hue augmentation
        hsv_s=0.7,                     # HSV saturation augmentation
        hsv_v=0.4,                     # HSV value augmentation
        degrees=0,                     # Rotation degrees
        translate=0.1,                 # Translation
        scale=0.5,                     # Scaling
        flipud=0.0,                    # Flip upside down
        fliplr=0.5,                    # Flip left-right
        mosaic=1.0,                    # Mosaic augmentation
        close_mosaic=15,               # Close mosaic at epoch 15
    )
    
    # Display results
    print(f"\n{'─'*80}")
    print("Step 3: Training Complete!")
    print(f"{'─'*80}\n")
    
    # Check if training was successful
    if results:
        print("✓ Training completed successfully!")
        print(f"\nResults folder: runs/detect/saliva_biomarkers/")
        print(f"Best model: runs/detect/saliva_biomarkers/weights/best.pt")
        print(f"Last model: runs/detect/saliva_biomarkers/weights/last.pt")
        
        # Display model info
        print(f"\n{'─'*80}")
        print("Model Information:")
        print(f"{'─'*80}")
        print(f"Model: {model}")
        
        print(f"\n{'='*80}")
        print("✓ YOUR MODEL HAS BEEN SUCCESSFULLY RETRAINED!")
        print(f"{'='*80}")
        print("\nNext Steps:")
        print("1. Use the new model for inference:")
        print("   python strip_analysis_new.py --image <path_to_image> \\")
        print("     --model runs/detect/saliva_biomarkers/weights/best.pt")
        print("\n2. The model will now correctly detect:")
        print("   - Circle (Class 0) → Cysteine")
        print("   - Circle (Class 1) → Glutathione")
        print("   - Circle (Class 2) → Sialic Acid")
        print(f"{'='*80}\n")
        
        return True
    else:
        print("❌ Training failed!")
        return False

def validate_model(model_path):
    """
    Validate the trained model on the validation set.
    
    Args:
        model_path: Path to the trained model
    """
    print(f"\n{'─'*80}")
    print("Step 4: Validating Model...")
    print(f"{'─'*80}\n")
    
    current_dir = Path(__file__).parent
    data_yaml = current_dir / "saliva_data.yaml"
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Validate
    metrics = model.val(data=str(data_yaml))
    
    print(f"\n✓ Validation Metrics:")
    print(f"  - mAP50: {metrics.box.map50:.3f}")
    print(f"  - mAP50-95: {metrics.box.map:.3f}")

if __name__ == "__main__":
    # Retrain the model
    success = retrain_model()
    
    # If training was successful, validate the model
    if success:
        model_path = "runs/detect/saliva_biomarkers/weights/best.pt"
        if Path(model_path).exists():
            validate_model(model_path)
