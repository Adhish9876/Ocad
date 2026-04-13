"""
Test the trained YOLOv8 model directly.
"""

from ultralytics import YOLO
import cv2

model_path = "runs/detect/runs/detect/saliva_biomarkers6/weights/best.pt"
image_path = "images/train/spot_002.png"

print("Loading model...")
model = YOLO(model_path)

print("Running inference...")
results = model.predict(source=image_path, conf=0.1, verbose=True)

print("\nResults:")
for result in results:
    print(f"Detections found: {len(result.boxes)}")
    for box in result.boxes:
        print(f"  Class: {box.cls}, Confidence: {box.conf:.3f}")

# Also test with visualization
image = cv2.imread(image_path)
img_with_boxes = results[0].plot()
cv2.imwrite("test_model_output.jpg", img_with_boxes)
print("\nVisualization saved to: test_model_output.jpg")
