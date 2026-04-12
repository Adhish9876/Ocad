# Strip Analysis Pipeline

This project implements a modular pipeline for detecting and analyzing urine test strips (URS10T) using YOLOv5 and OpenCV.

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- PyTorch
- YOLOv5 (included in the workspace)

## Usage

Run the `strip_analysis.py` script with an input image:

```bash
python strip_analysis.py --image "test/images/your_image.jpg"
```

You can optionally specify the weights file:

```bash
python strip_analysis.py --image "image.jpg" --weights "yolov5/runs/train/exp6/weights/best.pt"
```

## detailed Pipeline Steps

1.  **Detection**: Loads the YOLOv5 model and finds the bounding box of the strip.
2.  **Cropping**: Extracts the region of interest.
3.  **Alignment**: Uses edge detection and perspective transformation to align the strip vertically.
4.  **Segmentation**: Splits the strip into 10 equal vertical pads.
5.  **Feature Extraction**: Computes Mean RGB, HSV, and LAB color values for each pad.
6.  **Visualization**: Saves debug images for each step.

## Outputs

The script prints the feature vector and saves:
- `debug_cropped.jpg`
- `debug_aligned.jpg`
- `result_visualization.jpg`
