#!/usr/bin/env bash

# Inference script with visualization
# Usage: ./inference_with_viz.sh <input_image_folder> <output_folder> [model_path]

set -e

# Check if required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_image_folder> <output_folder> [model_path]"
    echo ""
    echo "Arguments:"
    echo "  input_image_folder  : Directory containing input images (.png, .jpg, .jpeg)"
    echo "  output_folder       : Directory where outputs will be saved"
    echo "  model_path         : (Optional) Path to trained model (default: runs_obb/cattle_y8n_obb10/weights/best.pt)"
    echo ""
    echo "Output structure:"
    echo "  output_folder/"
    echo "    ├── labels/         # YOLO-OBB format prediction files (.txt)"
    echo "    ├── predictions/    # Raw YOLO prediction output"
    echo "    └── visualizations/ # Visualized predictions on images"
    exit 1
fi

# Parse arguments
INPUT_DIR="$1"
OUTPUT_DIR="$2"
MODEL_PATH="${3:-runs_obb/cattle_y8n_obb10/weights/best.pt}"

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist!"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file '$MODEL_PATH' does not exist!"
    echo "Please check the model path or train a model first."
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/labels"
mkdir -p "$OUTPUT_DIR/predictions" 
mkdir -p "$OUTPUT_DIR/visualizations"

# Get absolute paths
INPUT_DIR=$(realpath "$INPUT_DIR")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
MODEL_PATH=$(realpath "$MODEL_PATH")

echo "=========================================="
echo "CATTLE INFERENCE WITH VISUALIZATION"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_PATH"
echo ""

# Count input images
IMG_COUNT=$(find "$INPUT_DIR" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l)
echo "Found $IMG_COUNT images to process"

if [ "$IMG_COUNT" -eq 0 ]; then
    echo "No images found in input directory!"
    exit 1
fi

echo ""
echo "Step 1: Running YOLO inference..."
echo "----------------------------------------"

# Run YOLO prediction
yolo task=obb mode=predict \
    model="$MODEL_PATH" \
    source="$INPUT_DIR" \
    save_txt=True \
    save_conf=True \
    project="$OUTPUT_DIR/predictions" \
    name=inference_run \
    exist_ok=True

echo "YOLO inference completed!"

# Move generated labels to clean labels folder
echo ""
echo "Step 2: Organizing prediction labels..."
echo "----------------------------------------"

PRED_LABELS_DIR="$OUTPUT_DIR/predictions/inference_run/labels"
if [ -d "$PRED_LABELS_DIR" ]; then
    find "$PRED_LABELS_DIR" -name '*.txt' -exec cp {} "$OUTPUT_DIR/labels/" \;
    LABEL_COUNT=$(find "$OUTPUT_DIR/labels" -name '*.txt' | wc -l)
    echo "Copied $LABEL_COUNT label files to $OUTPUT_DIR/labels/"
else
    echo "Warning: No prediction labels directory found at $PRED_LABELS_DIR"
fi

echo ""
echo "Step 3: Creating visualizations..."
echo "----------------------------------------"

# Create visualization using the viz_check.py script logic
python3 << EOF
import cv2
import os
import glob
import numpy as np
from pathlib import Path

def denorm(poly_norm, w, h):
    """Denormalize polygon coordinates."""
    pts = np.array(poly_norm, dtype=float).reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts.astype(int)

def visualize_predictions(image_dir, label_dir, output_dir):
    """Create visualizations of predictions."""
    viz_count = 0
    
    for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
        stem = Path(label_file).stem
        
        # Find corresponding image
        image_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(image_dir, stem + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break
        
        if image_path is None:
            print(f"Warning: No image found for {stem}")
            continue
            
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load image {image_path}")
            continue
            
        h, w = img.shape[:2]
        
        # Process labels
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                
                # Extract coordinates (skip class_id and optional confidence)
                coords = parts[1:9]
                try:
                    pts = denorm(coords, w, h)
                    
                    # Draw polygon
                    cv2.polylines(img, [pts], True, (0, 255, 0), 3)
                    
                    # Draw corner points
                    for (x, y) in pts:
                        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
                        
                    # Add class label and confidence if available
                    class_id = parts[0]
                    confidence = parts[9] if len(parts) > 9 else "N/A"
                    
                    # Get bounding box for text placement
                    x_min, y_min = pts.min(axis=0)
                    label_text = f"Class: {class_id}"
                    if confidence != "N/A":
                        label_text += f" ({float(confidence):.2f})"
                    
                    cv2.putText(img, label_text, (x_min, y_min - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                              
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse coordinates in {label_file}: {e}")
                    continue
        
        # Save visualization
        output_path = os.path.join(output_dir, stem + "_viz.jpg")
        cv2.imwrite(output_path, img)
        viz_count += 1
        
    return viz_count

# Run visualization
viz_count = visualize_predictions("$INPUT_DIR", "$OUTPUT_DIR/labels", "$OUTPUT_DIR/visualizations")
print(f"Created {viz_count} visualizations")
EOF

VIZ_COUNT=$(find "$OUTPUT_DIR/visualizations" -name '*_viz.jpg' | wc -l)
echo "Generated $VIZ_COUNT visualization images"

echo ""
echo "=========================================="
echo "INFERENCE COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output structure:"
echo "├── labels/         ($LABEL_COUNT files) - YOLO-OBB format predictions"
echo "├── predictions/    - Raw YOLO output"
echo "└── visualizations/ ($VIZ_COUNT files) - Visualized predictions"
echo ""
echo "You can find:"
echo "- Prediction labels in: $OUTPUT_DIR/labels/"
echo "- Visualized results in: $OUTPUT_DIR/visualizations/"
echo ""

# Optional: Show sample of results
if [ "$VIZ_COUNT" -gt 0 ]; then
    echo "Sample visualization files created:"
    find "$OUTPUT_DIR/visualizations" -name '*_viz.jpg' | head -3
fi

echo ""
echo "Inference pipeline completed!"
