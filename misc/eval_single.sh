#!/bin/bash

# Configuration
YOLO_MODEL_PATH="/home/sankalp/flake_classification/models/best.pt"
CLASSIFIER_MODEL_PATH="/home/sankalp/flake_classification/models/flake_classifier.pth"
CLASS_NAMES_FILE="/home/sankalp/flake_classification/classes.txt"
OUTPUT_DIR="/home/sankalp/flake_classification/single_image_results"
CONFIDENCE_THRESHOLD=0.95
DEVICE="cuda:2"  # or cuda:1, cuda:0, cpu

# Check if image path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_path> [options]"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/image.jpg"
    echo "  $0 /path/to/image.jpg --save_viz --show_viz"
    echo "  $0 /path/to/image.jpg --confidence 0.3"
    echo ""
    echo "Available options:"
    echo "  --save_viz    Save visualization image"
    echo "  --show_viz    Show visualization (requires display)"
    echo "  --confidence  Detection confidence threshold (default: 0.5)"
    echo "  --output_dir  Output directory (default: $OUTPUT_DIR)"
    echo "  --device      Device to use (default: $DEVICE)"
    exit 1
fi

IMAGE_PATH="$1"
shift  # Remove first argument (image path)

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image not found at $IMAGE_PATH"
    exit 1
fi

# Check if models exist
if [ ! -f "$YOLO_MODEL_PATH" ]; then
    echo "Error: YOLO model not found at $YOLO_MODEL_PATH"
    exit 1
fi

if [ ! -f "$CLASSIFIER_MODEL_PATH" ]; then
    echo "Error: Classifier model not found at $CLASSIFIER_MODEL_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Single Image Flake Detection and Classification"
echo "=============================================="
echo "Image: $IMAGE_PATH"
echo "YOLO Model: $YOLO_MODEL_PATH"
echo "Classifier: $CLASSIFIER_MODEL_PATH"
echo "Class Names: $CLASS_NAMES_FILE"
echo "Output Dir: $OUTPUT_DIR"
echo "Confidence: $CONFIDENCE_THRESHOLD"
echo "Device: $DEVICE"
echo "=============================================="

# Parse additional arguments for visualization and other options
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --save_viz)
            EXTRA_ARGS="$EXTRA_ARGS --save_viz"
            shift
            ;;
        --show_viz)
            EXTRA_ARGS="$EXTRA_ARGS --show_viz"
            shift
            ;;
        --confidence)
            CONFIDENCE_THRESHOLD="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run the single image detection and classification
python3 /home/sankalp/flake_classification/misc/single_image_pipeline.py \
    --image "$IMAGE_PATH" \
    --yolo_model "$YOLO_MODEL_PATH" \
    --classifier_model "$CLASSIFIER_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --confidence "$CONFIDENCE_THRESHOLD" \
    --device "$DEVICE" \
    --class_names "$CLASS_NAMES_FILE" \
    $EXTRA_ARGS

# Check if processing completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Processing completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"
    echo ""
    echo "View results:"
    IMAGE_NAME=$(basename "$IMAGE_PATH" | cut -d. -f1)
    echo "  - Detailed results: $OUTPUT_DIR/${IMAGE_NAME}_results.json"
    if [ -f "$OUTPUT_DIR/${IMAGE_NAME}_visualization.png" ]; then
        echo "  - Visualization: $OUTPUT_DIR/${IMAGE_NAME}_visualization.png"
    fi
else
    echo "Processing failed. Check the error messages above."
    exit 1
fi