#!/bin/bash

YOLO_MODEL_PATH="/home/sankalp/flake_classification/models/best.pt"
CLASSIFIER_MODEL_PATH="/home/sankalp/flake_classification/models/flake_classifier.pth"
VALIDATION_FORMAT="yolo"  # folder or "yolo"
VALIDATION_DIR_FOLDER=""

VALIDATION_DIR_YOLO="/home/sankalp/flake_classification/datasets/YOLO_ready/AllFlakes/"  # should contain images/ and labels/
CLASS_NAMES_FILE="/home/sankalp/flake_classification/classes.txt"

if [ "$VALIDATION_FORMAT" == "yolo" ]; then
    VALIDATION_DIR="$VALIDATION_DIR_YOLO"
else
    VALIDATION_DIR="$VALIDATION_DIR_FOLDER"
fi

OUTPUT_DIR="/home/sankalp/flake_classification/eval_results"
CONFIDENCE_THRESHOLD=0.5
DEVICE="cuda:2"  # or cuda:1, cuda:0, cpu

mkdir -p "$OUTPUT_DIR"

echo "Starting Flake Detection and Classification Evaluation"
echo "=================================================="
echo "YOLO Model: $YOLO_MODEL_PATH"
echo "Classifier: $CLASSIFIER_MODEL_PATH"
echo "Validation Dir: $VALIDATION_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Confidence: $CONFIDENCE_THRESHOLD"
echo "Device: $DEVICE"
echo "=================================================="

# Check if files exist
if [ ! -f "$YOLO_MODEL_PATH" ]; then
    echo "Error: YOLO model not found at $YOLO_MODEL_PATH"
    exit 1
fi

if [ ! -f "$CLASSIFIER_MODEL_PATH" ]; then
    echo "Error: Classifier model not found at $CLASSIFIER_MODEL_PATH"
    exit 1
fi

if [ ! -d "$VALIDATION_DIR" ]; then
    echo "Error: Validation directory not found at $VALIDATION_DIR"
    exit 1
fi

# Run evaluation
if [ "$VALIDATION_FORMAT" == "yolo" ]; then
    echo "Using YOLO format validation"
    python3 /home/sankalp/flake_classification/misc/detect_classify_pipeline.py \
        --yolo_model "$YOLO_MODEL_PATH" \
        --classifier_model "$CLASSIFIER_MODEL_PATH" \
        --val_dir "$VALIDATION_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --confidence "$CONFIDENCE_THRESHOLD" \
        --device "$DEVICE" \
        --format "yolo" \
        --class_names "$CLASS_NAMES_FILE" \
        --save_predictions
else
    echo "Using folder format validation"
    python3 detect_classify_pipeline.py \
        --yolo_model "$YOLO_MODEL_PATH" \
        --classifier_model "$CLASSIFIER_MODEL_PATH" \
        --val_dir "$VALIDATION_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --confidence "$CONFIDENCE_THRESHOLD" \
        --device "$DEVICE" \
        --format "folder" \
        --save_predictions
fi

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -la "$OUTPUT_DIR"
    echo ""
    echo "View results:"
    echo "  - Classification report: $OUTPUT_DIR/classification_report.txt"
    echo "  - Confusion matrix: $OUTPUT_DIR/confusion_matrix.png"
    echo "  - Detailed metrics: $OUTPUT_DIR/metrics.json"
    echo "  - Detection stats: $OUTPUT_DIR/detection_stats.json"
    echo "  - Detailed predictions: $OUTPUT_DIR/detailed_predictions.json"
else
    echo "Evaluation failed. Check the error messages above."
    exit 1
fi