#!/bin/bash

# Updated paths to match your dataset structure
DATA_PATH="/home/sankalp/flake_classification/datasets/YOLO_ready/AllFlakes/dataset.yaml"
MODEL="yolo11x.pt"
RUN_DIR="/home/sankalp/flake_classification/runs/detect/flake_yolo11x"

# Validate dataset exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Dataset YAML not found at $DATA_PATH"
    exit 1
fi

echo "Starting YOLO11x training..."
echo "Dataset: $DATA_PATH"
echo "Model: $MODEL"
echo "Output: $RUN_DIR"

yolo detect train \
    model=$MODEL \
    data=$DATA_PATH \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    lr0=0.01 \
    patience=50 \
    device=1 \
    workers=4 \
    project=$(dirname "$RUN_DIR") \
    name=$(basename "$RUN_DIR") \
    save=true \
    plots=true \
    val=true \
    cache=false \
    verbose=true \
    pretrained=true \
    optimizer=SGD \
    save_period=10

echo "Training completed! Results saved to: $RUN_DIR"