DATA_PATH="/home/sankalp/flake_classification/YOLO_flakes/data.yaml"
MODEL="yolo11x.pt"
RUN_DIR="/home/sankalp/flake_classification/YOLO_flakes/runs/train/flake_yolov11x"

yolo detect train \
    model=$MODEL \
    data=$DATA_PATH \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    device=0 \
    project=$(dirname "$RUN_DIR") \
    name=$(basename "$RUN_DIR") \
    save=true \
    verbose=true
