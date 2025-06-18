#!/usr/bin/env python3
from ultralytics import YOLO
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on the flake dataset")
    p.add_argument('--data',    type=str, default='/home/sankalp/flake_classification/YOLO_flakes/data.yaml',
                   help='path to data.yaml')
    p.add_argument('--model',   type=str, default='yolov8n.pt',
                   help='pretrained model to start from')
    p.add_argument('--epochs',  type=int, default=50, 
                   help='number of training epochs')
    p.add_argument('--imgsz',   type=int, default=640, 
                   help='image size for training')
    p.add_argument('--batch',   type=int, default=16, 
                   help='batch size')
    p.add_argument('--device',  type=str, default='0', 
                   help='CUDA device or CPU ("cpu")')
    p.add_argument('--project', type=str, default='/home/sankalp/flake_classification/YOLO_flakes/runs/train',
                   help='where to save runs')
    p.add_argument('--name',    type=str, default='flake_yolov8n',
                   help='run name')
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)  # load pretrained
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,            # save best and last weights
        verbose=True
    )

if __name__ == "__main__":
    main()
