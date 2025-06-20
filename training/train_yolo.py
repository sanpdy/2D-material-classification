#!/usr/bin/env python3
from ultralytics import YOLO
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on the flake dataset")
    p.add_argument('--data', type=str, 
                   default='/home/sankalp/flake_classification/datasets/YOLO_ready/AllFlakes/dataset.yaml',
                   help='path to dataset.yaml')
    p.add_argument('--model', type=str, default='yolov8n.pt',
                   help='pretrained model (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    p.add_argument('--epochs', type=int, default=100, 
                   help='number of training epochs')
    p.add_argument('--imgsz', type=int, default=640, 
                   help='image size for training')
    p.add_argument('--batch', type=int, default=16, 
                   help='batch size (-1 for auto)')
    p.add_argument('--lr0', type=float, default=0.01,
                   help='initial learning rate')
    p.add_argument('--patience', type=int, default=50,
                   help='early stopping patience (epochs)')
    p.add_argument('--device', type=str, default='0', 
                   help='CUDA device or CPU ("cpu")')
    p.add_argument('--workers', type=int, default=8,
                   help='number of data loading workers')
    p.add_argument('--project', type=str, 
                   default='/home/sankalp/flake_classification/runs/detect',
                   help='where to save runs')
    p.add_argument('--name', type=str, default='flake_detection',
                   help='run name')
    p.add_argument('--pretrained', action='store_true', default=True,
                   help='use pretrained weights')
    p.add_argument('--optimizer', type=str, default='SGD',
                   choices=['SGD', 'Adam', 'AdamW'], help='optimizer')
    p.add_argument('--resume', type=str, default=None,
                   help='resume training from checkpoint')
    p.add_argument('--cache', action='store_true',
                   help='cache images for faster training')
    
    return p.parse_args()

def validate_paths(args):
    """Validate that required paths exist"""
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    
    Path(args.project).mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset: {data_path}")
    print(f"Output: {Path(args.project) / args.name}")

def main():
    args = parse_args()
    
    try:
        validate_paths(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"🚀 Loading model: {args.model}")
    model = YOLO(args.model)
    
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    
    # Training config
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'lr0': args.lr0,
        'patience': args.patience,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'pretrained': args.pretrained,
        'optimizer': args.optimizer,
        'verbose': True,
        'save': True,
        'plots': True,        # Save training plots
        'val': True,          # Validate during training
        'save_period': 10,    # Save checkpoint every N epochs
    }
    
    if args.resume:
        train_args['resume'] = args.resume
    if args.cache:
        train_args['cache'] = True
    
    print("\nStarting training...")
    try:
        results = model.train(**train_args)
        print("\n✅ Training completed successfully!")
        print(f"📈 Best results saved to: {Path(args.project) / args.name}")
        
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"Final mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.3f}")
            print(f"Final mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
            
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()