"""
Convert YOLO dataset format to validation format for pipeline evaluation
This script reads your YOLO dataset and creates ground truth annotations
"""

import os
import json
import yaml
from pathlib import Path
import argparse
from PIL import Image
import shutil


def parse_yolo_annotation(label_file, img_width, img_height):
    """
    Parse YOLO format annotation file
    
    Args:
        label_file: Path to YOLO .txt annotation file
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        List of annotation dictionaries
    """
    annotations = []
    
    if not os.path.exists(label_file):
        return annotations
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            x1 = int((x_center - width/2) * img_width)
            y1 = int((y_center - height/2) * img_height)
            x2 = int((x_center + width/2) * img_width)
            y2 = int((y_center + height/2) * img_height)
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            if x2 > x1 and y2 > y1:
                annotation = {
                    "bbox": [x1, y1, x2, y2],
                    "class": str(class_id),
                    "confidence": 1.0
                }
                annotations.append(annotation)
    
    return annotations


def load_yolo_dataset_config(dataset_yaml):
    """Load YOLO dataset configuration"""
    with open(dataset_yaml, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_validation_ground_truth(dataset_yaml, output_file, split='val', class_mapping=None):
    """
    Create ground truth file from YOLO dataset
    
    Args:
        dataset_yaml: Path to YOLO dataset.yaml file
        output_file: Output path for ground truth JSON
        split: Dataset split to use ('train', 'val', 'test')
        class_mapping: Optional mapping from YOLO class IDs to layer numbers
    """
    config = load_yolo_dataset_config(dataset_yaml)
    dataset_root = Path(dataset_yaml).parent
    
    if split in config:
        images_dir = dataset_root / config[split]
    else:
        print(f"Warning: Split '{split}' not found in dataset config. Available splits: {list(config.keys())}")
        return
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    class_names = config.get('names', {})
    print(f"Class names from dataset: {class_names}")
    
    if class_mapping is None:
        class_mapping = {i: str(i+1) for i in range(len(class_names))}  # 0->1, 1->2, etc.
    
    print(f"Using class mapping: {class_mapping}")
    
    labels_dir = images_dir.parent / 'labels' / images_dir.name
    if not labels_dir.exists():
        labels_dir = str(images_dir).replace('images', 'labels')
        labels_dir = Path(labels_dir)
    
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    
    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        return
    
    ground_truth = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    processed_count = 0
    for image_file in images_dir.iterdir():
        if image_file.suffix.lower() in image_extensions:
            label_file = labels_dir / (image_file.stem + '.txt')
            
            try:
                with Image.open(image_file) as img:
                    img_width, img_height = img.size
                
                annotations = parse_yolo_annotation(label_file, img_width, img_height)
                
                for ann in annotations:
                    original_class = int(ann['class'])
                    if original_class in class_mapping:
                        ann['class'] = class_mapping[original_class]
                    else:
                        print(f"Warning: Class {original_class} not in mapping, keeping as is")
                
                # Create ground truth entry
                gt_entry = {
                    "filename": image_file.name,
                    "image_path": str(image_file),
                    "width": img_width,
                    "height": img_height,
                    "annotations": annotations
                }
                
                ground_truth.append(gt_entry)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} images...")
                    
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    total_images = len(ground_truth)
    total_annotations = sum(len(item['annotations']) for item in ground_truth)
    images_with_annotations = sum(1 for item in ground_truth if len(item['annotations']) > 0)
    
    print(f"\nGround truth file created: {output_path}")
    print(f"Total images: {total_images}")
    print(f"Images with annotations: {images_with_annotations}")
    print(f"Total annotations: {total_annotations}")
    print(f"Average annotations per image: {total_annotations/total_images:.2f}")
    
    class_counts = {}
    for item in ground_truth:
        for ann in item['annotations']:
            class_name = ann['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\nClass distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  Layer {class_name}: {count} annotations")
    
    return output_path


def copy_validation_images(ground_truth_file, output_dir):
    """
    Copy validation images to a separate directory for easier access
    
    Args:
        ground_truth_file: Path to ground truth JSON file
        output_dir: Directory to copy images to
    """
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying validation images to: {output_path}")
    
    for item in ground_truth:
        source_path = Path(item['image_path'])
        dest_path = output_path / item['filename']
        
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
        else:
            print(f"Warning: Source image not found: {source_path}")
    
    print(f"Copied {len(ground_truth)} images to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to validation format")
    parser.add_argument('--dataset-yaml', required=True, 
                       help='Path to YOLO dataset.yaml file')
    parser.add_argument('--output-gt', default='validation_ground_truth.json',
                       help='Output path for ground truth JSON file')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--copy-images', 
                       help='Directory to copy validation images to')
    parser.add_argument('--class-mapping', 
                       help='JSON file with class mapping (YOLO ID -> Layer number)')
    
    args = parser.parse_args()
    
    class_mapping = None
    if args.class_mapping:
        with open(args.class_mapping, 'r') as f:
            class_mapping = json.load(f)
        class_mapping = {int(k): v for k, v in class_mapping.items()}
        print(f"Loaded class mapping: {class_mapping}")
    
    gt_file = create_validation_ground_truth(
        args.dataset_yaml, 
        args.output_gt, 
        args.split,
        class_mapping
    )
    
    if args.copy_images and gt_file:
        copy_validation_images(gt_file, args.copy_images)


if __name__ == "__main__":
    main()