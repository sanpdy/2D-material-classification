import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json
import argparse
from torchvision import transforms
from ultralytics import YOLO
import torch.nn as nn
from transformers import ResNetModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FlakeLayerClassifier(nn.Module):
    def __init__(self, num_materials, material_dim, num_classes=4, dropout_prob=0.1, freeze_cnn=False):
        super().__init__()
        self.cnn = ResNetModel.from_pretrained("microsoft/resnet-18")
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        img_feat_dim = self.cnn.config.hidden_sizes[-1]
        self.material_embedding = nn.Embedding(num_materials, material_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc_img = nn.Sequential(
            nn.Linear(img_feat_dim, img_feat_dim),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(img_feat_dim, num_classes)
        )
        combined_dim = img_feat_dim + material_dim
        self.fc_comb = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(combined_dim, num_classes)
        )

    def forward(self, pixel_values, material=None):
        outputs = self.cnn(pixel_values=pixel_values)
        img_feats = outputs.pooler_output
        img_feats = img_feats.view(img_feats.size(0), -1)
        if material is None:
            return self.fc_img(img_feats)
        material_embeds = self.material_embedding(material)
        combined_feats = torch.cat((img_feats, material_embeds), dim=1)
        return self.fc_comb(combined_feats)

def load_models(yolo_path, classifier_path, device):
    """Load YOLO detection and classification models"""
    print("Loading YOLO model...")
    yolo_model = YOLO(yolo_path)
    
    print("Loading classification model...")
    checkpoint = torch.load(classifier_path, map_location=device)
    class_to_idx = checkpoint['class_to_idx']
    num_classes = len(class_to_idx)
    
    classifier = FlakeLayerClassifier(
        num_materials=num_classes,
        material_dim=64,
        num_classes=num_classes,
        dropout_prob=0.1,
        freeze_cnn=False
    )
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.to(device)
    classifier.eval()
    
    return yolo_model, classifier, class_to_idx

def get_classification_transform():
    """Transform for classification input"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_class_names(class_names_file):
    """Load class names from file"""
    if class_names_file and Path(class_names_file).exists():
        with open(class_names_file, 'r') as f:
            return [line.strip() for line in f]
    return None

def load_ground_truth_annotations(image_path, labels_dir=None):
    """Load ground truth annotations from YOLO polygon txt file,
    convert to pixel bbox, and return list of dicts with class_id, bbox, and poly."""
    image_path = Path(image_path)
    # Infer labels directory if not provided
    if labels_dir:
        labels_dir = Path(labels_dir)
    else:
        parent = image_path.parent
        if 'images' in parent.parts:
            parts = list(parent.parts)
            parts[parts.index('images')] = 'labels'
            labels_dir = Path(*parts)
        else:
            raise ValueError("Cannot infer labels_dir; please pass --labels_dir")

    txt_file = labels_dir / f"{image_path.stem}.txt"
    if not txt_file.exists():
        print(f"Warning: No ground-truth file at {txt_file}")
        return []

    # Load image to get dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to load image for dimensions: {image_path}")
    h, w = img.shape[:2]

    ground_truth = []
    with open(txt_file, 'r') as f:
        for ln, line in enumerate(f, start=1):
            parts = line.strip().split()
            # Must have class + even number of coords
            if len(parts) < 3 or (len(parts)-1) % 2 != 0:
                print(f"  Skipping invalid line {ln}: {line.strip()}")
                continue

            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            xs = coords[0::2]
            ys = coords[1::2]
            # Convert normalized to pixels
            xs_px = [int(x * w) for x in xs]
            ys_px = [int(y * h) for y in ys]
            # Rect hull
            x1, x2 = min(xs_px), max(xs_px)
            y1, y2 = min(ys_px), max(ys_px)

            ground_truth.append({
                'class_id': class_id,
                'bbox': [x1, y1, x2, y2],
                'poly': list(zip(xs_px, ys_px))
            })
    print(f"Loaded {len(ground_truth)} GT annotations from {txt_file.name}")
    return ground_truth


def visualize_results(image, detections, ground_truth=None,
                      class_names=None, save_path=None, show=True):
    """Draw predictions vs. ground-truth (rect + polygon) side-by-side."""
    # Setup subplots
    if ground_truth:
        fig, (ax_pred, ax_gt) = plt.subplots(1, 2, figsize=(20, 10))
    else:
        fig, ax_pred = plt.subplots(1, 1, figsize=(12, 8))
        ax_gt = None

    # Convert BGR→RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # COLORS
    colors = ['red','blue','green','orange','purple','brown','pink','gray','cyan','magenta']

    # ---- PREDICTIONS ----
    ax_pred.imshow(img_rgb)
    ax_pred.set_title(f"Predictions ({len(detections)} boxes)")
    for det in detections:
        x1,y1,x2,y2 = det['bbox']
        cls = det['predicted_class'] % len(colors)
        col = colors[cls]
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor=col, facecolor='none')
        ax_pred.add_patch(rect)
        label = det['class_name']
        confs = f"{det['detection_confidence']:.2f}/{det['classification_confidence']:.2f}"
        ax_pred.text(x1, y1-8, f"{label} {confs}",
                     color='white', backgroundcolor=col, fontsize=9, weight='bold')
    ax_pred.axis('off')

    # ---- GROUND TRUTH ----
    if ax_gt:
        ax_gt.imshow(img_rgb)
        ax_gt.set_title(f"Ground Truth ({len(ground_truth)} annots)")
        for gt in ground_truth:
            x1,y1,x2,y2 = gt['bbox']
            cid = gt['class_id'] % len(colors)
            col = colors[cid]
            # rect
            ax_gt.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                             linewidth=2, edgecolor=col, facecolor='none'))
            # polygon outline
            poly = gt['poly'] + [gt['poly'][0]]
            xs, ys = zip(*poly)
            ax_gt.plot(xs, ys, linestyle='--', color=col, linewidth=1)
            # label
            name = class_names[cid] if class_names and cid < len(class_names) else f"Class {gt['class_id']}"
            ax_gt.text(x1, y1-8, name,
                       color='white', backgroundcolor=col, fontsize=9, weight='bold')
        ax_gt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Visualization {'saved to '+str(save_path) if save_path else 'complete'}")

def detect_and_classify_single_image(image_path, yolo_model, classifier, transform, device, 
                                   confidence_threshold=0.5, class_names=None):
    """Process single image through detection and classification pipeline"""
    print(f"Processing image: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, []
    
    print(f"Image shape: {image.shape}")
    
    # Run YOLO detection
    print("Running YOLO detection...")
    results = yolo_model(image, conf=confidence_threshold, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            print(f"Found {len(boxes)} detections")
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                print(f"Detection {i+1}: bbox=[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], conf={conf:.3f}")
                
                # Extract bbox region
                bbox_img = image[int(y1):int(y2), int(x1):int(x2)]
                if bbox_img.size == 0:
                    print(f"  Warning: Empty bbox region")
                    continue
                
                # Convert to PIL and preprocess for classification
                bbox_pil = Image.fromarray(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))
                bbox_tensor = transform(bbox_pil).unsqueeze(0).to(device)
                
                # Run classification
                with torch.no_grad():
                    outputs = classifier(bbox_tensor)
                    _, predicted = torch.max(outputs, 1)
                    class_confidence = torch.softmax(outputs, dim=1)[0]
                    
                predicted_class = predicted.item()
                class_conf = class_confidence[predicted_class].item()
                
                if class_names:
                    class_name = class_names[predicted_class]
                else:
                    class_name = str(predicted_class)
                
                print(f"  Predicted class: {class_name} (confidence: {class_conf:.3f})")
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'detection_confidence': float(conf),
                    'predicted_class': predicted_class,
                    'class_name': class_name,
                    'classification_confidence': float(class_conf),
                    'all_class_probs': class_confidence.cpu().numpy().tolist()
                })
        else:
            print("No detections found")
    
    return image, detections

def load_ground_truth_annotations(image_path, labels_dir=None):
    """Load ground truth annotations from YOLO polygon txt file,
    convert to pixel bbox, and return list of dicts with class_id, bbox, and poly."""
    image_path = Path(image_path)
    # Infer labels directory if not provided
    if labels_dir:
        labels_dir = Path(labels_dir)
    else:
        parent = image_path.parent
        if 'images' in parent.parts:
            parts = list(parent.parts)
            parts[parts.index('images')] = 'labels'
            labels_dir = Path(*parts)
        else:
            raise ValueError("Cannot infer labels_dir; please pass --labels_dir")

    txt_file = labels_dir / f"{image_path.stem}.txt"
    if not txt_file.exists():
        print(f"Warning: No ground-truth file at {txt_file}")
        return []

    # Load image to get dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to load image for dimensions: {image_path}")
    h, w = img.shape[:2]

    ground_truth = []
    with open(txt_file, 'r') as f:
        for ln, line in enumerate(f, start=1):
            parts = line.strip().split()
            # Must have class + even number of coords
            if len(parts) < 3 or (len(parts)-1) % 2 != 0:
                print(f"  Skipping invalid line {ln}: {line.strip()}")
                continue

            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            xs = coords[0::2]
            ys = coords[1::2]
            # Convert normalized to pixels
            xs_px = [int(x * w) for x in xs]
            ys_px = [int(y * h) for y in ys]
            # Rect hull
            x1, x2 = min(xs_px), max(xs_px)
            y1, y2 = min(ys_px), max(ys_px)

            ground_truth.append({
                'class_id': class_id,
                'bbox': [x1, y1, x2, y2],
                'poly': list(zip(xs_px, ys_px))
            })
    print(f"Loaded {len(ground_truth)} GT annotations from {txt_file.name}")
    return ground_truth


def visualize_results(image, detections, ground_truth=None,
                      class_names=None, save_path=None, show=True):
    """Draw predictions vs. ground-truth (rect + polygon) side-by-side."""
    # Setup subplots
    if ground_truth:
        fig, (ax_pred, ax_gt) = plt.subplots(1, 2, figsize=(20, 10))
    else:
        fig, ax_pred = plt.subplots(1, 1, figsize=(12, 8))
        ax_gt = None

    # Convert BGR→RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # COLORS
    colors = ['red','blue','green','orange','purple','brown','pink','gray','cyan','magenta']

    # ---- PREDICTIONS ----
    ax_pred.imshow(img_rgb)
    ax_pred.set_title(f"Predictions ({len(detections)} boxes)")
    for det in detections:
        x1,y1,x2,y2 = det['bbox']
        cls = det['predicted_class'] % len(colors)
        col = colors[cls]
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor=col, facecolor='none')
        ax_pred.add_patch(rect)
        label = det['class_name']
        confs = f"{det['detection_confidence']:.2f}/{det['classification_confidence']:.2f}"
        ax_pred.text(x1, y1-8, f"{label} {confs}",
                     color='white', backgroundcolor=col, fontsize=9, weight='bold')
    ax_pred.axis('off')

    # ---- GROUND TRUTH ----
    if ax_gt:
        ax_gt.imshow(img_rgb)
        ax_gt.set_title(f"Ground Truth ({len(ground_truth)} annots)")
        for gt in ground_truth:
            x1,y1,x2,y2 = gt['bbox']
            cid = gt['class_id'] % len(colors)
            col = colors[cid]
            # rect
            ax_gt.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                             linewidth=2, edgecolor=col, facecolor='none'))
            # polygon outline
            poly = gt['poly'] + [gt['poly'][0]]
            xs, ys = zip(*poly)
            ax_gt.plot(xs, ys, linestyle='--', color=col, linewidth=1)
            # label
            name = class_names[cid] if class_names and cid < len(class_names) else f"Class {gt['class_id']}"
            ax_gt.text(x1, y1-8, name,
                       color='white', backgroundcolor=col, fontsize=9, weight='bold')
        ax_gt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Visualization {'saved to '+str(save_path) if save_path else 'complete'}")


def save_results(image_path, detections, ground_truth, output_dir, class_names=None):
    """Save detection and classification results including ground truth comparison"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create results dictionary
    results = {
        'image_path': str(image_path),
        'image_name': Path(image_path).name,
        'num_detections': len(detections),
        'num_ground_truth': len(ground_truth),
        'detections': detections,
        'ground_truth': ground_truth
    }
    
    # Add class names mapping if available
    if class_names:
        results['class_names'] = class_names
    
    # Save detailed results as JSON
    results_file = output_path / f"{Path(image_path).stem}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("DETECTION SUMMARY")
    print("="*50)
    print(f"Image: {Path(image_path).name}")
    print(f"Total detections: {len(detections)}")
    print(f"Ground truth annotations: {len(ground_truth)}")
    
    if ground_truth:
        print("\nGround Truth:")
        for i, gt in enumerate(ground_truth, 1):
            class_name = class_names[gt['class_id']] if class_names and gt['class_id'] < len(class_names) else f"Class {gt['class_id']}"
            print(f"  GT {i}: {class_name} at {gt['bbox']}")
    
    if detections:
        print("\nPredicted Results:")
        for i, det in enumerate(detections, 1):
            print(f"Detection {i}:")
            print(f"  Class: {det['class_name']}")
            print(f"  Detection confidence: {det['detection_confidence']:.3f}")
            print(f"  Classification confidence: {det['classification_confidence']:.3f}")
            print(f"  Bounding box: {det['bbox']}")
            
            # Show top class probabilities
            if class_names and len(det['all_class_probs']) > 1:
                print("  All class probabilities:")
                for j, prob in enumerate(det['all_class_probs']):
                    print(f"    {class_names[j]}: {prob:.3f}")
            print()
    else:
        print("No detections found!")
    
    return results_file

def main():
    parser = argparse.ArgumentParser(description="Single Image Flake Detection and Classification with Ground Truth")
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--yolo_model', type=str, required=True, 
                       help='Path to YOLO detection model (.pt file)')
    parser.add_argument('--classifier_model', type=str, required=True,
                       help='Path to classification model (.pth file)')
    parser.add_argument('--labels_dir', type=str,
                       help='Path to directory containing ground truth labels (will try to infer if not provided)')
    parser.add_argument('--output_dir', type=str, default='single_image_results',
                       help='Output directory for results')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda:0, etc.)')
    parser.add_argument('--class_names', type=str, 
                       help='Path to class names file')
    parser.add_argument('--save_viz', action='store_true',
                       help='Save visualization image')
    parser.add_argument('--show_viz', action='store_true',
                       help='Show visualization (requires display)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found at {args.image}")
        return
    
    # Check if models exist
    if not Path(args.yolo_model).exists():
        print(f"Error: YOLO model not found at {args.yolo_model}")
        return
    
    if not Path(args.classifier_model).exists():
        print(f"Error: Classifier model not found at {args.classifier_model}")
        return
    
    # Load class names if provided
    class_names = load_class_names(args.class_names)
    if class_names:
        print(f"Loaded class names: {class_names}")
    
    # Load ground truth annotations
    ground_truth = load_ground_truth_annotations(args.image, args.labels_dir)
    
    # Load models
    yolo_model, classifier, class_to_idx = load_models(
        args.yolo_model, args.classifier_model, device
    )
    print("Models loaded successfully!")
    
    # Get classification transform
    transform = get_classification_transform()
    
    # Process image
    image, detections = detect_and_classify_single_image(
        args.image, yolo_model, classifier, transform, device, 
        args.confidence, class_names
    )
    
    if image is None:
        print("Failed to process image")
        return
    
    # Save results
    results_file = save_results(args.image, detections, ground_truth, args.output_dir, class_names)
    
    # Create visualization if requested
    if args.save_viz or args.show_viz:
        viz_path = None
        if args.save_viz:
            viz_path = Path(args.output_dir) / f"{Path(args.image).stem}_visualization.png"
        
        visualize_results(image, detections, ground_truth, class_names, viz_path, args.show_viz)
    
    print(f"\nProcessing complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()