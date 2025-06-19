import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict, Counter
import argparse
from torchvision import transforms
from ultralytics import YOLO
import torch.nn as nn
from transformers import ResNetModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    yolo_model = YOLO(yolo_path)
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

def detect_and_classify_image(image_path, yolo_model, classifier, transform, device, 
                            confidence_threshold=0.5, class_names=None):
    """Process single image through detection and classification pipeline"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None, []
    
    results = yolo_model(image, conf=confidence_threshold, verbose=False)
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Extract bbox region
                bbox_img = image[int(y1):int(y2), int(x1):int(x2)]
                if bbox_img.size == 0:
                    continue
                
                # Convert to PIL and preprocess for classification
                bbox_pil = Image.fromarray(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))
                bbox_tensor = transform(bbox_pil).unsqueeze(0).to(device)
                
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
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'detection_confidence': float(conf),
                    'predicted_class': predicted_class,
                    'class_name': class_name,
                    'classification_confidence': float(class_conf),
                    'all_class_probs': class_confidence.cpu().numpy().tolist()
                })
    
    return image, detections

def load_yolo_labels(label_file):
    """Load YOLO format labels (class x_center y_center width height)"""
    labels = []
    if label_file.exists():
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    labels.append({
                        'class': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
    return labels

def process_validation_set(val_dir, yolo_model, classifier, transform, device, 
                         confidence_threshold=0.5, output_dir="results", 
                         format_type="folder", class_names_file=None):
    """Process validation set and compute metrics
    
    Args:
        format_type: 'folder' or 'yolo'
        class_names_file: path to classes.txt for YOLO format
    """
    
    val_path = Path(val_dir)
    all_predictions = []
    all_ground_truths = []
    detection_stats = defaultdict(int)
    results_per_image = []
    
    if format_type == "folder":
        class_dirs = [d for d in val_path.iterdir() if d.is_dir()]
        class_to_idx = {cls_dir.name: idx for idx, cls_dir in enumerate(sorted(class_dirs))}
        idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
        
        print(f"Processing validation set (folder format): {val_dir}")
        print(f"Found classes: {list(class_to_idx.keys())}")
        
        total_images = 0
        for class_dir in class_dirs:
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            total_images += len(image_files)
        
        processed = 0
        for class_dir in class_dirs:
            true_class = class_to_idx[class_dir.name]
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            for img_path in image_files:
                processed += 1
                if processed % 50 == 0:
                    print(f"Processed {processed}/{total_images} images...")
                
                pred_class, detections = process_single_image(
                    img_path, yolo_model, classifier, transform, device, 
                    confidence_threshold, idx_to_class, detection_stats
                )
                
                if pred_class is not None:
                    all_predictions.append(pred_class)
                    all_ground_truths.append(true_class)
                    
                    results_per_image.append({
                        'image_path': str(img_path),
                        'true_class': true_class,
                        'true_class_name': class_dir.name,
                        'predicted_class': pred_class,
                        'predicted_class_name': idx_to_class[pred_class],
                        'num_detections': len(detections),
                        'detections': detections
                    })
    
    elif format_type == "yolo":
        images_dir = val_path / "images" / "val"
        labels_dir = val_path / "labels" / "val"
        
        if not images_dir.exists() or not labels_dir.exists():
            raise ValueError(f"YOLO format requires {images_dir} and {labels_dir} directories")
        
        if class_names_file and Path(class_names_file).exists():
            with open(class_names_file, 'r') as f:
                class_names = [line.strip() for line in f]
        else:
            class_names = ["0", "1", "2", "3"]
        
        idx_to_class = {idx: name for idx, name in enumerate(class_names)}
        
        print(f"Processing validation set (YOLO format): {val_dir}")
        print(f"Classes: {class_names}")
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        total_images = len(image_files)
        
        for idx, img_path in enumerate(image_files):
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{total_images} images...")
            
            label_file = labels_dir / (img_path.stem + ".txt")
            gt_labels = load_yolo_labels(label_file)
            
            if not gt_labels:
                continue
            
            true_class = gt_labels[0]['class']
            pred_class, detections = process_single_image(
                img_path, yolo_model, classifier, transform, device, 
                confidence_threshold, idx_to_class, detection_stats
            )
            
            if pred_class is not None:
                all_predictions.append(pred_class)
                all_ground_truths.append(true_class)
                
                results_per_image.append({
                    'image_path': str(img_path),
                    'true_class': true_class,
                    'true_class_name': idx_to_class.get(true_class, str(true_class)),
                    'predicted_class': pred_class,
                    'predicted_class_name': idx_to_class[pred_class],
                    'num_detections': len(detections),
                    'detections': detections,
                    'ground_truth_boxes': gt_labels
                })
    
    return all_predictions, all_ground_truths, detection_stats, results_per_image, idx_to_class

def process_single_image(img_path, yolo_model, classifier, transform, device, 
                        confidence_threshold, idx_to_class, detection_stats):
    """Process a single image and return prediction"""
    
    image, detections = detect_and_classify_image(
        img_path, yolo_model, classifier, transform, device, 
        confidence_threshold, idx_to_class
    )
    
    if image is None:
        return None, []
    
    detection_stats['total_images'] += 1
    detection_stats['total_detections'] += len(detections)
    
    if len(detections) == 0:
        detection_stats['no_detections'] += 1
        return None, []
    elif len(detections) == 1:
        detection_stats['single_detection'] += 1
        pred_class = detections[0]['predicted_class']
    else:
        detection_stats['multiple_detections'] += 1
        best_det = max(detections, key=lambda x: x['classification_confidence'])
        pred_class = best_det['predicted_class']
    
    return pred_class, detections

def compute_metrics(predictions, ground_truths, class_names, output_dir):
    """Compute and save classification metrics"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    accuracy = accuracy_score(ground_truths, predictions)
    report = classification_report(
        ground_truths, predictions, 
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
    
    cm = confusion_matrix(ground_truths, predictions)
    with open(output_path / "classification_report.txt", "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(ground_truths, predictions, target_names=class_names, zero_division=0))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    metrics_dict = {
        'overall_accuracy': accuracy,
        'per_class_metrics': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    return metrics_dict

def save_detection_stats(detection_stats, output_dir):
    """Save detection statistics"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    stats_summary = {
        'total_images_processed': detection_stats['total_images'],
        'total_detections': detection_stats['total_detections'],
        'images_with_no_detections': detection_stats['no_detections'],
        'images_with_single_detection': detection_stats['single_detection'],
        'images_with_multiple_detections': detection_stats['multiple_detections'],
        'average_detections_per_image': detection_stats['total_detections'] / max(detection_stats['total_images'], 1),
        'detection_rate': (detection_stats['total_images'] - detection_stats['no_detections']) / max(detection_stats['total_images'], 1)
    }
    
    with open(output_path / "detection_stats.json", "w") as f:
        json.dump(stats_summary, f, indent=2)
    
    print("\n" + "="*50)
    print("DETECTION STATISTICS")
    print("="*50)
    for key, value in stats_summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    return stats_summary

def main():
    parser = argparse.ArgumentParser(description="Flake Detection and Classification Pipeline")
    parser.add_argument('--yolo_model', type=str, required=True, 
                       help='Path to YOLO detection model (.pt file)')
    parser.add_argument('--classifier_model', type=str, required=True,
                       help='Path to classification model (.pth file)')
    parser.add_argument('--val_dir', type=str, required=True,
                       help='Path to validation directory with class subdirectories')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda:0, etc.)')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save detailed predictions for each image')
    parser.add_argument('--format', type=str, choices=['folder', 'yolo'], default='folder',
                       help='Validation data format (folder or yolo)')
    parser.add_argument('--class_names', type=str, 
                       help='Path to class names file for YOLO format')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    print("Loading models...")
    yolo_model, classifier, class_to_idx = load_models(
        args.yolo_model, args.classifier_model, device
    )
    
    transform = get_classification_transform()
    print("Processing validation set...")
    predictions, ground_truths, detection_stats, results_per_image, idx_to_class = process_validation_set(
        args.val_dir, yolo_model, classifier, transform, device, args.confidence, 
        args.output_dir, args.format, args.class_names
    )
    
    if len(predictions) == 0:
        print("No valid predictions found. Check your models and data.")
        return
    
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print("Computing metrics...")
    metrics = compute_metrics(predictions, ground_truths, class_names, args.output_dir)
    det_stats = save_detection_stats(detection_stats, args.output_dir)
    if args.save_predictions:
        output_path = Path(args.output_dir)
        with open(output_path / "detailed_predictions.json", "w") as f:
            json.dump(results_per_image, f, indent=2)
        print(f"Detailed predictions saved to {output_path / 'detailed_predictions.json'}")
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Detection Rate: {det_stats['detection_rate']:.4f}")
    print(f"Average Detections per Image: {det_stats['average_detections_per_image']:.2f}")
    
    print("\nPer-class Accuracy:")
    for class_name in class_names:
        if class_name in metrics['per_class_metrics']:
            f1 = metrics['per_class_metrics'][class_name]['f1-score']
            precision = metrics['per_class_metrics'][class_name]['precision']
            recall = metrics['per_class_metrics'][class_name]['recall']
            print(f"  {class_name}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()