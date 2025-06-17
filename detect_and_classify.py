import os
import sys
import glob
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEG_MODEL_PATH  = "/home/sankalp/flake_classification/models/best.pt"
CLS_MODEL_PATH  = "/home/sankalp/flake_classification/models/resnet18_layer_classifier2.pth"
CONF_THRESH     = 0.5   # min detection confidence
MIN_BOX_AREA    = 1200   # min bbox area (px²) to keep
IOU_THRESH      = 0.5    # IoU threshold for detection matching

print(f"Using device: {DEVICE}")

test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225]),
])

print("Loading models...")
det_model = YOLO(SEG_MODEL_PATH)

clf = resnet18(weights=None)
clf.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(clf.fc.in_features, 4)
)
ckpt = torch.load(CLS_MODEL_PATH, map_location=DEVICE)
clf.load_state_dict(ckpt['model_state_dict'])
clf.to(DEVICE).eval()
print("Models loaded successfully!")

class PerformanceTracker:
    """Track and calculate model performance metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Detection metrics
        self.detection_stats = {
            'total_predictions': 0,
            'total_ground_truth': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'confidence_scores': [],
            'processing_times': []
        }
        
        # Classification metrics  
        self.classification_stats = {
            'predictions': [],
            'ground_truth': [],
            'confidence_scores': [],
            'per_layer_correct': {1: 0, 2: 0, 3: 0, 4: 0},
            'per_layer_total': {1: 0, 2: 0, 3: 0, 4: 0}
        }
        
        # Model quality metrics
        self.quality_metrics = {
            'low_confidence_detections': 0,  # < 0.7
            'high_confidence_errors': 0,     # > 0.8 but wrong
            'detection_consistency': [],     # Same flake detected across similar images
        }
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes [x1,y1,x2,y2]"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
            
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_detection_metrics(self, predictions, ground_truth, processing_time):
        """Update detection performance metrics"""
        self.detection_stats['total_predictions'] += len(predictions)
        self.detection_stats['total_ground_truth'] += len(ground_truth)
        self.detection_stats['processing_times'].append(processing_time)
        
        # Convert ground truth format: [x,y,w,h] -> [x1,y1,x2,y2]
        gt_boxes = []
        for gt in ground_truth:
            x, y, w, h = gt['bbox']
            gt_boxes.append([x, y, x+w, y+h])
        
        pred_boxes = [pred['bbox'] for pred in predictions]
        pred_confidences = [pred.get('detection_confidence', 0.5) for pred in predictions]
        
        # Match predictions to ground truth using IoU
        matched_gt = set()
        matched_pred = set()
        
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                    
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= IOU_THRESH:
                # True positive
                self.detection_stats['true_positives'] += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
            else:
                # False positive
                self.detection_stats['false_positives'] += 1
            
            # Track confidence scores
            self.detection_stats['confidence_scores'].append(pred_confidences[i])
            
            # Quality metrics
            if pred_confidences[i] < 0.7:
                self.quality_metrics['low_confidence_detections'] += 1
        
        # False negatives (unmatched ground truth)
        self.detection_stats['false_negatives'] += len(gt_boxes) - len(matched_gt)
    
    def update_classification_metrics(self, predictions, ground_truth):
        gt_dict = {}
        for gt in ground_truth:
            x, y, w, h = gt['bbox']
            gt_box = [x, y, x+w, y+h]
            gt_dict[tuple(gt_box)] = gt['category_id']
        
        for pred in predictions:
            pred_box = pred['bbox']
            pred_layer = pred['layer']
            pred_conf = pred.get('classification_confidence', 0.5)
            
            # Find matching ground truth
            best_match = None
            best_iou = 0
            
            for gt_box_tuple, gt_layer in gt_dict.items():
                gt_box = list(gt_box_tuple)
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = gt_layer
            
            if best_match is not None and best_iou >= IOU_THRESH:
                # Valid classification to evaluate
                self.classification_stats['predictions'].append(pred_layer)
                self.classification_stats['ground_truth'].append(best_match)
                self.classification_stats['confidence_scores'].append(pred_conf)
                
                # Per-layer tracking
                self.classification_stats['per_layer_total'][best_match] += 1
                if pred_layer == best_match:
                    self.classification_stats['per_layer_correct'][best_match] += 1
                elif pred_conf > 0.8:
                    # High confidence but wrong
                    self.quality_metrics['high_confidence_errors'] += 1
    
    def calculate_detection_metrics(self):
        tp = self.detection_stats['true_positives']
        fp = self.detection_stats['false_positives']
        fn = self.detection_stats['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_confidence = np.mean(self.detection_stats['confidence_scores']) if self.detection_stats['confidence_scores'] else 0
        avg_processing_time = np.mean(self.detection_stats['processing_times']) if self.detection_stats['processing_times'] else 0
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'average_confidence': round(avg_confidence, 4),
            'average_processing_time_ms': round(avg_processing_time, 2),
            'total_detections': tp + fp,
            'detection_accuracy': round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0
        }
    
    def calculate_classification_metrics(self):
        if not self.classification_stats['predictions']:
            return {}
        
        y_true = self.classification_stats['ground_truth']
        y_pred = self.classification_stats['predictions']
        
        # Overall accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(y_true)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[1,2,3,4])
        macro_f1 = np.mean(f1)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[1,2,3,4])
        
        # Per-layer accuracy
        per_layer_acc = {}
        for layer in [1, 2, 3, 4]:
            total = self.classification_stats['per_layer_total'][layer]
            correct = self.classification_stats['per_layer_correct'][layer]
            per_layer_acc[layer] = round(correct / total, 4) if total > 0 else 0.0
        
        # Average classification confidence
        avg_conf = np.mean(self.classification_stats['confidence_scores'])
        
        return {
            'overall_accuracy': round(accuracy, 4),
            'macro_f1_score': round(macro_f1, 4),
            'per_layer_accuracy': per_layer_acc,
            'per_layer_precision': {i+1: round(p, 4) for i, p in enumerate(precision)},
            'per_layer_recall': {i+1: round(r, 4) for i, r in enumerate(recall)},
            'per_layer_f1': {i+1: round(f, 4) for i, f in enumerate(f1)},
            'confusion_matrix': cm.tolist(),
            'average_confidence': round(avg_conf, 4),
            'total_classified': len(y_true)
        }
    
    def get_quality_analysis(self):
        """Get model quality and reliability metrics"""
        total_detections = self.detection_stats['total_predictions']
        
        return {
            'low_confidence_rate': round(self.quality_metrics['low_confidence_detections'] / max(total_detections, 1), 4),
            'high_confidence_error_rate': round(self.quality_metrics['high_confidence_errors'] / max(total_detections, 1), 4),
            'confidence_distribution': {
                'mean': round(np.mean(self.detection_stats['confidence_scores']), 4),
                'std': round(np.std(self.detection_stats['confidence_scores']), 4),
                'min': round(np.min(self.detection_stats['confidence_scores']), 4),
                'max': round(np.max(self.detection_stats['confidence_scores']), 4)
            } if self.detection_stats['confidence_scores'] else {}
        }

def detect_and_classify(img_bgr: np.ndarray) -> Tuple[List[Dict], float]:
    """
    Detection with performance tracking
    Returns: (detections, processing_time_ms)
    """
    start_time = time.time()
    img_rgb = img_bgr[...,::-1]
    
    results = det_model(img_rgb)[0]
    
    if results.masks is None or len(results.boxes) == 0:
        return [], (time.time() - start_time) * 1000
    
    masks  = results.masks.data.cpu().numpy()
    boxes  = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    
    crops, coords, det_scores = [], [], []
    
    for mask, box, score in zip(masks, boxes, scores):
        if score < CONF_THRESH:
            continue
            
        x1, y1, x2, y2 = box.astype(int)
        if (x2-x1)*(y2-y1) < MIN_BOX_AREA:
            continue
            
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
            
        y0, y1_ = ys.min(), ys.max()
        x0, x1_ = xs.min(), xs.max()
        
        crop_img  = img_rgb[y0:y1_+1, x0:x1_+1]
        crop_mask = mask[y0:y1_+1, x0:x1_+1].astype(np.uint8)
        crop_img  = (crop_img * crop_mask[..., None]).astype(np.uint8)
        
        crops.append(crop_img)
        coords.append([int(x1), int(y1), int(x2), int(y2)])
        det_scores.append(float(score))
    
    detections = []
    if crops:
        batch = torch.stack([test_tf(Image.fromarray(c)) for c in crops]).to(DEVICE)
        with torch.no_grad():
            logits = clf(batch)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()
        
        for coords_j, det_score, pred, conf in zip(coords, det_scores, predictions, confidences):
            detections.append({
                'bbox': coords_j,
                'layer': int(pred) + 1,
                'detection_confidence': round(det_score, 4),
                'classification_confidence': round(float(conf), 4)
            })
    
    processing_time = (time.time() - start_time) * 1000
    return detections, processing_time

def load_ground_truth(gt_json_path: str) -> Dict:
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
    
    img_id_map = {img['file_name']: img['id'] for img in gt_data['images']}
    anns_by_image = {}
    
    for ann in gt_data['annotations']:
        fname = next((fn for fn, iid in img_id_map.items() if iid == ann['image_id']), None)
        if fname:
            anns_by_image.setdefault(fname, []).append(ann)
    
    return anns_by_image

def print_performance_report(tracker: PerformanceTracker):
    """Print comprehensive performance report"""
    print("\n" + "="*80)
    print("🎯 MODEL PERFORMANCE REPORT")
    print("="*80)
    
    # Detection metrics
    det_metrics = tracker.calculate_detection_metrics()
    print("\n📍 DETECTION PERFORMANCE:")
    print(f"  Precision:     {det_metrics['precision']:.4f}")
    print(f"  Recall:        {det_metrics['recall']:.4f}")
    print(f"  F1-Score:      {det_metrics['f1_score']:.4f}")
    print(f"  Avg Confidence: {det_metrics['average_confidence']:.4f}")
    print(f"  Avg Processing: {det_metrics['average_processing_time_ms']:.2f} ms/image")
    
    # Classification metrics
    cls_metrics = tracker.calculate_classification_metrics()
    if cls_metrics:
        print("\n🏷️  CLASSIFICATION PERFORMANCE:")
        print(f"  Overall Accuracy: {cls_metrics['overall_accuracy']:.4f}")
        print(f"  Macro F1-Score:   {cls_metrics['macro_f1_score']:.4f}")
        print(f"  Avg Confidence:   {cls_metrics['average_confidence']:.4f}")
        
        print("\n  Per-Layer Performance:")
        for layer in [1, 2, 3, 4]:
            acc = cls_metrics['per_layer_accuracy'][layer]
            prec = cls_metrics['per_layer_precision'][layer]
            rec = cls_metrics['per_layer_recall'][layer]
            f1 = cls_metrics['per_layer_f1'][layer]
            print(f"    Layer {layer}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
        
        print("\n  Confusion Matrix (Rows=True, Cols=Predicted):")
        cm = cls_metrics['confusion_matrix']
        print("         1    2    3    4")
        for i, row in enumerate(cm):
            print(f"    {i+1}:  {row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}")
    
    # Quality analysis
    quality = tracker.get_quality_analysis()
    print("\n🔍 MODEL QUALITY ANALYSIS:")
    print(f"  Low Confidence Rate:     {quality['low_confidence_rate']:.4f}")
    print(f"  High Conf Error Rate:    {quality['high_confidence_error_rate']:.4f}")
    
    if quality['confidence_distribution']:
        conf_dist = quality['confidence_distribution']
        print(f"  Confidence Distribution:")
        print(f"    Mean: {conf_dist['mean']:.3f} ± {conf_dist['std']:.3f}")
        print(f"    Range: [{conf_dist['min']:.3f}, {conf_dist['max']:.3f}]")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python detect_and_classify_with_metrics.py <img_dir> <ground_truth.json> [output.json]")
        print("  img_dir: Directory containing test images")
        print("  ground_truth.json: COCO format ground truth annotations")
        print("  output.json: Optional output file for detailed results")
        sys.exit(1)
    
    IMG_DIR = sys.argv[1]
    GT_JSON = sys.argv[2]
    OUTPUT_JSON = sys.argv[3] if len(sys.argv) > 3 else "performance_results.json"
    
    print(f"🔍 Evaluating model performance on: {IMG_DIR}")
    print(f"📋 Using ground truth: {GT_JSON}")
    
    try:
        ground_truth = load_ground_truth(GT_JSON)
        print(f"✅ Loaded ground truth for {len(ground_truth)} images")
    except Exception as e:
        print(f"❌ Failed to load ground truth: {e}")
        sys.exit(1)
    
    # Initialize performance tracker
    tracker = PerformanceTracker()
    results = {}
    
    # Process images
    for img_file in os.listdir(IMG_DIR):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            continue
            
        img_path = os.path.join(IMG_DIR, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        print(f"Processing: {img_file}")
        
        # Run detection
        detections, proc_time = detect_and_classify(img)
        
        # Get ground truth for this image
        gt_anns = ground_truth.get(img_file, [])
        
        # Update performance metrics
        tracker.update_detection_metrics(detections, gt_anns, proc_time)
        tracker.update_classification_metrics(detections, gt_anns)
        
        # Store results
        results[img_file] = {
            'detections': detections,
            'ground_truth': gt_anns,
            'processing_time_ms': proc_time
        }
        
        print(f"  Found {len(detections)} flakes (GT: {len(gt_anns)})")
    
    # Calculate and display performance
    print_performance_report(tracker)
    
    # Save detailed results
    final_results = {
        'performance_metrics': {
            'detection': tracker.calculate_detection_metrics(),
            'classification': tracker.calculate_classification_metrics(),
            'quality_analysis': tracker.get_quality_analysis()
        },
        'detailed_results': results,
        'evaluation_config': {
            'confidence_threshold': CONF_THRESH,
            'iou_threshold': IOU_THRESH,
            'min_box_area': MIN_BOX_AREA
        }
    }
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {OUTPUT_JSON}")