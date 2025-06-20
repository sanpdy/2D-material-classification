import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from transformers import AutoModel, ResNetConfig

def get_device():
    if torch.cuda.is_available():
        for idx in [2, 1, 0]:
            try:
                torch.cuda.set_device(idx)
                return torch.device(f"cuda:{idx}")
            except:
                continue
    return torch.device("cpu")

DEVICE = get_device()

class FlakeLayerClassifier(nn.Module):
    def __init__(self, num_materials, material_dim, num_classes=4, pretrained=True):
        super().__init__()
        
        if pretrained:
            self.cnn = AutoModel.from_pretrained("microsoft/resnet-18")
        else:
            config = ResNetConfig()
            self.cnn = AutoModel.from_config(config)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.cnn(pixel_values=dummy_input)
            if hasattr(dummy_output, 'last_hidden_state'):
                self.image_dim = dummy_output.last_hidden_state.shape[1]
            elif hasattr(dummy_output, 'pooler_output'):
                self.image_dim = dummy_output.pooler_output.shape[1]
            else:
                first_output = list(dummy_output.values())[0]
                if len(first_output.shape) == 4:
                    self.image_dim = first_output.shape[1]
                else:
                    self.image_dim = first_output.shape[1]
        
        self.embedding = nn.Embedding(num_materials, material_dim)
        self.fc_img = nn.Linear(self.image_dim, num_classes)
        self.fc_comb = nn.Linear(self.image_dim + material_dim, num_classes)
        self.material_dim = material_dim
        self.num_classes = num_classes

    def forward(self, pixel_values, material_ids=None):
        cnn_outputs = self.cnn(pixel_values=pixel_values)
        if hasattr(cnn_outputs, 'last_hidden_state'):
            feat_map = cnn_outputs.last_hidden_state
            if len(feat_map.shape) == 4:
                img_feats = feat_map.mean(dim=(-2, -1))
            else:
                img_feats = feat_map
        elif hasattr(cnn_outputs, 'pooler_output'):
            img_feats = cnn_outputs.pooler_output
        else:
            first_output = list(cnn_outputs.values())[0]
            if len(first_output.shape) == 4:
                img_feats = first_output.mean(dim=(-2, -1))
            else:
                img_feats = first_output
        
        if material_ids is None:
            return self.fc_img(img_feats)
        
        mat_emb = self.embedding(material_ids)
        combined = torch.cat((img_feats, mat_emb), dim=1)
        return self.fc_comb(combined)

class FlakeDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, material2idx, tf):
        self.samples = samples
        self.labels = labels
        self.material2idx = material2idx
        self.tf = tf
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        image = Image.open(path).convert('RGB')
        image = self.tf(image)
        label = self.labels[idx]
        mat = os.path.basename(path).split('_')[0]
        mat_id = self.material2idx[mat]
        return image, mat_id, label

print("Loading trained model...")
checkpoint = torch.load('flake_layer_classifier_best.pth', map_location=DEVICE)
label_to_idx = checkpoint['label_to_idx']
idx_to_label = checkpoint['idx_to_label']
material2idx = checkpoint['material2idx']
num_classes = checkpoint['num_classes']
num_materials = checkpoint['num_materials']
material_dim = checkpoint['material_dim']

train_idx = checkpoint.get('train_idx', None)
val_idx = checkpoint.get('val_idx', None)

model = FlakeLayerClassifier(num_materials, material_dim, num_classes, pretrained=True).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Data paths
data_root = "/home/sankalp/flake_classification/GMMClassifier_bbox"
train_dir = os.path.join(data_root, "train")
test_dir = os.path.join(data_root, "test")

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def evaluate_set(samples, labels, set_name):
    """Common evaluation function"""
    dataset = FlakeDataset(samples, labels, material2idx, val_tf)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, mats, labs in loader:
            imgs, mats, labs = imgs.to(DEVICE), mats.to(DEVICE), labs.to(DEVICE)
            outputs = model(imgs, mats)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print(f"\n{'='*50}")
    print(f"{set_name} Results")
    print(f"{'='*50}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=[str(idx_to_label[i]) for i in range(num_classes)]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print(f"\nBalanced Accuracy: {balanced_accuracy_score(all_labels, all_preds):.4f}")
    print(f"Overall Accuracy: {(all_preds == all_labels).mean():.4f}")
    
    return all_preds, all_labels

if train_idx is not None and val_idx is not None:
    train_samples, train_labels = [], []
    for root, _, files in os.walk(train_dir):
        for f in files:
            if f.lower().endswith(('png','jpg','jpeg')):
                train_samples.append(os.path.join(root, f))
                train_labels.append(int(os.path.basename(root)))
    
    train_labels_mapped = [label_to_idx[label] for label in train_labels]
    val_samples = [train_samples[i] for i in val_idx]
    val_labels = [train_labels_mapped[i] for i in val_idx]
    
    val_preds, val_true = evaluate_set(val_samples, val_labels, "Validation Set")

print("Evaluating on test set...")
test_samples, test_labels = [], []
for root, _, files in os.walk(test_dir):
    for f in files:
        if f.lower().endswith(('png','jpg','jpeg')):
            test_samples.append(os.path.join(root, f))
            test_labels.append(int(os.path.basename(root)))

test_labels_mapped = []
unseen_labels = []
for label in test_labels:
    if label in label_to_idx:
        test_labels_mapped.append(label_to_idx[label])
    else:
        unseen_labels.append(label)

if unseen_labels:
    print(f"Warning: Found {len(unseen_labels)} samples with unseen labels: {set(unseen_labels)}")
    filtered_samples = []
    filtered_labels = []
    for sample, label in zip(test_samples, test_labels):
        if label in label_to_idx:
            filtered_samples.append(sample)
            filtered_labels.append(label_to_idx[label])
    test_samples = filtered_samples
    test_labels_mapped = filtered_labels

print(f"Evaluating on {len(test_samples)} test samples...")
test_preds, test_true = evaluate_set(test_samples, test_labels_mapped, "Test Set")

print(f"\n{'='*50}")
print("FINAL SUMMARY")
print(f"{'='*50}")
if 'val_preds' in locals():
    print(f"Validation Balanced Accuracy: {balanced_accuracy_score(val_true, val_preds):.4f}")
print(f"Test Balanced Accuracy: {balanced_accuracy_score(test_true, test_preds):.4f}")
print("Evaluation complete!")