import os
import numpy as np
import torch
import random
from copy import deepcopy
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
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
print(f"Device: {DEVICE}")

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
                print("Available keys in CNN output:", dummy_output.keys())
                first_output = list(dummy_output.values())[0]
                if len(first_output.shape) == 4:  # (batch, channels, H, W)
                    self.image_dim = first_output.shape[1]
                else:
                    self.image_dim = first_output.shape[1]
        
        print(f"CNN output dimension: {self.image_dim}")
        self.embedding = nn.Embedding(num_materials, material_dim)
        
        # Branch 1: Image only classification
        self.fc_img = nn.Linear(self.image_dim, num_classes)
        
        # Branch 2: Combined image + material classification
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
        
        # Branch 1: Image-only classification
        if material_ids is None:
            return self.fc_img(img_feats)
        
        # Branch 2: Combined image + material classification
        mat_emb = self.embedding(material_ids)  # (batch, material_dim)
        combined = torch.cat((img_feats, mat_emb), dim=1)  # (batch, image_dim + material_dim)
        return self.fc_comb(combined)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() if self.reduction=='mean' else loss.sum()

data_root = "/home/sankalp/flake_classification/GMMClassifier_bbox"
train_dir = os.path.join(data_root, "train")
test_dir  = os.path.join(data_root, "test")
batch_size = 16
num_workers = 4
lr = 3e-4
weight_decay = 1e-4
epochs = 50
test_size = 0.2
material_dim = 4

train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

all_samples, all_labels = [], []
for root, _, files in os.walk(train_dir):
    for f in files:
        if f.lower().endswith(('png','jpg','jpeg')):
            all_samples.append(os.path.join(root, f))
            all_labels.append(int(os.path.basename(root)))
for root, _, files in os.walk(test_dir):
    for f in files:
        if f.lower().endswith(('png','jpg','jpeg')):
            all_samples.append(os.path.join(root, f))
            all_labels.append(int(os.path.basename(root)))

unique_labels = sorted(set(all_labels))
print(f"Unique labels found: {unique_labels}")
print(f"Label counts: {np.bincount(all_labels)}")

label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
print(f"Label mapping: {label_to_idx}")
all_labels_mapped = [label_to_idx[label] for label in all_labels]
num_classes = len(unique_labels)
print(f"Number of classes: {num_classes}")

materials = sorted({os.path.basename(p).split('_')[0] for p in all_samples})
material2idx = {m:i for i,m in enumerate(materials)}
num_materials = len(materials)
print(f"Materials: {materials}")
print(f"Number of materials: {num_materials}")

class FlakeDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, tf):
        self.samples = samples
        self.labels = labels
        self.tf = tf
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        image = Image.open(path).convert('RGB')
        image = self.tf(image)
        label = self.labels[idx]
        mat = os.path.basename(path).split('_')[0]
        mat_id = material2idx[mat]
        return image, mat_id, label

# Train/val split
indices = list(range(len(all_samples)))
train_idx, val_idx = train_test_split(
    indices, 
    test_size=test_size, 
    stratify=all_labels_mapped,
    random_state=42
)

train_ds = FlakeDataset(
    [all_samples[i] for i in train_idx],
    [all_labels_mapped[i] for i in train_idx], 
    train_tf
)
val_ds = FlakeDataset(
    [all_samples[i] for i in val_idx],
    [all_labels_mapped[i] for i in val_idx], 
    val_tf
)

train_labels_mapped = [all_labels_mapped[i] for i in train_idx]
counts = np.bincount(train_labels_mapped, minlength=num_classes)
print(f"Class counts in training set: {counts}")

max_count = counts.max()
class_weights_list = [max_count / (counts[i] + 1e-8) for i in range(num_classes)]
sample_weights = [class_weights_list[label] for label in train_labels_mapped]

sampler = WeightedRandomSampler(
    sample_weights, 
    num_samples=len(train_ds) * 2, 
    replacement=True
)

# DataLoaders
train_loader = DataLoader(
    train_ds, 
    batch_size=batch_size, 
    sampler=sampler, 
    num_workers=num_workers
)
val_loader = DataLoader(
    val_ds, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers
)

model = FlakeLayerClassifier(
    num_materials, 
    material_dim, 
    num_classes, 
    pretrained=True
).to(DEVICE)

class_weights = torch.FloatTensor([max_count / (counts[i] + 1e-8) for i in range(num_classes)]).to(DEVICE)
print(f"Class weights shape: {class_weights.shape}")
print(f"Class weights: {class_weights}")

criterion = FocalLoss(alpha=class_weights, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

best_bal_acc = 0.0
no_improve = 0
patience = 12

print("Starting training...")
for epoch in range(1, epochs+1):
    model.train()
    total_loss = total_correct = total = 0
    
    for batch_idx, (imgs, mats, labs) in enumerate(train_loader):
        imgs, mats, labs = imgs.to(DEVICE), mats.to(DEVICE), labs.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(imgs, mats)
        loss = criterion(outputs, labs)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labs).sum().item()
        total += labs.size(0)
            
    scheduler.step()
    train_acc = total_correct / total
    
    model.eval()
    class_tot = np.zeros(num_classes)
    class_corr = np.zeros(num_classes)
    
    with torch.no_grad():
        for imgs, mats, labs in val_loader:
            imgs, mats, labs = imgs.to(DEVICE), mats.to(DEVICE), labs.to(DEVICE)
            outputs = model(imgs, mats)
            preds = outputs.argmax(dim=1)
            
            for i in range(labs.size(0)):
                label = labs[i].item()
                class_tot[label] += 1
                class_corr[label] += (preds[i] == labs[i]).item()
    
    class_acc = np.divide(class_corr, class_tot, out=np.zeros_like(class_corr), where=class_tot!=0)
    bal_acc = class_acc[class_tot > 0].mean()  # Only average over classes that exist in validation
    
    print(f"Epoch {epoch}/{epochs} | Train Acc: {train_acc:.4f} | Val Bal Acc: {bal_acc:.4f}")
    
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_state = deepcopy(model.state_dict())
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
torch.save({
    'model_state_dict': model.state_dict(),
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'material2idx': material2idx,
    'num_classes': num_classes,
    'num_materials': num_materials,
    'material_dim': material_dim
}, 'flake_layer_classifier_best.pth')

print(f"Training complete. Best Balanced Acc: {best_bal_acc:.4f}")
print(f"Model saved with label mappings: {label_to_idx}")