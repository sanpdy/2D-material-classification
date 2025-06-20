import torch
import torch.nn as nn
from transformers import ResNetModel

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


import os
import numpy as np
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import random
from copy import deepcopy

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DATA_ROOT = "./GMMClassifier_bbox"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR  = os.path.join(DATA_ROOT, "test")
BATCH_SIZE = 16
NUM_WORKERS = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
TEST_SIZE = 0.2

def get_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(2)
            device = torch.device("cuda:2")
            print(f"Using GPU 2: {torch.cuda.get_device_name(2)}")
            return device
        except:
            print("GPU 2 not available, trying GPU 1...")
            try:
                torch.cuda.set_device(1)
                device = torch.device("cuda:1")
                print(f"Using GPU 1: {torch.cuda.get_device_name(1)}")
                return device
            except:
                print("No working CUDA GPUs found, using CPU")
                return torch.device("cpu")
    else:
        print("CUDA not available, using CPU")
        return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")
print(f"Training with 80/20 train/validation split")

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),            
    #transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std =[0.229,0.224,0.225]),
    ])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),            
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225]),
])

train_ds = ImageFolder(TRAIN_DIR, transform=train_tf)
test_ds = ImageFolder(TEST_DIR, transform=val_tf)

print("\n" + "="*50)
print("ORIGINAL DATASET INFORMATION")
print("="*50)
print(f"Training samples: {len(train_ds)}")
print(f"Test samples: {len(test_ds)}")
print(f"Number of classes: {len(train_ds.classes)}")

print("\nCombining train and test datasets for rebalancing...")
all_samples = []
all_labels = []

for sample_path, label in train_ds.samples:
    all_samples.append(sample_path)
    all_labels.append(label)

test_ds_temp = ImageFolder(TEST_DIR, transform=train_tf)
for sample_path, label in test_ds_temp.samples:
    all_samples.append(sample_path)
    all_labels.append(label)

print(f"Combined dataset size: {len(all_samples)}")
combined_labels = np.array(all_labels)
combined_counts = np.bincount(combined_labels)

print("\nCOMBINED CLASS DISTRIBUTION:")
total_combined = len(all_samples)
for i, count in enumerate(combined_counts):
    percentage = (count / total_combined) * 100
    layer_num = int(train_ds.classes[i])
    print(f"  Layer {layer_num}: {count:4d} samples ({percentage:5.1f}%)")

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_model():
    model = FlakeLayerClassifier(
        num_materials=len(train_ds.classes),
        material_dim=64,
        num_classes=len(train_ds.classes),
        dropout_prob=0.1,
        freeze_cnn=False
    )
    model.to(DEVICE)
    return model


def evaluate_model(model, data_loader, dataset_name=""):
    model.eval()
    correct = 0
    total = 0
    class_correct = np.zeros(len(train_ds.classes))
    class_total = np.zeros(len(train_ds.classes))
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labs in data_loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            
            correct += (preds == labs).sum().item()
            total += labs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
            
            for i in range(labs.size(0)):
                label = labs[i].item()
                class_total[label] += 1
                if preds[i] == labs[i]:
                    class_correct[label] += 1
    
    accuracy = correct / total
    per_class_acc = []
    for i in range(len(train_ds.classes)):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            per_class_acc.append(acc)
    balanced_acc = np.mean(per_class_acc) if per_class_acc else 0.0
    
    return accuracy, balanced_acc, all_preds, all_labels, class_correct, class_total

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        
        from PIL import Image
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 80/20 split
print("\n" + "="*50)
print("80/20 TRAIN/VALIDATION SPLIT")
print("="*50)

train_idx, val_idx = train_test_split(
    range(len(all_samples)), 
    test_size=TEST_SIZE, 
    stratify=all_labels, 
    random_state=42
)

print(f"Train indices: {len(train_idx)}")
print(f"Validation indices: {len(val_idx)}")

train_samples = [all_samples[i] for i in train_idx]
train_labels = [all_labels[i] for i in train_idx]
val_samples = [all_samples[i] for i in val_idx]
val_labels = [all_labels[i] for i in val_idx]
train_dataset = CombinedDataset(train_samples, train_labels, transform=train_tf)
val_dataset = CombinedDataset(val_samples, val_labels, transform=val_tf)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
train_label_counts = np.bincount(train_labels, minlength=len(train_ds.classes))
val_label_counts = np.bincount(val_labels, minlength=len(train_ds.classes))

print("\nTRAIN SET CLASS DISTRIBUTION:")
for i, count in enumerate(train_label_counts):
    percentage = (count / len(train_labels)) * 100
    layer_num = int(train_ds.classes[i])
    print(f"  Layer {layer_num}: {count:4d} samples ({percentage:5.1f}%)")

print("\nVALIDATION SET CLASS DISTRIBUTION:")
for i, count in enumerate(val_label_counts):
    percentage = (count / len(val_labels)) * 100
    layer_num = int(train_ds.classes[i])
    print(f"  Layer {layer_num}: {count:4d} samples ({percentage:5.1f}%)")

max_count = max(train_label_counts)
sample_weights = [max_count / train_label_counts[label] for label in train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset) * 2,
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                         sampler=sampler, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                       shuffle=False, num_workers=NUM_WORKERS)

print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)

model = create_model()

class_weights = max_count / (train_label_counts + 1e-8)
class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print(f"Class weights: {class_weights}")

best_val_acc = 0.0
best_balanced_acc = 0.0
best_model_state = None
patience = 12
no_improve = 0

training_history = {
    'train_acc': [],
    'val_acc': [],
    'balanced_acc': [],
    'train_loss': []
}

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for imgs, labs in train_loader:
        imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labs)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labs).sum().item()
        train_total += labs.size(0)
    
    train_acc = train_correct / train_total
    train_loss = train_loss / train_total
    scheduler.step()
    val_acc, val_balanced_acc, val_preds, val_labels_list, class_correct, class_total = evaluate_model(model, val_loader, "Validation")
    
    training_history['train_acc'].append(train_acc)
    training_history['val_acc'].append(val_acc)
    training_history['balanced_acc'].append(val_balanced_acc)
    training_history['train_loss'].append(train_loss)
    
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch:2d}/{EPOCHS} | LR: {current_lr:.6f} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Acc: {val_acc:.4f} | Balanced: {val_balanced_acc:.4f}")
    
    if val_balanced_acc > best_balanced_acc:
        best_val_acc = val_acc
        best_balanced_acc = val_balanced_acc
        best_model_state = deepcopy(model.state_dict())
        no_improve = 0
        print(f"  → New best balanced accuracy: {best_balanced_acc:.4f}")
    else:
        no_improve += 1
    
    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\nLoaded best model with balanced accuracy: {best_balanced_acc:.4f}")

print("\n" + "="*50)
print("FINAL EVALUATION")
print("="*50)

# Final evaluation on validation set
val_acc, val_balanced_acc, val_preds, val_labels_list, class_correct, class_total = evaluate_model(model, val_loader, "Final Validation")

print(f"Final Validation Accuracy: {val_acc:.4f}")
print(f"Final Balanced Accuracy: {val_balanced_acc:.4f}")
print(f"\nPer-class accuracy on validation set:")
for i in range(len(train_ds.classes)):
    if class_total[i] > 0:
        acc = class_correct[i] / class_total[i]
        layer_num = int(train_ds.classes[i])
        print(f"  Layer {layer_num}: {acc:.4f} ({int(class_correct[i])}/{int(class_total[i])})")

cm = confusion_matrix(val_labels_list, val_preds)
print("\nConfusion Matrix:")
print("Pred:  1    2    3    4")
for i, row in enumerate(cm):
    layer_num = int(train_ds.classes[i])
    print(f"{layer_num}:   {row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}")

target_names = [f'Layer {int(cls)}' for cls in train_ds.classes]
print("\nClassification Report:")
print(classification_report(val_labels_list, val_preds, target_names=target_names))
torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': train_ds.class_to_idx,
    'training_history': training_history,
    'final_results': {
        'val_acc': val_acc,
        'balanced_acc': val_balanced_acc,
        'best_val_acc': best_val_acc,
        'best_balanced_acc': best_balanced_acc,
        'confusion_matrix': cm.tolist(),
        'class_correct': class_correct.tolist(),
        'class_total': class_total.tolist()
    },
    'hyperparameters': {
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'epochs': EPOCHS,
        'test_size': TEST_SIZE
    }
}, 'flake_classifier.pth')

print(f"\nModel saved to: resnet18_layer_bbox_classifier_8020_new_augmentation.pth")
print("Training complete!")
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"Total epochs trained: {len(training_history['train_acc'])}")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Best balanced accuracy: {best_balanced_acc:.4f}")
print(f"Final validation accuracy: {val_acc:.4f}")
print(f"Final balanced accuracy: {val_balanced_acc:.4f}")