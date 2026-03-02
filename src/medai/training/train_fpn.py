
import os
import argparse
import time
import copy
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvmodels
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

# Device config
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# Dataset
class FractureDataset(Dataset):
    def __init__(self, df, img_root, transform=None):
        self.entries = df
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries.iloc[idx]
        img_path = row['image_path']
        # Handle relative/absolute paths
        if not os.path.isabs(img_path):
             img_path = os.path.join(self.img_root, img_path)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img = Image.new('RGB', (224, 224))

        label = int(row['label'])
        if self.transform:
            img = self.transform(img)
        return img, label

# Transforms
def get_transforms(split, img_size=224):
    if split == 'train':
        return T.Compose([
            T.Resize((int(img_size*1.1), int(img_size*1.1))),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# FPN Classifier
class FPNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FPNClassifier, self).__init__()
        # Backbone (ResNet50)
        self.backbone = tvmodels.resnet50(pretrained=True)
        # We need features from different stages.
        # ResNet50 has layers: layer1 (256), layer2 (512), layer3 (1024), layer4 (2048)
        
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * 4, num_classes) # Concatenate pooled features from 4 levels

    def forward(self, x):
        # Extract features
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)

        features = OrderedDict()
        features['feat0'] = c2
        features['feat1'] = c3
        features['feat2'] = c4
        features['feat3'] = c5

        fpn_features = self.fpn(features)
        
        # Pool and Concatenate
        pooled = []
        # FPN returns keys matching input keys if OrderedDict, or indices if list. 
        # FeaturePyramidNetwork input is Dict[str, Tensor], output is Dict[str, Tensor]
        # Keys are preserved.
        
        # Correct order: feat0 -> feat3
        # Note: FPN output usually same keys.
        for k in ['feat0', 'feat1', 'feat2', 'feat3']:
            p = self.avg_pool(fpn_features[k])
            pooled.append(p.view(p.size(0), -1))
        
        cat_feat = torch.cat(pooled, dim=1)
        out = self.fc(cat_feat)
        return out

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
    return running_loss / len(loader.dataset), accuracy_score(all_targets, all_preds)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    return running_loss / len(loader.dataset), accuracy_score(all_targets, all_preds), (p, r, f1)

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_dir = os.path.join(root_dir, "balanced_augmented_dataset")
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    
    out_dir = os.path.join(root_dir, "outputs", "cross_validation")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Dataset & Loader
    batch_size = 16 
    num_classes = len(train_df['label'].unique())
    print(f"Num classes: {num_classes}")
    
    train_ds = FractureDataset(train_df, img_root=root_dir, transform=get_transforms('train'))
    val_ds = FractureDataset(val_df, img_root=root_dir, transform=get_transforms('val'))
    test_ds = FractureDataset(test_df, img_root=root_dir, transform=get_transforms('val'))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    print("Initializing FPN Model...")
    model = FPNClassifier(num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 5 
    
    best_f1 = 0.0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        train_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, (p, r, f1) = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{num_epochs} [{time.time()-train_start:.1f}s] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(out_dir, "best_fpn.pth"))
            
    print(f"Best Val F1: {best_f1:.4f}")
    
    # Final Eval on Test
    print("Evaluating on Test Set...")
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_fpn.pth")))
    test_loss, test_acc, (tp, tr, tf1) = validate(model, test_loader, criterion, DEVICE)
    print(f"Test Accuracy: {test_acc:.4f}, Test F1: {tf1:.4f}")
    
    # Simple comparison output
    results_file = os.path.join(out_dir, "fpn_results.txt")
    with open(results_file, "w") as f:
        f.write(f"FPN Model Results:\nAccuracy: {test_acc}\nF1 Score: {tf1}\n")
    print(f"Results saved to {results_file}")

    print("\nComparison Check:")
    model_files = [f for f in os.listdir(out_dir) if f.endswith(".pth")]
    print(f"Available models in {out_dir}: {model_files}")
    print("You can evaluate other models using similar validation logic if needed.")

if __name__ == "__main__":
    main()
