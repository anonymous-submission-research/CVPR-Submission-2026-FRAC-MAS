import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from vssm import vssm_base

# 1. Environment Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset_root = os.path.join(_PROJECT_ROOT, 'balanced_augmented_dataset')
checkpoint_path = os.path.join(_PROJECT_ROOT, 'models', 'vssm_small_0229_ckpt_epoch_222.pth')

# 2. Data Loading
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(os.path.join(dataset_root, 'train'), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(dataset_root, 'val'), transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# 3. Model Initialization & Weight Loading
print("Building VSSM-Base Model...")
model = vssm_base(num_classes=1000) # Load with 1000 classes initially

if os.path.exists(checkpoint_path):
    print(f"Loading Pre-trained Weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    sd = checkpoint.get('model', checkpoint)
    model.load_state_dict(sd)
    print("✅ Pre-trained weights loaded successfully.")
else:
    print(f"❌ Warning: Checkpoint not found at {checkpoint_path}. Starting from scratch.")

# Replace head for Fracture Detection (8 classes)
print(f"Adapting model head for {len(train_ds.classes)} classes...")
model.classifier.head = nn.Linear(model.classifier.head.in_features, len(train_ds.classes))
model.to(device)

# 4. Training loop (Simplified)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
criterion = nn.CrossEntropyLoss()

print("Starting training...")
for epoch in range(3): # Short session
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for imgs, labs in loop:
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labs)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        
    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labs in val_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
    
    print(f"\nEpoch {epoch+1} Results:")
    print(classification_report(all_labels, all_preds, target_names=train_ds.classes))

    # Save best
    torch.save(model.state_dict(), "vssm_fracture_finetuned.pth")

print("Training finished.")
