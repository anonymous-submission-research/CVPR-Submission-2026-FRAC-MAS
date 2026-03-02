import os
import math
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# Try to import architecture from current project files
try:
    from train_xfmamba_fast import XFMambaClassifier
    print("✅ Loaded XFMambaClassifier from train_xfmamba_fast.py")
except ImportError:
    print("❌ Could not import XFMambaClassifier from train_xfmamba_fast.py")
    sys.exit(1)

def run_evaluation(weights_path, dataset_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Running on device: {device}")

    # Dataset Setup
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_path = os.path.join(dataset_root, 'val')
    if not os.path.exists(val_path):
        print(f"❌ Error: Path {val_path} not found.")
        return

    val_ds = datasets.ImageFolder(val_path, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    num_classes = len(val_ds.classes)
    print(f"📊 Found {num_classes} classes: {val_ds.classes}")

    # Model Init
    # We use default params from train_xfmamba_fast.py
    model = XFMambaClassifier(num_classes=num_classes).to(device)

    # Weights Loading
    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights file not found at {weights_path}")
        return

    print(f"📥 Loading weights from: {weights_path}")
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        
        # Check for architecture mismatch
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        common_keys = model_keys.intersection(ckpt_keys)
        
        print(f"🔍 Key Match Analysis:")
        print(f"   - Model Keys: {len(model_keys)}")
        print(f"   - Checkpoint Keys: {len(ckpt_keys)}")
        print(f"   - Common Keys: {len(common_keys)}")
        
        if len(common_keys) < len(model_keys) * 0.5:
             print("⚠️  WARNING: Critical architecture mismatch detected!")
             print("Comparing samples:")
             print(f"   Model example key: {list(model_keys)[0] if model_keys else 'None'}")
             print(f"   Ckpt example key: {list(ckpt_keys)[0] if ckpt_keys else 'None'}")
        
        # Attempt to load
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model weights loaded (with strict=False).")
        
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # Evaluation loop
    print("⏳ Starting validation inference...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labs in tqdm(val_loader, desc="Evaluating"):
            imgs, labs = imgs.to(device), labs.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())

    # Report
    print("\n" + "="*50)
    print("📈 XFMAMBA EVALUATION REPORT")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=val_ds.classes))
    print("="*50)

if __name__ == "__main__":
    weights = os.path.join(_PROJECT_ROOT, 'models', 'vssm_small_0229_ckpt_epoch_222.pth')
    dataset = os.path.join(_PROJECT_ROOT, 'balanced_augmented_dataset')
    
    run_evaluation(weights, dataset)
