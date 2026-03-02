import os
import sys
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms
import logging
import torchvision.models as models
import timm
import warnings

# Suppress Pydantic warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
warnings.filterwarnings('ignore')

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Evaluate")

# Constants
CLASS_NAMES = [
    "Comminuted", "Greenstick", "Healthy", "Oblique", 
    "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"
]
DATA_DIR_NAME = "balanced_augmented_dataset"
MODELS_DIR = "models"
BATCH_SIZE = 32
NUM_WORKERS = 0 # Reduced for safety on Mac
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME_RAD_DINO = "microsoft/rad-dino"

# ==================================================================================
# SHARED COMPONENTs
# ==================================================================================

class BackendChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(BackendChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class BackendSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(BackendSpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class BackendCBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(BackendCBAM, self).__init__()
        self.ca = BackendChannelAttention(in_planes, ratio)
        self.sa = BackendSpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ==================================================================================
# LOGIC VARIANT 1: ORIGINAL (Legacy) - Matches trained weights for some models
# ==================================================================================

class BackendHypercolumnCBAMDenseNet_Original(nn.Module):
    def __init__(self, num_classes=8, growth_rate=32, bn_size=4, drop_rate=0.0):
        super(BackendHypercolumnCBAMDenseNet_Original, self).__init__()
        densenet = models.densenet169(weights=None)
        
        self.features = densenet.features
        # ORIGINAL LOGIC: Separate init_conv definition (random weights initially)
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
        )

        self.db1 = self.features.denseblock1
        self.db2 = self.features.denseblock2
        self.db3 = self.features.denseblock3
        self.db4 = self.features.denseblock4
        self.t1 = self.features.transition1
        self.t2 = self.features.transition2
        self.t3 = self.features.transition3
        self.norm_final = self.features.norm5
        
        self.fusion_conv = nn.Conv2d(2688, 1024, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(1024)
        self.cbam = BackendCBAM(1024)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.init_conv(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.db1(x)
        t1_out = self.t1(x)
        x = self.db2(t1_out)
        t2_out = self.t2(x)
        x = self.db3(t2_out)
        t3_out = self.t3(x)
        x = self.db4(t3_out)
        x_final = self.norm_final(x)
        target_size = x_final.shape[2:]
        t1_resized = nn.functional.interpolate(t1_out, size=target_size, mode='bilinear', align_corners=False)
        t2_resized = nn.functional.interpolate(t2_out, size=target_size, mode='bilinear', align_corners=False)
        t3_resized = nn.functional.interpolate(t3_out, size=target_size, mode='bilinear', align_corners=False)
        hypercolumn = torch.cat([x_final, t3_resized, t2_resized, t1_resized], dim=1)
        x = self.fusion_conv(hypercolumn)
        x = self.bn_fusion(x)
        x = nn.functional.relu(x)
        x = self.cbam(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_model_original(model_name, num_classes=8):
    model_name = model_name.lower()
    if "hypercolumn" in model_name: 
        return BackendHypercolumnCBAMDenseNet_Original(num_classes=num_classes)
    elif "densenet169" in model_name:
        model = models.densenet169(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model
    elif "maxvit" in model_name:
        model = models.maxvit_t(weights=None)
        in_features = model.classifier[5].in_features
        model.classifier[5] = nn.Linear(in_features, num_classes)
        return model
    elif "efficientnet" in model_name:
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    elif "mobilenet" in model_name:
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    elif "swin" in model_name:
        model = models.swin_t(weights=None)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return model
    return None

# ==================================================================================
# LOGIC VARIANT 2: FIXED (Backend Compliance) - Matches standard densenet weights
# ==================================================================================

class BackendHypercolumnCBAMDenseNet_Fixed(nn.Module):
    def __init__(self, num_classes=8, growth_rate=32, bn_size=4, drop_rate=0.0):
        super(BackendHypercolumnCBAMDenseNet_Fixed, self).__init__()
        densenet = models.densenet169(weights=None)
        
        self.features = densenet.features
        # FIXED LOGIC: Reuse features.conv0
        self.init_conv = nn.Sequential(self.features.conv0, self.features.norm0, self.features.relu0, self.features.pool0)

        self.db1 = self.features.denseblock1
        self.db2 = self.features.denseblock2
        self.db3 = self.features.denseblock3
        self.db4 = self.features.denseblock4
        self.t1 = self.features.transition1
        self.t2 = self.features.transition2
        self.t3 = self.features.transition3
        self.norm_final = self.features.norm5
        self.fusion_conv = nn.Conv2d(2688, 1024, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(1024)
        self.cbam = BackendCBAM(1024)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.db1(x)
        t1_out = self.t1(x)
        x = self.db2(t1_out)
        t2_out = self.t2(x)
        x = self.db3(t2_out)
        t3_out = self.t3(x)
        x = self.db4(t3_out)
        x_final = self.norm_final(x)
        target_size = x_final.shape[2:]
        t1_resized = nn.functional.interpolate(t1_out, size=target_size, mode='bilinear', align_corners=False)
        t2_resized = nn.functional.interpolate(t2_out, size=target_size, mode='bilinear', align_corners=False)
        t3_resized = nn.functional.interpolate(t3_out, size=target_size, mode='bilinear', align_corners=False)
        hypercolumn = torch.cat([x_final, t3_resized, t2_resized, t1_resized], dim=1)
        x = self.fusion_conv(hypercolumn)
        x = self.bn_fusion(x)
        x = nn.functional.relu(x)
        x = self.cbam(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

MODEL_CONFIGS_TIMM = {
    "swin": "swin_small_patch4_window7_224",
    "densenet169": "densenet169",
    "efficientnetv2": "efficientnet_b0",
    "efficientnet": "efficientnet_b0",
    "mobilenetv2": "mobilenetv2_100",
    "mobilenet": "mobilenetv2_100",
    "maxvit": "maxvit_tiny_tf_224",
}

def get_model_fixed(model_name, num_classes=8):
    model_name_lower = model_name.lower()
    
    # 1. Custom Hypercolumn (Fixed)
    if "hypercolumn" in model_name_lower: 
        return BackendHypercolumnCBAMDenseNet_Fixed(num_classes=num_classes)
    
    # 2. Standard Models via timm
    timm_model_name = None
    for key, config_name in MODEL_CONFIGS_TIMM.items():
        if key in model_name_lower:
            timm_model_name = config_name
            break
            
    if timm_model_name:
        try:
            model = timm.create_model(timm_model_name, pretrained=False)
            if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
                model.head = nn.Linear(model.head.in_features, num_classes)
            elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            else:
                model.reset_classifier(num_classes=num_classes)
            return model
        except Exception as e:
            return None
    return None

# ==================================================================================
# MAIN EVALUATION
# ==================================================================================

class RadDinoClassifier(nn.Module):
    def __init__(self, num_classes=8, head_type="linear"):
        super(RadDinoClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME_RAD_DINO)
        self.hidden_size = self.backbone.config.hidden_size
        
        if head_type == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Linear(self.hidden_size, num_classes)
        
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits

class UnifiedFractureDataset(Dataset):
    def __init__(self, csv_file, root_dir, rad_processor, legacy_transform):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.rad_processor = rad_processor
        self.legacy_transform = legacy_transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        local_path = row['image_path']
        label = int(row['label']) if 'label' in row else 0 
        full_path = local_path
        if not os.path.isabs(full_path): full_path = os.path.join(self.root_dir, local_path)
        if not os.path.exists(full_path):
             if os.path.exists(local_path): full_path = local_path
             elif os.path.exists(os.path.join("data", local_path)): full_path = os.path.join("data", local_path)
        image = Image.open(full_path).convert("RGB")
        pixel_values = self.rad_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        legacy_tensor = self.legacy_transform(image)
        return {'pixel_values': pixel_values, 'legacy_tensor': legacy_tensor, 'label': torch.tensor(label, dtype=torch.long)}

def evaluate_yolo_model(yolo_model_path, val_csv, root_dir="data"):
    """Evaluate a YOLO classification model on the validation set.

    Ground-truth class is derived from the image path's parent folder name so
    that the integer label in the CSV (which may use a different ordering) never
    causes a mismatch against the model's own class index.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return None, None

    model = YOLO(yolo_model_path)
    task = getattr(model, 'task', 'detect')
    logger.info(f"  YOLO task detected: {task}")

    # Build a name→index mapping from the model's own class list
    inv_names = {name: idx for idx, name in model.names.items()}

    data = pd.read_csv(val_csv)
    y_true, y_pred = [], []

    for _, row in tqdm(data.iterrows(), total=len(data), desc="YOLO Eval", leave=False):
        local_path = row['image_path']

        # Resolve image path
        full_path = local_path
        if not os.path.isabs(full_path):
            full_path = os.path.join(root_dir, local_path)
        if not os.path.exists(full_path):
            for candidate in [local_path, os.path.join("data", local_path)]:
                if os.path.exists(candidate):
                    full_path = candidate
                    break

        # Derive true class from folder name (robust to CSV label ordering)
        folder_name = os.path.basename(os.path.dirname(local_path))
        true_idx = inv_names.get(folder_name)
        if true_idx is None:
            # Try replacing spaces with underscores and vice-versa
            true_idx = inv_names.get(folder_name.replace(' ', '_'))
        if true_idx is None:
            true_idx = inv_names.get(folder_name.replace('_', ' '))
        if true_idx is None:
            continue  # unknown class — skip

        try:
            results = model.predict(full_path, verbose=False)
            result = results[0]
            if task == 'classify':
                pred_class = int(result.probs.top1)
            else:
                if result.boxes is not None and len(result.boxes) > 0:
                    best_idx = int(result.boxes.conf.argmax())
                    pred_class = int(result.boxes.cls[best_idx].item())
                else:
                    pred_class = 0
        except Exception:
            pred_class = -1  # will not match any true class

        y_true.append(true_idx)
        y_pred.append(pred_class)

    if not y_true:
        return 0.0, 0.0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, f1


def evaluate_single_pass(model, loader, is_rad_dino=False):
    model.to(DEVICE)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Batch", leave=False):
            labels = batch['label'].to(DEVICE)
            if is_rad_dino:
                inputs = batch['pixel_values'].to(DEVICE)
                outputs = model(inputs)
            else:
                inputs = batch['legacy_tensor'].to(DEVICE)
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')

def main():
    try:
        rad_processor = AutoImageProcessor.from_pretrained(MODEL_NAME_RAD_DINO)
    except:
        logger.error("RadDino processor fetch failed")
        return
    
    legacy_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_csv = None
    for p in [os.path.join(DATA_DIR_NAME, 'val.csv'), os.path.join("data", DATA_DIR_NAME, 'val.csv'), "val.csv"]:
        if os.path.exists(p): val_csv = p; break
    if not val_csv:
        logger.error("Val CSV not found")
        return
    
    val_dataset = UnifiedFractureDataset(val_csv, "data", rad_processor, legacy_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # 1. Collect all model files
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pth"))
    outputs_files = glob.glob(os.path.join("outputs", "**", "best.pth"), recursive=True)
    model_files.extend(outputs_files)

    # Collect YOLO .pt files: only include models whose class list matches the
    # 8 fracture classes (skips pretrained ImageNet checkpoints).
    _raw_yolo_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    # Also pick up fine-tuned best/last checkpoints saved under outputs/
    _raw_yolo_files += glob.glob(os.path.join("outputs", "**", "*.pt"), recursive=True)

    _fracture_class_names = {c.replace(' ', '_') for c in CLASS_NAMES} | set(CLASS_NAMES)
    yolo_files = []
    for _pt in _raw_yolo_files:
        try:
            from ultralytics import YOLO as _YOLO
            _m = _YOLO(_pt)
            _model_classes = set(_m.names.values())
            if len(_m.names) == 8 and _model_classes <= (_fracture_class_names | {c.replace('_', ' ') for c in _fracture_class_names}):
                yolo_files.append(_pt)
            else:
                logger.info(f"  Skipping {os.path.basename(_pt)} (nc={len(_m.names)}, not a fracture classifier)")
        except Exception:
            pass  # unreadable or non-YOLO .pt — skip silently

    if not model_files and not yolo_files:
        logger.error("No models found")
        return

    benchmark_results = []
    logger.info(f"Found {len(model_files)} models to evaluate.")

    for pth_path in model_files:
        model_filename = os.path.basename(pth_path)
        display_name = model_filename
        if model_filename == "best.pth":
            parent = os.path.basename(os.path.dirname(pth_path))
            display_name = f"{parent}_{model_filename}"
        
        logger.info(f"--- Evaluating {display_name} ---")

        strategies = []
        if "rad_dino" in display_name or "rad_dino" in pth_path:
            strategies.append(("RadDino", None))
        else:
            strategies.append(("Original", get_model_original))
            strategies.append(("Fixed", get_model_fixed))
        
        try:
             checkpoint = torch.load(pth_path, map_location=DEVICE)
             if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                 state_dict = checkpoint['model_state_dict']
             elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                 state_dict = checkpoint['state_dict']
             else:
                 state_dict = checkpoint 
        except Exception as e:
            logger.error(f"Cannot load file {display_name}: {e}")
            continue
            
        best_acc, best_f1, best_strat = -1.0, -1.0, "None"
        
        for strat_name, constructor in strategies:
            try:
                model, is_rad = None, False
                if strat_name == "RadDino":
                    is_rad = True
                    head_type = "linear"
                    for k in state_dict.keys():
                        if "classifier.0.weight" in k: head_type = "mlp"; break
                    model = RadDinoClassifier(len(CLASS_NAMES), head_type=head_type)
                    model.load_state_dict(state_dict, strict=False)
                else:
                    model = constructor(model_filename, len(CLASS_NAMES))
                    if model is None: continue
                    try: model.load_state_dict(state_dict, strict=True)
                    except: model.load_state_dict(state_dict, strict=False)
                
                acc, f1 = evaluate_single_pass(model, val_loader, is_rad_dino=is_rad)
                logger.info(f"  [{strat_name}] Acc: {acc:.4f}")
                
                if acc > best_acc:
                    best_acc, best_f1, best_strat = acc, f1, strat_name
                
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:
                pass
        
        if best_acc > -1.0:
            logger.info(f"  >> BEST: {best_strat} ({best_acc:.4f})")
            benchmark_results.append({
                "Model": display_name,
                "Accuracy": best_acc,
                "F1 Macro": best_f1,
                "Best Logic": best_strat
            })
        else:
            logger.warning("  >> ALL FAILED")

    # Evaluate YOLO (.pt) models
    if yolo_files:
        logger.info(f"Found {len(yolo_files)} YOLO model(s) to evaluate.")
        for pt_path in yolo_files:
            # Build a readable display name (e.g. outputs/.../best.pt → parent/best.pt)
            rel = os.path.relpath(pt_path)
            parts = rel.replace("\\", "/").split("/")
            display_name = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            logger.info(f"--- Evaluating YOLO: {display_name} ---")
            acc, f1 = evaluate_yolo_model(pt_path, val_csv)
            if acc is not None:
                logger.info(f"  >> YOLO Acc: {acc:.4f}, F1 Macro: {f1:.4f}")
                benchmark_results.append({
                    "Model": display_name,
                    "Accuracy": acc,
                    "F1 Macro": f1,
                    "Best Logic": "YOLO"
                })
            else:
                logger.warning(f"  >> YOLO evaluation failed for {display_name}")
    else:
        logger.info("No fracture-class YOLO (.pt) models found.")

    if benchmark_results:
        df = pd.DataFrame(benchmark_results).sort_values(by="F1 Macro", ascending=False)
        print("\n" + "="*60)
        print(" FINAL BENCHMARK (Combined Logic) ")
        print("="*60)
        print(df.to_string(index=False))
        out_csv = os.path.join("outputs", "model_benchmark_combined.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nSaved to {out_csv}")
    else:
        print("No results.")

if __name__ == "__main__":
    main()
