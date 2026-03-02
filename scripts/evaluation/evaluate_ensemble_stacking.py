import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torchvision import transforms
from PIL import Image
import joblib
import logging

# Add project root and src to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
project_root = _PROJECT_ROOT
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from medai.modules.ensemble_module import EnsembleModule

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("EvaluateEnsemble")

# Constants
CLASS_NAMES = [
    "Comminuted", "Greenstick", "Healthy", "Oblique", 
    "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"
]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
BATCH_SIZE = 16

class SimpleTestDataset(Dataset):
    def __init__(self, root_dir, class_names):
        self.root_dir = root_dir
        self.class_names = class_names
        self.samples = []
        
        for idx, cls in enumerate(class_names):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                cls_dir = os.path.join(root_dir, cls.replace(" ", "_"))
                if not os.path.exists(cls_dir):
                    continue
            
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, img_name), idx))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return path, label

def evaluate_stacking_ensemble():
    checkpoints_dir = os.path.join(project_root, "outputs", "cross_validation")
    stacker_path = os.path.join(project_root, "outputs", "stacker.joblib")
    dataset_test_dir = os.path.join(_PROJECT_ROOT, "balanced_augmented_dataset", "test")
    
    # Models to include
    model_names = ["maxvit", "yolo", "hypercolumn_cbam_densenet169", "rad_dino"]

    # 1. Initialize Ensemble Module
    logger.info("Initializing Ensemble Module...")
    ensemble = EnsembleModule(
        class_names=CLASS_NAMES,
        model_names=model_names,
        checkpoints_dir=checkpoints_dir,
        num_classes=8,
        device=DEVICE
    )
    
    # 2. Load Stacker
    if os.path.exists(stacker_path):
        ensemble.stacker = joblib.load(stacker_path)
        logger.info(f"✅ Loaded stacker from {stacker_path}")
    else:
        logger.error(f"❌ Stacker not found at {stacker_path}")
        return

    # 3. Setup Dataset
    dataset = SimpleTestDataset(dataset_test_dir, CLASS_NAMES)
    loader = DataLoader(dataset, batch_size=1, shuffle=False) # Stick to batch size 1 for run_ensemble compatibility
    
    logger.info(f"Evaluating {len(dataset)} images...")

    results = {
        "Ensemble (Stacked)": {"y_true": [], "y_pred": []}
    }
    for name in model_names:
        results[name] = {"y_true": [], "y_pred": []}

    # 4. Inference Loop
    for path, label in tqdm(loader, desc="Testing"):
        img_path = path[0]
        true_idx = label.item()
        
        # Run ensemble inference
        # This returns individual predictions AND ensemble prediction
        with torch.no_grad():
            out = ensemble.run_ensemble(img_path, use_stacking=True)
            
        if "error" in out:
            continue
            
        # Record Ensemble Prediction
        try:
            ensemble_pred_idx = CLASS_NAMES.index(out["ensemble_prediction"])
            results["Ensemble (Stacked)"]["y_true"].append(true_idx)
            results["Ensemble (Stacked)"]["y_pred"].append(ensemble_pred_idx)
        except:
            pass
            
        # Record Individual Model Predictions
        for name in model_names:
            if name in out["individual_predictions"]:
                try:
                    pred_class = out["individual_predictions"][name]["class"]
                    pred_idx = CLASS_NAMES.index(pred_class)
                    results[name]["y_true"].append(true_idx)
                    results[name]["y_pred"].append(pred_idx)
                except:
                    pass

    # 5. Calculate Metrics and Compare
    logger.info("\n" + "="*60)
    logger.info(" FINAL COMPARISON: INDIVIDUAL MODELS VS STACKED ENSEMBLE ")
    logger.info("="*60)
    
    comparison_data = []
    
    for name, data in results.items():
        if not data["y_true"]:
            continue
        acc = accuracy_score(data["y_true"], data["y_pred"])
        f1 = f1_score(data["y_true"], data["y_pred"], average='macro')
        comparison_data.append({
            "Model": name,
            "Accuracy": acc,
            "F1 Macro": f1
        })
        
    df = pd.DataFrame(comparison_data).sort_values(by="Accuracy", ascending=False)
    print(df.to_string(index=False))
    
    # Save the dataframe
    out_path = os.path.join(project_root, "outputs", "stacking_comparison_results.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved comparison to {out_path}")

if __name__ == "__main__":
    main_dir = os.path.dirname(os.path.abspath(__file__))
    evaluate_stacking_ensemble()
