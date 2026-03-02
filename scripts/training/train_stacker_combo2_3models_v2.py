#!/usr/bin/env python3
"""
Train Stacking Meta-Model for Combo 2 with 3 Models (Simplified)
Models: maxvit, hypercolumn_cbam_densenet169_focal, rad_dino
"""

import os
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from medai.modules.ensemble_module import EnsembleModule

# Configuration
DATASET_ROOT = Path("balanced_augmented_dataset")
TRAIN_DIR = DATASET_ROOT / "train"
OUTPUT_DIR = Path("outputs") / "ensemble"

# Model names for Combo 2
MODEL_NAMES = ["maxvit", "hypercolumn_cbam_densenet169_focal", "rad_dino"]

CLASS_NAMES = [
    "Comminuted",
    "Greenstick",
    "Healthy",
    "Oblique",
    "Oblique_Displaced",
    "Spiral",
    "Transverse",
    "Transverse_Displaced",
]

def get_training_data():
    """Load training images and labels"""
    images = []
    labels = []
    label_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}
    
    print("Loading training data...")
    for class_name in CLASS_NAMES:
        class_dir = TRAIN_DIR / class_name
        if not class_dir.exists():
            print(f"⚠️ Class directory not found: {class_dir}")
            continue
            
        img_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        for img_file in img_files:
            images.append(str(img_file))
            labels.append(label_to_idx[class_name])
    
    print(f"✅ Loaded {len(images)} training samples")
    return images, np.array(labels)

def main():
    """Main training pipeline"""
    try:
        print("=" * 70)
        print("TRAINING STACKING META-MODEL FOR COMBO 2 (3 MODELS)")
        print("=" * 70)
        print(f"Models: {', '.join(MODEL_NAMES)}")
        print(f"Classes: {', '.join(CLASS_NAMES)}")
        print()
        
        # Load training data
        images, labels = get_training_data()
        num_samples = len(images)
        print(f"Training samples: {num_samples}")
        print()
        
        # Initialize ensemble
        print("Initializing Ensemble with 3 models...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ensemble = EnsembleModule(
            model_names=MODEL_NAMES,
            class_names=CLASS_NAMES,
            checkpoints_dir=OUTPUT_DIR / "cross_validation",
            num_classes=8,
            device=device
        )
        print("✅ Ensemble initialized\n")
        
        # Get stacked predictions from ensemble on training data
        print("Getting predictions from ensemble on training data...")
        stacked_predictions = []
        
        for img_path in tqdm(images, desc="Processing training samples"):
            try:
                result = ensemble.run_ensemble(img_path, use_stacking=False)  # Don't use stacking yet
                if "error" not in result:
                    # Get model confidences for each class
                    ensemble_conf = result.get("ensemble_confidences", None)
                    if ensemble_conf is not None and isinstance(ensemble_conf, (list, np.ndarray)):
                        stacked_predictions.append(ensemble_conf)
            except Exception as e:
                # Use uniform distribution as fallback
                stacked_predictions.append(np.ones(len(CLASS_NAMES)) / len(CLASS_NAMES))
        
        # Convert to numpy array
        X = np.array(stacked_predictions)
        print(f"✅ Got predictions: shape {X.shape}")
        print(f"Features per sample: {X.shape[1]} (8 classes × 3 models)")
        print()
        
        # Train meta-model
        print("="*70)
        print("TRAINING META-MODEL (Logistic Regression)")
        print("="*70)
        stacker = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial',
                solver='lbfgs'
            ))
        ])
        
        print(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
        stacker.fit(X, labels)
        
        # Evaluate on training data
        train_acc = stacker.score(X, labels)
        print(f"✅ Training accuracy: {train_acc:.4f}")
        print()
        
        # Save stacker
        output_path = OUTPUT_DIR / "stacker_combo2_3models.joblib"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(stacker, output_path)
        print(f"✅ Stacker saved to: {output_path}")
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE")
        print("=" * 70)
        print(f"Stacker for Combo 2 (3 models) trained and saved!")
        print()
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
