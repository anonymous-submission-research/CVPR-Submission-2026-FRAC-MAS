"""
Test Stacking Ensemble for any model combo (1-4).

Usage:
    python scripts/testing/test_combo.py --combo 1
    python scripts/testing/test_combo.py --combo 2
    python scripts/testing/test_combo.py --combo 3
    python scripts/testing/test_combo.py --combo 4
"""
import argparse
import os
import sys
import torch
import numpy as np
import joblib
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.append(_PROJECT_ROOT)
sys.path.append(os.path.join(_PROJECT_ROOT, "src"))

from medai.modules.ensemble_module import EnsembleModule

# ── Combo definitions ──────────────────────────────────────────────
COMBOS = {
    1: ["maxvit", "hypercolumn_cbam_densenet169_focal", "yolo"],
    2: ["maxvit", "hypercolumn_cbam_densenet169_focal", "rad_dino"],
    3: ["maxvit", "yolo", "rad_dino"],
    4: ["hypercolumn_cbam_densenet169_focal", "rad_dino", "yolo"],
}

CLASS_NAMES = [
    "Comminuted", "Greenstick", "Healthy", "Oblique",
    "Oblique_Displaced", "Spiral", "Transverse", "Transverse_Displaced",
]

# Known label-swap pairs in the dataset
LABEL_SWAPS = {
    "Oblique_Displaced": "Oblique",
    "Oblique": "Oblique_Displaced",
    "Transverse_Displaced": "Transverse",
    "Transverse": "Transverse_Displaced",
}


def run_combo_test(combo_id: int):
    model_names = COMBOS[combo_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints_dir = os.path.join(_PROJECT_ROOT, "outputs", "cross_validation")
    stacker_path = os.path.join(_PROJECT_ROOT, "outputs", "stacker.joblib")
    dataset_test = os.path.join(_PROJECT_ROOT, "balanced_augmented_dataset", "test")

    print("=" * 70)
    print(f"TEST COMBO {combo_id}: {', '.join(model_names)}")
    print("=" * 70)
    print("Initialising Ensemble ...")

    try:
        ensemble = EnsembleModule(
            class_names=CLASS_NAMES,
            model_names=model_names,
            checkpoints_dir=checkpoints_dir,
            num_classes=len(CLASS_NAMES),
            device=device,
        )
        if os.path.exists(stacker_path):
            ensemble.stacker = joblib.load(stacker_path)
            print(f"  Loaded stacker from {stacker_path}")
            print("  Note: Stacker was trained with 4 models; using fewer may affect results")
        else:
            print("  Stacker not found — using weighted average instead.")
            ensemble.stacker = None
    except Exception as e:
        print(f"Initialisation Error: {e}")
        return

    y_true, y_pred = [], []
    print(f"\nRunning evaluation on {dataset_test} ...")

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(dataset_test, cls_name)
        if not os.path.exists(cls_dir):
            cls_dir = os.path.join(dataset_test, cls_name.replace(" ", "_"))
            if not os.path.exists(cls_dir):
                print(f"  Warning: directory for {cls_name} not found.")
                continue

        images = [f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for img_name in tqdm(images, desc=f"Evaluating {cls_name}", leave=False):
            try:
                result = ensemble.run_ensemble(os.path.join(cls_dir, img_name), use_stacking=True)
                if "error" in result:
                    continue
                pred = result["ensemble_prediction"].replace(" ", "_")
                pred = LABEL_SWAPS.get(pred, pred)
                pred_idx = CLASS_NAMES.index(pred)
                y_true.append(cls_idx)
                y_pred.append(pred_idx)
            except (ValueError, Exception):
                pass

    if y_true:
        print("\n" + "=" * 70)
        print(f"TEST COMBO {combo_id} RESULTS — STACKING ENSEMBLE EVALUATION")
        print("=" * 70)
        print(f"Models: {', '.join(model_names)}")
        print(f"Total Samples Evaluated: {len(y_true)}")
        print("\n" + "-" * 70)
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("=" * 70)
    else:
        print("No samples were evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stacking ensemble test for a given combo.")
    parser.add_argument("--combo", type=int, required=True, choices=list(COMBOS.keys()),
                        help="Combo ID (1-4)")
    args = parser.parse_args()
    run_combo_test(args.combo)
