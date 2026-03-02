"""
Test Weighted Probability Ensemble for any model combo (1-4).

Usage:
    python scripts/testing/test_weighted_combo.py --combo 1
    python scripts/testing/test_weighted_combo.py --combo 2
    python scripts/testing/test_weighted_combo.py --combo all
"""
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.append(_PROJECT_ROOT)
sys.path.append(os.path.join(_PROJECT_ROOT, "src"))

from medai.modules.ensemble_module import EnsembleModule

# ── Combo definitions with per-model weights ──────────────────────
COMBOS = {
    1: {
        "models": ["maxvit", "hypercolumn_cbam_densenet169_focal", "yolo"],
        "weights": {"maxvit": 0.30, "hypercolumn_cbam_densenet169_focal": 0.40, "yolo": 0.30},
    },
    2: {
        "models": ["maxvit", "hypercolumn_cbam_densenet169_focal", "rad_dino"],
        "weights": {"maxvit": 0.30, "hypercolumn_cbam_densenet169_focal": 0.40, "rad_dino": 0.30},
    },
    3: {
        "models": ["maxvit", "yolo", "rad_dino"],
        "weights": {"maxvit": 0.35, "yolo": 0.35, "rad_dino": 0.30},
    },
    4: {
        "models": ["hypercolumn_cbam_densenet169_focal", "rad_dino", "yolo"],
        "weights": {"hypercolumn_cbam_densenet169_focal": 0.40, "rad_dino": 0.30, "yolo": 0.30},
    },
}

CLASS_NAMES = [
    "Comminuted", "Greenstick", "Healthy", "Oblique",
    "Oblique_Displaced", "Spiral", "Transverse", "Transverse_Displaced",
]

LABEL_SWAPS = {
    "Oblique_Displaced": "Oblique",
    "Oblique": "Oblique_Displaced",
    "Transverse_Displaced": "Transverse",
    "Transverse": "Transverse_Displaced",
}


def get_model_predictions(ensemble, img, class_names):
    """Return per-model softmax probability vectors."""
    try:
        input_tensor = ensemble.transforms(img).unsqueeze(0).to(ensemble.device)
        all_probs, valid_models = [], []
        for name, model in ensemble.models.items():
            try:
                if name.lower() == "yolo":
                    probs = model.predict_pil(img)
                elif "rad_dino" in name.lower():
                    from medai.modules.ensemble_module import get_rad_dino_input_tensor
                    rad_tensor = get_rad_dino_input_tensor(img, ensemble.device)
                    with torch.no_grad():
                        logits = model(rad_tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                else:
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                all_probs.append(probs)
                valid_models.append(name)
            except Exception:
                pass
        return all_probs, valid_models
    except Exception:
        return [], []


def weighted_inference(all_probs, valid_models, model_weights, n_classes):
    """Weighted probability averaging → predicted class index."""
    if not all_probs:
        return None
    weighted = np.zeros(n_classes)
    total_w = 0.0
    for probs, name in zip(all_probs, valid_models):
        w = model_weights.get(name, 1.0 / len(valid_models))
        weighted += probs * w
        total_w += w
    if total_w > 0:
        weighted /= total_w
    return int(np.argmax(weighted))


def run_weighted_combo(combo_id: int):
    cfg = COMBOS[combo_id]
    model_names = cfg["models"]
    model_weights = cfg["weights"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints_dir = os.path.join(_PROJECT_ROOT, "outputs", "cross_validation")
    dataset_test = os.path.join(_PROJECT_ROOT, "balanced_augmented_dataset", "test")

    print("=" * 80)
    print(f"WEIGHTED ENSEMBLE — COMBO {combo_id} ({', '.join(model_names)})")
    print("=" * 80)
    for n in model_names:
        print(f"  {n}: {model_weights[n]:.2f}")

    try:
        ensemble = EnsembleModule(
            class_names=CLASS_NAMES, model_names=model_names,
            checkpoints_dir=checkpoints_dir, num_classes=len(CLASS_NAMES), device=device,
        )
        print("  Ensemble loaded.\n")
    except Exception as e:
        print(f"  Error: {e}")
        return

    y_true, y_pred = [], []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(dataset_test, cls_name)
        if not os.path.exists(cls_dir):
            continue
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for img_name in tqdm(imgs, desc=f"Evaluating {cls_name}", leave=False):
            try:
                img = Image.open(os.path.join(cls_dir, img_name)).convert("RGB")
                all_probs, valid = get_model_predictions(ensemble, img, CLASS_NAMES)
                if not all_probs:
                    continue
                pred_idx = weighted_inference(all_probs, valid, model_weights, len(CLASS_NAMES))
                if pred_idx is None:
                    continue
                pred_class = LABEL_SWAPS.get(CLASS_NAMES[pred_idx], CLASS_NAMES[pred_idx])
                y_true.append(cls_idx)
                y_pred.append(CLASS_NAMES.index(pred_class))
            except Exception:
                pass

    if y_true:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        print("\n" + "=" * 80)
        print(f"RESULTS — COMBO {combo_id} (WEIGHTED)")
        print("=" * 80)
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print("-" * 80)
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
        print("=" * 80)
    else:
        print("No samples were evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run weighted-probability ensemble test.")
    parser.add_argument("--combo", type=str, required=True,
                        help="Combo ID (1-4) or 'all'")
    args = parser.parse_args()

    if args.combo.lower() == "all":
        for cid in COMBOS:
            run_weighted_combo(cid)
            print("\n")
    else:
        run_weighted_combo(int(args.combo))
