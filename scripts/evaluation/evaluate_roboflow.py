#!/usr/bin/env python3
"""
Binary fracture-detection evaluation on the Roboflow external test dataset.

Every image in the Roboflow dataset is a fracture (10 fracture-type folders).
A model prediction is **correct** when it predicts *any* class other than
"Healthy".

Evaluation runs (10 total)
--------------------------
  1-4.  Four individual models  (MaxViT, YOLO, HyperColumn-CBAM-DenseNet169, RAD-DINO)
  5.    Full ensemble – weighted-average strategy
  6.    Full ensemble – stacking strategy
  7-10. Leave-one-out triplets (each omitting one of the four models),
        using weighted-average strategy

Metrics captured per run
------------------------
  Accuracy, Precision, Recall (= Sensitivity = Detection Rate), F1,
  Specificity, NPV, MCC, Cohen's Kappa, AUC (where applicable),
  and per-fracture-type detection breakdown.

Outputs
-------
  outputs/roboflow_eval/roboflow_binary_eval_results.json   (full detail)
  outputs/roboflow_eval/roboflow_binary_eval_summary.csv    (summary table)

Usage
-----
    python scripts/evaluate_roboflow.py
"""

import os
import sys
import json
import warnings
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torchvision.models as models
import timm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Comminuted", "Greenstick", "Healthy", "Oblique",
    "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced",
]
NUM_CLASSES = len(CLASS_NAMES)
HEALTHY_IDX = CLASS_NAMES.index("Healthy")  # 2

# All Roboflow folders (every image is a fracture)
ROBOFLOW_FRACTURE_FOLDERS = [
    "Avulsion fracture",
    "Comminuted fracture",
    "Fracture Dislocation",
    "Greenstick fracture",
    "Hairline Fracture",
    "Impacted fracture",
    "Longitudinal fracture",
    "Oblique fracture",
    "Pathological fracture",
    "Spiral Fracture",
]

ROBOFLOW_DIR = "roboflow_dataset"
MODELS_DIR = "models"
OUTPUT_DIR = "outputs/roboflow_eval"
STACKER_PATH = "outputs/ensemble/stacker.joblib"

DEVICE = torch.device(
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Stacker was trained with models in this exact order
STACKER_MODEL_ORDER = ["maxvit", "yolo", "hypercolumn_cbam_densenet169", "rad_dino"]

# ──────────────────────────────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.shared_mlp(self.avg_pool(x)) + self.shared_mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class HypercolumnCBAMDenseNet169(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        densenet = models.densenet169(weights=None)
        self.features = densenet.features
        self.init_conv = nn.Sequential(
            self.features.conv0, self.features.norm0,
            self.features.relu0, self.features.pool0,
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
        self.cbam = CBAM(1024)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, num_classes))

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
        sz = x_final.shape[2:]
        hyper = torch.cat([
            x_final,
            nn.functional.interpolate(t3_out, size=sz, mode="bilinear", align_corners=False),
            nn.functional.interpolate(t2_out, size=sz, mode="bilinear", align_corners=False),
            nn.functional.interpolate(t1_out, size=sz, mode="bilinear", align_corners=False),
        ], dim=1)
        x = nn.functional.relu(self.bn_fusion(self.fusion_conv(hyper)))
        x = self.cbam(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class RadDinoClassifier(nn.Module):
    def __init__(self, num_classes=8, head_type="linear"):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained("microsoft/rad-dino")
        hidden = self.backbone.config.hidden_size
        if head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(hidden, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )
        else:
            self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, pixel_values):
        cls_emb = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        return self.classifier(cls_emb)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def get_legacy_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_rad_dino_processor():
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained("microsoft/rad-dino")


def _load_state_dict(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def collect_all_images(roboflow_dir: str):
    """Return list of (image_path, folder_name) for all fracture images."""
    samples = []
    for folder_name in ROBOFLOW_FRACTURE_FOLDERS:
        folder = os.path.join(roboflow_dir, folder_name)
        test_folder = os.path.join(folder, "Test")
        search_dir = test_folder if os.path.isdir(test_folder) else folder
        if not os.path.isdir(search_dir):
            print(f"  WARNING: folder not found: {search_dir}")
            continue
        for img_name in sorted(os.listdir(search_dir)):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                samples.append((os.path.join(search_dir, img_name), folder_name))
    return samples


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────

def load_hypercolumn():
    model = HypercolumnCBAMDenseNet169(NUM_CLASSES)
    sd = _load_state_dict(os.path.join(MODELS_DIR, "best_hypercolumn_cbam_densenet169.pth"))
    model.load_state_dict(sd, strict=False)
    return model


def load_maxvit():
    model = timm.create_model("maxvit_tiny_tf_224", pretrained=False)
    if hasattr(model, "head") and hasattr(model.head, "fc") and isinstance(model.head.fc, nn.Linear):
        model.head.fc = nn.Linear(model.head.fc.in_features, NUM_CLASSES)
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
    else:
        model.reset_classifier(num_classes=NUM_CLASSES)
    sd = _load_state_dict(os.path.join(MODELS_DIR, "best_maxvit.pth"))
    model.load_state_dict(sd, strict=True)
    return model


def load_rad_dino():
    sd = _load_state_dict(os.path.join(MODELS_DIR, "best_rad_dino_classifier.pth"))
    head_type = "mlp" if any("classifier.0." in k for k in sd.keys()) else "linear"
    model = RadDinoClassifier(NUM_CLASSES, head_type=head_type)
    model.load_state_dict(sd, strict=False)
    return model


def load_yolo():
    from ultralytics import YOLO
    search = [
        "outputs/yolo_cls_finetune/yolo_cls_ft/weights/best.pt",
        "models/best.pt",
    ]
    for p in search:
        if os.path.exists(p):
            return YOLO(p)
    raise FileNotFoundError(f"YOLO checkpoint not found in {search}")


# ──────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_pytorch(model, image: Image.Image, transform, is_rad_dino=False, rad_processor=None):
    """Return softmax probabilities (numpy, shape [NUM_CLASSES])."""
    model.eval()
    if is_rad_dino:
        inputs = rad_processor(images=image, return_tensors="pt")["pixel_values"].to(DEVICE)
        logits = model(inputs)
    else:
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        logits = model(tensor)
    return torch.softmax(logits, dim=1).cpu().numpy()[0]


def predict_yolo(yolo_model, image_path: str):
    """Return softmax probabilities aligned to CLASS_NAMES."""
    results = yolo_model.predict(image_path, verbose=False)
    raw_probs = results[0].probs.data.cpu().numpy()
    yolo_names = yolo_model.names
    aligned = np.zeros(NUM_CLASSES, dtype=np.float32)
    for yidx, yname in yolo_names.items():
        for cidx, cname in enumerate(CLASS_NAMES):
            if yname.strip().replace("_", " ").lower() == cname.lower():
                aligned[cidx] = raw_probs[yidx]
                break
    s = aligned.sum()
    if s > 0:
        aligned /= s
    return aligned


# ──────────────────────────────────────────────────────────────────────
# Ensemble combination strategies
# ──────────────────────────────────────────────────────────────────────

HYPERCOLUMN_PRIORITY_CLASSES = {"Oblique", "Oblique Displaced", "Transverse", "Transverse Displaced"}


def _is_hypercolumn(name: str) -> bool:
    return "hypercolumn" in name.lower() or "cbam" in name.lower()


def combine_weighted(all_probs: dict):
    """Weighted-average ensemble with hypercolumn-priority logic.

    all_probs: {model_key: np.ndarray of shape (NUM_CLASSES,)}
    Returns: combined probabilities (NUM_CLASSES,)
    """
    names = list(all_probs.keys())
    probs_list = [all_probs[n] for n in names]

    # First pass: equal-weighted to detect preliminary class
    equal_avg = np.mean(probs_list, axis=0)
    prelim_class = CLASS_NAMES[np.argmax(equal_avg)]
    use_hc_priority = prelim_class in HYPERCOLUMN_PRIORITY_CLASSES

    weights = []
    for n in names:
        if use_hc_priority and _is_hypercolumn(n):
            weights.append(1.0)
        else:
            weights.append(1.0)
    weights = np.array(weights)
    weights /= weights.sum()

    combined = np.zeros(NUM_CLASSES, dtype=np.float64)
    for p, w in zip(probs_list, weights):
        combined += p * w
    return combined


def combine_stacking(all_probs: dict, stacker):
    """Stacking ensemble: feed concatenated per-model probs into trained meta-classifier.

    The stacker expects features ordered as STACKER_MODEL_ORDER.
    all_probs: {model_key: np.ndarray of shape (NUM_CLASSES,)}
    Returns: combined probabilities (NUM_CLASSES,)
    """
    feat_parts = []
    for model_key in STACKER_MODEL_ORDER:
        if model_key in all_probs:
            feat_parts.append(all_probs[model_key])
        else:
            feat_parts.append(np.zeros(NUM_CLASSES, dtype=np.float32))
    feat = np.concatenate(feat_parts).reshape(1, -1)
    return stacker.predict_proba(feat)[0]


# ──────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────

def compute_binary_metrics(y_true, y_pred, y_scores):
    """Compute binary fracture-detection metrics.

    y_true:   1 = fracture (ground truth, all 1 for roboflow)
    y_pred:   1 if model predicts any non-Healthy class, else 0
    y_scores: P(fracture) = 1 - P(Healthy)
    """
    m = {}
    m["n_samples"] = len(y_true)

    # Core classification metrics
    m["accuracy"] = float(accuracy_score(y_true, y_pred))
    m["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    m["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    m["sensitivity"] = m["recall"]
    m["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    m["detection_rate"] = m["recall"]

    # MCC and Cohen's Kappa
    try:
        m["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        m["mcc"] = None
    try:
        m["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        m["cohen_kappa"] = None

    # Confusion matrix: [TN, FP; FN, TP]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    m["tp"] = int(tp)
    m["tn"] = int(tn)
    m["fp"] = int(fp)
    m["fn"] = int(fn)

    # Specificity = TN / (TN + FP)
    m["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    # NPV = TN / (TN + FN)
    m["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else None

    # AUC – only meaningful when both classes present in y_true
    try:
        if len(set(y_true)) > 1:
            m["auc"] = float(roc_auc_score(y_true, y_scores))
        else:
            m["auc"] = None
            m["mean_fracture_score"] = float(np.mean(y_scores))
    except Exception:
        m["auc"] = None

    return m


def compute_per_class_breakdown(predictions_by_folder):
    """Per fracture-type detection stats."""
    breakdown = {}
    for folder, preds in sorted(predictions_by_folder.items()):
        total = len(preds)
        correct = sum(preds)
        breakdown[folder] = {
            "total": total,
            "detected": correct,
            "missed": total - correct,
            "detection_rate": round(correct / total, 4) if total else 0.0,
        }
    return breakdown


# ──────────────────────────────────────────────────────────────────────
# Core evaluation driver
# ──────────────────────────────────────────────────────────────────────

def run_binary_eval(run_name, samples, prob_fn, description=""):
    """Evaluate a single config on all roboflow images (binary: fracture vs healthy).

    prob_fn(image_path, pil_image) -> np.ndarray of shape (NUM_CLASSES,)
    """
    print(f"\n{'='*70}")
    print(f"  {run_name}")
    if description:
        print(f"  {description}")
    print(f"{'='*70}")
    print(f"  Samples: {len(samples)}")

    y_true, y_pred, y_scores = [], [], []
    by_folder = defaultdict(list)

    for img_path, folder_name in tqdm(samples, desc=f"  {run_name}", leave=False):
        try:
            img = Image.open(img_path).convert("RGB")
            probs = prob_fn(img_path, img)
            pred_idx = int(np.argmax(probs))
            is_fracture = int(pred_idx != HEALTHY_IDX)
            fracture_score = 1.0 - float(probs[HEALTHY_IDX])

            y_true.append(1)  # all images are fractures
            y_pred.append(is_fracture)
            y_scores.append(fracture_score)
            by_folder[folder_name].append(is_fracture)
        except Exception as e:
            print(f"    SKIP {os.path.basename(img_path)}: {e}")

    metrics = compute_binary_metrics(y_true, y_pred, y_scores)
    per_class = compute_per_class_breakdown(by_folder)
    metrics["per_class"] = per_class

    # Print summary
    print(f"    Detection rate: {metrics['detection_rate']:.4f}  "
          f"({metrics['tp']}/{metrics['n_samples']})")
    print(f"    Accuracy:       {metrics['accuracy']:.4f}")
    print(f"    Precision:      {metrics['precision']:.4f}")
    print(f"    Recall:         {metrics['recall']:.4f}")
    print(f"    F1:             {metrics['f1']:.4f}")
    if metrics['mcc'] is not None:
        print(f"    MCC:            {metrics['mcc']:.4f}")
    if metrics.get('mean_fracture_score') is not None:
        print(f"    Mean P(frac):   {metrics['mean_fracture_score']:.4f}")
    print(f"    Per-class breakdown:")
    for folder, stats in per_class.items():
        print(f"      {folder:30s}  {stats['detected']}/{stats['total']}  "
              f"({stats['detection_rate']:.2%})")

    return {"run_name": run_name, "description": description, "metrics": metrics}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Roboflow dataset: {ROBOFLOW_DIR}")

    # Collect ALL images (all 10 fracture folders)
    samples = collect_all_images(ROBOFLOW_DIR)
    print(f"Total fracture images: {len(samples)}")
    if not samples:
        print("ERROR: No images found.")
        return

    # Preprocessing
    legacy_transform = get_legacy_transform()
    rad_processor = get_rad_dino_processor()

    # ── Load all four models once ─────────────────────────────────────
    print("\nLoading models...")

    print("  Loading HyperColumn-CBAM-DenseNet169...")
    hypercolumn_model = load_hypercolumn()
    hypercolumn_model.to(DEVICE).eval()

    print("  Loading MaxViT...")
    maxvit_model = load_maxvit()
    maxvit_model.to(DEVICE).eval()

    print("  Loading RAD-DINO...")
    rad_dino_model = load_rad_dino()
    rad_dino_model.to(DEVICE).eval()

    print("  Loading YOLO...")
    yolo_model = load_yolo()

    # Load stacker
    print("  Loading stacker meta-classifier...")
    stacker = joblib.load(STACKER_PATH)
    print(f"  Stacker features: {stacker.n_features_in_}, "
          f"model order: {STACKER_MODEL_ORDER}")

    print("All models loaded.\n")

    # ── Build prediction functions ────────────────────────────────────
    # Use factory functions (closures) to avoid late-binding issues with lambdas

    def _make_pytorch_predictor(model, is_rad_dino=False):
        def fn(path, img):
            return predict_pytorch(model, img, legacy_transform,
                                   is_rad_dino=is_rad_dino,
                                   rad_processor=rad_processor if is_rad_dino else None)
        return fn

    def _make_yolo_predictor(model):
        def fn(path, img):
            return predict_yolo(model, path)
        return fn

    model_registry = {
        "maxvit": {
            "display_name": "MaxViT",
            "predict": _make_pytorch_predictor(maxvit_model),
        },
        "yolo": {
            "display_name": "YOLO",
            "predict": _make_yolo_predictor(yolo_model),
        },
        "hypercolumn_cbam_densenet169": {
            "display_name": "HyperColumn-CBAM-DenseNet169",
            "predict": _make_pytorch_predictor(hypercolumn_model),
        },
        "rad_dino": {
            "display_name": "RAD-DINO",
            "predict": _make_pytorch_predictor(rad_dino_model, is_rad_dino=True),
        },
    }

    all_results = []

    # ── Runs 1-4: Individual models ───────────────────────────────────
    print("\n" + "#"*70)
    print("  INDIVIDUAL MODEL EVALUATION (Runs 1-4)")
    print("#"*70)

    for key in STACKER_MODEL_ORDER:
        entry = model_registry[key]
        result = run_binary_eval(
            run_name=entry["display_name"],
            samples=samples,
            prob_fn=entry["predict"],
            description=f"Individual model: {entry['display_name']}",
        )
        all_results.append(result)

    # ── Run 5: Full ensemble – weighted average ───────────────────────
    print("\n" + "#"*70)
    print("  FULL ENSEMBLE EVALUATION (Runs 5-6)")
    print("#"*70)

    def ensemble_weighted_fn(path, img):
        probs_dict = {}
        for key in STACKER_MODEL_ORDER:
            probs_dict[key] = model_registry[key]["predict"](path, img)
        return combine_weighted(probs_dict)

    result = run_binary_eval(
        run_name="Ensemble (Weighted Avg)",
        samples=samples,
        prob_fn=ensemble_weighted_fn,
        description="Full 4-model ensemble with weighted-average strategy",
    )
    all_results.append(result)

    # ── Run 6: Full ensemble – stacking ───────────────────────────────
    def ensemble_stacking_fn(path, img):
        probs_dict = {}
        for key in STACKER_MODEL_ORDER:
            probs_dict[key] = model_registry[key]["predict"](path, img)
        return combine_stacking(probs_dict, stacker)

    result = run_binary_eval(
        run_name="Ensemble (Stacking)",
        samples=samples,
        prob_fn=ensemble_stacking_fn,
        description="Full 4-model ensemble with stacking meta-classifier",
    )
    all_results.append(result)

    # ── Runs 7-10: Leave-one-out triplets ─────────────────────────────
    print("\n" + "#"*70)
    print("  LEAVE-ONE-OUT TRIPLET EVALUATION (Runs 7-10)")
    print("#"*70)

    for leave_out_key in STACKER_MODEL_ORDER:
        triplet_keys = [k for k in STACKER_MODEL_ORDER if k != leave_out_key]
        triplet_names = [model_registry[k]["display_name"] for k in triplet_keys]
        left_out_name = model_registry[leave_out_key]["display_name"]

        run_name = f"Triplet (w/o {left_out_name})"
        description = f"3-model ensemble: {', '.join(triplet_names)} (weighted avg)"

        def make_triplet_fn(keys):
            def fn(path, img):
                probs_dict = {}
                for k in keys:
                    probs_dict[k] = model_registry[k]["predict"](path, img)
                return combine_weighted(probs_dict)
            return fn

        result = run_binary_eval(
            run_name=run_name,
            samples=samples,
            prob_fn=make_triplet_fn(triplet_keys),
            description=description,
        )
        all_results.append(result)

    # ── Save results ──────────────────────────────────────────────────
    # Detailed JSON
    json_path = os.path.join(OUTPUT_DIR, "roboflow_binary_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {json_path}")

    # Summary CSV
    summary_rows = []
    for r in all_results:
        m = r["metrics"]
        row = {
            "Run": r["run_name"],
            "Description": r["description"],
            "N": m["n_samples"],
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "Sensitivity": m["sensitivity"],
            "Specificity": m["specificity"],
            "NPV": m["npv"],
            "MCC": m["mcc"],
            "Cohen_Kappa": m["cohen_kappa"],
            "AUC": m["auc"],
            "Mean_Fracture_Score": m.get("mean_fracture_score"),
            "TP": m["tp"],
            "FN": m["fn"],
            "Detection_Rate": m["detection_rate"],
        }
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUTPUT_DIR, "roboflow_binary_eval_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

    # Print final summary table
    display_cols = ["Run", "Accuracy", "Precision", "Recall", "F1",
                    "MCC", "Detection_Rate", "Mean_Fracture_Score"]
    print("\n" + "="*100)
    print("  ROBOFLOW BINARY FRACTURE DETECTION – ALL 10 RUNS")
    print("="*100)
    print(df[display_cols].to_string(index=False, float_format="%.4f"))
    print("="*100)


if __name__ == "__main__":
    main()
