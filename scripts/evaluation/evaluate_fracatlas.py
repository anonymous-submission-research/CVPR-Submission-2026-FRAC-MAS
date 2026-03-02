#!/usr/bin/env python3
"""
Binary fracture-detection evaluation on the FracAtlas external test dataset.

Instead of using the official test.csv (which only contains fractured images),
we randomly sample N images from each of the two folders:
    fracatlas/images/Fractured/      → label = 1 (fracture)
    fracatlas/images/Non_fractured/  → label = 0 (healthy)

This gives a balanced binary evaluation with true positives AND true negatives,
enabling meaningful Specificity, NPV, MCC, Cohen's Kappa, and AUC.

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
  Specificity, NPV, MCC, Cohen's Kappa, AUC,
  and per-class (Fractured / Non_fractured) breakdown.

  Results are reported at TWO thresholds:
    1. Default (argmax / 0.5) — standard softmax-based decision
    2. Optimal threshold — found via Youden's J statistic on the test set
       (maximises Sensitivity + Specificity − 1)

Outputs
-------
  outputs/fracatlas_eval/fracatlas_binary_eval_results.json   (full detail)
  outputs/fracatlas_eval/fracatlas_binary_eval_summary.csv    (summary table)
  outputs/fracatlas_eval/fracatlas_roc_curves.png             (ROC curves)
  outputs/fracatlas_eval/fracatlas_roc_curves_optimal.png     (ROC with optimal pts)

Usage
-----
    python scripts/evaluate_fracatlas.py
    python scripts/evaluate_fracatlas.py --n 150   # override sample size per class
"""

import os
import sys
import json
import random
import argparse
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

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve,
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

FRACATLAS_DIR = "fracatlas/images"
FRACTURED_DIR = os.path.join(FRACATLAS_DIR, "Fractured")
NON_FRACTURED_DIR = os.path.join(FRACATLAS_DIR, "Non_fractured")

DATASET_CSV = "fracatlas/dataset.csv"
MODELS_DIR = "models"
OUTPUT_DIR = "outputs/fracatlas_eval"
STACKER_PATH = "outputs/ensemble/stacker.joblib"

SAMPLES_PER_CLASS = 100  # default: 100 fractured + 100 non-fractured = 200 total
RANDOM_SEED = 42

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


def collect_samples(n_per_class: int, seed: int, hand_only: bool = False):
    """Randomly sample n_per_class images from Fractured and Non_fractured folders.

    If hand_only=True, filter to images where the 'hand' column == 1 in
    fracatlas/dataset.csv before sampling.

    Returns list of (image_path, label, source_folder) tuples.
      label = 1  → fracture
      label = 0  → healthy / non-fractured
    """
    rng = random.Random(seed)

    # Build whitelist of hand images if requested
    hand_whitelist = None
    if hand_only:
        df = pd.read_csv(DATASET_CSV)
        hand_whitelist = set(df.loc[df["hand"] == 1, "image_id"].tolist())
        print(f"  Hand-only filter: {len(hand_whitelist)} hand images in dataset.csv")

    samples = []
    for folder, label, source_name in [
        (FRACTURED_DIR, 1, "Fractured"),
        (NON_FRACTURED_DIR, 0, "Non_fractured"),
    ]:
        all_imgs = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
        if hand_whitelist is not None:
            all_imgs = [f for f in all_imgs if f in hand_whitelist]
            print(f"    {source_name}: {len(all_imgs)} hand images available")
        n = min(n_per_class, len(all_imgs))
        chosen = rng.sample(all_imgs, n)
        for fname in chosen:
            samples.append((os.path.join(folder, fname), label, source_name))

    rng.shuffle(samples)
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
    """Weighted-average ensemble with hypercolumn-priority logic."""
    names = list(all_probs.keys())
    probs_list = [all_probs[n] for n in names]

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
    """Stacking ensemble: feed concatenated per-model probs into trained meta-classifier."""
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

def compute_binary_metrics_at_threshold(y_true, y_scores, threshold):
    """Compute binary fracture-detection metrics at a given threshold.

    y_true:     1 = fracture, 0 = healthy (ground truth)
    y_scores:   P(fracture) = 1 - P(Healthy)
    threshold:  classify as fracture when y_scores >= threshold
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    m = {}
    m["threshold"] = float(threshold)
    m["n_samples"] = len(y_true)
    m["n_positive"] = int(y_true.sum())
    m["n_negative"] = m["n_samples"] - m["n_positive"]

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

    # AUC (threshold-independent, but store here for convenience)
    try:
        if len(set(y_true)) > 1:
            m["auc"] = float(roc_auc_score(y_true, y_scores))
        else:
            m["auc"] = None
    except Exception:
        m["auc"] = None

    return m


def find_optimal_threshold(y_true, y_scores, method="youden"):
    """Find optimal binary classification threshold.

    Methods:
      'youden'  — maximise Youden's J = Sensitivity + Specificity − 1
      'f1'      — maximise F1 score

    Returns (best_threshold, best_score).
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    if method == "youden":
        j_scores = tpr - fpr  # = sensitivity + specificity - 1
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx]), float(j_scores[best_idx])
    elif method == "f1":
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.01, 0.99, 200):
            preds = (y_scores >= t).astype(int)
            f = f1_score(y_true, preds, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = t
        return float(best_t), float(best_f1)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_per_class_breakdown(predictions_by_folder):
    """Per source-folder detection stats."""
    breakdown = {}
    for folder, entries in sorted(predictions_by_folder.items()):
        total = len(entries)
        correct = sum(entries)
        breakdown[folder] = {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
        }
    return breakdown


# ──────────────────────────────────────────────────────────────────────
# Core evaluation driver
# ──────────────────────────────────────────────────────────────────────

def run_binary_eval(run_name, samples, prob_fn, description=""):
    """Evaluate a single config on FracAtlas samples (binary: fracture vs healthy).

    Returns results at BOTH the default threshold (argmax) and the optimal
    threshold (Youden's J), plus ROC curve data for plotting.

    samples: list of (image_path, label, source_folder)
    prob_fn(image_path, pil_image) -> np.ndarray of shape (NUM_CLASSES,)
    """
    print(f"\n{'='*70}")
    print(f"  {run_name}")
    if description:
        print(f"  {description}")
    print(f"{'='*70}")
    print(f"  Samples: {len(samples)}")

    y_true, y_scores = [], []
    by_folder_scores = defaultdict(list)  # folder -> list of (gt_label, score)

    for img_path, gt_label, source_folder in tqdm(samples, desc=f"  {run_name}", leave=False):
        try:
            img = Image.open(img_path).convert("RGB")
            probs = prob_fn(img_path, img)
            fracture_score = 1.0 - float(probs[HEALTHY_IDX])

            y_true.append(gt_label)
            y_scores.append(fracture_score)
            by_folder_scores[source_folder].append((gt_label, fracture_score))
        except Exception as e:
            print(f"    SKIP {os.path.basename(img_path)}: {e}")

    y_true_np = np.asarray(y_true)
    y_scores_np = np.asarray(y_scores)

    # ── Default threshold (argmax ≈ 0.5) ──
    DEFAULT_THRESHOLD = 0.5
    default_metrics = compute_binary_metrics_at_threshold(y_true_np, y_scores_np, DEFAULT_THRESHOLD)

    # Per-class breakdown at default threshold
    by_folder_default = defaultdict(list)
    for folder, entries in by_folder_scores.items():
        for gt, score in entries:
            pred = int(score >= DEFAULT_THRESHOLD)
            by_folder_default[folder].append(int(pred == gt))
    default_metrics["per_class"] = compute_per_class_breakdown(dict(by_folder_default))

    # ── Optimal threshold (Youden's J) ──
    opt_threshold, opt_j = find_optimal_threshold(y_true_np, y_scores_np, method="youden")
    optimal_metrics = compute_binary_metrics_at_threshold(y_true_np, y_scores_np, opt_threshold)
    optimal_metrics["youden_j"] = opt_j

    # Per-class breakdown at optimal threshold
    by_folder_opt = defaultdict(list)
    for folder, entries in by_folder_scores.items():
        for gt, score in entries:
            pred = int(score >= opt_threshold)
            by_folder_opt[folder].append(int(pred == gt))
    optimal_metrics["per_class"] = compute_per_class_breakdown(dict(by_folder_opt))

    # ── ROC curve data ──
    fpr, tpr, _ = roc_curve(y_true_np, y_scores_np)

    # ── Print summary ──
    def _print_metrics(label, m):
        print(f"\n    --- {label} (threshold = {m['threshold']:.4f}) ---")
        print(f"    Accuracy:       {m['accuracy']:.4f}")
        print(f"    Precision:      {m['precision']:.4f}")
        print(f"    Recall:         {m['recall']:.4f}  "
              f"({m['tp']}/{m['n_positive']} fractures detected)")
        print(f"    F1:             {m['f1']:.4f}")
        if m['specificity'] is not None:
            print(f"    Specificity:    {m['specificity']:.4f}  "
                  f"({m['tn']}/{m['n_negative']} healthy correct)")
        if m['npv'] is not None:
            print(f"    NPV:            {m['npv']:.4f}")
        if m['mcc'] is not None:
            print(f"    MCC:            {m['mcc']:.4f}")
        if m.get('auc') is not None:
            print(f"    AUC:            {m['auc']:.4f}")
        if m.get('cohen_kappa') is not None:
            print(f"    Cohen's Kappa:  {m['cohen_kappa']:.4f}")
        if "per_class" in m:
            print(f"    Per-class breakdown:")
            for folder, stats in m["per_class"].items():
                label_str = "fracture" if folder == "Fractured" else "healthy"
                print(f"      {folder:20s} ({label_str:>8s})  "
                      f"{stats['correct']}/{stats['total']}  ({stats['accuracy']:.2%})")

    _print_metrics("Default", default_metrics)
    _print_metrics("Optimal (Youden's J)", optimal_metrics)

    return {
        "run_name": run_name,
        "description": description,
        "default_metrics": default_metrics,
        "optimal_metrics": optimal_metrics,
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FracAtlas binary fracture evaluation")
    parser.add_argument("--n", type=int, default=SAMPLES_PER_CLASS,
                        help=f"Number of images to sample per class (default: {SAMPLES_PER_CLASS})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed for reproducibility (default: {RANDOM_SEED})")
    parser.add_argument("--hand-only", action="store_true",
                        help="Filter to hand X-rays only (anatomically matched to training data)")
    args = parser.parse_args()

    # Adjust output directory when hand-only
    output_dir = OUTPUT_DIR + ("_hand" if args.hand_only else "")
    os.makedirs(output_dir, exist_ok=True)

    subset_label = "Hand X-rays only" if args.hand_only else "All body parts"
    print(f"Device: {DEVICE}")
    print(f"FracAtlas image dir: {FRACATLAS_DIR}")
    print(f"Subset: {subset_label}")
    print(f"Samples per class: {args.n}")
    print(f"Random seed: {args.seed}")

    # Collect balanced samples
    samples = collect_samples(args.n, args.seed, hand_only=args.hand_only)
    n_pos = sum(1 for _, l, _ in samples if l == 1)
    n_neg = sum(1 for _, l, _ in samples if l == 0)
    print(f"Total samples: {len(samples)}  (Fractured: {n_pos}, Non_fractured: {n_neg})")
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
    # Save sampling info for reproducibility
    sampling_info = {
        "dataset": "FracAtlas",
        "subset": "hand" if args.hand_only else "all",
        "samples_per_class": args.n,
        "random_seed": args.seed,
        "total_samples": len(samples),
        "n_fractured": n_pos,
        "n_non_fractured": n_neg,
        "fractured_dir": FRACTURED_DIR,
        "non_fractured_dir": NON_FRACTURED_DIR,
        "sampled_files": {
            "Fractured": sorted([os.path.basename(p) for p, l, _ in samples if l == 1]),
            "Non_fractured": sorted([os.path.basename(p) for p, l, _ in samples if l == 0]),
        },
    }

    # Strip ROC arrays from JSON (store separately) to keep JSON readable
    results_for_json = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k != "roc"}
        results_for_json.append(r_copy)

    # Detailed JSON
    output_data = {
        "sampling_info": sampling_info,
        "results": results_for_json,
    }
    json_path = os.path.join(output_dir, "fracatlas_binary_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nDetailed results saved to {json_path}")

    # ── ROC Curves ────────────────────────────────────────────────────
    # Plot 1: All 10 ROC curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for r in all_results:
        auc_val = r["default_metrics"].get("auc")
        auc_str = f" (AUC={auc_val:.3f})" if auc_val is not None else ""
        ax.plot(r["roc"]["fpr"], r["roc"]["tpr"],
                label=f"{r['run_name']}{auc_str}", linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    subset_suffix = " (Hand X-rays)" if args.hand_only else ""
    ax.set_title(f"FracAtlas{subset_suffix} \u2013 ROC Curves (All 10 Runs)", fontsize=14)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)
    roc_path = os.path.join(output_dir, "fracatlas_roc_curves.png")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)
    print(f"ROC curves saved to {roc_path}")

    # Plot 2: ROC curves with optimal operating points marked
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = plt.cm.tab10.colors
    for i, r in enumerate(all_results):
        auc_val = r["default_metrics"].get("auc")
        auc_str = f" (AUC={auc_val:.3f})" if auc_val is not None else ""
        color = colors[i % len(colors)]
        ax.plot(r["roc"]["fpr"], r["roc"]["tpr"],
                label=f"{r['run_name']}{auc_str}", linewidth=1.5, color=color)
        # Mark optimal operating point
        opt = r["optimal_metrics"]
        if opt["specificity"] is not None:
            opt_fpr = 1.0 - opt["specificity"]
            opt_tpr = opt["sensitivity"]
            ax.plot(opt_fpr, opt_tpr, "o", color=color, markersize=8,
                    markeredgecolor="black", markeredgewidth=1.0)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate (1 \u2212 Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(f"FracAtlas{subset_suffix} \u2013 ROC Curves with Optimal Thresholds (\u25cf)", fontsize=14)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)
    roc_opt_path = os.path.join(output_dir, "fracatlas_roc_curves_optimal.png")
    fig.tight_layout()
    fig.savefig(roc_opt_path, dpi=200)
    plt.close(fig)
    print(f"ROC curves (with optimal points) saved to {roc_opt_path}")

    # ── Summary CSV (both thresholds) ─────────────────────────────────
    summary_rows = []
    for r in all_results:
        for label, mkey in [("default", "default_metrics"), ("optimal", "optimal_metrics")]:
            m = r[mkey]
            row = {
                "Run": r["run_name"],
                "Threshold_Type": label,
                "Threshold": m["threshold"],
                "Description": r["description"],
                "N": m["n_samples"],
                "N_Positive": m["n_positive"],
                "N_Negative": m["n_negative"],
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
                "TP": m["tp"],
                "TN": m["tn"],
                "FP": m["fp"],
                "FN": m["fn"],
                "Detection_Rate": m["detection_rate"],
            }
            if label == "optimal":
                row["Youden_J"] = m.get("youden_j")
            summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, "fracatlas_binary_eval_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

    # ── Print final summary tables ────────────────────────────────────
    display_cols = ["Run", "Threshold", "Accuracy", "Precision", "Recall",
                    "Specificity", "F1", "MCC", "AUC"]

    print("\n" + "="*110)
    print(f"  FRACATLAS{subset_suffix.upper()} BINARY FRACTURE DETECTION \u2013 DEFAULT THRESHOLD (0.5)  "
          f"({n_pos} fracture + {n_neg} healthy = {len(samples)} images)")
    print("="*110)
    df_default = df[df["Threshold_Type"] == "default"]
    print(df_default[display_cols].to_string(index=False, float_format="%.4f"))

    print("\n" + "="*110)
    print(f"  FRACATLAS{subset_suffix.upper()} BINARY FRACTURE DETECTION \u2013 OPTIMAL THRESHOLD (Youden's J)  "
          f"({n_pos} fracture + {n_neg} healthy = {len(samples)} images)")
    print("="*110)
    df_optimal = df[df["Threshold_Type"] == "optimal"]
    print(df_optimal[display_cols].to_string(index=False, float_format="%.4f"))
    print("="*110)


if __name__ == "__main__":
    main()
