"""
Generate additional publication-quality figures for the DeepCK results section.

Outputs (all saved to outputs/figures/):
  1. per_class_coverage.pdf       — Grouped bar chart: accuracy vs conformal coverage per class
  2. confusion_matrix.pdf         — Confusion matrix heatmap for ensemble argmax predictions
  3. gradcam_attention_comparison.pdf — Grouped bar chart: active fraction per model per fracture type
  4. critic_per_class.pdf         — Per-class Critic verdicts with correct/incorrect overlay
  5. confidence_distribution.pdf  — Box/strip plot of confidence for correct vs incorrect predictions

Usage:
    python scripts/additional_viz.py
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFORMAL_RESULTS = ROOT / "outputs" / "conformal_results.json"
CRITIC_RESULTS = ROOT / "outputs" / "critic_evaluation.json"
GRADCAM_RESULTS = ROOT / "outputs" / "gradcam_comparison_results.json"

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ACM-friendly color palette
COLORS = {
    "accuracy":  "#2166ac",  # blue
    "cov_05":    "#d6604d",  # red-orange
    "cov_10":    "#4393c3",  # light blue
    "maxvit":    "#e66101",  # orange
    "hc_cbam":   "#5e3c99",  # purple
    "correct":   "#1b9e77",  # teal
    "incorrect": "#d95f02",  # dark orange
    "confirmed": "#66c2a5",  # green
    "rejected":  "#fc8d62",  # salmon
    "uncertain": "#8da0cb",  # lavender
}

# Abbreviated class names for plots
SHORT_NAMES = {
    "Comminuted": "COM",
    "Greenstick": "GRN",
    "Healthy": "HLT",
    "Oblique": "OBL",
    "Oblique Displaced": "OBL-D",
    "Spiral": "SPR",
    "Transverse": "TRV",
    "Transverse Displaced": "TRV-D",
}

CLASS_ORDER = [
    "Comminuted", "Greenstick", "Healthy", "Oblique",
    "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced",
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Per-class accuracy & conformal coverage grouped bar chart
# ═════════════════════════════════════════════════════════════════════════════
def plot_per_class_coverage():
    cr = load_json(CONFORMAL_RESULTS)

    # Gather per-class stats at both alpha levels
    stats = {}
    for cls in CLASS_ORDER:
        stats[cls] = {"n": 0, "correct": 0, "cov_05": 0, "cov_10": 0}

    for alpha_key, cov_key in [("0.05", "cov_05"), ("0.1", "cov_10")]:
        for s in cr[alpha_key]["per_sample"]:
            cls = s["true_class"]
            if alpha_key == "0.05":
                stats[cls]["n"] += 1
                stats[cls]["correct"] += int(s["argmax_correct"])
            stats[cls][cov_key] += int(s["true_covered"])

    classes = CLASS_ORDER
    n_cls = len(classes)
    x = np.arange(n_cls)
    width = 0.25

    acc_vals = [stats[c]["correct"] / stats[c]["n"] * 100 for c in classes]
    cov05_vals = [stats[c]["cov_05"] / stats[c]["n"] * 100 for c in classes]
    cov10_vals = [stats[c]["cov_10"] / stats[c]["n"] * 100 for c in classes]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars1 = ax.bar(x - width, acc_vals, width, label="Accuracy",
                   color=COLORS["accuracy"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, cov05_vals, width, label=r"Coverage ($\alpha=0.05$)",
                   color=COLORS["cov_05"], edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + width, cov10_vals, width, label=r"Coverage ($\alpha=0.10$)",
                   color=COLORS["cov_10"], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES[c] for c in classes], rotation=0)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.legend(loc="lower left", framealpha=0.9)
    ax.axhline(y=90, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(y=95, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

    # Add value labels on bars below 100%
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h < 100:
                ax.annotate(f"{h:.0f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    out_path = OUT_DIR / "per_class_coverage.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"[1/5] Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Confusion matrix heatmap
# ═════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix():
    cr = load_json(CONFORMAL_RESULTS)

    # Build confusion matrix from alpha=0.1 per-sample (same predictions at both)
    n_cls = len(CLASS_ORDER)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    cls2idx = {c: i for i, c in enumerate(CLASS_ORDER)}

    for s in cr["0.1"]["per_sample"]:
        t_idx = cls2idx[s["true_class"]]
        p_idx = cls2idx[s["argmax_pred"]]
        cm[t_idx, p_idx] += 1

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    short_labels = [SHORT_NAMES[c] for c in CLASS_ORDER]
    ax.set_xticks(np.arange(n_cls))
    ax.set_yticks(np.arange(n_cls))
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_yticklabels(short_labels)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(n_cls):
        for j in range(n_cls):
            val = cm[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=9, fontweight="bold" if i == j else "normal")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    fig.tight_layout()
    out_path = OUT_DIR / "confusion_matrix.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"[2/5] Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. Grad-CAM attention comparison bar chart
# ═════════════════════════════════════════════════════════════════════════════
def plot_gradcam_attention():
    gr = load_json(GRADCAM_RESULTS)

    fracture_types = []
    maxvit_active = []
    hc_active = []

    for entry in gr:
        # Use true_class for label
        ft = entry["true_class"]
        fracture_types.append(ft)
        maxvit_active.append(entry["models"]["maxvit"]["active_fraction"] * 100)
        hc_active.append(entry["models"]["hypercolumn_cbam_densenet169"]["active_fraction"] * 100)

    x = np.arange(len(fracture_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars1 = ax.bar(x - width / 2, maxvit_active, width, label="MaxViT",
                   color=COLORS["maxvit"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, hc_active, width, label="HC-CBAM DenseNet-169",
                   color=COLORS["hc_cbam"], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Active Fraction (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES.get(ft, ft) for ft in fracture_types], rotation=0)
    ax.set_ylim(0, 80)
    ax.legend(loc="upper left", framealpha=0.9)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = OUT_DIR / "gradcam_attention_comparison.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"[3/5] Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Critic per-class results
# ═════════════════════════════════════════════════════════════════════════════
def plot_critic_per_class():
    ce = load_json(CRITIC_RESULTS)

    per_class = defaultdict(lambda: {"total": 0, "correct": 0, "confirmed": 0,
                                      "rejected": 0, "uncertain": 0})
    for s in ce["per_sample"]:
        cls = s["true_class"]
        per_class[cls]["total"] += 1
        per_class[cls]["correct"] += int(s["is_correct"])
        v = s["critic_verdict"].lower()
        if v == "yes":
            per_class[cls]["confirmed"] += 1
        elif v == "no":
            per_class[cls]["rejected"] += 1
        else:
            per_class[cls]["uncertain"] += 1

    classes = CLASS_ORDER
    n_cls = len(classes)
    x = np.arange(n_cls)

    totals = [per_class[c]["total"] for c in classes]
    corrects = [per_class[c]["correct"] for c in classes]
    incorrects = [per_class[c]["total"] - per_class[c]["correct"] for c in classes]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Stacked bar: correct (green) + incorrect (red) — all confirmed
    bars_correct = ax.bar(x, corrects, width=0.6,
                          label="Confirmed (correct)", color=COLORS["confirmed"],
                          edgecolor="white", linewidth=0.5)
    bars_incorrect = ax.bar(x, incorrects, width=0.6, bottom=corrects,
                            label="Confirmed (incorrect)", color=COLORS["incorrect"],
                            edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Number of Samples")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES[c] for c in classes], rotation=0)
    ax.set_ylim(0, max(totals) + 3)
    ax.legend(loc="upper right", framealpha=0.9)

    # Annotate totals
    for i, (t, c_val) in enumerate(zip(totals, corrects)):
        inc = t - c_val
        if inc > 0:
            # Label the incorrect portion
            ax.annotate(f"{inc}",
                        xy=(i, t),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=COLORS["incorrect"],
                        fontweight="bold")

    fig.tight_layout()
    out_path = OUT_DIR / "critic_per_class.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"[4/5] Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. Confidence distribution: correct vs incorrect
# ═════════════════════════════════════════════════════════════════════════════
def plot_confidence_distribution():
    ce = load_json(CRITIC_RESULTS)

    correct_conf = []
    incorrect_conf = []

    for s in ce["per_sample"]:
        conf = s["pred_confidence"]
        if s["is_correct"]:
            correct_conf.append(conf)
        else:
            incorrect_conf.append(conf)

    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Box plots side by side
    bp_data = [correct_conf, incorrect_conf]
    bp = ax.boxplot(bp_data, positions=[1, 2], widths=0.5,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=5),
                    medianprops=dict(color="black", linewidth=1.5))

    bp["boxes"][0].set_facecolor(COLORS["correct"])
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(COLORS["incorrect"])
    bp["boxes"][1].set_alpha(0.7)

    # Overlay strip points (jittered)
    rng = np.random.RandomState(42)
    for i, (data, pos) in enumerate(zip(bp_data, [1, 2])):
        jitter = rng.uniform(-0.12, 0.12, size=len(data))
        color = COLORS["correct"] if i == 0 else COLORS["incorrect"]
        ax.scatter(pos + jitter, data, s=12, alpha=0.5, color=color,
                   edgecolors="black", linewidth=0.3, zorder=3)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Correct\n($N = {len(correct_conf)}$)",
                        f"Incorrect\n($N = {len(incorrect_conf)}$)"])
    ax.set_ylabel("Max Softmax Probability")
    ax.set_ylim(-0.05, 1.1)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

    # Annotate medians and means
    for i, data in enumerate(bp_data):
        median = np.median(data)
        mean = np.mean(data)
        pos = i + 1
        ax.annotate(f"med={median:.3f}\nmean={mean:.3f}",
                    xy=(pos + 0.35, median), fontsize=7.5,
                    ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    fig.tight_layout()
    out_path = OUT_DIR / "confidence_distribution.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"[5/5] Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating additional figures for results.tex ...\n")

    plot_per_class_coverage()
    plot_confusion_matrix()
    plot_gradcam_attention()
    plot_critic_per_class()
    plot_confidence_distribution()

    print(f"\nAll figures saved to {OUT_DIR}/")
