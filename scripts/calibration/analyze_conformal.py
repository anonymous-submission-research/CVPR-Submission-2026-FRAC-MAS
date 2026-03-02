"""
Comprehensive Conformal Prediction Analysis for the Results section.

Produces:
  - outputs/conformal/conformal_calibration.json   (per-alpha calibration details)
  - outputs/conformal/conformal_thresholds.json    (thresholds at each alpha)
  - outputs/conformal/conformal_results.json       (full test-set results: coverage, set sizes, safety cases)
  - outputs/figures/conformal_set_sizes.pdf (histogram of prediction set sizes)
  - outputs/figures/conformal_coverage_table.pdf (coverage vs target table)

Usage:
  python scripts/analyze_conformal.py \
      --checkpoints ./models \
      --models maxvit,yolo,hypercolumn_cbam_densenet169,rad_dino \
      --alphas 0.05,0.10 \
      [--val-npz outputs/ensemble/val_calib.npz]  # skip recomputation if it exists
"""

import os
import sys
import json
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, os.path.abspath('src'))
from medai import app
from medai.uncertainty.conformal import calibrate_conformal, predict_conformal_set


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_csv(path):
    """Load image_path, label pairs from a CSV."""
    rows = []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append((r['image_path'], int(r['label'])))
    return rows


def resolve_path(p):
    for candidate in [p, os.path.join('data', p), os.path.join('.', p)]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(p)


def is_hypercolumn(name):
    return 'hypercolumn' in name.lower() or 'cbam' in name.lower()


def model_probs_single(name, model, pil_img, tensor, device):
    """Get probabilities for a single model."""
    import torch
    if app.is_yolo_model(model):
        return model.predict_pil(pil_img)
    elif app.is_rad_dino_model(name):
        rad_tensor = app.get_rad_dino_input_tensor(pil_img, device)
        with torch.no_grad():
            logits = model(rad_tensor)
        return torch.softmax(logits, dim=1).cpu().numpy()[0]
    else:
        with torch.no_grad():
            out = model(tensor)
        return torch.softmax(out, dim=1).cpu().numpy()[0]


def compute_probs_for_split(rows, models, model_names, device, transforms, hyper_weight):
    """Run inference on a split and return weighted-averaged ensemble probs + labels."""
    import torch
    from PIL import Image

    N = len(rows)
    M = len(model_names)
    C = len(app.CLASS_NAMES)

    model_probs = np.zeros((N, M, C), dtype=np.float32)
    labels = np.zeros((N,), dtype=np.int32)

    for i, (img_path, label) in enumerate(rows):
        try:
            p = resolve_path(img_path)
            pil = Image.open(p).convert('RGB')
        except Exception as e:
            print(f'  Warning: failed to open {img_path}: {e}')
            continue
        tensor = transforms(pil).unsqueeze(0).to(device)
        labels[i] = label
        for j, name in enumerate(model_names):
            model = models[name]
            try:
                probs = model_probs_single(name, model, pil, tensor, device)
            except Exception as e:
                print(f'  Warning: inference failed for {name} on sample {i}: {e}')
                continue
            model_probs[i, j] = probs
        if (i + 1) % 20 == 0 or i == N - 1:
            print(f'  Processed {i + 1}/{N}')

    # weighted average
    is_hyper = [is_hypercolumn(n) for n in model_names]
    ws = np.array([hyper_weight if h else 1.0 for h in is_hyper], dtype=np.float32)
    ws /= ws.sum()
    avg_probs = (model_probs * ws[None, :, None]).sum(axis=1)  # (N, C)
    return avg_probs, labels, model_probs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Conformal Prediction Analysis')
    parser.add_argument('--checkpoints', default='./models')
    parser.add_argument('--models', default='maxvit,yolo,hypercolumn_cbam_densenet169,rad_dino')
    parser.add_argument('--alphas', default='0.05,0.10', help='Comma-separated alpha (miscoverage) levels')
    parser.add_argument('--val-csv', default='balanced_augmented_dataset/val.csv')
    parser.add_argument('--test-csv', default='balanced_augmented_dataset/test.csv')
    parser.add_argument('--val-npz', default='outputs/ensemble/val_calib.npz',
                        help='If exists, skip recomputing val probs')
    parser.add_argument('--hyper-weight', type=float, default=None,
                        help='HyperColumn weight (auto-detect from outputs/ensemble/hypercolumn_weight.txt if omitted)')
    parser.add_argument('--out-dir', default='outputs')
    parser.add_argument('--fig-dir', default='outputs/figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    alphas = [float(a.strip()) for a in args.alphas.split(',')]
    selected_models = [m.strip() for m in args.models.split(',') if m.strip()]

    # Detect hyper-weight
    hyper_weight = args.hyper_weight
    if hyper_weight is None:
        wpath = os.path.join(args.out_dir, 'hypercolumn_weight.txt')
        if os.path.exists(wpath):
            hyper_weight = float(open(wpath).read().strip())
            print(f'Using hypercolumn weight from file: {hyper_weight}')
        else:
            hyper_weight = 1.5
            print(f'Using default hypercolumn weight: {hyper_weight}')

    device = app.get_device()
    print(f'Device: {device}')

    # Load models
    print('Loading models:', selected_models)
    models = app.load_models(args.checkpoints, selected_models, device)
    model_names = list(models.keys())
    print(f'Loaded {len(model_names)} models: {model_names}')
    transforms = app.get_transforms(app.IMG_SIZE)

    # ------------------------------------------------------------------
    # 1. Compute / load validation probabilities for calibration
    # ------------------------------------------------------------------
    if os.path.exists(args.val_npz):
        print(f'\nLoading validation probabilities from {args.val_npz}')
        d = np.load(args.val_npz)
        val_model_probs = d['model_probs']
        val_labels = d['labels']
        npz_model_names = list(d.get('model_names', model_names))
        print(f'  Val samples: {len(val_labels)}, models in npz: {npz_model_names}')

        # weighted average
        is_hyper = [is_hypercolumn(n) for n in npz_model_names]
        ws = np.array([hyper_weight if h else 1.0 for h in is_hyper], dtype=np.float32)
        ws /= ws.sum()
        val_probs = (val_model_probs * ws[None, :, None]).sum(axis=1)
    else:
        print(f'\nComputing validation probabilities...')
        val_rows = load_csv(args.val_csv)
        val_probs, val_labels, _ = compute_probs_for_split(
            val_rows, models, model_names, device, transforms, hyper_weight
        )
        print(f'  Val samples: {len(val_labels)}')

    # ------------------------------------------------------------------
    # 2. Calibrate conformal thresholds at each alpha
    # ------------------------------------------------------------------
    thresholds = {}
    calibration_details = {}
    for alpha in alphas:
        t = calibrate_conformal(val_probs, val_labels, alpha=alpha)
        thresholds[str(alpha)] = float(t)
        # Also compute val-set coverage as sanity check
        n_covered = 0
        set_sizes = []
        for i in range(len(val_labels)):
            cs = predict_conformal_set(val_probs[i], t, app.CLASS_NAMES)
            set_sizes.append(len(cs))
            true_class = app.CLASS_NAMES[val_labels[i]]
            if true_class in cs:
                n_covered += 1
        val_coverage = n_covered / len(val_labels)
        calibration_details[str(alpha)] = {
            'alpha': alpha,
            'target_coverage': 1.0 - alpha,
            'threshold': float(t),
            'val_empirical_coverage': float(val_coverage),
            'val_avg_set_size': float(np.mean(set_sizes)),
            'val_median_set_size': float(np.median(set_sizes)),
            'val_n_samples': len(val_labels),
        }
        print(f'  alpha={alpha:.2f} => threshold={t:.6f}, val coverage={val_coverage:.4f}, avg set size={np.mean(set_sizes):.2f}')

    # Save threshold artifacts
    with open(os.path.join(args.out_dir, 'conformal_thresholds.json'), 'w') as fh:
        json.dump(thresholds, fh, indent=2)
    print(f'Saved thresholds to {os.path.join(args.out_dir, "conformal_thresholds.json")}')

    with open(os.path.join(args.out_dir, 'conformal_calibration.json'), 'w') as fh:
        json.dump(calibration_details, fh, indent=2)
    print(f'Saved calibration details to {os.path.join(args.out_dir, "conformal_calibration.json")}')

    # ------------------------------------------------------------------
    # 3. Compute test-set probabilities
    # ------------------------------------------------------------------
    print(f'\nComputing test-set probabilities...')
    test_rows = load_csv(args.test_csv)
    test_probs, test_labels, test_model_probs = compute_probs_for_split(
        test_rows, models, model_names, device, transforms, hyper_weight
    )
    N_test = len(test_labels)
    print(f'  Test samples: {N_test}')

    # ------------------------------------------------------------------
    # 4. Evaluate conformal prediction on test set for each alpha
    # ------------------------------------------------------------------
    results = {}
    ensemble_argmax = test_probs.argmax(axis=1)
    ensemble_accuracy = float((ensemble_argmax == test_labels).mean())
    print(f'\nEnsemble argmax accuracy on test set: {ensemble_accuracy:.4f}')

    for alpha in alphas:
        t = thresholds[str(alpha)]
        target_cov = 1.0 - alpha

        covered = 0
        set_sizes = []
        set_size_dist = Counter()
        per_sample = []
        safety_cases = []  # cases where argmax is wrong but conformal set covers truth

        for i in range(N_test):
            cs = predict_conformal_set(test_probs[i], t, app.CLASS_NAMES)
            true_class = app.CLASS_NAMES[test_labels[i]]
            pred_class = app.CLASS_NAMES[ensemble_argmax[i]]
            is_covered = true_class in cs
            if is_covered:
                covered += 1
            sz = len(cs)
            set_sizes.append(sz)
            set_size_dist[sz] += 1

            sample_info = {
                'image': test_rows[i][0],
                'true_class': true_class,
                'argmax_pred': pred_class,
                'argmax_correct': bool(pred_class == true_class),
                'conformal_set': cs,
                'set_size': sz,
                'true_covered': bool(is_covered),
            }
            per_sample.append(sample_info)

            # Safety case: argmax wrong but conformal covers truth
            if pred_class != true_class and is_covered:
                safety_cases.append(sample_info)

        test_coverage = covered / N_test
        avg_set_size = float(np.mean(set_sizes))
        median_set_size = float(np.median(set_sizes))

        alpha_result = {
            'alpha': alpha,
            'target_coverage': target_cov,
            'test_coverage': float(test_coverage),
            'coverage_gap': float(test_coverage - target_cov),
            'n_test': N_test,
            'avg_set_size': avg_set_size,
            'median_set_size': median_set_size,
            'set_size_distribution': {str(k): v for k, v in sorted(set_size_dist.items())},
            'ensemble_argmax_accuracy': ensemble_accuracy,
            'n_safety_cases': len(safety_cases),
            'safety_cases': safety_cases[:10],  # top 10 for readability
            'per_sample': per_sample,
        }
        results[str(alpha)] = alpha_result

        print(f'\n  alpha={alpha:.2f} (target {target_cov:.0%}):')
        print(f'    Test coverage: {test_coverage:.4f} ({covered}/{N_test})')
        print(f'    Avg set size:  {avg_set_size:.2f}')
        print(f'    Median set size: {median_set_size:.1f}')
        print(f'    Set-size distribution: {dict(sorted(set_size_dist.items()))}')
        print(f'    Safety cases (argmax wrong, conformal covers): {len(safety_cases)}')

    # Save full results
    with open(os.path.join(args.out_dir, 'conformal_results.json'), 'w') as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f'\nSaved full results to {os.path.join(args.out_dir, "conformal_results.json")}')

    # ------------------------------------------------------------------
    # 5. Generate figures
    # ------------------------------------------------------------------
    # Figure 1: Histogram of prediction set sizes for each alpha
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 4), squeeze=False)
    for idx, alpha in enumerate(alphas):
        ax = axes[0, idx]
        r = results[str(alpha)]
        dist = r['set_size_distribution']
        sizes = sorted([int(k) for k in dist.keys()])
        counts = [dist[str(s)] for s in sizes]
        bars = ax.bar(sizes, counts, color='steelblue', edgecolor='white', width=0.7)
        ax.set_xlabel('Prediction Set Size', fontsize=12)
        ax.set_ylabel('Number of Test Samples', fontsize=12)
        ax.set_title(f'α = {alpha} (target {1-alpha:.0%} coverage)\n'
                     f'Achieved: {r["test_coverage"]:.1%}, '
                     f'Avg size: {r["avg_set_size"]:.2f}', fontsize=11)
        ax.set_xticks(sizes)
        # annotate bars
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(c), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(args.fig_dir, 'conformal_set_sizes.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'Saved set-size histogram to {fig_path}')
    plt.close()

    # Figure 2: Coverage comparison table as a figure
    fig, ax = plt.subplots(figsize=(6, 1.5 + 0.4 * len(alphas)))
    ax.axis('off')
    table_data = [['α', 'Target Coverage', 'Achieved Coverage', 'Avg Set Size', 'Median Set Size']]
    for alpha in alphas:
        r = results[str(alpha)]
        table_data.append([
            f'{alpha:.2f}',
            f'{r["target_coverage"]:.0%}',
            f'{r["test_coverage"]:.1%}',
            f'{r["avg_set_size"]:.2f}',
            f'{r["median_set_size"]:.1f}',
        ])
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    # Style header
    for j in range(len(table_data[0])):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    plt.title('Conformal Prediction: Coverage vs. Target', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    fig_path2 = os.path.join(args.fig_dir, 'conformal_coverage_table.pdf')
    plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path2.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'Saved coverage table to {fig_path2}')
    plt.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('CONFORMAL PREDICTION ANALYSIS SUMMARY')
    print('=' * 60)
    print(f'Ensemble models: {model_names}')
    print(f'HyperColumn weight: {hyper_weight}')
    print(f'Validation samples: {len(val_labels)}')
    print(f'Test samples: {N_test}')
    print(f'Ensemble argmax accuracy: {ensemble_accuracy:.4f}')
    for alpha in alphas:
        r = results[str(alpha)]
        print(f'\nα = {alpha}:')
        print(f'  Threshold:       {thresholds[str(alpha)]:.6f}')
        print(f'  Target coverage: {r["target_coverage"]:.0%}')
        print(f'  Test coverage:   {r["test_coverage"]:.1%}')
        print(f'  Avg set size:    {r["avg_set_size"]:.2f}')
        print(f'  Safety cases:    {r["n_safety_cases"]}')
    print('=' * 60)


if __name__ == '__main__':
    main()
