"""
Build validation NPZ from balanced_augmented_dataset/val.csv, compute per-model probabilities,
grid-search HYPERCOLUMN_WEIGHT, and calibrate conformal threshold for the ensemble.

Outputs:
 - outputs/ensemble/val_calib.npz (model_probs, labels, model_names)
 - conformal_threshold.txt (calibrated threshold)
 - outputs/ensemble/hypercolumn_weight.txt (best weight)

Usage:
  python scripts/prepare_val_and_calibrate.py --checkpoints ./models --models maxvit,yolo,hypercolumn_cbam_densenet169,rad_dino --alpha 0.10
"""
import os
import argparse
import sys
import numpy as np
from PIL import Image
import json
import torch

sys.path.insert(0, os.path.abspath('src'))
from medai import app
from medai.uncertainty.conformal import calibrate_conformal


def load_val_csv(path='balanced_augmented_dataset/val.csv'):
    import csv
    rows = []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append((r['image_path'], int(r['label'])))
    return rows


def is_hypercolumn(name):
    return 'hypercolumn' in name.lower() or 'cbam' in name.lower()


def _model_probs_single(name, model, pil_img, tensor, device):
    """Get probabilities for a single model, dispatching by type."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', default='./models')
    parser.add_argument('--models', default='maxvit,yolo,hypercolumn_cbam_densenet169,rad_dino')
    parser.add_argument('--alpha', type=float, default=0.10)
    parser.add_argument('--output-npz', default='outputs/ensemble/val_calib.npz')
    parser.add_argument('--threshold-out', default='conformal_threshold.txt')
    parser.add_argument('--weight-out', default='outputs/ensemble/hypercolumn_weight.txt')
    parser.add_argument('--weights', default='1.0,1.5,2.0,2.5,3.0,4.0,5.0')
    args = parser.parse_args()

    device = app.get_device()
    selected_models = [m.strip() for m in args.models.split(',') if m.strip()]

    print('Loading models:', selected_models)
    models = app.load_models(args.checkpoints, selected_models, device)
    if not models:
        print('No models loaded. Exiting.')
        return

    model_names = list(models.keys())
    M = len(model_names)

    rows = load_val_csv()
    N = len(rows)
    print(f'Loaded {N} validation entries')

    # Pre-allocate arrays
    C = len(app.CLASS_NAMES)  # known = 8
    print('Number of classes:', C)

    def resolve(p):
        if os.path.exists(p):
            return p
        p2 = os.path.join('data', p)
        if os.path.exists(p2):
            return p2
        p3 = os.path.join('.', p)
        if os.path.exists(p3):
            return p3
        raise FileNotFoundError(p)

    transforms = app.get_transforms(app.IMG_SIZE)
    model_probs = np.zeros((N, M, C), dtype=np.float32)
    labels = np.zeros((N,), dtype=np.int32)

    for i, (img_path, label) in enumerate(rows):
        try:
            p = resolve(img_path)
            pil = Image.open(p).convert('RGB')
        except Exception as e:
            print('Failed to open', img_path, '->', e)
            continue
        tensor = transforms(pil).unsqueeze(0).to(device)
        labels[i] = label
        for j, name in enumerate(model_names):
            model = models[name]
            try:
                probs = _model_probs_single(name, model, pil, tensor, device)
            except Exception as e:
                print(f'  ⚠️ Inference failed for {name} on sample {i}: {e}')
                continue
            model_probs[i, j] = probs
        if (i+1) % 20 == 0 or i == N-1:
            print(f'Processed {i+1}/{N}')

    os.makedirs(os.path.dirname(args.output_npz), exist_ok=True)
    np.savez(args.output_npz, model_probs=model_probs, labels=labels, model_names=np.array(model_names))
    print('Saved validation NPZ to', args.output_npz)

    # Grid search hypercolumn weight
    weights_to_try = [float(w) for w in args.weights.split(',')]
    best_weight = None
    best_acc = -1.0
    # prepare indicator for hypercolumn models
    is_hyper = [is_hypercolumn(n) for n in model_names]

    for w in weights_to_try:
        # compute weights vector
        ws = np.array([w if is_h else 1.0 for is_h in is_hyper], dtype=np.float32)
        ws = ws / ws.sum()
        # weighted avg
        avg = (model_probs * ws[None, :, None]).sum(axis=1)  # (N, C)
        preds = avg.argmax(axis=1)
        acc = (preds == labels).mean()
        print(f'Weight {w:.2f} -> accuracy {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            best_weight = w
    print('Best hypercolumn weight:', best_weight, 'acc:', best_acc)

    os.makedirs(os.path.dirname(args.weight_out), exist_ok=True)
    with open(args.weight_out, 'w') as fh:
        fh.write(str(best_weight))
    print('Wrote best weight to', args.weight_out)

    # compute threshold using best weight
    ws = np.array([best_weight if is_h else 1.0 for is_h in is_hyper], dtype=np.float32)
    ws = ws / ws.sum()
    avg = (model_probs * ws[None, :, None]).sum(axis=1)

    t = calibrate_conformal(avg, labels, alpha=args.alpha)
    with open(args.threshold_out, 'w') as fh:
        fh.write(str(float(t)))
    print('Calibrated threshold', t, 'written to', args.threshold_out)

    # also save full outputs for inspection
    out_json = {
        'model_names': model_names,
        'best_weight': best_weight,
        'best_acc': float(best_acc),
        'threshold': float(t)
    }
    with open('outputs/ensemble/val_calib_summary.json', 'w') as fh:
        json.dump(out_json, fh, indent=2)
    print('Wrote summary to outputs/ensemble/val_calib_summary.json')


if __name__ == '__main__':
    main()
