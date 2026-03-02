"""
Compute confusion matrix and per-class calibration statistics from outputs/ensemble/val_calib.npz

Outputs:
 - outputs/confusion_matrix.png
 - outputs/validation_calibration.csv
 - outputs/validation_confusion.json
"""
import os
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, brier_score_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='outputs/ensemble/val_calib.npz')
    parser.add_argument('--weight', type=float, default=1.0, help='hypercolumn weight')
    parser.add_argument('--outdir', default='outputs')
    args = parser.parse_args()

    data = np.load(args.input)
    model_probs = data['model_probs']  # (N, M, C)
    labels = data['labels']
    model_names = [n.decode('utf-8') if isinstance(n, bytes) else n for n in data['model_names']]

    # determine hypercolumn flags
    is_hyper = [ ('hypercolumn' in n.lower() or 'cbam' in n.lower()) for n in model_names]
    weights = np.array([args.weight if h else 1.0 for h in is_hyper], dtype=float)
    weights = weights / weights.sum()

    avg = (model_probs * weights[None,:,None]).sum(axis=1)  # (N,C)
    preds = avg.argmax(axis=1)

    cm = confusion_matrix(labels, preds)
    os.makedirs(args.outdir, exist_ok=True)

    # plot confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Validation Confusion Matrix')
    cm_path = os.path.join(args.outdir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

    # per-class calibration: for each class, compute avg predicted prob where predicted == class and accuracy
    C = avg.shape[1]
    per_class = []
    for c in range(C):
        idx = (preds == c)
        if idx.sum() == 0:
            avg_prob = 0.0
            acc = None
        else:
            avg_prob = avg[idx, c].mean()
            acc = (labels[idx] == c).mean()
        # overall reliability: Brier score for class c as one-vs-rest
        y_true = (labels == c).astype(int)
        y_prob = avg[:, c]
        brier = brier_score_loss(y_true, y_prob)
        per_class.append({'class_idx': c, 'class_name': data['model_names'].shape and None or str(c), 'avg_pred_prob_when_predicted': float(avg_prob), 'pred_count': int(idx.sum()), 'accuracy_when_predicted': (None if acc is None else float(acc)), 'brier_score': float(brier)})

    with open(os.path.join(args.outdir, 'validation_confusion.json'), 'w') as fh:
        json.dump({'confusion_matrix': cm.tolist(), 'per_class': per_class, 'model_names': model_names}, fh, indent=2)

    # write calibration csv
    import csv
    with open(os.path.join(args.outdir, 'validation_calibration.csv'), 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['class_idx','class_name','pred_count','avg_pred_prob_when_predicted','accuracy_when_predicted','brier_score'])
        for p in per_class:
            writer.writerow([p['class_idx'], app.CLASS_NAMES[p['class_idx']] if 'app' in globals() else p['class_idx'], p['pred_count'], p['avg_pred_prob_when_predicted'], p['accuracy_when_predicted'], p['brier_score']])

    print('Wrote', cm_path, 'and validation_calibration.csv and validation_confusion.json')

if __name__ == '__main__':
    main()
