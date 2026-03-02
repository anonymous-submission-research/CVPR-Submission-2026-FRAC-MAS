"""
Simple calibration script for split-conformal thresholding.

Usage examples:
  - Save validation probabilities and labels as a npz: `np.savez('calib.npz', probs=probs, labels=labels)`
    Then run: `python scripts/calibrate_conformal.py --input calib.npz --alpha 0.10 --output threshold.txt`

  - Or provide a CSV where the final column is `label` and other columns are class probabilities.

The script prints and optionally saves the calibrated nonconformity threshold.
"""
import argparse
import numpy as np
import os
from medai.uncertainty.conformal import calibrate_conformal


def load_npz(path: str):
    d = np.load(path)
    probs = d.get('probs')
    labels = d.get('labels')
    if probs is None or labels is None:
        raise ValueError('NPZ must contain arrays named "probs" and "labels"')
    return probs, labels


def load_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        raise ValueError('CSV must contain a "label" column with integer labels')
    labels = df['label'].to_numpy()
    probs = df.drop(columns=['label']).to_numpy()
    return probs, labels


def main():
    parser = argparse.ArgumentParser(description='Calibrate conformal threshold for classification')
    parser.add_argument('--input', required=True, help='Path to .npz or .csv with validation probs and labels')
    parser.add_argument('--alpha', type=float, default=0.10, help='Miscoverage level (default 0.10 => 90% coverage)')
    parser.add_argument('--output', help='Optional path to save threshold (text file)')
    args = parser.parse_args()

    if args.input.lower().endswith('.npz'):
        probs, labels = load_npz(args.input)
    elif args.input.lower().endswith('.csv'):
        probs, labels = load_csv(args.input)
    else:
        raise ValueError('Unsupported input format. Use .npz or .csv')

    threshold = calibrate_conformal(probs, labels, alpha=args.alpha)
    print(f'Calibrated nonconformity threshold: {threshold:.6f}  (alpha={args.alpha})')

    if args.output:
        with open(args.output, 'w') as fh:
            fh.write(str(float(threshold)))
        print(f'Saved threshold to {args.output}')


if __name__ == '__main__':
    main()
