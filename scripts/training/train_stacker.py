"""
Train a stacking meta-classifier on outputs/ensemble/val_calib.npz

Outputs:
 - outputs/ensemble/stacker.joblib
 - outputs/ensemble/stacker_eval.json

The stacker is a multinomial logistic regression trained on flattened per-model probabilities.
"""
import os
import sys
import argparse
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='outputs/ensemble/val_calib.npz')
    parser.add_argument('--out', default='outputs/ensemble/stacker.joblib')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    model_probs = data['model_probs']  # (N, M, C)
    labels = data['labels']
    model_names = data['model_names'].tolist() if 'model_names' in data else []
    N, M, C = model_probs.shape
    print(f'Loaded {N} samples, {M} models, {C} classes')
    if model_names:
        print(f'Models: {model_names}')

    X = model_probs.reshape(N, M*C)
    y = labels

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000))
    ])

    param_grid = {
        'clf__C': [0.01, 0.1, 1.0, 10.0],
        'clf__penalty': ['l2']
    }

    gs = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    probs_val = best.predict_proba(X_val)
    preds_val = probs_val.argmax(axis=1)
    acc = accuracy_score(y_val, preds_val)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(best, args.out)

    with open('outputs/ensemble/stacker_eval.json', 'w') as fh:
        json.dump({
            'val_accuracy': float(acc),
            'best_params': gs.best_params_,
            'model_names': model_names,
            'num_models': M,
            'num_classes': C,
        }, fh, indent=2)

    print('Best params:', gs.best_params_)
    print('Trained stacker, val acc:', acc)
    print('Saved to', args.out)

if __name__ == '__main__':
    main()
