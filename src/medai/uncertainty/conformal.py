"""
Simple split-conformal utilities for classification.

Provides:
- calibrate_conformal(validation_probs, validation_labels, alpha)
- predict_conformal_set(probs, threshold, class_names)

The implementation follows the simple nonconformity score 1 - p_true
and computes the (1-alpha) quantile on the calibration set.
"""
from typing import List, Sequence
import numpy as np


def calibrate_conformal(validation_probs: np.ndarray, validation_labels: np.ndarray, alpha: float = 0.10) -> float:
    """
    Calibrate a split-conformal nonconformity threshold.

    Args:
        validation_probs: array (N, C) probabilities from the model on the calibration set.
        validation_labels: array (N,) integer labels (0..C-1) for calibration set.
        alpha: desired miscoverage level (e.g., 0.10 for 90% coverage).

    Returns:
        threshold: float nonconformity threshold t such that examples with
                   1 - p_true <= t will be covered at approximately 1 - alpha.
    """
    validation_probs = np.asarray(validation_probs)
    validation_labels = np.asarray(validation_labels)

    if validation_probs.ndim != 2:
        raise ValueError("validation_probs must be shape (N, C)")
    if validation_probs.shape[0] != validation_labels.shape[0]:
        raise ValueError("validation_probs and validation_labels must have the same first dimension")

    p_true = validation_probs[np.arange(validation_labels.shape[0]), validation_labels]
    nonconformity = 1.0 - p_true

    # coverage quantile: we want the (1-alpha) quantile of nonconformity
    coverage = 1.0 - alpha
    # use interpolation='higher' to be conservative (ensure coverage)
    t = float(np.quantile(nonconformity, coverage, interpolation='higher')) if hasattr(np.quantile, '__call__') else float(np.quantile(nonconformity, coverage))
    return t


def predict_conformal_set(probs: Sequence[float], threshold: float, class_names: List[str]) -> List[str]:
    """
    Given class probabilities and a calibrated nonconformity threshold, return the conformal prediction set.

    Args:
        probs: array-like shape (C,) probabilities for each class.
        threshold: calibrated nonconformity threshold from `calibrate_conformal`.
        class_names: list of length C with class name strings.

    Returns:
        list of class names included in the conformal set. Guaranteed approximate coverage 1-alpha.
    """
    probs = np.asarray(probs)
    if probs.ndim != 1:
        raise ValueError("probs must be a 1-D array of class probabilities")
    if probs.shape[0] != len(class_names):
        raise ValueError("length of probs must match length of class_names")

    cutoff = 1.0 - float(threshold)
    included = [class_names[i] for i, p in enumerate(probs) if float(p) >= cutoff]

    # Ensure we never return an empty set: fall back to the argmax class.
    if not included:
        # fallback: return top-2 classes to better reflect ambiguity
        top_idxs = probs.argsort()[-2:][::-1]
        included = [class_names[int(i)] for i in top_idxs]

    return included
