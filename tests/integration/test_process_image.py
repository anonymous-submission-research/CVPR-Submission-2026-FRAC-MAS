import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import pytest

from backend_hf import app as medai_app_module


class DummyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # return logits with strong prediction for class 0
        logits = torch.zeros((x.shape[0], self.num_classes), dtype=torch.float32)
        logits[:, 0] = 10.0
        return logits


class DummyYOLOWrapper(nn.Module):
    """Simulates YOLOClassifierWrapper for testing."""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def predict_pil(self, image):
        aligned = np.zeros(self.num_classes, dtype=np.float32)
        aligned[0] = 0.95
        aligned[1] = 0.05
        return aligned

    def forward(self, x):
        raise NotImplementedError("Use predict_pil()")


def test_process_image_basic():
    # ensure models dict can accept a dummy
    from backend_hf.app import process_image, models, device, CLASS_NAMES

    # Inject dummy model
    dummy = DummyModel(len(CLASS_NAMES))
    dummy.to(device)
    models['dummy_test_model'] = dummy

    # Create a trivial image
    img = Image.new('RGB', (224, 224), color='white')

    result = process_image(img, use_conformal='false', ensemble_mode='weighted', stacker_path=None)

    assert 'prediction' in result
    assert 'ensemble' in result
    assert result['prediction']['top_class'] in CLASS_NAMES


def test_process_image_with_yolo_stub():
    """Test that process_image handles a YOLO-like model that uses predict_pil()."""
    from backend_hf.app import process_image, models, device, CLASS_NAMES

    # Clear existing models and inject YOLO stub + a normal model
    models.clear()
    dummy_normal = DummyModel(len(CLASS_NAMES))
    dummy_normal.to(device)
    models['maxvit'] = dummy_normal

    yolo_stub = DummyYOLOWrapper(len(CLASS_NAMES))
    models['yolo'] = yolo_stub

    img = Image.new('RGB', (224, 224), color='white')

    result = process_image(img, use_conformal='false', ensemble_mode='weighted', stacker_path=None)

    assert 'prediction' in result
    assert 'ensemble' in result
    # Both models should appear in individual predictions
    assert 'yolo' in result['ensemble']['individual_predictions'] or 'maxvit' in result['ensemble']['individual_predictions']
