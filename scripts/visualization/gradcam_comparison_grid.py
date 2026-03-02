"""
Generate publication-quality Grad-CAM comparison grids.

For each selected test image, this script generates a grid showing:
  Row = test image (different fracture types)
  Col = Original | Model-1 Grad-CAM | Model-2 Grad-CAM | ... | Model-N Grad-CAM

This produces the key figure for Section 3.3 / 4.4 of the paper.

Outputs:
  - outputs/figures/gradcam_comparison_grid.pdf
  - outputs/figures/gradcam_comparison_grid.png
  - outputs/evaluation/gradcam_comparison_results.json  (per-image, per-model attention stats)

Usage:
  python scripts/gradcam_comparison_grid.py \
      --checkpoints ./models \
      --models maxvit,hypercolumn_cbam_densenet169,yolo,rad_dino \
      --test-csv balanced_augmented_dataset/test.csv \
      --n-samples 4 \
      --classes "Comminuted,Oblique Displaced,Transverse,Spiral"

Notes:
  - YOLO and RAD-DINO use wrapper models that don't support standard backward hooks.
    For these models, we skip Grad-CAM and show "N/A" panels.
    Alternatively, pass only models that support Grad-CAM:
      --models maxvit,hypercolumn_cbam_densenet169
  - If pytorch-grad-cam is installed, uses its GradCAM; else falls back to custom hook-based.
"""

import os
import sys
import json
import argparse
import csv
import random
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torchvision.transforms as T

sys.path.insert(0, os.path.abspath('src'))
from medai import app

# Try pytorch-grad-cam first, fall back to hook-based implementation
try:
    from pytorch_grad_cam import GradCAM as PytorchGradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    USE_PYTORCH_GRADCAM = True
except ImportError:
    USE_PYTORCH_GRADCAM = False


# ---- Custom hook-based GradCAM (fallback) ----

class HookGradCAM:
    """Simple hook-based Grad-CAM for when pytorch-grad-cam is unavailable."""
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None
        self.handles = []
        self.handles.append(target_layer.register_forward_hook(self._fwd_hook))
        try:
            self.handles.append(target_layer.register_full_backward_hook(self._bwd_hook))
        except Exception:
            self.handles.append(target_layer.register_backward_hook(self._bwd_hook))

    def _fwd_hook(self, mod, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, mod, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx, device):
        self.model.zero_grad()
        input_tensor = input_tensor.to(device).requires_grad_(True)
        out = self.model(input_tensor)
        loss = out[0, class_idx]
        loss.backward(retain_graph=True)
        if self.gradients is None or self.activations is None:
            return None
        grads = self.gradients[0]
        acts = self.activations[0]
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * acts).sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / (cam.max() + 1e-8)
        H, W = input_tensor.shape[-2], input_tensor.shape[-1]
        import cv2
        cam = cv2.resize(cam, (W, H))
        return cam

    def clear(self):
        for h in self.handles:
            h.remove()


# ---- Target layer detection ----

def get_target_layers(model, model_name: str) -> Optional[list]:
    """Determine the target layer(s) for Grad-CAM based on model architecture."""
    name_lower = model_name.lower()

    # HyperColumn CBAM DenseNet - use the CBAM output or fusion_conv
    if 'hypercolumn' in name_lower or 'cbam' in name_lower:
        if hasattr(model, 'cbam') and hasattr(model.cbam, 'sa'):
            return [model.cbam.sa.conv]
        if hasattr(model, 'fusion_conv'):
            return [model.fusion_conv]
        if hasattr(model, 'features'):
            # Last conv in DenseNet backbone
            convs = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
            return [convs[-1]] if convs else None

    # MaxViT - use last stage or block
    if 'maxvit' in name_lower:
        if hasattr(model, 'stages'):
            last_stage = model.stages[-1]
            # Find last conv
            convs = [m for m in last_stage.modules() if isinstance(m, nn.Conv2d)]
            if convs:
                return [convs[-1]]
        # timm maxvit
        for attr_name in ['stages', 'blocks']:
            if hasattr(model, attr_name):
                blocks = getattr(model, attr_name)
                if hasattr(blocks, '__getitem__'):
                    last = blocks[-1]
                    convs = [m for m in last.modules() if isinstance(m, nn.Conv2d)]
                    if convs:
                        return [convs[-1]]

    # Swin
    if 'swin' in name_lower:
        if hasattr(model, 'layers'):
            convs = [m for m in model.layers[-1].modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
            if convs:
                return [convs[-1]]
        if hasattr(model, 'features') or hasattr(model, 'norm'):
            return [model.norm] if hasattr(model, 'norm') else None

    # DenseNet
    if 'densenet' in name_lower:
        if hasattr(model, 'features'):
            convs = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
            return [convs[-1]] if convs else None

    # EfficientNet
    if 'efficient' in name_lower:
        for attr in ['features', 'blocks', 'conv_head']:
            if hasattr(model, attr):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Conv2d):
                    return [mod]
                convs = [m for m in mod.modules() if isinstance(m, nn.Conv2d)]
                if convs:
                    return [convs[-1]]

    # MobileNet
    if 'mobile' in name_lower:
        if hasattr(model, 'features'):
            convs = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
            return [convs[-1]] if convs else None

    # Generic fallback: last Conv2d
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    return [convs[-1]] if convs else None


def compute_gradcam_for_model(model, model_name, input_tensor, class_idx, device):
    """Compute Grad-CAM array for a single model. Returns HxW numpy array in [0,1] or None."""
    # Skip models that don't support backward (YOLO, RAD-DINO)
    if app.is_yolo_model(model):
        return None
    if app.is_rad_dino_model(model_name):
        # RAD-DINO uses a transformer backbone; Grad-CAM is tricky but possible
        # We'll try to get the last attention block
        target_layers = get_target_layers(model, model_name)
        if target_layers is None:
            return None
    else:
        target_layers = get_target_layers(model, model_name)
        if target_layers is None:
            return None

    model.eval()
    model.to(device)

    if USE_PYTORCH_GRADCAM:
        try:
            with PytorchGradCAM(model=model, target_layers=target_layers) as cam:
                targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None
                grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=targets)
                return grayscale_cam[0]
        except Exception as e:
            print(f'    pytorch-grad-cam failed for {model_name}: {e}')
            # Fall through to hook-based
    
    # Hook-based fallback
    try:
        gcam = HookGradCAM(model, target_layers[0])
        cam_arr = gcam(input_tensor, class_idx, device)
        gcam.clear()
        return cam_arr
    except Exception as e:
        print(f'    Hook GradCAM failed for {model_name}: {e}')
        return None


def overlay_cam_on_image(img_rgb: np.ndarray, cam: np.ndarray, alpha=0.5) -> np.ndarray:
    """Overlay heatmap on RGB image. Both inputs should be in [0,1]."""
    import cv2
    cam_resized = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlaid = (1 - alpha) * img_rgb + alpha * heatmap_rgb
    return np.clip(overlaid, 0, 1)


def load_csv(path):
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


def compute_attention_stats(cam: np.ndarray) -> Dict:
    """Compute attention statistics for analysis."""
    if cam is None:
        return {'available': False}
    threshold = 0.5
    binary = cam > threshold
    total_px = cam.size
    active_px = int(binary.sum())
    centroid_y, centroid_x = 0.5, 0.5
    if active_px > 0:
        coords = np.argwhere(binary)
        centroid_y = float(coords[:, 0].mean() / cam.shape[0])
        centroid_x = float(coords[:, 1].mean() / cam.shape[1])
    return {
        'available': True,
        'mean_activation': float(cam.mean()),
        'max_activation': float(cam.max()),
        'active_fraction': float(active_px / total_px),
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
    }


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Comparison Grid for Paper')
    parser.add_argument('--checkpoints', default='./models')
    parser.add_argument('--models', default='maxvit,hypercolumn_cbam_densenet169',
                        help='Models for Grad-CAM (comma-separated). YOLO/RAD-DINO do not support Grad-CAM well.')
    parser.add_argument('--test-csv', default='balanced_augmented_dataset/test.csv')
    parser.add_argument('--n-samples', type=int, default=4, help='Number of representative images')
    parser.add_argument('--classes', default='Comminuted,Oblique Displaced,Transverse,Spiral',
                        help='Preferred classes to select one image from each')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fig-dir', default='outputs/figures')
    parser.add_argument('--out-dir', default='outputs')
    parser.add_argument('--img-size', type=int, default=224)
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    selected_models = [m.strip() for m in args.models.split(',') if m.strip()]
    preferred_classes = [c.strip() for c in args.classes.split(',')]

    device = app.get_device()
    print(f'Device: {device}')

    # Load models
    print('Loading models:', selected_models)
    models = app.load_models(args.checkpoints, selected_models, device)
    model_names = list(models.keys())
    print(f'Loaded: {model_names}')

    # Load test data and select representative images
    test_rows = load_csv(args.test_csv)
    by_class = defaultdict(list)
    for img_path, label in test_rows:
        class_name = app.CLASS_NAMES[label]
        by_class[class_name].append((img_path, label))

    selected_images = []
    for cls in preferred_classes:
        if cls in by_class and by_class[cls]:
            row = random.choice(by_class[cls])
            selected_images.append(row)
    # Fill remaining if needed
    remaining_classes = [c for c in app.CLASS_NAMES if c not in preferred_classes]
    while len(selected_images) < args.n_samples and remaining_classes:
        cls = remaining_classes.pop(0)
        if cls in by_class and by_class[cls]:
            selected_images.append(random.choice(by_class[cls]))

    print(f'\nSelected {len(selected_images)} images:')
    for img_path, label in selected_images:
        print(f'  [{app.CLASS_NAMES[label]}] {img_path}')

    transforms = app.get_transforms(args.img_size)

    # Compute Grad-CAM for each image × each model
    n_images = len(selected_images)
    n_models = len(model_names)
    n_cols = 1 + n_models  # Original + each model

    all_results = []
    grid_data = []  # list of list: [row][col] = image array

    for i, (img_path, label) in enumerate(selected_images):
        true_class = app.CLASS_NAMES[label]
        print(f'\nProcessing [{true_class}]: {img_path}')
        p = resolve_path(img_path)
        pil = Image.open(p).convert('RGB')
        img_rgb = np.array(pil.resize((args.img_size, args.img_size))) / 255.0
        input_tensor = transforms(pil).unsqueeze(0)

        row_images = [img_rgb]  # first column = original
        row_stats = {'image': img_path, 'true_class': true_class, 'models': {}}

        for j, mname in enumerate(model_names):
            model = models[mname]
            print(f'  Computing Grad-CAM for {mname}...')

            # Get predicted class for this model
            try:
                if app.is_yolo_model(model):
                    probs = model.predict_pil(pil)
                elif app.is_rad_dino_model(mname):
                    rad_tensor = app.get_rad_dino_input_tensor(pil, device)
                    with torch.no_grad():
                        logits = model(rad_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                else:
                    with torch.no_grad():
                        logits = model(input_tensor.to(device))
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_class_idx = int(probs.argmax())
                pred_class = app.CLASS_NAMES[pred_class_idx]
            except Exception as e:
                print(f'    Inference failed: {e}')
                row_images.append(np.ones_like(img_rgb) * 0.9)
                row_stats['models'][mname] = {'available': False, 'error': str(e)}
                continue

            # Compute Grad-CAM targeting the predicted class
            cam = compute_gradcam_for_model(model, mname, input_tensor, pred_class_idx, device)
            if cam is not None:
                overlay = overlay_cam_on_image(img_rgb, cam, alpha=0.5)
                row_images.append(overlay)
                stats = compute_attention_stats(cam)
                stats['pred_class'] = pred_class
                stats['confidence'] = float(probs[pred_class_idx])
                row_stats['models'][mname] = stats
                print(f'    OK: pred={pred_class} ({probs[pred_class_idx]:.3f}), active={stats["active_fraction"]:.2%}')
            else:
                # N/A panel
                na_img = np.ones_like(img_rgb) * 0.9
                row_images.append(na_img)
                row_stats['models'][mname] = {'available': False, 'reason': 'GradCAM not supported'}
                print(f'    Grad-CAM not available for {mname}')

        grid_data.append(row_images)
        all_results.append(row_stats)

    # Save results JSON
    with open(os.path.join(args.out_dir, 'gradcam_comparison_results.json'), 'w') as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f'\nSaved analysis to {os.path.join(args.out_dir, "gradcam_comparison_results.json")}')

    # ---- Create the figure ----
    print('\nGenerating comparison grid figure...')
    col_labels = ['Original'] + [n.replace('_', '\n') for n in model_names]
    row_labels = [app.CLASS_NAMES[lbl] for _, lbl in selected_images]

    fig_w = 3.0 * n_cols
    fig_h = 3.0 * n_images + 0.8
    fig, axes = plt.subplots(n_images, n_cols, figsize=(fig_w, fig_h))
    if n_images == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for r in range(n_images):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(grid_data[r][c])
            ax.axis('off')
            if r == 0:
                ax.set_title(col_labels[c], fontsize=10, fontweight='bold')
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=10, fontweight='bold', rotation=0,
                              labelpad=60, ha='right', va='center')

    plt.suptitle('Grad-CAM Comparison Across Ensemble Models', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig_path = os.path.join(args.fig_dir, f'gradcam_comparison_grid.{ext}')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Saved figure to {fig_path}')
    plt.close()

    # ---- Print analysis summary ----
    print('\n' + '=' * 60)
    print('GRAD-CAM COMPARISON SUMMARY')
    print('=' * 60)
    for res in all_results:
        print(f'\n  Image: {res["true_class"]}')
        for mname, stats in res['models'].items():
            if stats.get('available', False):
                print(f'    {mname}: pred={stats["pred_class"]} (conf={stats["confidence"]:.3f}), '
                      f'active={stats["active_fraction"]:.2%}, centroid=({stats["centroid_x"]:.2f},{stats["centroid_y"]:.2f})')
            else:
                print(f'    {mname}: N/A ({stats.get("reason", "error")})')
    print('=' * 60)


if __name__ == '__main__':
    main()
