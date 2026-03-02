"""
visualize_xgradcam.py

Generates XGrad-CAM overlays for misclassified examples listed in a CSV (format produced earlier):
    image_path,true,pred,top1,top2

For each row this script saves a PNG with:
  - original image
  - XGrad-CAM overlay for the **true** class
  - XGrad-CAM overlay for the **predicted** class
  - difference overlay (pred - true)

Usage:
    python visualize_xgradcam.py \
      --checkpoint outputs/swin_mps/best.pth \
      --misclassified outputs/analysis/misclassified.csv \
      --img-root . \
      --model swin --img-size 224 --out-dir outputs/analysis/xgradcam_overlays \
      --class-names "Comminuted,Greenstick,Healthy,Oblique,Oblique Displaced,Spiral,Transverse,Transverse Displaced"

Notes:
- Script prefers MPS (Apple Silicon) if available; if XGrad-CAM backward on MPS fails it will automatically fall back to CPU for CAM computation.
- Requires: torch, timm, torchvision, pillow, numpy, opencv-python

"""

import os
import csv
import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
import torchvision.models as tvmodels


def detect_device():
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_model(name: str, num_classes: int, pretrained=False):
    name = name.lower()
    if name.startswith('swin'):
        m = timm.create_model('swin_small_patch4_window7_224', pretrained=pretrained)
        # Check the existing head structure and adapt so the final module
        # matches the checkpoint's expected keys. Some timm versions use
        # `m.head.fc` (a submodule named "fc"); others expose `m.head`
        # directly as an nn.Linear. The checkpoint in this repo has
        # `head.fc.weight`/`head.fc.bias`, so prefer creating a `head`
        # module with a child `fc` when needed.
        orig_head = m.head
        # helper simple wrapper with .fc attribute when we need it
        class _HeadWithFC(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.fc = nn.Linear(in_features, out_features)
            def forward(self, x):
                return self.fc(x)

        if hasattr(orig_head, 'fc'):
            # existing head already has .fc (most compatible case)
            in_f = orig_head.fc.in_features
            m.head.fc = nn.Linear(in_f, num_classes)
        elif isinstance(orig_head, nn.Linear):
            # head is a Linear module — wrap it so state_dict keys like
            # `head.fc.weight` will exist
            in_f = orig_head.in_features
            m.head = _HeadWithFC(in_f, num_classes)
        else:
            # fallback: try to find an in_features attribute or replace
            # head with our wrapper using a best-effort in_features value
            in_f = getattr(getattr(orig_head, 'fc', orig_head), 'in_features', None)
            if in_f is None:
                # last resort: try model's default embed dim or raise
                try:
                    in_f = m.head.in_features
                except Exception:
                    raise RuntimeError('Unable to determine head in_features for Swin model')
            m.head = _HeadWithFC(in_f, num_classes)
        return m
    if name.startswith('convnext'):
        m = timm.create_model('convnext_tiny', pretrained=pretrained)
        m.head.fc = nn.Linear(m.head.fc.in_features, num_classes)
        return m
    if name.startswith('densenet'):
        m = tvmodels.densenet169(pretrained=pretrained)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    raise ValueError('unknown model')


class XGradCAM:
    """Hook-based X-GradCAM. Call with a model (in eval mode) and a target conv layer name (optional).
    If target_layer_name is None, the last nn.Conv2d module is chosen heuristically.
    """
    def __init__(self, model: nn.Module, target_layer_name: Optional[str] = None):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None
        self.handles = []

        # pick target layer
        if target_layer_name is None:
            target_layer = None
            for n, m in reversed(list(self.model.named_modules())):
                if isinstance(m, nn.Conv2d):
                    target_layer_name = n
                    target_layer = m
                    break
            if target_layer is None:
                raise RuntimeError('No Conv2d layer found for XGrad-CAM')
        else:
            target_layer = dict(self.model.named_modules()).get(target_layer_name, None)
            if target_layer is None:
                raise RuntimeError(f'layer name {target_layer_name} not found')

        # register hooks
        self.handles.append(target_layer.register_forward_hook(self._forward_hook))
        # backward hook
        try:
            self.handles.append(target_layer.register_backward_hook(self._backward_hook))
        except Exception:
            # fallback for newer pytorch versions: use register_full_backward_hook if available
            try:
                self.handles.append(target_layer.register_full_backward_hook(self._backward_hook))
            except Exception:
                # some builds won't allow backward hooks; we'll compute gradients by retaining graph and reading .grad from activations
                pass

    def _forward_hook(self, module, inp, out):
        # out: tensor shape (B,C,H,W)
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out[0] shape (B,C,H,W)
        self.gradients = grad_out[0].detach()

    def clear(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None, device: torch.device = torch.device('cpu')):
        """Compute CAM for a single input tensor (1,C,H,W). Returns cam resized to input HxW in numpy [0,1]."""
        self.model.zero_grad()
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True
        outputs = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(outputs.argmax(dim=1).item())
        loss = outputs[0, class_idx]
        loss.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError('XGradCAM failed to collect gradients/activations (hooks missing)')

        grads = self.gradients[0]  # C,H,W
        acts = self.activations[0]  # C,H,W
        
        # X-GradCAM Weights: sum(grads) / (sum(acts) + eps)
        # But we use the implementation often simplified as gradients * activations
        # The correct weights for X-GradCAM are essentially the normalized gradients
        # weighted by the activation maps themselves.
        # Implementation from 'X-GradCAM: Explainable Gradient-weighted Class Activation Mapping'
        # weights = sum(grads * acts) / sum(acts)
        
        weights = (grads * acts).sum(dim=(1, 2)) / (acts.sum(dim=(1, 2)) + 1e-8)
        cam = (weights[:, None, None] * acts).sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / (cam.max() + 1e-8)
        else:
            cam = np.zeros_like(cam)
        # resize to original input spatial size (assume square input)
        H = input_tensor.shape[-2]; W = input_tensor.shape[-1]
        cam = cv2.resize(cam, (W, H))
        return cam


def apply_colormap_on_image(org_img: np.ndarray, activation: np.ndarray, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """Overlay heatmap on image (org_img: HxW x 3 uint8, activation: HxW float in [0,1])"""
    if activation is None:
        raise ValueError('activation is None')
    # ensure activation is 2D and in [0,1]
    activation = np.asarray(activation)
    if activation.ndim == 3:
        # if somehow a channel dim exists, reduce to single channel
        activation = activation[..., 0]
    activation = np.clip(activation, 0.0, 1.0)

    # Convert activation -> heatmap (BGR) and resize heatmap to match original image
    heatmap = np.uint8(255 * activation)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # Resize heatmap to original image spatial size before blending
    h, w = org_img.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # convert heatmap to RGB to match org_img (which is RGB)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # ensure types match for addWeighted
    org_uint8 = org_img.astype('uint8')
    heat_uint8 = heatmap.astype('uint8')
    overlaid = cv2.addWeighted(org_uint8, 1.0 - alpha, heat_uint8, alpha, 0)
    return overlaid


def pil_to_numpy(img: Image.Image):
    arr = np.array(img.convert('RGB'))
    return arr


def get_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--misclassified', required=True)
    parser.add_argument('--img-root', default='.')
    parser.add_argument('--model', default='swin')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--out-dir', default='outputs/analysis/xgradcam_overlays')
    parser.add_argument('--class-names', required=True)
    parser.add_argument('--target-layer', default=None)
    parser.add_argument('--max-samples', type=int, default=200, help='max misclassified rows to process')
    args = parser.parse_args()

    class_names = [c.strip() for c in args.class_names.split(',')]
    num_classes = len(class_names)

    device_pref = detect_device()
    print('preferred device:', device_pref)

    model = get_model(args.model, num_classes, pretrained=False)
    ck = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ck['model_state_dict'])

    # We'll run forward on preferred device, but if backward (for CAM) fails on MPS we'll move to CPU for CAM computation
    model.to(device_pref)
    model.eval()

    transform = get_transform(args.img_size)

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    with open(args.misclassified, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    rows = rows[:args.max_samples]

    # initialize XGradCAM on device_pref; if backward fails, we will retry on CPU
    xgradcam = None
    try:
        xgradcam = XGradCAM(model, target_layer_name=args.target_layer)
        cam_device = device_pref
    except Exception as e:
        print('XGradCAM init failed on preferred device; will try CPU. Error:', e)
        cam_device = torch.device('cpu')
        model_cpu = get_model(args.model, num_classes, pretrained=False)
        model_cpu.load_state_dict(ck['model_state_dict'])
        model_cpu.to(cam_device)
        model_cpu.eval()
        xgradcam = XGradCAM(model_cpu, target_layer_name=args.target_layer)

    for i, r in enumerate(rows):
        img_path = r['image_path'] if os.path.isabs(r['image_path']) else os.path.join(args.img_root, r['image_path'])
        true_lbl = int(r['true'])
        pred_lbl = int(r['pred'])
        try:
            pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print('failed to open', img_path, e); continue

        org_np = pil_to_numpy(pil)
        inp = transform(pil).unsqueeze(0)

        # forward on preferred device to get outputs and predicted class
        try:
            inp_pref = inp.to(device_pref)
            with torch.no_grad():
                out_pref = model(inp_pref)
                probs = torch.softmax(out_pref, dim=1).cpu().numpy()[0]
        except Exception as e:
            print('forward failed on preferred device:', e)
            # fallback to CPU forward
            model.cpu(); inp_cpu = inp; model.eval()
            with torch.no_grad():
                out_cpu = model(inp_cpu)
                probs = torch.softmax(out_cpu, dim=1).numpy()[0]

        # compute CAMs on xgradcam.device (cam_device)
        cam_true = None; cam_pred = None
        try:
            # ensure model used for xgradcam is on cam_device
            cam_model = xgradcam.model
            # move input to cam_device
            inp_cam = inp.to(cam_device)
            cam_true = xgradcam(inp_cam, class_idx=true_lbl, device=cam_device)
            cam_pred = xgradcam(inp_cam, class_idx=pred_lbl, device=cam_device)
        except Exception as e:
            print('XGrad-CAM on preferred device failed for', img_path, 'error:', e)
            # try CPU
            try:
                # rebuild cpu model if needed
                cpu_dev = torch.device('cpu')
                model_cpu = get_model(args.model, num_classes, pretrained=False)
                model_cpu.load_state_dict(ck['model_state_dict'])
                model_cpu.to(cpu_dev); model_cpu.eval()
                xgradcam_cpu = XGradCAM(model_cpu, target_layer_name=args.target_layer)
                cam_true = xgradcam_cpu(inp.to(cpu_dev), class_idx=true_lbl, device=cpu_dev)
                cam_pred = xgradcam_cpu(inp.to(cpu_dev), class_idx=pred_lbl, device=cpu_dev)
                xgradcam_cpu.clear()
            except Exception as e2:
                print('XGrad-CAM CPU retry failed for', img_path, e2)
                continue

        # overlay
        try:
            over_true = apply_colormap_on_image(org_np, cam_true, alpha=0.5)
            over_pred = apply_colormap_on_image(org_np, cam_pred, alpha=0.5)
            diff = cam_pred - cam_true
            diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            over_diff = apply_colormap_on_image(org_np, diff, alpha=0.6)

            # concat: original | true | pred | diff
            h, w, _ = org_np.shape
            # resize overlays to original size if needed
            over_true = cv2.resize(over_true, (w, h))
            over_pred = cv2.resize(over_pred, (w, h))
            over_diff = cv2.resize(over_diff, (w, h))
            orig_bgr = cv2.cvtColor(org_np, cv2.COLOR_RGB2BGR)
            grid = np.vstack([np.hstack([orig_bgr, cv2.cvtColor(over_true, cv2.COLOR_RGB2BGR)]),
                              np.hstack([cv2.cvtColor(over_pred, cv2.COLOR_RGB2BGR), cv2.cvtColor(over_diff, cv2.COLOR_RGB2BGR)])])

            out_name = f"{i:04d}_true{true_lbl}_pred{pred_lbl}_{os.path.basename(img_path)}.png"
            out_path = os.path.join(args.out_dir, out_name)
            cv2.imwrite(out_path, grid)
        except Exception as e:
            print('failed to create overlay for', img_path, e)
            continue

    xgradcam.clear()
    print('Saved overlays to', args.out_dir)

def generate_xgradcam_overlay(
    model: nn.Module,
    image_path: str,
    target_layer_name: Optional[str] = None,
    true_label: Optional[int] = None,
    pred_label: Optional[int] = None,
    device: torch.device = torch.device('cpu'),
    img_size: int = 224
):
    """
    Generates an XGrad-CAM overlay for a single image.
    Returns the overlay image (PIL Image) and the raw heatmaps (true and pred).
    """
    model.eval()
    transform = get_transform(img_size)
    
    try:
        pil_img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None, None, None

    org_np = pil_to_numpy(pil_img)
    inp = transform(pil_img).unsqueeze(0).to(device)
    
    # Initialize XGradCAM
    # Note: reusing the model instance. 
    # Ensure wrap this call or manage lifecycle if called repeatedly in a loop with different models, 
    # but here we assume model is passed ready.
    
    # If the model is on a different device than 'device', move input.
    # We assume model and device match or let pytorch handle mismatch errors (which we should avoid).
    
    xgradcam = None
    try:
        xgradcam = XGradCAM(model, target_layer_name=target_layer_name)
        
        # We need to run forward to get the prediction if not provided?
        # The caller might provide pred_label.
        
        if pred_label is None:
            with torch.no_grad():
                out = model(inp)
                pred_label = int(out.argmax(dim=1).item())
        
        # Calculate CAM for Pred
        cam_pred = xgradcam(inp, class_idx=pred_label, device=device)
        
        cam_true = None
        if true_label is not None:
            cam_true = xgradcam(inp, class_idx=true_label, device=device)
            
        # Create overlay for Predicted Class
        over_pred = apply_colormap_on_image(org_np, cam_pred, alpha=0.5)
        
        # Cleanup hooks
        xgradcam.clear()
        
        return Image.fromarray(over_pred), cam_pred, cam_true
        
    except Exception as e:
        print(f"XGradCAM generation failed: {e}")
        if xgradcam:
            xgradcam.clear()
        return None, None, None


if __name__ == '__main__':
    main()
