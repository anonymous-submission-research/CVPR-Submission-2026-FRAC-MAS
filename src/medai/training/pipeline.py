"""
Fracture classification pipeline — Mac MPS only with Weights & Biases logging

Features:
- Enforces MPS device on Apple Silicon (exits if not available).
- Supports three backbones: swin, convnext, densenet (via timm / torchvision).
- Local checkpointing (best.pth) and automatic upload of checkpoints to Weights & Biases using `wandb.save`.
- WandB logging of train/val metrics, lr, and confusion matrix artifact.
- Stage-2 Grad-CAM cropping and retrain supported.

Usage (example):
    # install dependencies (see terminal commands below for mac-specific instructions)
    python fracture_classification_pipeline_mps_wandb.py \
        --train-csv data/train.csv --val-csv data/val.csv --test-csv data/test.csv \
        --model swin --num-classes 8 --epochs 20 --batch-size 6 --img-size 224 \
        --out-dir outputs/swin_mps --wandb-project fracture-mps --wandb-entity your_entity

Notes:
- This script *requires* MPS (Apple Silicon). It will exit if MPS is unavailable.
- Use small batch sizes (4-8) depending on your GPU/VRAM. The Mac M4 Pro Max 36GB UM should handle moderate sizes but training is slower than CUDA GPUs.
- For WandB: run `wandb login` beforehand or set `WANDB_API_KEY` env var.

"""

import os
import argparse
import time
import copy
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvmodels
import timm

import wandb
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import cv2

# ----------------------------- Device (MPS only) -----------------------------

def require_mps():
    if getattr(torch.backends, 'mps', None) is None or not torch.backends.mps.is_available():
        raise RuntimeError('MPS (Apple Silicon) is required for this script but was not detected on this machine.')
    return torch.device('mps')

DEVICE = require_mps()
print(f"Using device: {DEVICE}")

# ----------------------------- Dataset -----------------------------
class FractureDataset(Dataset):
    def __init__(self, df, img_root: str = '.', transform=None, use_bbox: bool=False):
        self.entries = df
        self.img_root = img_root
        self.transform = transform
        self.use_bbox = use_bbox

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]
        img_path = row['image_path']
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.img_root, img_path)
        img = Image.open(img_path).convert('RGB')

        if self.use_bbox and all(k in row for k in ('bbox_xmin','bbox_ymin','bbox_xmax','bbox_ymax')):
            xmin = int(row['bbox_xmin']); ymin = int(row['bbox_ymin']); xmax = int(row['bbox_xmax']); ymax = int(row['bbox_ymax'])
            img = img.crop((xmin, ymin, xmax, ymax))

        label = int(row['label'])
        if self.transform:
            img = self.transform(img)
        return img, label, img_path

# ----------------------------- Transforms -----------------------------

def get_transforms(split: str, img_size: int = 224):
    if split == 'train':
        return T.Compose([
            T.Resize((int(img_size*1.1), int(img_size*1.1))),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomRotation(15),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

# ----------------------------- Model selection -----------------------------

def get_model(name: str, num_classes: int, pretrained: bool=True):
    name = name.lower()
    if name.startswith('swin'):
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=pretrained)
        if hasattr(model, 'reset_classifier'):
            model.reset_classifier(num_classes=num_classes)
        else:
            model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    if name.startswith('convnext'):
        model = timm.create_model('convnext_tiny', pretrained=pretrained)
        if hasattr(model, 'reset_classifier'):
            model.reset_classifier(num_classes=num_classes)
        else:
            model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
        return model
    if name.startswith('densenet'):
        model = tvmodels.densenet169(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f'Unknown model: {name}')

# ----------------------------- Training & Evaluation -----------------------------

def save_checkpoint(state, is_best, out_dir, name='checkpoint.pth', upload_to_wandb: bool=False):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(out_dir, 'best.pth')
        torch.save(state, best_path)
        if upload_to_wandb:
            try:
                wandb.save(best_path)
                print('Uploaded best checkpoint to WandB:', best_path)
            except Exception as e:
                print('WandB save failed:', e)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.softmax(dim=1).argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(labels.detach().cpu().numpy().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    return epoch_loss, p, r, f1


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.softmax(dim=1).argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(labels.detach().cpu().numpy().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    return epoch_loss, p, r, f1, cm

# ----------------------------- Grad-CAM utilities -----------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer_name: str = None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        if target_layer_name is None:
            for n, m in reversed(list(self.model.named_modules())):
                if isinstance(m, (nn.Conv2d,)):
                    target_layer_name = n
                    break
        self.target_layer_name = target_layer_name
        if target_layer_name is None:
            raise ValueError('Cannot find a convolutional layer for Grad-CAM')
        target_module = dict(self.model.named_modules())[self.target_layer_name]
        self.hook_handles.append(target_module.register_forward_hook(self._forward_hook))
        # Note: register_full_backward_hook not supported in all versions; use backward hook where available
        try:
            self.hook_handles.append(target_module.register_backward_hook(self._backward_hook))
        except Exception:
            pass

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None, device: torch.device = DEVICE):
        self.model.zero_grad()
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True
        outputs = self.model(input_tensor)
        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()
        loss = outputs[0, class_idx]
        loss.backward(retain_graph=True)
        if self.gradients is None or self.activations is None:
            raise RuntimeError('GradCAM failed to collect gradients/activations — try a different target layer name')
        grads = self.gradients[0]
        acts = self.activations[0]
        weights = grads.mean(dim=(1,2))
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = np.maximum(cam.cpu().numpy(), 0)
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self):
        for h in self.hook_handles:
            try:
                h.remove()
            except Exception:
                pass

# ----------------------------- Heatmap -> bbox -----------------------------

def heatmap_to_bbox(cam: np.ndarray, thr: float = 0.5, min_area: int = 100):
    H, W = cam.shape
    thr_val = cam.max() * thr
    mask = (cam >= thr_val).astype('uint8') * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        return (x, y, x+w, y+h)
    return None

# ----------------------------- Generate crops from Grad-CAM (stage 2 prep) -----------------------------

def generate_crops_from_gradcam(model, entries: List[Dict], out_dir: str, transform_for_cam, device: torch.device, cam_layer: str=None, thr: float=0.5, padding: float=0.15):
    os.makedirs(out_dir, exist_ok=True)
    gradcam = GradCAM(model, target_layer_name=cam_layer)
    new_entries = []
    for i, row in enumerate(entries):
        path = row['image_path']
        img = Image.open(path).convert('RGB')
        tensor = transform_for_cam(img).unsqueeze(0).to(device)
        try:
            cam = gradcam(tensor, class_idx=None, device=device)
        except Exception as e:
            print('GradCAM failed for', path, e)
            continue
        bbox = heatmap_to_bbox(cam, thr=thr)
        if bbox is None:
            w, h = img.size
            cx, cy = w//2, h//2
            side = int(min(w,h)*0.6)
            xmin = max(0, cx-side//2); ymin = max(0, cy-side//2); xmax = min(w, cx+side//2); ymax = min(h, cy+side//2)
        else:
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin; h = ymax - ymin
            px = int(w * padding); py = int(h * padding)
            xmin = max(0, xmin - px); ymin = max(0, ymin - py); xmax = min(img.size[0], xmax + px); ymax = min(img.size[1], ymax + py)
        crop = img.crop((xmin, ymin, xmax, ymax)).resize((224,224))
        fname = f"crop_{i}_{os.path.basename(path)}"
        out_path = os.path.join(out_dir, fname)
        crop.save(out_path)
        new_entries.append({'image_path': out_path, 'label': row['label']})
    gradcam.close()
    return new_entries

# ----------------------------- Inference with simple TTA -----------------------------

def tta_predict(model, pil_img: Image.Image, device, img_size=224):
    base = T.Compose([T.Resize((img_size,img_size)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    img1 = base(pil_img).unsqueeze(0).to(device)
    img2 = base(pil_img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out1 = model(img1).softmax(dim=1)
        out2 = model(img2).softmax(dim=1)
    probs = (out1 + out2) / 2.0
    return probs.squeeze(0).cpu().numpy()

# ----------------------------- Helpers: CSV loader -----------------------------

def load_csv_like(path: str) -> List[Dict]:
    import csv
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

# ----------------------------- Main -----------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=str, help='train csv', required=True)
    parser.add_argument('--val-csv', type=str, help='val csv', required=True)
    parser.add_argument('--test-csv', type=str, help='test csv', required=True)
    parser.add_argument('--img-root', type=str, default='.', help='root for images')
    parser.add_argument('--model', type=str, default='swin', choices=['swin','convnext','densenet'])
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--out-dir', type=str, default='outputs')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--stage2', action='store_true', help='run stage 2: generate crops from gradcam and retrain')
    parser.add_argument('--stage2-crop-dir', type=str, default='crops')
    parser.add_argument('--cam-layer', type=str, default=None, help='module name for Grad-CAM hook (optional)')

    # wandb args
    parser.add_argument('--wandb-project', type=str, default='fracture-mps')
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument('--wandb-mode', type=str, default='online', choices=['online','offline','disabled'])

    args = parser.parse_args(argv)

    # initialize wandb
    if args.wandb_mode != 'disabled':
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, mode=args.wandb_mode)
        wandb.config.update(vars(args))
    else:
        wandb.init(mode='disabled')

    device = DEVICE

    # load CSVs
    train_rows = load_csv_like(args.train_csv)
    val_rows = load_csv_like(args.val_csv)
    test_rows = load_csv_like(args.test_csv)

    train_tf = get_transforms('train', img_size=args.img_size)
    val_tf = get_transforms('val', img_size=args.img_size)

    model = get_model(args.model, args.num_classes, pretrained=True).to(device)

    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        print('Loaded checkpoint', args.checkpoint)

    train_ds = FractureDataset(train_rows, img_root=args.img_root, transform=train_tf)
    val_ds = FractureDataset(val_rows, img_root=args.img_root, transform=val_tf)
    test_ds = FractureDataset(test_rows, img_root=args.img_root, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1,args.epochs))

    best_f1 = 0.0
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_p, train_r, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_p, val_r, val_f1, cm = validate(model, val_loader, criterion, device)
        scheduler.step()
        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
        ck_name = f'epoch_{epoch}.pth'
        save_checkpoint({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_f1': val_f1}, is_best, out_dir, name=ck_name, upload_to_wandb=(args.wandb_mode!='disabled'))

        # wandb logging
        metrics = {'epoch': epoch, 'train_loss': train_loss, 'train_macro_f1': train_f1, 'val_loss': val_loss, 'val_macro_f1': val_f1, 'lr': scheduler.get_last_lr()[0]}
        print(f"Epoch {epoch}/{args.epochs} time={time.time()-start:.1f}s")
        print(metrics)
        if args.wandb_mode != 'disabled':
            wandb.log(metrics, step=epoch)
            # log confusion matrix as an image
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6,6))
                ax.imshow(cm, interpolation='nearest')
                ax.set_title('Confusion matrix')
                wandb.log({"confusion_matrix": wandb.Image(fig)}, step=epoch)
                plt.close(fig)
            except Exception as e:
                print('Failed to log confusion matrix plot to wandb:', e)

    # load best and final test evaluation
    best_ck = os.path.join(out_dir, 'best.pth')
    if os.path.exists(best_ck):
        ck = torch.load(best_ck, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        print('Loaded best checkpoint for final evaluation')

    test_loss, test_p, test_r, test_f1, test_cm = validate(model, test_loader, criterion, device)
    print('Test results:', test_loss, test_p, test_r, test_f1)
    np.savetxt(os.path.join(out_dir, 'confusion_matrix.txt'), test_cm, fmt='%d')

    if args.wandb_mode != 'disabled':
        # save confusion matrix as artifact
        try:
            wandb.log({'test_macro_f1': test_f1})
            wandb.save(os.path.join(out_dir, 'confusion_matrix.txt'))
        except Exception as e:
            print('WandB final save failed:', e)

    # Stage 2: Grad-CAM cropping and retrain
    if args.stage2:
        print('Starting Stage-2: generating crops via Grad-CAM and retraining on cropped ROIs')
        cam_transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        crops_out = args.stage2_crop_dir
        new_train = generate_crops_from_gradcam(model, train_rows, out_dir=crops_out, transform_for_cam=cam_transform, device=device, cam_layer=args.cam_layer or None, thr=0.5)
        train_ds2 = FractureDataset(new_train, transform=get_transforms('train', img_size=args.img_size))
        train_loader2 = DataLoader(train_ds2, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1,args.epochs//2))
        best_f1_stage2 = 0.0
        for epoch in range(max(5, args.epochs//2)):
            train_loss, train_p, train_r, train_f1 = train_one_epoch(model, train_loader2, optimizer, criterion, device)
            val_loss, val_p, val_r, val_f1, cm = validate(model, val_loader, criterion, device)
            is_best = val_f1 > best_f1_stage2
            if is_best:
                best_f1_stage2 = val_f1
            save_checkpoint({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_f1': val_f1}, is_best, out_dir, name=f'stage2_epoch_{epoch}.pth', upload_to_wandb=(args.wandb_mode!='disabled'))
            scheduler.step()
            if args.wandb_mode != 'disabled':
                wandb.log({'stage2_epoch': epoch, 'stage2_val_macro_f1': val_f1, 'stage2_train_macro_f1': train_f1}, step=epoch)
        print('Stage-2 finished. Best val macro-F1:', best_f1_stage2)

    print('Finished.')
    if args.wandb_mode != 'disabled':
        wandb.finish()

if __name__ == '__main__':
    main()
