"""
Fine-tune yolo26m-cls classification model on the balanced augmented fracture dataset.

Dataset expected structure (already present at data/balanced_augmented_dataset/):
  train/
    Comminuted/  Greenstick/  Healthy/  Oblique/
    Oblique_Displaced/  Spiral/  Transverse/  Transverse_Displaced/
  val/  (same class folders)
  test/ (same class folders)

Usage:
  python scripts/train_yolo_cls.py [--epochs N] [--imgsz S] [--batch B] [--device DEV]
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "yolo26m-cls.pt"
DATA_PATH  = ROOT / "data" / "balanced_augmented_dataset"
SAVE_DIR   = ROOT / "outputs" / "yolo_cls_finetune"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune yolo26m-cls on fracture dataset")
    p.add_argument("--epochs",    type=int,   default=50,    help="Number of training epochs")
    p.add_argument("--imgsz",     type=int,   default=224,   help="Input image size (square)")
    p.add_argument("--batch",     type=int,   default=32,    help="Batch size (-1 = auto-batch)")
    p.add_argument("--lr0",       type=float, default=1e-3,  help="Initial learning rate")
    p.add_argument("--lrf",       type=float, default=0.01,  help="Final LR as fraction of lr0")
    p.add_argument("--dropout",   type=float, default=0.0,   help="Classifier dropout (0–1)")
    p.add_argument("--freeze",    type=int,   default=0,
                   help="Freeze first N backbone layers (0 = train all)")
    p.add_argument("--optimizer", type=str,   default="AdamW",
                   choices=["SGD", "Adam", "AdamW", "RMSProp", "auto"],
                   help="Optimizer")
    p.add_argument("--patience",  type=int,   default=15,
                   help="Early-stopping patience (0 = disabled)")
    p.add_argument("--workers",   type=int,   default=8,     help="Dataloader workers")
    p.add_argument("--device",    type=str,   default="",
                   help="Device: '' = auto, 'cpu', '0', 'mps'")
    p.add_argument("--name",      type=str,   default="yolo_cls_ft",
                   help="Run name under outputs/yolo_cls_finetune/")
    p.add_argument("--resume",    action="store_true",
                   help="Resume from last checkpoint if it exists")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {MODEL_PATH.name}")
    print(f"  Dataset    : {DATA_PATH}")
    print(f"  Epochs     : {args.epochs}  |  ImgSz: {args.imgsz}  |  Batch: {args.batch}")
    print(f"  LR         : {args.lr0} → ×{args.lrf} final")
    print(f"  Optimizer  : {args.optimizer}  |  Freeze: {args.freeze} layers")
    print(f"  Device     : {args.device or 'auto'}")
    print(f"{'='*60}\n")

    # Load pretrained model
    model = YOLO(str(MODEL_PATH))

    # Train / fine-tune
    results = model.train(
        data        = str(DATA_PATH),
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        lr0         = args.lr0,
        lrf         = args.lrf,
        dropout     = args.dropout,
        freeze      = args.freeze if args.freeze > 0 else None,
        optimizer   = args.optimizer,
        patience    = args.patience,
        workers     = args.workers,
        device      = args.device if args.device else None,
        project     = str(SAVE_DIR),
        name        = args.name,
        exist_ok    = args.resume,
        resume      = args.resume,
        # Augmentation (mild — data already augmented)
        augment     = True,
        hsv_h       = 0.015,
        hsv_s       = 0.4,
        hsv_v       = 0.4,
        degrees     = 10.0,
        translate   = 0.1,
        scale       = 0.5,
        flipud      = 0.0,
        fliplr      = 0.5,
        # Logging
        plots       = True,
        verbose     = True,
    )

    # ── Evaluate on test split ────────────────────────────────────────────────
    print("\nRunning evaluation on test split …")
    test_results = model.val(
        data    = str(DATA_PATH),
        split   = "test",
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device if args.device else None,
        project = str(SAVE_DIR),
        name    = args.name + "_test_eval",
    )

    print("\n✓ Fine-tuning complete.")
    print(f"  Best weights : {SAVE_DIR / args.name / 'weights' / 'best.pt'}")
    print(f"  Last weights : {SAVE_DIR / args.name / 'weights' / 'last.pt'}")
    print(f"  Results dir  : {SAVE_DIR / args.name}")


if __name__ == "__main__":
    main()
