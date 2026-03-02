"""
Inspect images: compute per-model logits/probs and save Grad-CAM overlays.

Usage:
  python scripts/inspect_images.py --images test_images/Oblique_Displaced_51_jpg... test_images/Transverse_153_jpg... --checkpoints ./models --models swin,mobilenetv2,... --out outputs/inspection.json

Outputs:
 - outputs/inspection.json with per-model probs/logits
 - outputs/gradcam_<image_basename>_<model>.png for overlays
"""
import os
import sys
import argparse
import json
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath('src'))
from medai import app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--checkpoints', default='./models')
    parser.add_argument('--models', default='swin,mobilenetv2,efficientnetv2,maxvit,densenet169')
    parser.add_argument('--out', default='outputs/inspection.json')
    args = parser.parse_args()

    device = app.get_device()
    selected_models = [m.strip() for m in args.models.split(',') if m.strip()]
    models = app.load_models(args.checkpoints, selected_models, device)
    if not models:
        print('No models loaded')
        return

    results = []
    for img_path in args.images:
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.open(os.path.join('test_images', img_path)).convert('RGB') if os.path.exists(os.path.join('test_images', img_path)) else None
        if img is None:
            print('Failed to open', img_path); continue

        record = {'image': img_path, 'models': {}}
        # per-model probs
        for name, model in models.items():
            agent = app.DiagnosticAgent(model, app.CLASS_NAMES, device)
            res = agent.diagnose(img)
            record['models'][name] = res
            # gradcam
            try:
                explain = app.ExplainabilityAgent(model, app.CLASS_NAMES, device)
                pred_class = res['predicted_class']
                pred_idx = app.CLASS_NAMES.index(pred_class) if pred_class in app.CLASS_NAMES else None
                cam = explain.generate_gradcam(img, pred_idx)
                if cam is not None:
                    outdir = 'outputs'
                    os.makedirs(outdir, exist_ok=True)
                    fname = os.path.join(outdir, f"gradcam_{os.path.basename(img_path)}_{name}.png")
                    vis = explain.visualize_gradcam(img, cam)
                    vis.save(fname)
                    record['models'][name]['gradcam'] = fname
            except Exception as e:
                record['models'][name]['gradcam_error'] = str(e)

        results.append(record)
        print('Inspected', img_path)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2)
    print('Wrote', args.out)

if __name__ == '__main__':
    main()
