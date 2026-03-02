"""
CLI to run diagnosis + ensemble on images in a directory and output results (including conformal sets).

Usage:
  python scripts/test_with_conformal.py --test-dir test_images --checkpoint-dir ./models --models swin --threshold-file ./conformal_threshold.txt

Outputs a JSON file `outputs/conformal/test_with_conformal.json` containing a list of results.
"""
import argparse
import json
import os
import sys
from PIL import Image

# Ensure repo src is on path
sys.path.insert(0, os.path.abspath('src'))

from medai import app


def load_models(checkpoint_dir, selected_models, device):
    return app.load_models(checkpoint_dir, selected_models, device)


def main():
    parser = argparse.ArgumentParser(description='Run tests on images directory and include conformal sets')
    parser.add_argument('--test-dir', type=str, default='test_images', help='Directory with test images')
    parser.add_argument('--checkpoint-dir', type=str, default='./models', help='Directory with model checkpoints')
    parser.add_argument('--models', type=str, default='', help='Comma-separated model names to load (defaults to all available)')
    parser.add_argument('--threshold-file', type=str, default='./conformal_threshold.txt', help='Path to threshold file')
    parser.add_argument('--threshold', type=float, default=None, help='Manual threshold value if file missing')
    parser.add_argument('--output', type=str, default='outputs/conformal/test_with_conformal.json', help='Output JSON path')
    args = parser.parse_args()

    device = app.get_device()

    # Determine model list
    if args.models:
        selected_models = [m.strip() for m in args.models.split(',')]
    else:
        selected_models = list(app.MODEL_CONFIGS.keys())

    models = load_models(args.checkpoint_dir, selected_models, device)
    if not models:
        print('No models loaded. Exiting.')
        return

    # determine threshold
    conformal_threshold = None
    if os.path.exists(args.threshold_file):
        try:
            with open(args.threshold_file, 'r') as fh:
                conformal_threshold = float(fh.read().strip())
                print(f'Loaded threshold from {args.threshold_file}: {conformal_threshold}')
        except Exception as e:
            print('Failed to read threshold file, will use manual if provided:', e)
    if conformal_threshold is None and args.threshold is not None:
        conformal_threshold = float(args.threshold)
        print(f'Using manual threshold: {conformal_threshold}')

    primary_model_name = list(models.keys())[0]
    primary_model = models[primary_model_name]

    diag_agent = app.DiagnosticAgent(primary_model, app.CLASS_NAMES, device, conformal_threshold=conformal_threshold)
    ensemble_agent = app.ModelEnsembleAgent(models, app.CLASS_NAMES, device, conformal_threshold=conformal_threshold)

    results = []

    for fname in sorted(os.listdir(args.test_dir)):
        fpath = os.path.join(args.test_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            img = Image.open(fpath).convert('RGB')
        except Exception as e:
            print('Skipping', fpath, 'failed to open:', e)
            continue

        diag = diag_agent.diagnose(img)
        ens = ensemble_agent.run_ensemble(img)

        results.append({
            'image': fpath,
            'diagnosis': diag,
            'ensemble': ens
        })
        print('Processed', fpath)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as fh:
        json.dump(results, fh, indent=2)

    print('Wrote results to', args.output)


if __name__ == '__main__':
    main()
