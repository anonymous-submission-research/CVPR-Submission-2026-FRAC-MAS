"""
Evaluate the Critic Agent on all test-set predictions.

This script:
  1. Runs the ensemble on all test set images to get predictions
  2. For each prediction, invokes the Critic Agent (MedGemma VLM)
  3. Computes: true rejection rate, false rejection rate, uncertainty rate,
     post-Critic accuracy, and safety value metrics

Outputs:
  - outputs/evaluation/critic_evaluation.json   (full per-sample results + aggregate metrics)
  - outputs/figures/critic_metrics.pdf (summary table figure)

Usage:
  python scripts/evaluate_critic.py \
      --checkpoints ./models \
      --models maxvit,yolo,hypercolumn_cbam_densenet169,rad_dino \
      --test-csv balanced_augmented_dataset/test.csv \
      --mode hf_spaces   # or --mode simulate (for offline testing without API)

Environment variables for MedGemma:
  MEDGEMMA_API_TOKEN  or  HF_TOKEN
  MEDGEMMA_SPACES_URL (optional)
  MEDGEMMA_MODE  (hf_spaces | local)
"""

import os
import sys
import json
import argparse
import csv
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath('src'))
from medai import app
from medai.uncertainty.conformal import calibrate_conformal, predict_conformal_set

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path):
    rows = []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append((r['image_path'], int(r['label'])))
    return rows


def resolve_path(p):
    for c in [p, os.path.join('data', p), os.path.join('.', p)]:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(p)


def is_hypercolumn(name):
    return 'hypercolumn' in name.lower() or 'cbam' in name.lower()


def model_probs_single(name, model, pil_img, tensor, device):
    import torch
    if app.is_yolo_model(model):
        return model.predict_pil(pil_img)
    elif app.is_rad_dino_model(name):
        rad_tensor = app.get_rad_dino_input_tensor(pil_img, device)
        with torch.no_grad():
            logits = model(rad_tensor)
        return torch.softmax(logits, dim=1).cpu().numpy()[0]
    else:
        with torch.no_grad():
            out = model(tensor)
        return torch.softmax(out, dim=1).cpu().numpy()[0]


# ---------------------------------------------------------------------------
# Simulated Critic (for offline evaluation without VLM API)
# ---------------------------------------------------------------------------

class SimulatedCriticAgent:
    """
    Simulates the Critic Agent's behavior for offline evaluation.
    
    Heuristic: if the ensemble confidence is very high (>0.85) and the prediction
    looks consistent, confirm. If confidence is low (<0.5), reject. Otherwise uncertain.
    This approximates what a VLM would do when comparing image features to the prediction.
    """
    def review_diagnosis(self, image, prediction_label, prediction_confidence, context_definition):
        # Simulate VLM review based on confidence + randomness
        import random
        
        if prediction_confidence > 0.85:
            # High confidence: usually confirm, small chance of uncertain
            r = random.random()
            if r < 0.85:
                verdict = "yes"
            elif r < 0.95:
                verdict = "uncertain"
            else:
                verdict = "no"
        elif prediction_confidence > 0.5:
            # Medium confidence: mix of outcomes
            r = random.random()
            if r < 0.5:
                verdict = "yes"
            elif r < 0.8:
                verdict = "uncertain"
            else:
                verdict = "no"
        else:
            # Low confidence: likely reject
            r = random.random()
            if r < 0.3:
                verdict = "yes"
            elif r < 0.5:
                verdict = "uncertain"
            else:
                verdict = "no"
        
        critic_confidence = 0.8 if verdict in ["yes", "no"] else 0.5
        
        return {
            "verdict": verdict,
            "critic_confidence": critic_confidence,
            "explanation": f"[Simulated] Verdict: {verdict} for {prediction_label} "
                          f"(ensemble conf: {prediction_confidence:.3f})",
            "flagged_for_human": verdict == "no",
            "critic_response_text": f"Simulated review of {prediction_label}"
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate Critic Agent on test set')
    parser.add_argument('--checkpoints', default='./models')
    parser.add_argument('--models', default='maxvit,yolo,hypercolumn_cbam_densenet169,rad_dino')
    parser.add_argument('--test-csv', default='balanced_augmented_dataset/test.csv')
    parser.add_argument('--mode', default='simulate',
                        choices=['hf_spaces', 'local', 'simulate'],
                        help='Critic agent mode: hf_spaces/local (real VLM) or simulate (offline)')
    parser.add_argument('--hyper-weight', type=float, default=None)
    parser.add_argument('--out-dir', default='outputs')
    parser.add_argument('--fig-dir', default='outputs/figures')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between API calls (secs)')
    parser.add_argument('--use-hybrid', action='store_true', help='Use hybrid decision rule (critic + conformal + ensemble signals)')
    parser.add_argument('--margin-threshold', type=float, default=0.15, help='Top1-top2 margin below which prediction is treated as ambiguous')
    parser.add_argument('--confidence-threshold', type=float, default=0.6, help='Ensemble confidence below which critic will tend to reject/uncertain')
    parser.add_argument('--conformal-threshold', type=float, default=None, help='Optional calibrated conformal threshold (nonconformity t); when provided, used to compute conformal set')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # Detect hyper-weight
    hyper_weight = args.hyper_weight
    if hyper_weight is None:
        wpath = os.path.join(args.out_dir, 'hypercolumn_weight.txt')
        if os.path.exists(wpath):
            hyper_weight = float(open(wpath).read().strip())
        else:
            hyper_weight = 1.5

    device = app.get_device()
    selected_models = [m.strip() for m in args.models.split(',') if m.strip()]

    print(f'Device: {device}')
    print(f'Loading models: {selected_models}')
    models = app.load_models(args.checkpoints, selected_models, device)
    model_names = list(models.keys())
    print(f'Loaded: {model_names}')

    transforms = app.get_transforms(app.IMG_SIZE)

    # Initialize Critic Agent
    if args.mode == 'simulate':
        print('Using SIMULATED Critic Agent (no API calls)')
        critic = SimulatedCriticAgent()
    else:
        try:
            from medai.agents.critic_agent import CriticAgent
            critic = CriticAgent(mode=args.mode)
            print(f'Using real CriticAgent in {args.mode} mode')
        except Exception as e:
            print(f'Failed to init CriticAgent: {e}. Falling back to simulation.')
            critic = SimulatedCriticAgent()

    # Load test data
    test_rows = load_csv(args.test_csv)
    N = len(test_rows)
    print(f'Test samples: {N}')

    # Medical knowledge for critic context
    knowledge_base = app.MEDICAL_KNOWLEDGE_BASE

    # Process each test sample
    results = []
    import torch

    for i, (img_path, label) in enumerate(test_rows):
        true_class = app.CLASS_NAMES[label]
        try:
            p = resolve_path(img_path)
            pil = Image.open(p).convert('RGB')
        except Exception as e:
            print(f'  Skipping {img_path}: {e}')
            continue

        tensor = transforms(pil).unsqueeze(0).to(device)

        # Ensemble prediction
        M = len(model_names)
        C = len(app.CLASS_NAMES)
        model_probs_arr = np.zeros((M, C), dtype=np.float32)
        for j, mname in enumerate(model_names):
            model = models[mname]
            try:
                probs = model_probs_single(mname, model, pil, tensor, device)
                model_probs_arr[j] = probs
            except Exception as e:
                print(f'  Inference failed for {mname}: {e}')

        # Weighted average
        is_hyper = [is_hypercolumn(n) for n in model_names]
        ws = np.array([hyper_weight if h else 1.0 for h in is_hyper], dtype=np.float32)
        ws /= ws.sum()
        avg_probs = (model_probs_arr * ws[:, None]).sum(axis=0)
        pred_idx = int(avg_probs.argmax())
        pred_class = app.CLASS_NAMES[pred_idx]
        pred_conf = float(avg_probs[pred_idx])

        # Get context definition for critic
        context_def = knowledge_base.get(pred_class, {}).get('definition', f'{pred_class} fracture')

        # Run Critic Agent
        try:
            critic_result = critic.review_diagnosis(
                image=pil,
                prediction_label=pred_class,
                prediction_confidence=pred_conf,
                context_definition=context_def
            )
        except Exception as e:
            print(f'  Critic failed for sample {i}: {e}')
            critic_result = {
                'verdict': 'uncertain',
                'critic_confidence': 0.0,
                'explanation': f'Critic error: {e}',
                'flagged_for_human': True,
            }

        is_correct = pred_class == true_class
        raw_critic_verdict = critic_result.get('verdict', 'uncertain')

        # --- Hybrid decision rule (optional) ---
        final_verdict = raw_critic_verdict
        # compute ensemble ambiguity signals
        sorted_probs = np.sort(avg_probs)[::-1]
        top1 = float(sorted_probs[0])
        top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = top1 - top2

        # If conformal threshold provided, compute conformal set and check membership
        conformal_set = None
        if args.conformal_threshold is not None:
            try:
                conformal_set = predict_conformal_set(avg_probs, args.conformal_threshold, app.CLASS_NAMES)
            except Exception:
                conformal_set = None

        if args.use_hybrid:
            # 1) If Critic explicitly rejects -> final 'no'
            if raw_critic_verdict == 'no':
                final_verdict = 'no'
            else:
                # 2) If Critic provided an independent top diagnosis that differs and is confident -> reject
                c_top = critic_result.get('top_diagnosis')
                c_top_conf = critic_result.get('top_diagnosis_confidence', critic_result.get('critic_confidence', 0.0))
                if c_top and c_top.lower() != pred_class.lower() and c_top_conf >= 0.6:
                    final_verdict = 'no'
                # 3) If conformal set is provided and does not include the predicted class -> reject
                elif conformal_set is not None and pred_class not in conformal_set:
                    final_verdict = 'no'
                # 4) If ensemble confidence is very low -> mark uncertain/reject
                elif pred_conf < args.confidence_threshold:
                    final_verdict = 'uncertain'
                # 5) If margin is small (ambiguous top-2) -> mark uncertain
                elif margin < args.margin_threshold:
                    final_verdict = 'uncertain'
                else:
                    final_verdict = 'yes'

        # Expose final verdict separately while keeping original critic fields
        verdict = final_verdict

        sample_result = {
            'index': i,
            'image': img_path,
            'true_class': true_class,
            'pred_class': pred_class,
            'pred_confidence': pred_conf,
            'is_correct': is_correct,
            'critic_verdict': critic_result.get('verdict', 'uncertain'),
            'critic_confidence': critic_result.get('critic_confidence', 0.0),
            'critic_explanation': critic_result.get('explanation', ''),
            'flagged_for_human': critic_result.get('flagged_for_human', False),
            # Hybrid / post-processed decision (may differ from raw critic)
            'final_verdict': verdict,
            'final_flagged_for_human': True if verdict in ['no', 'uncertain'] else False,
            'conformal_set': conformal_set,
            'ensemble_margin': margin,
            'ensemble_top_confidence': pred_conf,
        }
        results.append(sample_result)

        if (i + 1) % 10 == 0:
            print(f'  Processed {i + 1}/{N}')

        if args.mode != 'simulate' and args.delay > 0:
            time.sleep(args.delay)

    # ------------------------------------------------------------------
    # Compute Metrics
    # ------------------------------------------------------------------
    total = len(results)
    correct_preds = [r for r in results if r['is_correct']]
    wrong_preds = [r for r in results if not r['is_correct']]

    # Ensemble raw accuracy
    raw_accuracy = len(correct_preds) / total if total > 0 else 0

    # Critic verdict counts
    # Prefer final post-processed verdict if available
    def _get_verdict(r):
        return r.get('final_verdict', r.get('critic_verdict', 'uncertain'))

    confirmed = [r for r in results if _get_verdict(r) == 'yes']
    rejected = [r for r in results if _get_verdict(r) == 'no']
    uncertain = [r for r in results if _get_verdict(r) == 'uncertain']

    # True rejection rate: among wrong predictions, how often does critic reject?
    wrong_rejected = [r for r in wrong_preds if _get_verdict(r) == 'no']
    true_rejection_rate = len(wrong_rejected) / len(wrong_preds) if wrong_preds else 0

    # False rejection rate: among correct predictions, how often does critic reject?
    correct_rejected = [r for r in correct_preds if _get_verdict(r) == 'no']
    false_rejection_rate = len(correct_rejected) / len(correct_preds) if correct_preds else 0

    # Uncertainty rate: fraction flagged as uncertain
    uncertainty_rate = len(uncertain) / total if total > 0 else 0

    # Post-critic accuracy: among confirmed predictions, what is accuracy?
    confirmed_correct = [r for r in confirmed if r['is_correct']]
    post_critic_accuracy = len(confirmed_correct) / len(confirmed) if confirmed else 0

    # Rejected error rate: among rejected predictions, what was the original error rate?
    rejected_wrong = [r for r in rejected if not r['is_correct']]
    rejected_error_rate = len(rejected_wrong) / len(rejected) if rejected else 0

    # Safety margin = post_critic_accuracy - raw_accuracy
    safety_margin = post_critic_accuracy - raw_accuracy

    metrics = {
        'total_samples': total,
        'raw_ensemble_accuracy': float(raw_accuracy),
        'post_critic_accuracy': float(post_critic_accuracy),
        'safety_margin': float(safety_margin),
        'n_confirmed': len(confirmed),
        'n_rejected': len(rejected),
        'n_uncertain': len(uncertain),
        'true_rejection_rate': float(true_rejection_rate),
        'false_rejection_rate': float(false_rejection_rate),
        'uncertainty_rate': float(uncertainty_rate),
        'rejected_error_rate': float(rejected_error_rate),
        'n_correct': len(correct_preds),
        'n_wrong': len(wrong_preds),
        'n_wrong_rejected': len(wrong_rejected),
        'n_correct_rejected': len(correct_rejected),
        'n_confirmed_correct': len(confirmed_correct),
    }

    output = {
        'metrics': metrics,
        'mode': args.mode,
        'models': model_names,
        'hyper_weight': hyper_weight,
        'per_sample': results,
    }

    out_path = os.path.join(args.out_dir, 'critic_evaluation.json')
    with open(out_path, 'w') as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f'\nSaved evaluation to {out_path}')

    # ------------------------------------------------------------------
    # Generate figure: metrics summary table
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    table_data = [
        ['Metric', 'Value'],
        ['Total test samples', str(total)],
        ['Raw ensemble accuracy', f'{raw_accuracy:.1%}'],
        ['Post-Critic accuracy (confirmed only)', f'{post_critic_accuracy:.1%}'],
        ['Safety margin (Post-Critic − Raw)', f'{safety_margin:+.1%}'],
        ['', ''],
        ['Confirmed by Critic', f'{len(confirmed)} ({len(confirmed)/total:.0%})'],
        ['Rejected by Critic', f'{len(rejected)} ({len(rejected)/total:.0%})'],
        ['Uncertain', f'{len(uncertain)} ({len(uncertain)/total:.0%})'],
        ['', ''],
        ['True rejection rate (wrong preds rejected)', f'{true_rejection_rate:.1%}'],
        ['False rejection rate (correct preds rejected)', f'{false_rejection_rate:.1%}'],
        ['Error rate among rejected predictions', f'{rejected_error_rate:.1%}'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                     colWidths=[0.55, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    table[0, 0].set_facecolor('#4472C4')
    table[0, 0].set_text_props(color='white', fontweight='bold')
    table[0, 1].set_facecolor('#4472C4')
    table[0, 1].set_text_props(color='white', fontweight='bold')

    # Highlight key rows
    for row_idx in [3, 4]:
        for col in [0, 1]:
            if row_idx < len(table_data):
                table[row_idx, col].set_facecolor('#E8F0FE')

    plt.title('Critic Agent Evaluation Results', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        fig_path = os.path.join(args.fig_dir, f'critic_metrics.{ext}')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Saved metrics figure to {os.path.join(args.fig_dir, "critic_metrics.pdf")}')
    plt.close()

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('CRITIC AGENT EVALUATION SUMMARY')
    print('=' * 60)
    print(f'Mode: {args.mode}')
    print(f'Total test samples: {total}')
    print(f'Raw ensemble accuracy: {raw_accuracy:.1%}')
    print(f'Post-Critic accuracy: {post_critic_accuracy:.1%}')
    print(f'Safety margin: {safety_margin:+.1%}')
    print(f'')
    print(f'Confirmed: {len(confirmed)}/{total} ({len(confirmed)/total:.0%})')
    print(f'Rejected:  {len(rejected)}/{total} ({len(rejected)/total:.0%})')
    print(f'Uncertain: {len(uncertain)}/{total} ({len(uncertain)/total:.0%})')
    print(f'')
    print(f'True rejection rate:  {true_rejection_rate:.1%} ({len(wrong_rejected)}/{len(wrong_preds)} wrong preds rejected)')
    print(f'False rejection rate: {false_rejection_rate:.1%} ({len(correct_rejected)}/{len(correct_preds)} correct preds rejected)')
    print(f'Error rate among rejected: {rejected_error_rate:.1%}')
    print('=' * 60)


if __name__ == '__main__':
    main()
