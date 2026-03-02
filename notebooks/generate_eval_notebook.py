import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

code_cells = [
r"""import os
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import numpy as np
import pandas as pd
import warnings
from tqdm.auto import tqdm
import timm
import traceback

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

dataset_root = os.path.join(os.path.abspath("."), "balanced_augmented_dataset", "test")
classes = sorted(os.listdir(dataset_root))
print("Classes:", classes)

# Helper for standard models
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def evaluate_model(model_fn, dataset_path, predict_fn):
    y_true = []
    y_pred = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_path, cls)
        for img_name in tqdm(os.listdir(cls_dir), desc=cls, leave=False):
            img_path = os.path.join(cls_dir, img_name)
            try:
                pred_idx = predict_fn(img_path)
                y_true.append(idx)
                y_pred.append(pred_idx)
            except Exception as e:
                pass
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred))
    
    # Return macro results
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {'Precision': p, 'Recall': r, 'F1-Score': f1}

results_summary = {}
""",
r"""# 1. Evaluate MaxViT
import torch.nn as nn

try:
    model_maxvit = timm.create_model('maxvit_rmlp_small_rw_224.sw_in1k', pretrained=False)
    model_maxvit.head = nn.Linear(model_maxvit.head.in_features, 8)
    
    ck = torch.load(os.path.join(os.path.abspath("."), "outputs", "cross_validation", "best_maxvit.pth"), map_location='cpu')
    model_maxvit.load_state_dict(ck.get('model_state_dict', ck), strict=False)
    model_maxvit.to(device)
    model_maxvit.eval()

    def predict_maxvit(img_path):
        img = Image.open(img_path).convert('RGB')
        inp = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model_maxvit(inp)
        return int(out.argmax(dim=1).item())

    print("\nEvaluating MaxViT...")
    results_summary['MaxViT'] = evaluate_model(model_maxvit, dataset_root, predict_maxvit)
except Exception as e:
    print("MaxViT Error:", e)
    traceback.print_exc()
""",
r"""# 2. Evaluate hypercolumn_cbam_densenet169_focal
import sys
_repo_root = os.path.abspath(".")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import visualize_xgradcam as vxc
    model_densenet = vxc.get_model('densenet169', num_classes=8, pretrained=False)
    ck2 = torch.load(os.path.join(os.path.abspath("."), "outputs", "cross_validation", "best_hypercolumn_cbam_densenet169_focal.pth"), map_location='cpu')
    model_densenet.load_state_dict(ck2.get('model_state_dict', ck2), strict=False)
    model_densenet.to(device)
    model_densenet.eval()

    def predict_densenet(img_path):
        img = Image.open(img_path).convert('RGB')
        inp = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model_densenet(inp)
        return int(out.argmax(dim=1).item())

    print("\nEvaluating Hypercolumn CBAM DenseNet169 Focal...")
    results_summary['HC_CBAM_DenseNet169_Focal'] = evaluate_model(model_densenet, dataset_root, predict_densenet)
except Exception as e:
    print("Hypercolumn DenseNet Error:", e)
    traceback.print_exc()
""",
r"""# 3. Evaluate YOLOv26m-cls (using best.pt from finetuning)
try:
    from ultralytics import YOLO

    yolo_path = os.path.join(os.path.abspath("."), "outputs", "yolo_cls_finetune", "yolo_cls_ft", "weights", "best.pt")
    if not os.path.exists(yolo_path):
        yolo_path = os.path.join(os.path.abspath("."), "yolov8n-cls.pt")  # fallback
    model_yolo = YOLO(yolo_path)

    yolo_names = model_yolo.names
    name_to_idx = {name: idx for idx, name in enumerate(classes)}

    def predict_yolo(img_path):
        res = model_yolo(img_path, verbose=False)
        yolo_cls_name = res[0].names[res[0].probs.top1]
        yolo_cls_name = yolo_cls_name.replace(' ', '_')
        return name_to_idx.get(yolo_cls_name, 0)

    print("\nEvaluating YOLO Model...")
    results_summary['YOLOv26m-cls'] = evaluate_model(model_yolo, dataset_root, predict_yolo)
except Exception as e:
    print("YOLO Error:", e)
    traceback.print_exc()
""",
r"""# 4. Evaluate RAD-DINO
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model_id = "microsoft/rad-dino"
    image_processor = AutoImageProcessor.from_pretrained(model_id)

    model_dino = AutoModelForImageClassification.from_pretrained(model_id, num_labels=8, ignore_mismatched_sizes=True)

    pth_path = os.path.join(os.path.abspath("."), "outputs", "dinorad", "dinorad_best.pth")
    if os.path.exists(pth_path):
        ck = torch.load(pth_path, map_location='cpu')
        
        # Determine if we need to map keys (e.g. if it has backbone. prefixes)
        state_dict = ck.get('model_state_dict', ck)
        mapped_sd = {}
        for k, v in state_dict.items():
            new_k = k.replace('backbone.', 'dinov2.')
            if new_k.startswith('classifier.'):
                if 'weight' in new_k: new_k = 'classifier.weight'
                elif 'bias' in new_k: new_k = 'classifier.bias'
                else: continue
            mapped_sd[new_k] = v
            
        print('Loading mapped state dict from dinorad_best.pth...')
        model_dino.load_state_dict(mapped_sd, strict=False)

    model_dino.to(device)
    model_dino.eval()

    def predict_dino(img_path):
        img = Image.open(img_path).convert('RGB')
        inputs = image_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_dino(**inputs)
        pred_dino_idx = int(outputs.logits.argmax(dim=1).item())
        return pred_dino_idx

    print("\nEvaluating RAD-DINO...")
    results_summary['RAD-DINO'] = evaluate_model(model_dino, dataset_root, predict_dino)
except Exception as e:
    print("RAD-DINO Error:", e)
    traceback.print_exc()
""",
r"""# 5. Summary
try:
    if results_summary:
        summary_df = pd.DataFrame(results_summary).T
        print("\n===========================================")
        print("Overall Macro Metrics for Entire Dataset:")
        print("===========================================")
        print(summary_df.to_string())
    else:
        print("No models evaluated successfully.")
except Exception as e:
    print("Summary Error:", e)
"""
]

cells = []
for code in code_cells:
    cells.append(nbf.v4.new_code_cell(code))

nb.cells = cells

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate_models.ipynb"), 'w') as f:
    nbf.write(nb, f)
