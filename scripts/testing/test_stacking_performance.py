import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torchvision import transforms
from PIL import Image
import joblib
import logging
import timm
from ultralytics import YOLO

# Add project root and src to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
project_root = _PROJECT_ROOT
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("EvaluateStacking")

CLASS_NAMES = [
    "Comminuted", "Greenstick", "Healthy", "Oblique", 
    "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"
]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleTestDataset(Dataset):
    def __init__(self, root_dir, class_names):
        self.samples = []
        for idx, cls in enumerate(class_names):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                cls_dir = os.path.join(root_dir, cls.replace(" ", "_"))
            if not os.path.exists(cls_dir): continue
            for img in os.listdir(cls_dir):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, img), idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def get_base_model(name, device):
    if name == 'maxvit':
        model = timm.create_model('maxvit_rmlp_small_rw_224.sw_in1k', pretrained=False, num_classes=8)
        pth = os.path.join(project_root, "outputs", "cross_validation", "best_maxvit.pth")
        ck = torch.load(pth, map_location='cpu')
        model.load_state_dict(ck.get('model_state_dict', ck), strict=False)
    elif name == 'hypercolumn_cbam_densenet169':
        import visualize_xgradcam as vxc
        model = vxc.get_model('densenet169', num_classes=8, pretrained=False)
        pth = os.path.join(project_root, "outputs", "cross_validation", "best_hypercolumn_cbam_densenet169_focal.pth")
        ck = torch.load(pth, map_location='cpu')
        model.load_state_dict(ck.get('model_state_dict', ck), strict=False)
    elif name == 'yolo':
        pth = os.path.join(project_root, "outputs", "yolo_cls_finetune", "yolo_cls_ft", "weights", "best.pt")
        model = YOLO(pth)
        return model
    elif name == 'rad_dino':
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        model_id = "microsoft/rad-dino"
        model = AutoModelForImageClassification.from_pretrained(model_id, num_labels=8, ignore_mismatched_sizes=True)
        processor = AutoImageProcessor.from_pretrained(model_id)
        pth = os.path.join(project_root, "outputs", "dinorad", "dinorad_best.pth")
        if os.path.exists(pth):
            ck = torch.load(pth, map_location='cpu')
            sd = ck.get('model_state_dict', ck)
            mapped = {}
            for k, v in sd.items():
                nk = k.replace('backbone.', 'dinov2.')
                if nk.startswith('classifier.'):
                    if 'weight' in nk: nk = 'classifier.weight'
                    elif 'bias' in nk: nk = 'classifier.bias'
                    else: continue
                mapped[nk] = v
            model.load_state_dict(mapped, strict=False)
        return (model.to(device).eval(), processor)
    
    return model.to(device).eval()

def main():
    logger.info("Starting Stacking Ensemble Evaluation (Independent Logic)")
    test_dir = os.path.join(_PROJECT_ROOT, "balanced_augmented_dataset", "test")
    stacker_path = os.path.join(project_root, "outputs", "stacker.joblib")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load All Components
    logger.info("Loading Models...")
    models = {
        'maxvit': get_base_model('maxvit', DEVICE),
        'hc_densenet': get_base_model('hypercolumn_cbam_densenet169', DEVICE),
        'yolo': get_base_model('yolo', DEVICE),
        'rad_dino': get_base_model('rad_dino', DEVICE) # returns (model, processor)
    }
    stacker = joblib.load(stacker_path)
    
    dataset = SimpleTestDataset(test_dir, CLASS_NAMES)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    y_true = []
    y_preds = {k: [] for k in list(models.keys()) + ['ensemble']}
    
    logger.info(f"Running Inference on {len(dataset)} images...")
    
    for img_path, label in tqdm(loader):
        img_path = img_path[0]
        label = label.item()
        y_true.append(label)
        
        pil_img = Image.open(img_path).convert('RGB')
        tensor_img = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        all_probs = [] # For stacker: [maxvit, yolo, hc, dino] - order matters!
        # According to stacker_eval.json order is: maxvit, yolo, hypercolumn_cbam_densenet169, rad_dino
        
        # 1. MaxViT
        with torch.no_grad():
            p_maxvit = torch.softmax(models['maxvit'](tensor_img), dim=1).cpu().numpy()[0]
        y_preds['maxvit'].append(np.argmax(p_maxvit))
        
        # 2. YOLO
        y_res = models['yolo'](img_path, verbose=False)[0]
        # Align YOLO indices to CLASS_NAMES
        y_probs_raw = y_res.probs.data.cpu().numpy()
        y_probs_aligned = np.zeros(8)
        for y_idx, y_name in y_res.names.items():
            try: 
                c_idx = CLASS_NAMES.index(y_name.replace(' ', '_'))
                y_probs_aligned[c_idx] = y_probs_raw[y_idx]
            except: pass
        y_preds['yolo'].append(np.argmax(y_probs_aligned))
        
        # 3. HC DenseNet
        with torch.no_grad():
            p_hc = torch.softmax(models['hc_densenet'](tensor_img), dim=1).cpu().numpy()[0]
        y_preds['hc_densenet'].append(np.argmax(p_hc))
        
        # 4. RAD-DINO
        m_dino, proc_dino = models['rad_dino']
        dino_inputs = proc_dino(images=pil_img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            p_dino = torch.softmax(m_dino(**dino_inputs).logits, dim=1).cpu().numpy()[0]
        y_preds['rad_dino'].append(np.argmax(p_dino))
        
        # Ensemble Step
        # Order for stacker: maxvit, yolo, hypercolumn, rad_dino
        probs_stack = np.concatenate([p_maxvit, y_probs_aligned, p_hc, p_dino]).reshape(1, -1)
        # scaler + clf usually in stacker pipe
        ens_probs = stacker.predict_proba(probs_stack)[0]
        y_preds['ensemble'].append(np.argmax(ens_probs))

    # Reporting
    logger.info("\n" + "="*70)
    logger.info(" FINAL COMPARISON: STACKING ENSEMBLE PERFORMANCE ")
    logger.info("="*70)
    
    rows = []
    for k, v in y_preds.items():
        acc = accuracy_score(y_true, v)
        f1 = f1_score(y_true, v, average='macro')
        rows.append({'Model': k, 'Accuracy': acc, 'F1-Macro': f1})
        
    df = pd.DataFrame(rows).sort_values('Accuracy', ascending=False)
    print(df.to_string(index=False))
    
    print("\nEnsemble Detailed Report:")
    print(classification_report(y_true, y_preds['ensemble'], target_names=CLASS_NAMES))

if __name__ == "__main__":
    main()
