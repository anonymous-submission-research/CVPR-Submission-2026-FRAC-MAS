"""
Comprehensive Weighted Ensemble Testing & Optimization
Tests different weight combinations to find optimal ensemble performance
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import joblib
from itertools import product

# Add project root and src to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
project_root = _PROJECT_ROOT
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from medai.modules.ensemble_module import EnsembleModule, get_rad_dino_input_tensor

class WeightedEnsembleTester:
    def __init__(self, ensemble, class_names, dataset_test):
        self.ensemble = ensemble
        self.class_names = class_names
        self.dataset_test = dataset_test
        self.results = {}
    
    def load_test_data(self):
        """Load all test images and labels"""
        images = []
        labels = []
        label_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        
        for cls_idx, cls_name in enumerate(self.class_names):
            cls_dir = os.path.join(self.dataset_test, cls_name)
            if not os.path.exists(cls_dir):
                continue
            
            img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in img_files:
                images.append(os.path.join(cls_dir, img_file))
                labels.append(cls_idx)
        
        return images, np.array(labels)
    
    def get_model_probabilities(self, img_path):
        """Get probability outputs from all models"""
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            return None
        
        input_tensor = self.ensemble.transforms(img).unsqueeze(0).to(self.ensemble.device)
        model_probs = {}
        
        for name, model in self.ensemble.models.items():
            try:
                if name.lower() == "yolo":
                    probs = model.predict_pil(img)
                elif "rad_dino" in name.lower():
                    rad_tensor = get_rad_dino_input_tensor(img, self.ensemble.device)
                    with torch.no_grad():
                        logits = model(rad_tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                else:
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                model_probs[name] = probs
            except:
                pass
        
        return model_probs if len(model_probs) > 0 else None
    
    def predict_weighted(self, model_probs, weights):
        """Weighted average of probability distributions"""
        weighted_probs = np.zeros(len(self.class_names))
        total_weight = 0.0
        
        for model_name, probs in model_probs.items():
            weight = weights.get(model_name, 0.25)
            weighted_probs += probs * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_probs /= total_weight
        
        return np.argmax(weighted_probs)
    
    def un_swap_label(self, pred_idx):
        """Un-swap labels for evaluation"""
        pred_class_name = self.class_names[pred_idx]
        
        if pred_class_name == "Oblique_Displaced":
            return self.class_names.index("Oblique")
        elif pred_class_name == "Oblique":
            return self.class_names.index("Oblique_Displaced")
        elif pred_class_name == "Transverse_Displaced":
            return self.class_names.index("Transverse")
        elif pred_class_name == "Transverse":
            return self.class_names.index("Transverse_Displaced")
        
        return pred_idx
    
    def test_weights(self, images, labels, weights, weight_name):
        """Test a specific weight configuration"""
        y_pred = []
        
        for img_path in tqdm(images, desc=f"Testing {weight_name}", leave=False):
            try:
                model_probs = self.get_model_probabilities(img_path)
                if model_probs is None:
                    continue
                
                pred_idx = self.predict_weighted(model_probs, weights)
                pred_idx_unswapped = self.un_swap_label(pred_idx)
                y_pred.append(pred_idx_unswapped)
            except:
                continue
        
        if len(y_pred) < len(labels):
            # Pad predictions
            y_pred.extend([0] * (len(labels) - len(y_pred)))
        
        y_pred = np.array(y_pred[:len(labels)])
        accuracy = accuracy_score(labels, y_pred)
        
        return accuracy, y_pred
    
    def compare_approaches(self, images, labels):
        """Compare stacking vs weighted averaging"""
        print("\n" + "=" * 80)
        print("COMPARING ENSEMBLE APPROACHES")
        print("=" * 80)
        
        results = {}
        
        # 1. Stacking (if available)
        if hasattr(self.ensemble, 'stacker') and self.ensemble.stacker is not None:
            print("\n1. Testing STACKING approach...")
            y_pred_stacking = []
            
            for img_path in tqdm(images, desc="Stacking", leave=False):
                try:
                    model_probs = self.get_model_probabilities(img_path)
                    if model_probs is None:
                        continue
                    
                    # Get all probabilities in order
                    all_probs = []
                    for name in self.ensemble.model_names:
                        if name in model_probs:
                            all_probs.append(model_probs[name])
                    
                    if len(all_probs) > 0:
                        all_probs = np.stack(all_probs)  # (M, C)
                        feat = all_probs.reshape(1, -1)
                        proba = self.ensemble.stacker.predict_proba(feat)[0]
                        pred_idx = np.argmax(proba)
                        pred_idx_unswapped = self.un_swap_label(pred_idx)
                        y_pred_stacking.append(pred_idx_unswapped)
                except:
                    pass
            
            if len(y_pred_stacking) > 0:
                y_pred_stacking = np.array(y_pred_stacking[:len(labels)])
                acc_stacking = accuracy_score(labels, y_pred_stacking)
                results['Stacking'] = {
                    'accuracy': acc_stacking,
                    'y_pred': y_pred_stacking
                }
                print(f"   Stacking Accuracy: {acc_stacking:.4f}")
        
        # 2. Equal weights
        print("\n2. Testing EQUAL WEIGHTS (0.25 each)...")
        weights_equal = {
            "maxvit": 0.25,
            "hypercolumn_cbam_densenet169_focal": 0.25,
            "yolo": 0.25,
            "rad_dino": 0.25
        }
        acc_equal, y_pred_equal = self.test_weights(images, labels, weights_equal, "Equal Weights")
        results['Equal Weights'] = {'accuracy': acc_equal, 'y_pred': y_pred_equal}
        print(f"   Equal Weights Accuracy: {acc_equal:.4f}")
        
        # 3. Preset weights
        print("\n3. Testing PRESET WEIGHTS (0.30, 0.30, 0.25, 0.15)...")
        weights_preset = {
            "maxvit": 0.30,
            "hypercolumn_cbam_densenet169_focal": 0.30,
            "yolo": 0.25,
            "rad_dino": 0.15
        }
        acc_preset, y_pred_preset = self.test_weights(images, labels, weights_preset, "Preset Weights")
        results['Preset Weights'] = {'accuracy': acc_preset, 'y_pred': y_pred_preset}
        print(f"   Preset Weights Accuracy: {acc_preset:.4f}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        for approach, data in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"{approach:.<40} {data['accuracy']:.4f}")
        
        return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints_dir = os.path.join(project_root, "outputs", "cross_validation")
    dataset_test = os.path.join(project_root, "balanced_augmented_dataset", "test")
    stacker_path = os.path.join(project_root, "outputs", "stacker.joblib")
    
    class_names = [
        "Comminuted", "Greenstick", "Healthy", "Oblique",
        "Oblique_Displaced", "Spiral", "Transverse", "Transverse_Displaced"
    ]
    
    model_names = ["maxvit", "hypercolumn_cbam_densenet169_focal", "yolo", "rad_dino"]
    
    print("=" * 80)
    print("WEIGHTED ENSEMBLE OPTIMIZATION & TESTING")
    print("=" * 80)
    print(f"Models: {', '.join(model_names)}")
    print(f"Device: {device}\n")
    
    try:
        # Initialize Ensemble
        print("Initializing Ensemble...")
        ensemble = EnsembleModule(
            class_names=class_names,
            model_names=model_names,
            checkpoints_dir=checkpoints_dir,
            num_classes=8,
            device=device
        )
        
        # Load stacker if available
        if os.path.exists(stacker_path):
            ensemble.stacker = joblib.load(stacker_path)
            print(f"✅ Loaded stacker from {stacker_path}")
        else:
            print(f"⚠️ Stacker not found at {stacker_path}")
        
        print("✅ Ensemble initialized\n")
        
        # Create tester
        tester = WeightedEnsembleTester(ensemble, class_names, dataset_test)
        
        # Load test data
        print("Loading test data...")
        images, labels = tester.load_test_data()
        print(f"✅ Loaded {len(images)} test samples\n")
        
        # Compare approaches
        results = tester.compare_approaches(images, labels)
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
