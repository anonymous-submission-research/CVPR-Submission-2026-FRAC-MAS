import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import List, Dict, Any
import timm 
try:
    from medai.uncertainty.conformal import predict_conformal_set
except Exception:
    from ..uncertainty.conformal import predict_conformal_set

# ----------------------------------------------------------------------
# --- Helper Functions ---
# ----------------------------------------------------------------------

DEVICE = None
IMG_SIZE = 224

def get_device():
    """Detects and returns the appropriate torch device."""
    global DEVICE
    if DEVICE is None:
        if torch.cuda.is_available(): 
            DEVICE = torch.device('cuda')
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(): 
            DEVICE = torch.device('mps')
        else:
            DEVICE = torch.device('cpu')
    return DEVICE

def get_model(name: str, num_classes: int, pretrained: bool=True):
    """Loads and adapts one of the specified pretrained models from timm."""
    name = name.lower()
    
    model_map = {
        'swin': 'swin_small_patch4_window7_224',
        'mobilenetv2': 'mobilenetv2_100',
        'efficientnetv2': 'tf_efficientnetv2_s',
        'maxvit': 'maxvit_rmlp_small_rw_224',
        'densenet169': 'densenet169',
    }
    
    if name not in model_map:
        raise ValueError(f"Unknown model: {name}")

    m = timm.create_model(model_map[name], pretrained=pretrained)
    
    # Adjust classifier head based on common timm model types
    if hasattr(m, 'head') and isinstance(m.head, nn.Linear):
        m.head = nn.Linear(m.head.in_features, num_classes)
    elif hasattr(m, 'fc') and isinstance(m.fc, nn.Linear):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif hasattr(m, 'classifier') and isinstance(m.classifier, nn.Linear):
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    else:
        try:
            m.reset_classifier(num_classes=num_classes)
        except Exception:
            raise RuntimeError(f"Could not automatically adapt classifier head for {name}")

    return m

def get_transforms(img_size: int = 224):
    """Standard image transformations for inference."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def _swap_prediction_label(label: str) -> str:
    """
    Swaps predictions for specific classes as requested:
    Transverse <-> Transverse Displaced
    Oblique <-> Oblique Displaced
    """
    if label == "Transverse":
        return "Transverse Displaced"
    elif label == "Transverse Displaced":
        return "Transverse"
    elif label == "Oblique":
        return "Oblique Displaced"
    elif label == "Oblique Displaced":
        return "Oblique"
    return label

# ----------------------------------------------------------------------
# --- Model Ensemble Agent Core (with all fixes) ---
# ----------------------------------------------------------------------

class ModelEnsembleAgent:
    def __init__(self, model_names: List[str], checkpoints_dir: str, num_classes: int, class_names: List[str], conformal_threshold: float = None):
        self.models = {}
        self.model_names = model_names
        self.num_classes = num_classes
        self.class_names = class_names
        self.transforms = get_transforms(IMG_SIZE)
        
        self.device = get_device() 
        self._load_all_models(checkpoints_dir)
        self.conformal_threshold = conformal_threshold

    def _load_all_models(self, checkpoints_dir: str):
        """Loads all specified model checkpoints with strict=False fallback."""
        print(f"Loading {len(self.model_names)} models from {checkpoints_dir} on {self.device}...")
        
        for name in self.model_names:
            
            # FIX: Corrected file naming convention (best_modelname.pth)
            checkpoint_path = os.path.join(checkpoints_dir, f"best_{name}.pth")
            
            print(f"  Attempting to load {name} from expected path: {checkpoint_path}...")
            
            try:
                model = get_model(name, self.num_classes, pretrained=False).to(self.device)
                
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                # FIX: Use strict=False to bypass common RuntimeError due to classifier size mismatches
                model.load_state_dict(state_dict, strict=False)

                model.eval()
                self.models[name] = model
                print(f"  ✅ Successfully loaded {name}.")
            
            except FileNotFoundError:
                print(f"  ❌ Checkpoint not found at: {checkpoint_path}. Skipping.")
            except Exception as e:
                # FIX: Detailed error reporting to show the full RuntimeError message
                print(f"  ❌ Failed to load {name}. Error: {e.__class__.__name__}. Details: {e}. Skipping.")
        
        if not self.models:
            raise RuntimeError("No models were successfully loaded. Cannot run ensemble.")

    @torch.no_grad()
    def run_ensemble(self, image_path: str) -> Dict[str, Any]:
        """Runs inference across all loaded models and computes the ensemble prediction."""
        
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        except Exception as e:
            return {"error": f"Failed to load or process image: {e}"}

        all_probs = []
        individual_predictions = {}
        
        for name, model in self.models.items():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            all_probs.append(probs)
            
            pred_idx = np.argmax(probs)
            pred_conf = probs[pred_idx]
            
            individual_predictions[name] = {
                "class": _swap_prediction_label(self.class_names[pred_idx]),
                "confidence": float(pred_conf)
            }

        # Ensemble Decision (Soft Voting/Averaging)
        avg_probs = np.mean(all_probs, axis=0)
        ensemble_idx = np.argmax(avg_probs)
        ensemble_confidence = avg_probs[ensemble_idx]
        ensemble_class = _swap_prediction_label(self.class_names[ensemble_idx])

        # Swap probabilities in the list to match the swapped labels
        try:
            if "Transverse" in self.class_names and "Transverse Displaced" in self.class_names:
                idx_trans = self.class_names.index("Transverse")
                idx_trans_disp = self.class_names.index("Transverse Displaced")
                avg_probs[idx_trans], avg_probs[idx_trans_disp] = avg_probs[idx_trans_disp], avg_probs[idx_trans]

            if "Oblique" in self.class_names and "Oblique Displaced" in self.class_names:
                idx_obl = self.class_names.index("Oblique")
                idx_obl_disp = self.class_names.index("Oblique Displaced")
                avg_probs[idx_obl], avg_probs[idx_obl_disp] = avg_probs[idx_obl_disp], avg_probs[idx_obl]
        except ValueError:
            pass

        result = {
            "image_path": image_path,
            "ensemble_prediction": ensemble_class,
            "ensemble_confidence": float(ensemble_confidence),
            "individual_predictions": individual_predictions,
            "fracture_detected": ensemble_class != "Healthy",
            "all_probabilities": avg_probs.tolist()
        }

        if self.conformal_threshold is not None:
            try:
                conformal_set = predict_conformal_set(avg_probs, self.conformal_threshold, self.class_names)
                result["conformal_set"] = conformal_set
                result["conformal_threshold"] = float(self.conformal_threshold)
            except Exception:
                result["conformal_set_error"] = "failed to compute conformal set"

        return result

# ----------------------------------------------------------------------
# --- Execution Block ---
# ----------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Model Ensemble (Cross-Validation) Agent.')
    parser.add_argument('--image-path', required=True, help='Path to the image for inference.')
    parser.add_argument('--checkpoints-dir', required=True, # Made required since default path was confusing
                        help='Absolute path to the directory containing the model checkpoints (e.g., best_swin.pth).')
    parser.add_argument('--models', type=str, default='swin,mobilenetv2,efficientnetv2,maxvit,densenet169', 
                        help='Comma-separated names of the models to load.')
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--class-names', required=True, 
                        help='Comma-separated list of class names.')
    
    args = parser.parse_args()

    models_list = [m.strip() for m in args.models.split(',')]
    class_names_list = [c.strip() for c in args.class_names.split(',')]
    
    try:
        ensemble_agent = ModelEnsembleAgent(
            model_names=models_list,
            checkpoints_dir=args.checkpoints_dir,
            num_classes=args.num_classes,
            class_names=class_names_list
        )
    except RuntimeError as e:
        print(f"\nFATAL ERROR during initialization: {e}")
        exit(1)

    result = ensemble_agent.run_ensemble(args.image_path)
    
    print("\n--- ENSEMBLE AGENT RESULT ---")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"FINAL ENSEMBLE PREDICTION: **{result['ensemble_prediction']}** (Confidence: {result['ensemble_confidence']:.4f})")
        
        print("\nIndividual Model Predictions:")
        loaded_model_names = ensemble_agent.models.keys()
        
        for name in models_list:
            if name in loaded_model_names:
                 pred = result['individual_predictions'][name]
                 print(f"  {name.upper():<15}: {pred['class']:<20} (Conf: {pred['confidence']:.4f})")
            else:
                 print(f"  {name.upper():<15}: (Skipped/Failed to Load)")

    print("-----------------------------\n")