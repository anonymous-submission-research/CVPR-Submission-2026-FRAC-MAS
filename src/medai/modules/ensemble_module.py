import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Union
import timm 
from medai.uncertainty.conformal import predict_conformal_set

# RAD-DINO / YOLO support
try:
    from transformers import AutoModel, AutoImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# ----------------------------------------------------------------------
# --- Helper Functions (Duplicated for standalone capability) ---
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
    
    # Simple mapping, can be expanded
    model_map = {
        'swin': 'swin_small_patch4_window7_224',
        'mobilenetv2': 'mobilenetv2_100',
        'efficientnetv2': 'tf_efficientnetv2_s',
        'maxvit': 'maxvit_rmlp_small_rw_224',
        'densenet169': 'densenet169',
    }
    # Check for hypercolumn variants or exact matches
    if name in model_map:
        timm_name = model_map[name]
    elif name.startswith('swin'): timm_name = 'swin_small_patch4_window7_224'
    elif 'densenet' in name: timm_name = 'densenet169'
    elif 'efficientnet' in name: timm_name = 'tf_efficientnetv2_s'
    else:
        # Fallback: try to us name directly
        timm_name = name

    try:
        m = timm.create_model(timm_name, pretrained=pretrained)
    except Exception:
        raise ValueError(f"Unknown or unavailable model: {name}")
    
    # Adjust classifier head
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
            # Some models might need custom logic
            pass

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
    Transverse <-> Transverse_Displaced
    Oblique <-> Oblique_Displaced
    """
    # Normalize spaces to underscores
    label = label.replace(" ", "_")
    if label == "Transverse":
        return "Transverse_Displaced"
    elif label == "Transverse_Displaced":
        return "Transverse"
    elif label == "Oblique":
        return "Oblique_Displaced"
    elif label == "Oblique_Displaced":
        return "Oblique"
    return label

# ----------------------------------------------------------------------
# --- RAD-DINO and YOLO helpers ---
# ----------------------------------------------------------------------

RAD_DINO_MODEL_NAME = "microsoft/rad-dino"

CLASS_NAMES = [
    "Comminuted", "Greenstick", "Healthy", "Oblique",
    "Oblique_Displaced", "Spiral", "Transverse", "Transverse_Displaced"
]

YOLO_SEARCH_PATHS = [
    "outputs/yolo_cls_finetune/yolo_cls_ft/weights/best.pt",
    "models/yolo_best.pt",
    "models/best.pt",
    "outputs/weights/best.pt",
    "weights/best.pt",
]


def _detect_rad_dino_head_type(state_dict):
    """Detect whether the saved RAD-DINO checkpoint uses a linear or MLP head."""
    for key in state_dict:
        if key.startswith("head.") and "head.0." in key:
            return "mlp"
    return "linear"


class RadDinoClassifier(nn.Module):
    """Wrapper that loads microsoft/rad-dino backbone + trained classification head."""
    def __init__(self, num_classes: int = 8, head_type: str = "linear"):
        super(RadDinoClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(RAD_DINO_MODEL_NAME)
        hidden = self.backbone.config.hidden_size
        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.head = nn.Linear(hidden, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        cls_token = out.last_hidden_state[:, 0]
        return self.head(cls_token)


def get_rad_dino_processor():
    return AutoImageProcessor.from_pretrained(RAD_DINO_MODEL_NAME)


def get_rad_dino_input_tensor(image: Image.Image, dev) -> torch.Tensor:
    processor = get_rad_dino_processor()
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].to(dev)


def _is_rad_dino_model_name(name: str) -> bool:
    return "rad_dino" in name.lower()


class YOLOClassifierWrapper(nn.Module):
    """Wraps an ultralytics YOLO classification model for ensemble use."""
    def __init__(self, yolo_model, class_names: List[str]):
        super().__init__()
        self.yolo = yolo_model
        self.class_names = class_names
        self._build_index_map()

    def _build_index_map(self):
        yolo_names = self.yolo.names if hasattr(self.yolo, "names") else {}
        self.index_map = {}
        for yidx, yname in yolo_names.items():
            for cidx, cname in enumerate(self.class_names):
                if yname.strip().lower() == cname.strip().lower():
                    self.index_map[yidx] = cidx
                    break

    def predict_pil(self, image: Image.Image) -> np.ndarray:
        results = self.yolo.predict(source=image, verbose=False)
        raw_probs = results[0].probs.data.cpu().numpy()
        aligned = np.zeros(len(self.class_names), dtype=np.float32)
        for yidx, prob in enumerate(raw_probs):
            cidx = self.index_map.get(yidx)
            if cidx is not None:
                aligned[cidx] = prob
        s = aligned.sum()
        if s > 0:
            aligned /= s
        return aligned

    def forward(self, x):
        raise NotImplementedError("Use predict_pil() for YOLO-based inference")


def _is_yolo_model(model) -> bool:
    return isinstance(model, YOLOClassifierWrapper)

# ----------------------------------------------------------------------
# --- Ensemble Module Core ---
# ----------------------------------------------------------------------

class EnsembleModule:
    """Runs inference across multiple models and combines predictions."""
    
    # Classes where hypercolumn models should get more weight
    HYPERCOLUMN_PRIORITY_CLASSES = {"Oblique", "Oblique_Displaced", "Transverse", "Transverse_Displaced"}
    # Weight for hypercolumn models when priority class is detected
    HYPERCOLUMN_WEIGHT = 1.0
    # Weight for other models
    DEFAULT_WEIGHT = 1.0
    
    def __init__(self, 
                 class_names: List[str], 
                 models: Optional[Dict[str, nn.Module]] = None, 
                 model_names: Optional[List[str]] = None,
                 checkpoints_dir: Optional[str] = None,
                 num_classes: int = 8, 
                 device=None,
                 img_size: int = 224, 
                 conformal_threshold: float = None):
        
        self.class_names = class_names
        self.device = device if device else get_device()
        self.transforms = get_transforms(img_size)
        self.conformal_threshold = conformal_threshold
        self.num_classes = num_classes
        
        self.models = {}
        if models is not None:
             self.models = models
             # validation?
        elif model_names and checkpoints_dir:
            self.model_names = model_names
            self._load_all_models(checkpoints_dir)
        else:
            raise ValueError("Either 'models' dict OR ('model_names' list AND 'checkpoints_dir') must be provided.")
            
    def _load_all_models(self, checkpoints_dir: str):
        """Loads all specified model checkpoints including RAD-DINO and YOLO."""
        print(f"Loading {len(self.model_names)} models from {checkpoints_dir} on {self.device}...")
        
        for name in self.model_names:
            try:
                # --- RAD-DINO ---
                if _is_rad_dino_model_name(name):
                    if not TRANSFORMERS_AVAILABLE:
                        print(f"  ⚠️ Skipping {name}: transformers not installed.")
                        continue
                    # Try multiple checkpoint paths
                    ckpt_path = os.path.join(checkpoints_dir, "best_rad_dino_classifier.pth")
                    alt_ckpt_path = os.path.join(os.path.dirname(checkpoints_dir), "dinorad", "dinorad_best.pth")
                    
                    if os.path.exists(ckpt_path):
                        pass  # Use default path
                    elif os.path.exists(alt_ckpt_path):
                        ckpt_path = alt_ckpt_path
                    else:
                        print(f"  ❌ RAD-DINO checkpoint not found at {ckpt_path} or {alt_ckpt_path}. Skipping.")
                        continue
                    
                    sd = torch.load(ckpt_path, map_location=self.device)
                    state_dict = sd.get("model_state_dict", sd)
                    head_type = _detect_rad_dino_head_type(state_dict)
                    model = RadDinoClassifier(self.num_classes, head_type=head_type)
                    model.load_state_dict(state_dict, strict=False)
                    model.to(self.device).eval()
                    self.models[name] = model
                    print(f"  ✅ Successfully loaded {name} (RAD-DINO, head={head_type}).")
                    continue

                # --- YOLO ---
                if name.lower() in ("yolo", "yolov26m", "yolo26m"):
                    if not ULTRALYTICS_AVAILABLE:
                        print(f"  ⚠️ Skipping {name}: ultralytics not installed.")
                        continue
                    yolo_path = None
                    for sp in YOLO_SEARCH_PATHS:
                        if os.path.exists(sp):
                            yolo_path = sp
                            break
                    if yolo_path is None:
                        print(f"  ❌ YOLO checkpoint not found. Searched: {YOLO_SEARCH_PATHS}. Skipping.")
                        continue
                    yolo_raw = YOLO(yolo_path, task="classify")
                    wrapper = YOLOClassifierWrapper(yolo_raw, self.class_names)
                    self.models[name] = wrapper
                    print(f"  ✅ Successfully loaded {name} (YOLO) from {yolo_path}.")
                    continue

                # --- Standard timm / hypercolumn models ---
                checkpoint_path = os.path.join(checkpoints_dir, f"best_{name}.pth")
                base_arch = 'densenet169' if 'densenet' in name else name
                model = get_model(base_arch, self.num_classes, pretrained=False).to(self.device)
                
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                self.models[name] = model
                print(f"  ✅ Successfully loaded {name}.")
            except Exception as e:
                print(f"  ❌ Failed to load {name}. Error: {e}. Skipping.")
        
        if not self.models:
            raise RuntimeError("No models were successfully loaded.")

    def _is_hypercolumn_model(self, model_name: str) -> bool:
        """Check if a model is a hypercolumn/column model."""
        return "hypercolumn" in model_name.lower() or "cbam" in model_name.lower()
    
    def _get_weighted_average(self, all_probs: List[np.ndarray], model_names: List[str], 
                               use_hypercolumn_priority: bool) -> np.ndarray:
        """
        Compute weighted average of probabilities.
        """
        weights = []
        for name in model_names:
            if use_hypercolumn_priority and self._is_hypercolumn_model(name):
                weights.append(self.HYPERCOLUMN_WEIGHT)
            else:
                weights.append(self.DEFAULT_WEIGHT)
        
        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
             weights = np.ones(len(weights)) / len(weights)
        
        # Compute weighted average
        weighted_probs = np.zeros_like(all_probs[0])
        for prob, weight in zip(all_probs, weights):
            weighted_probs += prob * weight
        
        return weighted_probs

    def _predict_with_stacker(self, all_probs: List[np.ndarray], model_names: List[str]):
        """If a `stacker` is present on the instance, use it to predict class probabilities."""
        if not hasattr(self, 'stacker') or self.stacker is None:
            raise RuntimeError('No stacker available')
        
        probs = np.stack(all_probs, axis=0)  # (M, C)
        feat = probs.reshape(1, -1)
        proba = self.stacker.predict_proba(feat)[0]
        return proba
    
    @torch.no_grad()
    def run_ensemble(self, image_input: Union[str, Image.Image], use_stacking: bool = False) -> Dict[str, Any]:
        """Runs ensemble inference on an image (path or object)."""
        if not self.models:
            return {"error": "No models loaded"}
        
        image_path_str = "in-memory-image"
        
        # 1. Image Loading
        if isinstance(image_input, str):
            full_image_path = os.path.abspath(image_input)
            if not os.path.exists(full_image_path):
                return {"error": f"Image file not found at {image_input}"}
            try:
                img = Image.open(full_image_path).convert('RGB')
                image_path_str = image_input
            except Exception as e:
                return {"error": f"Failed to open image at {full_image_path}: {e}"}
        elif isinstance(image_input, Image.Image):
             img = image_input.convert('RGB')
        else:
             return {"error": "Invalid input type. Expected str (path) or PIL.Image."}
        
        input_tensor = self.transforms(img).unsqueeze(0).to(self.device)
        
        all_probs = []
        model_names = []
        individual_predictions = {}
        
        for name, model in self.models.items():
            try:
                if _is_yolo_model(model):
                    # YOLO has its own preprocessing pipeline
                    probs = model.predict_pil(img)
                elif _is_rad_dino_model_name(name):
                    # RAD-DINO uses HuggingFace AutoImageProcessor
                    rad_tensor = get_rad_dino_input_tensor(img, self.device)
                    logits = model(rad_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                else:
                    # Standard timm model
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            except Exception as e:
                print(f"  ⚠️ Inference failed for {name}: {e}. Skipping.")
                continue
                
            all_probs.append(probs)
            model_names.append(name)
            
            pred_idx = np.argmax(probs)
            pred_class_raw = self.class_names[pred_idx]
            
            individual_predictions[name] = {
                "class": _swap_prediction_label(pred_class_raw),
                "confidence": float(probs[pred_idx])
            }
        
        if not all_probs:
            return {"error": "All models failed during inference"}
        
        # First pass: compute equal-weighted average to determine likely class
        equal_avg_probs = np.mean(all_probs, axis=0)
        preliminary_idx = np.argmax(equal_avg_probs)
        preliminary_class = self.class_names[preliminary_idx]
        
        # Check if preliminary class is one where hypercolumn models should have priority
        use_hypercolumn_priority = preliminary_class in self.HYPERCOLUMN_PRIORITY_CLASSES
        
        # Second pass: compute final weighted average based on detected class
        if use_stacking:
            try:
                avg_probs = self._predict_with_stacker(all_probs, model_names)
            except Exception:
                avg_probs = self._get_weighted_average(all_probs, model_names, use_hypercolumn_priority)
        else:
            avg_probs = self._get_weighted_average(all_probs, model_names, use_hypercolumn_priority)
            
        ensemble_idx = np.argmax(avg_probs)
        ensemble_class_raw = self.class_names[ensemble_idx]
        ensemble_class = _swap_prediction_label(ensemble_class_raw)
        ensemble_confidence = float(avg_probs[ensemble_idx])
        
        # Prepare all probabilities with swapped labels
        all_probs_dict = {}
        for i in range(len(avg_probs)):
            class_name = self.class_names[i]
            swapped_name = _swap_prediction_label(class_name)
            all_probs_dict[swapped_name] = float(avg_probs[i])
            
        # Swap logic for list output
        # (Same as in DiagnosticModule)
        probs_np = avg_probs.copy()
        try:
             if "Transverse" in self.class_names and "Transverse Displaced" in self.class_names:
                idx_trans = self.class_names.index("Transverse")
                idx_trans_disp = self.class_names.index("Transverse Displaced")
                probs_np[idx_trans], probs_np[idx_trans_disp] = probs_np[idx_trans_disp], probs_np[idx_trans]

             if "Oblique" in self.class_names and "Oblique Displaced" in self.class_names:
                idx_obl = self.class_names.index("Oblique")
                idx_obl_disp = self.class_names.index("Oblique Displaced")
                probs_np[idx_obl], probs_np[idx_obl_disp] = probs_np[idx_obl_disp], probs_np[idx_obl]
        except ValueError:
            pass
        
        result = {
            "image_path": image_path_str,
            "ensemble_prediction": ensemble_class,
            "ensemble_confidence": ensemble_confidence,
            "individual_predictions": individual_predictions,
            "fracture_detected": ensemble_class != "Healthy",
            "all_probabilities": probs_np.tolist(),
            "all_probabilities_dict": all_probs_dict,
            "weighted_voting": use_hypercolumn_priority,
            "weighting_reason": f"Hypercolumn models prioritized for {preliminary_class}" if use_hypercolumn_priority else "Equal weights for all models",
            "is_label_swapped": True
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
    parser = argparse.ArgumentParser(description='Multi-Model Ensemble Module.')
    parser.add_argument('--image-path', required=True, help='Path to the image.')
    parser.add_argument('--checkpoints-dir', required=True, help='Path to checkpoints directory.')
    parser.add_argument('--models', type=str, default='swin,mobilenetv2,efficientnetv2,maxvit,densenet169', 
                        help='Comma-separated names of the models to load.')
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--class-names', required=True, help='Comma-separated class names.')
    
    args = parser.parse_args()

    models_list = [m.strip() for m in args.models.split(',')]
    class_names_list = [c.strip() for c in args.class_names.split(',')]
    
    try:
        module = EnsembleModule(
            model_names=models_list,
            checkpoints_dir=args.checkpoints_dir,
            num_classes=args.num_classes,
            class_names=class_names_list
        )
    except RuntimeError as e:
        print(f"\nFATAL ERROR during initialization: {e}")
        exit(1)

    result = module.run_ensemble(args.image_path)
    
    print("\n--- ENSEMBLE MODULE RESULT ---")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"Prediction: {result['ensemble_prediction']} (Conf: {result['ensemble_confidence']:.4f})")
    print("-----------------------------\n")
