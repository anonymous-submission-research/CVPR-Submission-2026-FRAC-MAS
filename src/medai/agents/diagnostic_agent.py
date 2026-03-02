import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from PIL import Image
import numpy as np
from typing import Dict, Any, List
try:
    # Prefer package-style import when installed as top-level package
    from medai.uncertainty.conformal import predict_conformal_set
except Exception:
    # Fallback to relative import when running from the repository (src in path)
    from ..uncertainty.conformal import predict_conformal_set

# --- 1. CONFIGURATION ---

def get_device():
    """Dynamically selects CUDA, MPS, or falls back to CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()

def get_model(name: str, num_classes: int, pretrained: bool=True):
    """Loads the model architecture (Swin, ConvNext, etc.)."""
    name = name.lower()
    if name.startswith('swin'):
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=pretrained)
        if hasattr(model, 'reset_classifier'):
            model.reset_classifier(num_classes=num_classes)
        else:
            model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    
    # Add other model loading logic here if needed (ConvNext, Densenet, etc.)
    raise ValueError(f'Unknown model: {name}')

def get_transforms(img_size: int = 224):
    """Returns the standard test/validation transforms."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
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

# --- 2. DIAGNOSTIC AGENT CORE ---

class DiagnosticAgent:
    def __init__(self, checkpoint_path: str, model_name: str, num_classes: int, img_size: int, class_names: List[str], conformal_threshold: float = None):
        self.device = DEVICE
        self.img_size = img_size
        self.class_names = class_names
        self.conformal_threshold = conformal_threshold
        
        # 1. Load Model Architecture
        self.model = get_model(model_name, num_classes, pretrained=False).to(self.device)
        
        # 2. Load Weights from Checkpoint
        try:
            ck = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ck.get('model_state_dict', ck)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"✅ Diagnostic Agent loaded model from {checkpoint_path} on {self.device}.")
        except FileNotFoundError:
            print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
            exit(1)
        except Exception as e:
            print(f"❌ Error loading model state: {e}")
            exit(1)

        # 3. Setup Transforms
        self.transform = get_transforms(self.img_size)

    def run_diagnosis(self, image_path: str) -> Dict[str, Any]:
        """
        Runs the image classification model, detects fractures, and outputs scores.
        
        This method includes the fix for FileNotFoundError by resolving the path.
        """
        
        # CRITICAL FIX: Convert relative path to absolute path for reliable file access
        full_image_path = os.path.abspath(image_path)

        if not os.path.exists(full_image_path):
            # Report the original path back to the user for clarity
            return {"error": f"Image file not found at {image_path}"}

        # 1. Image Loading and Preprocessing
        try:
            # Use the resolved full path for PIL to open
            img = Image.open(full_image_path).convert('RGB') 
        except Exception as e:
            return {"error": f"Failed to open image at {full_image_path}. Reason: {e}"}


        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # 2. Model Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
        # Softmax to get probabilities (confidence scores)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        
        # 3. Score Calculation
        
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
        uncertainty = 1.0 - confidence
        
        predicted_class_name_raw = self.class_names[predicted_idx]
        predicted_class_name = _swap_prediction_label(predicted_class_name_raw)

        # Determine Fracture Presence (assuming 'Healthy' is a known class)
        is_fracture_detected = (predicted_class_name != 'Healthy')
        
        # We need to be careful with all_probabilities as list.
        # Ideally we should return a dict, but if legacy code expects list, we might just swap values in the list
        # such that the value for 'Transverse' (index 6) is now the value that was at index 7?
        # No, that's confusing.
        # The user request is "predict the opposite".
        # So if model predicts "Transverse" (index 6), we say "Transverse Displaced".
        # The confidence should be coming from index 6.
        # But if we return a list of probs, and the consumer maps it to class_names, they will see high prob at index 6 -> Transverse.
        # So we MUST swap the values in the list if we want standard consumers to see "Transverse Displaced" having the high probability?
        # Wait, if we swap values: probs[6] <-> probs[7].
        # Then probs[6] (Transverse slot) gets the value of Transverse Displaced (which was low). So Transverse becomes low.
        # And probs[7] (Transverse Displaced slot) gets the value of Transverse (which was high). So Transverse Displaced becomes high.
        # If the consumer reads max(probs), they find index 7 is high -> Transverse Displaced. Correct.
        
        # So I will swap the probabilities in the list for the corresponding indices.
        # Assuming typical class order... if class_names is passed in, I should find indices from it.
        
        probs_np = probabilities.cpu().numpy()
        
        try:
            # Find indices safely
            if "Transverse" in self.class_names and "Transverse Displaced" in self.class_names:
                idx_trans = self.class_names.index("Transverse")
                idx_trans_disp = self.class_names.index("Transverse Displaced")
                # Swap
                probs_np[idx_trans], probs_np[idx_trans_disp] = probs_np[idx_trans_disp], probs_np[idx_trans]

            if "Oblique" in self.class_names and "Oblique Displaced" in self.class_names:
                idx_obl = self.class_names.index("Oblique")
                idx_obl_disp = self.class_names.index("Oblique Displaced")
                # Swap
                probs_np[idx_obl], probs_np[idx_obl_disp] = probs_np[idx_obl_disp], probs_np[idx_obl]
            
        except ValueError:
            pass
            
        uncertainty = 1.0 - confidence # This is approximate, really entropy
        
        result = {
            "image_path": image_path,
            "fracture_detected": is_fracture_detected,
            "predicted_class": predicted_class_name,
            "severity_type": predicted_class_name,  # Proxy for severity
            "confidence_score": confidence,
            "uncertainty_score": uncertainty,
            "all_probabilities": probs_np.tolist()
        }

        # Add conformal prediction set when a threshold is provided
        if self.conformal_threshold is not None:
            try:
                conformal_set = predict_conformal_set(probabilities.cpu().numpy(), self.conformal_threshold, self.class_names)
                result["conformal_set"] = conformal_set
                result["conformal_threshold"] = float(self.conformal_threshold)
            except Exception:
                # Don't fail inference if conformal post-process errors out
                result["conformal_set_error"] = "failed to compute conformal set"

        return result
# --- 3. EXECUTION ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a diagnostic agent on a single image.')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the image file to diagnose.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (e.g., outputs/swin_mps/best.pth)')
    parser.add_argument('--model', type=str, default='swin', choices=['swin', 'convnext', 'densenet'])
    parser.add_argument('--num-classes', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--class-names', type=str, required=True, 
                        help='Comma-separated list of class names (e.g., "A,B,C")')
    
    args = parser.parse_args()

    # Convert class names string to a list
    class_names_list = [c.strip() for c in args.class_names.split(',')]

    # Ensure 'Healthy' is in the list for the 'fracture_detected' check to work reliably
    if 'Healthy' not in class_names_list:
        print("Warning: 'Healthy' class not found in --class-names list. Fracture detection may be inaccurate.")
    
    # Initialize the Agent
    agent = DiagnosticAgent(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        num_classes=args.num_classes,
        img_size=args.img_size,
        class_names=class_names_list
    )

    # Run the Diagnosis
    result = agent.run_diagnosis(args.image_path)

    # Output Results
    print("\n--- DIAGNOSTIC RESULTS ---")
    if "error" in result:
        print(f"Status: FAILED\nReason: {result['error']}")
    else:
        print(f"Status: SUCCESS")
        print(f"Image: {result['image_path']}")
        print(f"Fracture Detected: {'YES' if result['fracture_detected'] else 'NO'}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"--- Scores ---")
        print(f"Severity Type: {result['severity_type']}")
        print(f"Confidence Score: {result['confidence_score']:.4f}")
        print(f"Uncertainty Score: {result['uncertainty_score']:.4f}")
    print("--------------------------\n")