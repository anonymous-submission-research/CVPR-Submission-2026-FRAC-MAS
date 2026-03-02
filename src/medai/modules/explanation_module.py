import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from typing import Dict, Any, List, Optional, Union
import logging

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logging.warning("pytorch-grad-cam not installed. Explanation module will be limited.")

def get_transforms(img_size: int = 224):
    """Standard image transformations for inference."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Explanation Module Core ---

class ExplanationModule:
    """Generates Grad-CAM visualizations and textual explanations."""
    
    def __init__(self, model: nn.Module, class_names: List[str], device, body_part: str = "bone"):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.body_part = body_part
        self.transforms = get_transforms()
        self.target_layer = self._get_target_layer()
    
    def _get_target_layer(self):
        """Gets the appropriate target layer for Grad-CAM."""
        if self.model is None:
            return None
        
        # Try common layer names
        for attr in ['layer4', 'features', 'stages', 'blocks']:
            if hasattr(self.model, attr):
                layer = getattr(self.model, attr)
                if isinstance(layer, nn.Sequential) and len(layer) > 0:
                    return [layer[-1]]
                return [layer]
        
        # Fallback: get last conv layer
        layers = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                layers.append(module)
        return [layers[-1]] if layers else None
    
    def generate_gradcam(self, image_input: Union[Image.Image, str], target_class: int = None) -> Optional[np.ndarray]:
        """Generates Grad-CAM heatmap."""
        if not GRADCAM_AVAILABLE or self.model is None or self.target_layer is None:
            return None
        
        # Handle input
        if isinstance(image_input, str):
            try:
                img = Image.open(image_input).convert('RGB')
            except Exception:
                return None
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            return None

        try:
            input_tensor = self.transforms(img).unsqueeze(0).to(self.device)
            self.model.eval() # Ensure eval mode
            
            with GradCAM(model=self.model, target_layers=self.target_layer) as cam:
                targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                return grayscale_cam[0]
        except Exception as e:
            logging.warning(f"Grad-CAM generation failed: {e}")
            return None
    
    def visualize_gradcam(self, image: Image.Image, cam_array: np.ndarray) -> Image.Image:
        """Overlays Grad-CAM on the original image."""
        if cam_array is None or not GRADCAM_AVAILABLE:
            return image
        
        try:
            # Normalize image to 0-1
            img_array = np.array(image.resize((224, 224))) / 255.0
            
            # Create heatmap overlay
            visualization = show_cam_on_image(img_array.astype(np.float32), cam_array, use_rgb=True)
            return Image.fromarray(visualization)
        except Exception:
            return image # Fallback to original image
    
    def generate_explanation(self, diagnosis_result: Dict[str, Any], cam_array: np.ndarray = None) -> str:
        """Generates textual explanation based on diagnosis and Grad-CAM."""
        predicted_class = diagnosis_result.get("predicted_class", diagnosis_result.get("ensemble_prediction", "Unknown"))
        confidence = diagnosis_result.get("confidence_score", diagnosis_result.get("ensemble_confidence", 0.0))
        
        if predicted_class == "Healthy":
            if confidence > 0.90:
                return f"The {self.body_part} appears **healthy** with high confidence ({confidence:.2f}). No fracture pattern was detected."
            else:
                return f"The {self.body_part} is likely **healthy** ({confidence:.2f}), though some areas warrant closer examination."
        
        # Analyze heatmap if available
        location_text = ""
        if cam_array is not None:
            norm_cam = cam_array / (cam_array.max() + 1e-8)
            y_indices, x_indices = np.where(norm_cam > 0.5)
            if len(y_indices) > 0 and len(x_indices) > 0:
                avg_x = np.mean(x_indices) / cam_array.shape[1]
                avg_y = np.mean(y_indices) / cam_array.shape[0]
                
                x_loc = "right side" if avg_x > 0.65 else ("left side" if avg_x < 0.35 else "center")
                y_loc = "distal end" if avg_y > 0.65 else ("proximal end" if avg_y < 0.35 else "middle region")
                location_text = f" The model's attention is focused on the **{y_loc}** of the **{x_loc}**."
        
        # Confidence description
        if confidence > 0.9:
            conf_desc = "high"
        elif confidence > 0.7:
            conf_desc = "moderate"
        else:
            conf_desc = "low"
        
        explanation = (
            f"A fracture pattern consistent with **{predicted_class}** is detected with {conf_desc} "
            f"confidence ({confidence:.2f}).{location_text}"
        )
        
        # Add simpler visual cue description
        if predicted_class in ["Transverse", "Oblique"]:
             explanation += " This is based on a distinct linear focus."
        
        return explanation

# --- Test Block ---
if __name__ == "__main__":
    print("Explanation module loaded.")
