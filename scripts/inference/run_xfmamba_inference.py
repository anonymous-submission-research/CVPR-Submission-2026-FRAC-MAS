import torch
import torch.nn as nn
import os
import sys
from PIL import Image
from torchvision import transforms

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# Import model architecture from the fast script
from train_xfmamba_fast import XFMambaClassifier

def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use 8 classes as per dataset
    num_classes = 8
    model = XFMambaClassifier(num_classes=num_classes).to(device)

    weights_path = os.path.join(_PROJECT_ROOT, 'models', 'vssm_small_0229_ckpt_epoch_222.pth')
    
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}")
        return

    print(f"Loading weights from {weights_path}...")
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get('model', checkpoint)
        
        # Attempt to load state dict
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
        
        if len(missing) > 100 or len(unexpected) > 100:
            print("WARNING: Architecture mismatch detected. The weights might not belong to this XFMamba implementation.")
        
        model.eval()
        
        # Load a sample image
        sample_img_path = os.path.join(_PROJECT_ROOT, 'data', 'test_images', 'sample_xray.png')
        if not os.path.exists(sample_img_path):
            print(f"Sample image {sample_img_path} not found. Using a dummy image.")
            img = Image.new('RGB', (224, 224), color='white')
        else:
            img = Image.open(sample_img_path).convert('RGB')
            
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = tf(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            
        class_names = ['Comminuted', 'Greenstick', 'Healthy', 'Oblique', 'Oblique Displaced', 'Spiral', 'Transverse', 'Transverse Displaced']
        print(f"Prediction: {class_names[pred.item()]} (Confidence: {conf.item():.4f})")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_inference()
