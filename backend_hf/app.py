import os
from dotenv import load_dotenv
import sys
import multiprocessing
# Prevent leaked-semaphore warning on macOS when SentenceTransformer / tokenizers
# fork background processes.  Must be called before any other multiprocessing use.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set

# Clean up any leaked semaphores at interpreter exit so the resource-tracker
# doesn't emit a noisy warning on Ctrl-C.
import atexit, multiprocessing.resource_tracker as _rt

def _cleanup_semaphores():
    """Silence 'leaked semaphore' warnings at shutdown."""
    try:
        _rt._resource_tracker._stop = None  # type: ignore[attr-defined]
    except Exception:
        pass

atexit.register(_cleanup_semaphores)

# Add src to path for imports - handles both local (../src) and container/HF (./src) structures
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../src')) # Local: src is sibling
sys.path.append(os.path.join(current_dir, '..'))     # Local: parent of medai
sys.path.append(os.path.join(current_dir, 'src'))    # HF: src is subdir
sys.path.append(current_dir)                         # HF: current dir is root

load_dotenv() # Load environment variables from .env file

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import io
import timm
import requests
import base64
import logging
import uuid
from datetime import datetime
from fastapi.responses import StreamingResponse, JSONResponse
import matplotlib.pyplot as plt
from io import BytesIO
import cv2

# Import Agents for Critic Flow (Loaded from self-contained module for cloud deployment)
try:
    from backend_hf.shared import IMAGE_STORE, CLASS_NAMES
    # We will redefine CLASS_NAMES locally if import fails but ideally we use shared
except ImportError:
    try:
        from shared import IMAGE_STORE, CLASS_NAMES
    except ImportError:
        # Fallback if shared module fails
        IMAGE_STORE = {}
        CLASS_NAMES = [
            "Comminuted", "Greenstick", "Healthy", "Oblique", 
            "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"
        ]

try:
    from medai_agent_module import CriticAgent, evaluate_consensus
except ImportError:
    logger.warning("medai_agent_module not found in local path. attempting Standard Import.")
    try:
        from medai.agents.critic_agent import CriticAgent
        from medai.utils.consensus import evaluate_consensus
    except ImportError:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Image Store for Chat Agent (In-Memory for Demo)
# In production, use Redis or S3/Blob storage
# IMAGE_STORE = {} # Now from shared.py

# Try optional imports
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logger.warning("pytorch-grad-cam not installed. Explainability features will be disabled.")

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not installed. Knowledge features will be disabled.")

app = FastAPI()

# ============================================================================
# CUSTOM MODEL ARCHITECTURES
# ============================================================================

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0.0):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
    
    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0:
            out = nn.functional.dropout(out, p=self.drop_rate, training=self.training)
        return out

class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0.0):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module(f'denselayer{i + 1}', layer)
    
    def forward(self, x):
        features = [x]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class HypercolumnCBAMDenseNet(nn.Module):
    def __init__(self, num_classes=8, growth_rate=32, bn_size=4, drop_rate=0.0):
        super(HypercolumnCBAMDenseNet, self).__init__()
        import torchvision.models as models
        densenet = models.densenet169(weights=None)
        self.features = densenet.features
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64)
        )
        self.db1 = self.features.denseblock1
        self.db2 = self.features.denseblock2
        self.db3 = self.features.denseblock3
        self.db4 = self.features.denseblock4
        self.t1 = self.features.transition1
        self.t2 = self.features.transition2
        self.t3 = self.features.transition3
        self.norm_final = self.features.norm5
        self.fusion_conv = nn.Conv2d(2688, 1024, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(1024)
        self.cbam = CBAM(1024)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.init_conv(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.db1(x)
        t1_out = self.t1(x)
        x = self.db2(t1_out)
        t2_out = self.t2(x)
        x = self.db3(t2_out)
        t3_out = self.t3(x)
        x = self.db4(t3_out)
        x_final = self.norm_final(x)
        target_size = x_final.shape[2:]
        t1_resized = nn.functional.interpolate(t1_out, size=target_size, mode='bilinear', align_corners=False)
        t2_resized = nn.functional.interpolate(t2_out, size=target_size, mode='bilinear', align_corners=False)
        t3_resized = nn.functional.interpolate(t3_out, size=target_size, mode='bilinear', align_corners=False)
        hypercolumn = torch.cat([x_final, t3_resized, t2_resized, t1_resized], dim=1)
        x = self.fusion_conv(hypercolumn)
        x = self.bn_fusion(x)
        x = nn.functional.relu(x)
        x = self.cbam(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============================================================================
# CONSTANTS & CONFIG
# ============================================================================

# CLASS_NAMES imported from shared if available, else defined here as fallback

if not CLASS_NAMES:
    CLASS_NAMES = [
        "Comminuted", "Greenstick", "Healthy", "Oblique", 
        "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"
    ]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

MODEL_FILES = {
    "best_swin.pth": "swin",
    "best_densenet169.pth": "densenet169",
    "best_efficientnetv2.pth": "efficientnetv2",
    "best_mobilenetv2.pth": "mobilenetv2",
    "best_maxvit.pth": "maxvit",
    "best_hypercolumn_cbam_densenet169.pth": "hypercolumn_cbam_densenet169",
    "best_hypercolumn_cbam_densenet169_focal.pth": "hypercolumn_cbam_densenet169_focal",
    "best_hypercolumn_cbam_densenet169_old.pth": "hypercolumn_cbam_densenet169_old",
    "best_hypercolumn_densenet169.pth": "hypercolumn_densenet169",
    "best_hypercolumn_densenet169_old.pth": "hypercolumn_densenet169_old",
    "best_rad_dino_classifier.pth": "rad_dino",
}

MODEL_CONFIGS = {
    "swin": "swin_small_patch4_window7_224",
    "densenet169": "densenet169",
    "efficientnetv2": "efficientnet_b0",
    "mobilenetv2": "mobilenetv2_100",
    "maxvit": "maxvit_tiny_tf_224",
    "hypercolumn_cbam_densenet169": "custom",
    "hypercolumn_cbam_densenet169_focal": "custom",
    "hypercolumn_cbam_densenet169_old": "custom",
    "hypercolumn_densenet169": "custom",
    "hypercolumn_densenet169_old": "custom",
    "rad_dino": "rad_dino",
    "yolo": "yolo",
}

# Active models for ensemble inference (override via ACTIVE_MODELS env var)
# Only these models are loaded at startup and used for inference / Grad-CAM.
# Other checkpoints in ./models are treated as baselines and NOT loaded.
ACTIVE_MODELS = [
    m.strip()
    for m in os.environ.get(
        "ACTIVE_MODELS", "maxvit,yolo,hypercolumn_cbam_densenet169,rad_dino"
    ).split(",")
    if m.strip()
]

# RAD-DINO constants
RAD_DINO_MODEL_NAME = "microsoft/rad-dino"

# YOLO model search paths (relative to models dir or project root)
YOLO_SEARCH_PATHS = [
    "outputs/yolo_cls_finetune/yolo_cls_ft/weights/best.pt",
    "models/yolo_best.pt",
    "models/best.pt",
    "outputs/weights/best.pt",
    "weights/best.pt",
]

MEDICAL_KNOWLEDGE_BASE = {
    "Comminuted": {
        "definition": "A fracture where the bone is shattered into three or more fragments.",
        "icd_code": "S42.35",
        "severity": "Severe",
        "treatment_guidelines": [
            "Immediate orthopedic consultation required",
            "Surgical intervention often necessary (ORIF)",
            "Extended immobilization period (8-12 weeks)",
            "Physical therapy post-healing"
        ],
        "prognosis": "Recovery typically 3-6 months with proper surgical management."
    },
    "Greenstick": {
        "definition": "An incomplete fracture where the bone bends and cracks but does not break completely.",
        "icd_code": "S42.31",
        "severity": "Mild to Moderate",
        "treatment_guidelines": [
            "Often treated with casting or splinting",
            "Immobilization for 4-6 weeks",
            "Common in children due to bone flexibility",
            "Follow-up X-rays to monitor healing"
        ],
        "prognosis": "Excellent prognosis, typically heals within 4-8 weeks."
    },
    "Healthy": {
        "definition": "No fracture detected. Bone structure appears normal.",
        "icd_code": "Z03.89",
        "severity": "None",
        "treatment_guidelines": [
            "No treatment required for fracture",
            "Address any other symptoms if present",
            "Follow up if pain persists"
        ],
        "prognosis": "N/A - No fracture present."
    },
    "Oblique": {
        "definition": "A fracture with an angled break across the bone shaft.",
        "icd_code": "S42.33",
        "severity": "Moderate",
        "treatment_guidelines": [
            "May require reduction if displaced",
            "Casting for 6-8 weeks typical",
            "Monitor for displacement during healing",
            "Physical therapy may be beneficial"
        ],
        "prognosis": "Good prognosis with proper alignment, 6-10 weeks healing."
    },
    "Oblique Displaced": {
        "definition": "An angled fracture where bone fragments have shifted from normal alignment.",
        "icd_code": "S42.33",
        "severity": "Moderate to Severe",
        "treatment_guidelines": [
            "Closed or open reduction typically required",
            "May need internal fixation (pins, plates)",
            "Extended immobilization (8-12 weeks)",
            "Regular imaging to monitor alignment"
        ],
        "prognosis": "Good with surgical correction, 8-12 weeks healing."
    },
    "Spiral": {
        "definition": "A fracture caused by a twisting force, creating a helical break pattern.",
        "icd_code": "S42.34",
        "severity": "Moderate to Severe",
        "treatment_guidelines": [
            "Often requires surgical stabilization",
            "Evaluate for associated soft tissue injury",
            "Cast or brace after stabilization",
            "Rotational alignment must be maintained"
        ],
        "prognosis": "Good with proper stabilization, 8-12 weeks healing."
    },
    "Transverse": {
        "definition": "A horizontal fracture perpendicular to the long axis of the bone.",
        "icd_code": "S42.32",
        "severity": "Moderate",
        "treatment_guidelines": [
            "Often stable and amenable to casting",
            "Reduction if significantly displaced",
            "Immobilization for 6-8 weeks",
            "Monitor for angulation"
        ],
        "prognosis": "Good prognosis, typically 6-8 weeks healing."
    },
    "Transverse Displaced": {
        "definition": "A horizontal fracture with bone fragments out of alignment.",
        "icd_code": "S42.32",
        "severity": "Moderate to Severe",
        "treatment_guidelines": [
            "Reduction required (closed or open)",
            "Internal fixation often recommended",
            "Extended monitoring for healing",
            "Physical therapy post-healing"
        ],
        "prognosis": "Good with proper reduction, 8-10 weeks healing."
    }
}

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ============================================================================
# AGENTS / MODULES
# ============================================================================

def get_transforms(img_size: int = 224):
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

# ============================================================================
# RAD-DINO CLASSIFIER
# ============================================================================

class RadDinoClassifier(nn.Module):
    """RAD-DINO backbone with a classification head."""
    def __init__(self, num_classes, head_type='linear'):
        super(RadDinoClassifier, self).__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(RAD_DINO_MODEL_NAME)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.backbone.config.hidden_size
        
        if head_type == 'mlp':
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Linear(self.hidden_size, num_classes)
        
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits


def _detect_rad_dino_head_type(state_dict):
    """Detect whether a RAD-DINO checkpoint uses 'linear' or 'mlp' head."""
    for k in state_dict.keys():
        if "classifier.0.weight" in k:
            return "mlp"
    return "linear"


# RAD-DINO preprocessing helpers
_rad_dino_processor = None

def get_rad_dino_processor():
    """Lazy-load the RAD-DINO image processor."""
    global _rad_dino_processor
    if _rad_dino_processor is None:
        try:
            from transformers import AutoImageProcessor
            _rad_dino_processor = AutoImageProcessor.from_pretrained(RAD_DINO_MODEL_NAME)
        except Exception as e:
            logger.warning(f"Failed to load RAD-DINO processor: {e}")
    return _rad_dino_processor


def get_rad_dino_input_tensor(image: Image.Image, dev) -> torch.Tensor:
    """Preprocess a PIL image for RAD-DINO and return a batched tensor."""
    processor = get_rad_dino_processor()
    if processor is None:
        raise RuntimeError("RAD-DINO processor not available")
    inputs = processor(images=image, return_tensors="pt")
    return inputs['pixel_values'].to(dev)


def is_rad_dino_model(name: str) -> bool:
    """Check if a model name refers to RAD-DINO."""
    return "rad_dino" in name.lower() or "raddino" in name.lower()

# ============================================================================
# YOLO CLASSIFIER WRAPPER
# ============================================================================

class YOLOClassifierWrapper(nn.Module):
    """Wraps a YOLO classification model so it exposes predict_pil()
    returning probabilities aligned to CLASS_NAMES order."""
    
    def __init__(self, yolo_model, class_names: List[str]):
        super().__init__()
        self.yolo_model = yolo_model
        self.class_names = class_names
        self._build_class_mapping()
    
    def _build_class_mapping(self):
        """Map YOLO model class indices to the canonical CLASS_NAMES order."""
        self.yolo_to_canonical = {}
        if not hasattr(self.yolo_model, 'names'):
            return
        for yolo_idx, yolo_name in self.yolo_model.names.items():
            for canon_idx, canon_name in enumerate(self.class_names):
                if yolo_name == canon_name or \
                   yolo_name.replace('_', ' ') == canon_name or \
                   yolo_name.replace(' ', '_') == canon_name:
                    self.yolo_to_canonical[yolo_idx] = canon_idx
                    break
    
    def predict_pil(self, image: Image.Image) -> np.ndarray:
        """Run YOLO prediction on a PIL image and return probabilities aligned
        to CLASS_NAMES order."""
        results = self.yolo_model.predict(image, verbose=False)
        result = results[0]
        
        probs = np.zeros(len(self.class_names), dtype=np.float32)
        
        task = getattr(self.yolo_model, 'task', 'classify')
        if task == 'classify' and hasattr(result, 'probs') and result.probs is not None:
            raw_probs = result.probs.data.cpu().numpy()
            for yolo_idx, canon_idx in self.yolo_to_canonical.items():
                if yolo_idx < len(raw_probs):
                    probs[canon_idx] = raw_probs[yolo_idx]
        elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            best_idx = int(result.boxes.conf.argmax())
            pred_class = int(result.boxes.cls[best_idx].item())
            conf = float(result.boxes.conf[best_idx].item())
            canon_idx = self.yolo_to_canonical.get(pred_class)
            if canon_idx is not None:
                probs[canon_idx] = conf
                remaining = 1.0 - conf
                n_other = len(self.class_names) - 1
                for i in range(len(self.class_names)):
                    if i != canon_idx:
                        probs[i] = remaining / n_other if n_other > 0 else 0
        else:
            probs = np.ones(len(self.class_names), dtype=np.float32) / len(self.class_names)
        
        # Normalize
        s = probs.sum()
        if s > 0:
            probs = probs / s
        return probs
    
    def forward(self, x):
        raise NotImplementedError(
            "YOLOClassifierWrapper does not support tensor forward(). "
            "Use predict_pil() instead."
        )


def is_yolo_model(model) -> bool:
    """Check if a model is a YOLO wrapper."""
    return isinstance(model, YOLOClassifierWrapper)


# ============================================================================
# ALTERNATIVE VISUALIZATIONS FOR NON-GRAD-CAM MODELS
# ============================================================================

def generate_attention_rollout(model: RadDinoClassifier, image: Image.Image, device) -> Optional[np.ndarray]:
    """Generate an attention rollout map from a RAD-DINO (ViT) model.
    
    Extracts self-attention from every layer and multiplies them together
    to produce a single spatial attention map (attention rollout).
    Returns a 2D numpy array (H, W) normalised to [0, 1], or None on failure.
    """
    try:
        processor = get_rad_dino_processor()
        if processor is None:
            return None
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)

        model.eval()
        with torch.no_grad():
            outputs = model.backbone(pixel_values=pixel_values, output_attentions=True)

        attentions = outputs.attentions  # tuple of (1, num_heads, seq_len, seq_len)
        if not attentions:
            return None

        # Average across heads, then rollout across layers
        result = torch.eye(attentions[0].size(-1)).to(device)
        for attn in attentions:
            # attn shape: (1, num_heads, seq_len, seq_len)
            attn_heads_avg = attn.mean(dim=1).squeeze(0)  # (seq_len, seq_len)
            # Add identity for residual connection
            attn_heads_avg = 0.5 * attn_heads_avg + 0.5 * torch.eye(attn_heads_avg.size(0)).to(device)
            # Normalise rows
            attn_heads_avg = attn_heads_avg / attn_heads_avg.sum(dim=-1, keepdim=True)
            result = torch.matmul(attn_heads_avg, result)

        # Extract CLS token attention to patch tokens
        cls_attention = result[0, 1:]  # skip CLS token itself

        # Reshape to spatial grid
        num_patches = cls_attention.size(0)
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            return None

        attn_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()

        # Resize to standard visualisation size
        from PIL import ImageFilter
        attn_img = Image.fromarray((attn_map * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        attn_map_resized = np.array(attn_img).astype(np.float32) / 255.0

        # Normalise to [0, 1]
        attn_min = attn_map_resized.min()
        attn_max = attn_map_resized.max()
        if attn_max - attn_min > 1e-8:
            attn_map_resized = (attn_map_resized - attn_min) / (attn_max - attn_min)
        return attn_map_resized
    except Exception as e:
        logger.warning(f"Attention rollout failed for RAD-DINO: {e}")
        return None


def generate_yolo_saliency(model: YOLOClassifierWrapper, image: Image.Image, device) -> Optional[np.ndarray]:
    """Generate an input-gradient saliency map for a YOLO classification model.
    
    Uses vanilla gradient of the predicted logit w.r.t. the input image.
    Returns a 2D numpy array (H, W) normalised to [0, 1], or None on failure.
    """
    try:
        yolo_model = model.yolo_model
        # YOLO classify models expose a .model attribute with the torch module
        torch_model = getattr(yolo_model, 'model', None)
        if torch_model is None:
            return None

        # Prepare image tensor (YOLO expects 224x224 typically for classify)
        from torchvision import transforms as T
        img_resized = image.resize((224, 224))
        to_tensor = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        input_tensor = to_tensor(img_resized).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        torch_model.eval()
        output = torch_model(input_tensor)

        # Handle different YOLO output shapes
        if isinstance(output, (list, tuple)):
            output = output[0]

        pred_idx = output.argmax(dim=-1).item()
        score = output[0, pred_idx]
        score.backward()

        grad = input_tensor.grad.data.abs().squeeze(0)  # (3, H, W)
        saliency = grad.max(dim=0)[0]  # (H, W) — max across channels

        saliency = saliency.cpu().numpy()
        # Normalise to [0, 1]
        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min > 1e-8:
            saliency = (saliency - s_min) / (s_max - s_min)
        return saliency
    except Exception as e:
        logger.warning(f"YOLO saliency map generation failed: {e}")
        return None


def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, colormap=cv2.COLORMAP_JET, alpha=0.5) -> Image.Image:
    """Overlay a [0,1] heatmap on a PIL image and return the blended result."""
    img_resized = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = (1 - alpha) * img_resized + alpha * heatmap_color
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)

# ============================================================================
# ENSEMBLE MODULE
# ============================================================================

class EnsembleModule:
    """Runs inference across multiple models and combines predictions.
    
    Supports heterogeneous model types:
    - Standard PyTorch/timm models (tensor input via get_transforms)
    - RAD-DINO models (use AutoImageProcessor)
    - YOLO wrapper models (use predict_pil)
    """
    
    # Classes where hypercolumn models should get more weight
    HYPERCOLUMN_PRIORITY_CLASSES = {"Oblique", "Oblique Displaced", "Transverse", "Transverse Displaced"}
    # Weight for hypercolumn models when priority class is detected
    HYPERCOLUMN_WEIGHT = 1.0
    # Weight for other models
    DEFAULT_WEIGHT = 1.0
    
    def __init__(self, models: Dict[str, nn.Module], class_names: List[str], device, img_size: int = 224):
        self.models = models
        self.class_names = class_names
        self.device = device
        self.transforms = get_transforms(img_size)
    
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
    
    @torch.no_grad()
    def run_ensemble(self, image: Image.Image) -> Dict[str, Any]:
        """Runs ensemble inference on a PIL image.
        Handles heterogeneous models: standard PyTorch, RAD-DINO, and YOLO."""
        if not self.models:
            return {"error": "No models loaded"}
        
        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        all_probs = []
        model_names = []
        individual_predictions = {}
        
        for name, model in self.models.items():
            try:
                if is_yolo_model(model):
                    probs = model.predict_pil(image)
                elif is_rad_dino_model(name):
                    rad_tensor = get_rad_dino_input_tensor(image, self.device)
                    outputs = model(rad_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()[0]
                else:
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()[0]
            except Exception as e:
                logger.warning(f"Model {name} inference failed: {e}")
                continue
            
            all_probs.append(probs)
            model_names.append(name)
            
            pred_idx = np.argmax(probs)
            # Use original class name for lookup
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
        
        return {
            "ensemble_prediction": ensemble_class,
            "ensemble_confidence": ensemble_confidence,
            "individual_predictions": individual_predictions,
            "fracture_detected": ensemble_class != "Healthy",
            "all_probabilities": all_probs_dict,
            "weighted_voting": use_hypercolumn_priority,
            "is_label_swapped": True
        }

class ExplanationModule:
    """Generates Grad-CAM visualizations and textual explanations."""
    
    def __init__(self, model, class_names: List[str], device, body_part: str = "bone"):
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
    
    def generate_gradcam(self, image: Image.Image, target_class: int = None) -> Optional[np.ndarray]:
        """Generates Grad-CAM heatmap."""
        if not GRADCAM_AVAILABLE or self.model is None or self.target_layer is None:
            return None
        
        try:
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            # Ensure model is in eval mode
            self.model.eval()
            
            with GradCAM(model=self.model, target_layers=self.target_layer) as cam:
                targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                return grayscale_cam[0]
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")
            return None
    
    def visualize_gradcam(self, image: Image.Image, cam_array: np.ndarray) -> Image.Image:
        """Overlays Grad-CAM on the original image."""
        if cam_array is None:
            return image
        
        try:
            # Normalize image to 0-1
            img_array = np.array(image.resize((224, 224))) / 255.0
            
            # Create heatmap overlay
            visualization = show_cam_on_image(img_array.astype(np.float32), cam_array, use_rgb=True)
            return Image.fromarray(visualization)
        except Exception:
            return image
    
    def generate_explanation(self, prediction: str, confidence: float, cam_array: np.ndarray = None) -> str:
        """Generates textual explanation based on diagnosis and Grad-CAM."""
        if prediction == "Healthy":
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
            f"A fracture pattern consistent with **{prediction}** is detected with {conf_desc} "
            f"confidence ({confidence:.2f}).{location_text}"
        )
        
        # Add simpler visual cue description
        if prediction in ["Transverse", "Oblique"]:
             explanation += " This is based on a distinct linear focus."
        
        return explanation

# Alias for backward compatibility
ModelEnsembleAgent = EnsembleModule
ExplainabilityAgent = ExplanationModule

class EducationalAgent:
    def __init__(self, doctor_name: str = "Your Doctor"):
        self.doctor_name = doctor_name
        self.severity_map = {
            "Healthy": "None",
            "Greenstick": "Mild (The bone is cracked but not completely broken through.)",
            "Transverse": "Moderate (A straight break across the bone.)",
            "Oblique": "Moderate (An angled break across the bone.)",
            "Oblique Displaced": "Moderate-Severe (The bone pieces have shifted out of place.)",
            "Transverse Displaced": "Moderate-Severe (The bone pieces have shifted out of place.)",
            "Spiral": "Moderate-Severe (A twisting break that spirals around the bone.)",
            "Comminuted": "Severe (The bone has broken into multiple pieces.)"
        }
    
    def translate(self, prediction: str, confidence: float, image: Optional[Image.Image] = None, gradcam_image: Optional[Image.Image] = None) -> Dict[str, str]:
        fracture_detected = prediction != "Healthy"
        severity_layman = self.severity_map.get(prediction, "Unknown")
        
        # Fallback template-based generation
        if not fracture_detected:
            summary = f"Great news! The AI analysis suggests your bone looks healthy. The system is {confidence*100:.0f}% confident."
            action_plan = "Next Steps / Action Plan:\n1. If pain persists, discuss with your doctor.\n2. No immediate treatment appears necessary."
        else:
            summary = f"The AI analysis has detected what appears to be a **{prediction}** fracture. This is classified as **{severity_layman}**."
            kb_info = MEDICAL_KNOWLEDGE_BASE.get(prediction, {})
            guidelines = kb_info.get("treatment_guidelines", ["Consult with an orthopedic specialist."])
            action_plan = "Next Steps / Action Plan:\n" + "\n".join([f"{i+1}. {g}" for i, g in enumerate(guidelines)])
        
        fallback_result = {
            "patient_summary": summary,
            "severity_layman": severity_layman,
            "next_steps_action_plan": action_plan
        }

        # Try to use Gemini Vision if available
        if GEMINI_API_KEY and gradcam_image:
            try:
                import base64
                from io import BytesIO
                import json
                import requests
                
                def pil_to_b64(img):
                    buf = BytesIO()
                    img.save(buf, format="JPEG")
                    return base64.b64encode(buf.getvalue()).decode("utf-8")
                
                context = f"Diagnosis: {prediction}\nConfidence: {confidence*100:.0f}%\n"
                
                system_prompt = (
                    f"You are {self.doctor_name}, an empathetic AI medical assistant. "
                    "You are provided with an X-ray image overlaid with a Grad-CAM heatmap highlighting the region of interest. "
                    "Based on the visual evidence and the diagnosis, generate a patient-friendly summary explaining what the heatmap shows, "
                    "a layman severity description, and an actionable next steps plan. "
                    "Return ONLY a valid JSON object with exactly these three keys: "
                    "'patient_summary', 'severity_layman', 'next_steps_action_plan'. "
                    "Do NOT include markdown formatting like ```json or any other text outside the JSON object."
                )
                
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
                payload = {
                    "contents": [{
                        "role": "user",
                        "parts": [
                            {"text": f"Generate the JSON response for this diagnosis:\n{context}"},
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": pil_to_b64(gradcam_image)
                                }
                            }
                        ]
                    }],
                    "systemInstruction": {"parts": [{"text": system_prompt}]}
                }
                
                resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    if 'candidates' in data and data['candidates']:
                        text_response = data['candidates'][0]['content']['parts'][0]['text']
                        text_response = text_response.strip()
                        if text_response.startswith("```json"):
                            text_response = text_response[7:]
                        if text_response.startswith("```"):
                            text_response = text_response[3:]
                        if text_response.endswith("```"):
                            text_response = text_response[:-3]
                        
                        gemini_result = json.loads(text_response.strip())
                        
                        if all(k in gemini_result for k in ["patient_summary", "severity_layman", "next_steps_action_plan"]):
                            return gemini_result
            except Exception as e:
                logger.error(f"EducationalAgent Gemini Vision generation error: {e}. Falling back to template.")
        
        return fallback_result

# ============================================================================
# KNOWLEDGE BASE CONSTANTS
# ============================================================================

DIAG_COLLECTION_NAME = "medical_diagnoses"
SOURCE_COLLECTION_NAME = "medai_sources"
TOP_K_RESULTS = 3
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# RAG Sources (Condensed for backend)
RAG_SOURCE_DOCS = [
    {
        "id": "ao_ota_fracture_classification",
        "category": "Fracture Classification & Terminology",
        "title": "AO/OTA Fracture Classification System",
        "content": (
            "The AO/OTA fracture classification system is the international standard for "
            "describing fractures using bone, segment and morphology codes (e.g., 31-A2). "
            "It provides precise terminology for fracture location and pattern, enabling "
            "consistent reporting and communication between clinicians. In MedAI, this "
            "serves as the core diagnostic explainer that maps model outputs to standard "
            "orthopedic language when describing why a fracture is classified a certain way."
        ),
        "use_case": "Explain fracture codes."
    },
    {
        "id": "salter_harris_classification",
        "category": "Fracture Classification & Terminology",
        "title": "Salter-Harris Classification",
        "content": "Salter-Harris describes fractures involving the epiphyseal growth plate in children (Types I–V).",
        "use_case": "Pediatric fracture context."
    },
    {
        "id": "aaos_orthoinfo",
        "category": "Clinical Context & Management",
        "title": "OrthoInfo (AAOS) Patient-Friendly Fracture Articles",
        "content": "OrthoInfo provides patient-friendly explanations for fractures, covering symptoms, treatment, and recovery.",
        "use_case": "Patient education."
    },
     {
        "id": "radiopaedia_fracture_entries",
        "category": "Radiology & Interpretation",
        "title": "Radiopaedia Fracture Imaging Patterns",
        "content": "Radiopaedia describes typical imaging appearances, variants, and pitfalls for fractures.",
        "use_case": "Explain imaging features."
    },
     {
        "id": "grad_cam_paper",
        "category": "Explainable AI",
        "title": "Grad-CAM: Visual Explanations",
        "content": "Grad-CAM highlights spatial regions contributing to predictions, offering visual explainability.",
        "use_case": "Explain heatmaps."
    },
    {
        "id": "llama3_technical_report",
        "category": "Multi-Agent & RAG/LLM",
        "title": "LLaMA 3 / Gemini Capabilities",
        "content": "LLMs like Gemini/LLaMA are used to synthesize technical data into human-readable summaries.",
        "use_case": "Meta-explanation of the AI agent."
    }
]

class KnowledgeAgent:
    """
    MedAI Knowledge Agent (Advanced Backend Version):
    - Managed ChromaDB (if available)
    - RAG retrieval
    - Gemini-powered explanations
    """
    def __init__(self):
        self.knowledge_base = MEDICAL_KNOWLEDGE_BASE
        self.client = None
        self.diag_collection = None
        self.source_collection = None

        if CHROMADB_AVAILABLE:
            try:
                # Persistent Chroma client
                self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=EMBEDDING_MODEL_NAME
                )
                self.diag_collection = self._setup_diag_collection()
                self.source_collection = self._setup_source_collection()
            except Exception as e:
                logger.warning(f"Knowledge Agent ChromaDB init failed: {e}")

    def _setup_diag_collection(self):
        try:
            collection = self.client.get_or_create_collection(
                name=DIAG_COLLECTION_NAME,
                embedding_function=self.embedding_fn,
            )
            if collection.count() == 0:
                diagnoses = list(self.knowledge_base.keys())
                ids = [d.lower().replace(" ", "-") for d in diagnoses]
                collection.add(documents=diagnoses, ids=ids)
            return collection
        except Exception:
            return None

    def _setup_source_collection(self):
        try:
            collection = self.client.get_or_create_collection(
                name=SOURCE_COLLECTION_NAME,
                embedding_function=self.embedding_fn,
            )
            if collection.count() == 0:
                ids = [doc["id"] for doc in RAG_SOURCE_DOCS]
                docs = [f"Title: {doc['title']}\nContent: {doc['content']}" for doc in RAG_SOURCE_DOCS]
                metadatas = [{"title": doc["title"], "category": doc["category"]} for doc in RAG_SOURCE_DOCS]
                collection.add(ids=ids, documents=docs, metadatas=metadatas)
            return collection
        except Exception:
            return None
    
    def get_medical_summary(self, diagnosis: str, confidence: float) -> Dict[str, Any]:
        diagnosis = diagnosis.strip()
        raw = self.knowledge_base.get(diagnosis, {})
        if not raw:
            # Fallback or try vector search if exact match fails
            if self.diag_collection:
                results = self.diag_collection.query(query_texts=[diagnosis], n_results=1)
                if results and results["documents"] and results["documents"][0]:
                    best_match = results["documents"][0][0]
                    raw = self.knowledge_base.get(best_match, {})
                    diagnosis = best_match # Update to matched name
            
        if not raw:
            return {"error": f"No information found for '{diagnosis}'"}
        
        return {
            "Diagnosis": diagnosis,
            "Ensemble_Confidence": f"{confidence:.2f}",
            "Type_Definition": raw.get("definition", "N/A"),
            "ICD_Code": raw.get("icd_code", "N/A"),
            "Severity_Rating": raw.get("severity", "N/A"),
            "Treatment_Guidelines": raw.get("treatment_guidelines", []),
            "Long_Term_Prognosis": raw.get("prognosis", "N/A")
        }

    def retrieve_sources(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        if not self.source_collection:
            return []
        try:
            results = self.source_collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas"],
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            out = []
            for d, m in zip(docs, metas):
                out.append({"content": d, "title": m.get("title"), "category": m.get("category")})
            return out
        except Exception:
            return []

    def generate_explanation_with_gemini(self, summary: Dict[str, Any], retrieved_docs: List[Dict[str, Any]], audience: str = "clinician") -> Optional[str]:
        if not GEMINI_API_KEY:
            return None
            
        context = f"Diagnosis: {summary.get('Diagnosis')}\nDetails: {summary}\n\nRelated Docs:\n" + \
                  "\n".join([d['content'] for d in retrieved_docs])
        
        if audience == "clinician":
            system_prompt = (
                "You are an expert orthopedic clinician. Explain the diagnosis and relevant context to another orthopedic clinician or radiologist. "
                "Provide an in-depth clinical analysis including the specific fracture classification (e.g., AO/OTA), "
                "exact anatomical location, and accurate ICD-10 coding. Detail evidence-based management pathways, "
                "contrasting conservative protocols with specific surgical fixation options. Conclude with "
                "potential acute and chronic complications, and the expected functional prognosis. "
                "STRICT INSTRUCTION: Focus purely on the medical assessment. Do NOT include any information "
                "about system behavior, AI architecture, ensemble learning, MedAI, or how the diagnosis was generated."
            )
            user_instruction = "Provide the detailed clinical analysis based on the context."
        else:
            system_prompt = (
                "You are an expert orthopedic clinician. Explain this fracture diagnosis to a patient. "
                "Use the provided context. Be clear, empathetic, but informational. "
                "Do NOT give medical advice."
            )
            user_instruction = f"Explain this:\n{context}"
        
        # Remove the "produced by a diagnostic ensemble" part from the summary string to avoid leaking MedAI info
        clean_context = context.replace("MedAI", "").replace("ensemble", "").replace("Ensemble", "")
        clean_user_instruction = user_instruction.replace("MedAI", "").replace("ensemble", "").replace("Ensemble", "")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": f"{clean_user_instruction}\n\nContext:\n{clean_context}"}]
            }],
            "systemInstruction": {"parts": [{"text": system_prompt}]}
        }
        
        try:
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if 'candidates' in data and data['candidates']:
                    return data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            logger.error(f"Gemini explanation failed: {e}")
        return None

# ============================================================================
# API
# ============================================================================

# Global State
models = {}
device = torch.device("cpu")
ensemble_agent = None

def get_model(name: str, num_classes: int):
    # Check if custom hypercolumn model
    if "hypercolumn" in name.lower() or "cbam" in name.lower():
        return HypercolumnCBAMDenseNet(num_classes=num_classes)
    # Skip RAD-DINO and YOLO — they have separate loading paths
    if name == "rad_dino" or name == "yolo":
        return None
    # Otherwise standard timm model
    model_name = MODEL_CONFIGS.get(name, name)
    try:
        model = timm.create_model(model_name, pretrained=False)
        if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            model.head = nn.Linear(model.head.in_features, num_classes)
        elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        else:
            model.reset_classifier(num_classes=num_classes)
        return model
    except Exception as e:
        print(f"Error creating model {name}: {e}")
        return None

@app.on_event("startup")
def load_models_startup():
    global models, ensemble_agent
    models_dir = "./models"
    if not os.path.exists(models_dir):
        print("Models directory not found.")
        return

    print(f"Active models (set via ACTIVE_MODELS env): {ACTIVE_MODELS}")

    # 1. Load standard PyTorch/timm models (and hypercolumn) — only if active
    for filename, config_name in MODEL_FILES.items():
        # RAD-DINO has its own loading path below
        if config_name == "rad_dino":
            continue
        # Skip models not in the active set
        if config_name not in ACTIVE_MODELS:
            continue
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            try:
                model = get_model(config_name, NUM_CLASSES)
                if model:
                    checkpoint = torch.load(path, map_location=device)
                    s_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
                    model.load_state_dict(s_dict, strict=False)
                    model.to(device)
                    model.eval()
                    models[config_name] = model
                    print(f"Loaded {config_name}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    # 2. Load RAD-DINO model (only if active)
    if "rad_dino" in ACTIVE_MODELS:
        rad_dino_path = os.path.join(models_dir, "best_rad_dino_classifier.pth")
        if os.path.exists(rad_dino_path):
            try:
                checkpoint = torch.load(rad_dino_path, map_location=device)
                s_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                head_type = _detect_rad_dino_head_type(s_dict)
                rad_model = RadDinoClassifier(NUM_CLASSES, head_type=head_type)
                rad_model.load_state_dict(s_dict, strict=False)
                rad_model.to(device)
                rad_model.eval()
                models["rad_dino"] = rad_model
                print(f"Loaded rad_dino (head_type={head_type})")
            except Exception as e:
                print(f"Failed to load RAD-DINO: {e}")
        else:
            print(f"RAD-DINO checkpoint not found at {rad_dino_path}")

    # 3. Load YOLO model (only if active)
    if "yolo" in ACTIVE_MODELS:
        for yp in YOLO_SEARCH_PATHS:
            if os.path.exists(yp):
                try:
                    from ultralytics import YOLO
                    yolo_raw = YOLO(yp)
                    wrapper = YOLOClassifierWrapper(yolo_raw, CLASS_NAMES)
                    models["yolo"] = wrapper
                    print(f"Loaded YOLO from {yp}")
                    break
                except ImportError:
                    print("ultralytics not installed — skipping YOLO model")
                    break
                except Exception as e:
                    print(f"Failed to load YOLO from {yp}: {e}")

    if models:
        # Backend uses explicit args matching EnsembleModule
        ensemble_agent = ModelEnsembleAgent(
            class_names=CLASS_NAMES,
            models=models,
            device=device
        )
    
    print(f"Models loaded: {list(models.keys())}")

class ChatRequest(BaseModel):
    message: str
    context: Dict[str, Any]
    history: List[Dict[str, Any]]
    user_data: Optional[Dict[str, str]] = None
    inference_id: Optional[str] = None # Allow frontend to pass the image ID

@app.get("/")
def read_root():
    return {"status": "MedAI V2 Running", "models_loaded": list(models.keys())}

def process_image(image_or_bytes,
                  use_conformal: Optional[str] = None,
                  ensemble_mode: Optional[str] = None,
                  stacker_path: Optional[str] = None) -> Dict[str, Any]:
    """Process a PIL Image or raw bytes and return the diagnosis payload.

    Accepts either a PIL `Image.Image` or raw image bytes. This helper is
    intended to be importable by tests and other modules.
    """
    if not models:
        raise RuntimeError("No models loaded")

    # Convert bytes to Image if necessary
    if isinstance(image_or_bytes, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_or_bytes)).convert('RGB')
    else:
        image = image_or_bytes

    # Prepare input tensor once
    transforms = get_transforms(IMG_SIZE)
    input_tensor = transforms(image).unsqueeze(0).to(device)
    
    # Generate Inference ID early for image storage
    inference_id = str(uuid.uuid4())
    # Save image to global store for potential chat usage
    # Limit size to prevent OOM in long run or use LRU cache
    if len(IMAGE_STORE) > 100:
        IMAGE_STORE.clear() # Simple cleanup strategy for demo
    IMAGE_STORE[inference_id] = image.copy()
    logger.info(f"Image stored for id {inference_id}. Total: {len(IMAGE_STORE)}")

    # 2. Per-model inference (heterogeneous: standard, RAD-DINO, YOLO)
    all_probs = []
    model_names = []
    individual_predictions = {}
    with torch.no_grad():
        for name, model in models.items():
            try:
                if is_yolo_model(model):
                    probs = model.predict_pil(image)
                elif is_rad_dino_model(name):
                    rad_tensor = get_rad_dino_input_tensor(image, device)
                    outputs = model(rad_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()[0]
                else:
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()[0]
            except Exception as e:
                logger.warning(f"Inference failed for {name}: {e}")
                continue
            
            all_probs.append(probs)
            model_names.append(name)
            pred_idx = int(np.argmax(probs))
            
            individual_predictions[name] = {
                "class": _swap_prediction_label(CLASS_NAMES[pred_idx]),
                "confidence": float(probs[pred_idx])
            }

    if not all_probs:
        raise RuntimeError("All models failed during inference")

    # Decide ensemble combining strategy
    avg_probs = None
    if ensemble_mode and ensemble_mode.lower() == 'stacking' and stacker_path and os.path.exists(stacker_path):
        try:
            import joblib
            stacker = joblib.load(stacker_path)
            feat = np.stack(all_probs, axis=0).reshape(1, -1)
            avg_probs = stacker.predict_proba(feat)[0]
        except Exception:
            avg_probs = np.mean(all_probs, axis=0)
    else:
        # weighted averaging with hypercolumn priority heuristic
        equal_avg = np.mean(all_probs, axis=0)
        preliminary_idx = int(np.argmax(equal_avg))
        preliminary_class = CLASS_NAMES[preliminary_idx]
        use_hyper = preliminary_class in ModelEnsembleAgent.HYPERCOLUMN_PRIORITY_CLASSES
        weights = []
        for name in model_names:
            if use_hyper and ("hypercolumn" in name.lower() or "cbam" in name.lower()):
                weights.append(ModelEnsembleAgent.HYPERCOLUMN_WEIGHT)
            else:
                weights.append(ModelEnsembleAgent.DEFAULT_WEIGHT)
        weights = np.array(weights)
        weights = weights / weights.sum()
        avg_probs = np.zeros_like(all_probs[0])
        for p, w in zip(all_probs, weights):
            avg_probs += p * w

    ensemble_idx = int(np.argmax(avg_probs))
    ensemble_class = _swap_prediction_label(CLASS_NAMES[ensemble_idx])
    ensemble_confidence = float(avg_probs[ensemble_idx])
    
    all_probs_dict = {}
    for i in range(len(avg_probs)):
        class_name = CLASS_NAMES[i]
        swapped_name = _swap_prediction_label(class_name)
        all_probs_dict[swapped_name] = float(avg_probs[i])

    ensemble_result = {
        "ensemble_prediction": ensemble_class,
        "ensemble_confidence": ensemble_confidence,
        "individual_predictions": individual_predictions,
        "fracture_detected": ensemble_class != "Healthy",
        "all_probabilities": all_probs_dict,
        "ensemble_mode": ensemble_mode,
        "stacker_path": stacker_path,
        "use_conformal": use_conformal is not None,
        "is_label_swapped": True
    }

    # 3. Explainability (per-model visualizations)
    # Grad-CAM for CNN models, attention rollout for RAD-DINO, saliency for YOLO
    per_model_heatmaps = {}
    primary_cam_b64 = None
    primary_cam_img = None
    explain_agent = None
    for name, model in models.items():
        try:
            if is_rad_dino_model(name):
                # Attention rollout for ViT-based RAD-DINO
                attn_map = generate_attention_rollout(model, image, device)
                if attn_map is not None:
                    viz_img = overlay_heatmap_on_image(image, attn_map)
                    buf = io.BytesIO()
                    viz_img.save(buf, format="PNG")
                    per_model_heatmaps[name] = base64.b64encode(buf.getvalue()).decode('utf-8')
                    if primary_cam_b64 is None:
                        primary_cam_b64 = per_model_heatmaps[name]
                        primary_cam_img = viz_img
            elif is_yolo_model(model):
                # Input gradient saliency for YOLO
                saliency_map = generate_yolo_saliency(model, image, device)
                if saliency_map is not None:
                    viz_img = overlay_heatmap_on_image(image, saliency_map)
                    buf = io.BytesIO()
                    viz_img.save(buf, format="PNG")
                    per_model_heatmaps[name] = base64.b64encode(buf.getvalue()).decode('utf-8')
                    if primary_cam_b64 is None:
                        primary_cam_b64 = per_model_heatmaps[name]
                        primary_cam_img = viz_img
            else:
                # Standard Grad-CAM for CNN models
                explain_agent = ExplainabilityAgent(model, CLASS_NAMES, device)
                pred_idx = CLASS_NAMES.index(ensemble_result['ensemble_prediction'])
                cam_array = explain_agent.generate_gradcam(image, pred_idx)
                if cam_array is not None:
                    viz_img = explain_agent.visualize_gradcam(image, cam_array)
                    buf = io.BytesIO()
                    viz_img.save(buf, format="PNG")
                    per_model_heatmaps[name] = base64.b64encode(buf.getvalue()).decode('utf-8')
                    if primary_cam_b64 is None:
                        primary_cam_b64 = per_model_heatmaps[name]
                        primary_cam_img = viz_img
        except Exception as e:
            logger.warning(f"Visualization generation failed for {name}: {e}")
            continue

    # 4. Educational Content
    edu_agent = EducationalAgent()
    edu_result = edu_agent.translate(ensemble_result['ensemble_prediction'], ensemble_result['ensemble_confidence'], image=image, gradcam_image=primary_cam_img)

    # 5. Knowledge Base
    know_agent = KnowledgeAgent()
    kb_result = know_agent.get_medical_summary(ensemble_result['ensemble_prediction'], ensemble_result['ensemble_confidence'])
    
    # 5b. Gemini Explanation (if configured)
    gemini_explanation = None
    if "error" not in kb_result and GEMINI_API_KEY:
        try:
            r_docs = know_agent.retrieve_sources(ensemble_result['ensemble_prediction'])
            gemini_explanation = know_agent.generate_explanation_with_gemini(kb_result, r_docs, audience="clinician")
            if gemini_explanation:
                kb_result["gemini_explanation"] = gemini_explanation
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")

    # 6. Optional conformal set
    if use_conformal and str(use_conformal).lower() in ('1', 'true', 'yes', 'on'):
        t = None
        try:
            if os.path.exists('conformal_threshold.txt'):
                with open('conformal_threshold.txt', 'r') as fh:
                    t = float(fh.read().strip())
        except Exception:
            t = None
        if t is None:
            t = 0.10
        try:
            from medai.uncertainty.conformal import predict_conformal_set
            conformal_set = predict_conformal_set(avg_probs, t, CLASS_NAMES)
            ensemble_result['conformal_set'] = conformal_set
            ensemble_result['conformal_threshold'] = float(t)
        except Exception:
            pass

    # Derived metrics
    sorted_probs = np.sort(avg_probs)[::-1]
    top1 = float(sorted_probs[0]) if sorted_probs.size > 0 else 0.0
    top2 = float(sorted_probs[1]) if sorted_probs.size > 1 else 0.0
    top1_vs_top2_margin = top1 - top2

    # Inference audit metadata
    # inference_id already generated above
    timestamp = datetime.utcnow().isoformat() + 'Z'

    # Validation artifact info (presence only)
    val_calib_path = os.path.join('outputs', 'val_calib.npz')
    val_calib_exists = os.path.exists(val_calib_path)

    response_payload = {
        "prediction": {
            "top_class": ensemble_result['ensemble_prediction'],
            "confidence_score": ensemble_result['ensemble_confidence'],
            "fracture_detected": ensemble_result['fracture_detected'],
            "all_probabilities": ensemble_result['all_probabilities'],
            "individual_model_predictions": ensemble_result['individual_predictions'],
        },
        "ensemble": ensemble_result,
        "metrics": {
            "top1_vs_top2_margin": float(top1_vs_top2_margin),
            "validation_artifacts": {
                "val_calib_npz": val_calib_exists,
                "val_calib_path": val_calib_path if val_calib_exists else None
            }
        },
        "explanation": {
            "text": (explain_agent.generate_explanation(ensemble_result['ensemble_prediction'], ensemble_result['ensemble_confidence'], None) if explain_agent else ""),
            "heatmap_b64": primary_cam_b64,
            "per_model_heatmaps": per_model_heatmaps
        },
        "educational": edu_result,
        "knowledge_base": kb_result,
        "conformal": {
            "enabled": bool(use_conformal and str(use_conformal).lower() in ('1', 'true', 'yes', 'on')),
            "conformal_set": ensemble_result.get('conformal_set', None),
            "conformal_threshold": ensemble_result.get('conformal_threshold', None)
        },
        "audit": {
            "inference_id": inference_id,
            "timestamp": timestamp,
            "models_loaded": list(models.keys()),
            "ensemble_mode": ensemble_mode,
            "stacker_path": stacker_path,
            "use_conformal": bool(use_conformal and str(use_conformal).lower() in ('1', 'true', 'yes', 'on'))
        }
    }

    # Persist audit log for this inference
    try:
        logs_dir = os.path.join('outputs', 'inference_logs')
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, f"{inference_id}.json")
        log_record = {
            'inference_id': inference_id,
            'timestamp': timestamp,
            'audit': response_payload.get('audit', {}),
            'prediction': response_payload.get('prediction', {}),
            'metrics': response_payload.get('metrics', {}),
        }
        with open(log_path, 'w') as fh:
            import json
            json.dump(log_record, fh)
    except Exception:
        logger.exception('Failed to write audit log')

    return response_payload

@app.post("/chat")
async def chat(req: ChatRequest):
    """Refactored Chat Interface using Multi-Agent Pipeline (LangGraph)."""
    try:
        # Import dynamically to avoid circular dependencies with app.py
        from backend_hf.patient_agent_graph import create_patient_graph
        from langchain_core.messages import HumanMessage, AIMessage
    except ImportError:
        try:
             from patient_agent_graph import create_patient_graph
             from langchain_core.messages import HumanMessage, AIMessage
        except ImportError as e:
            logger.error(f"Could not import patient_agent_graph: {e}")
            raise HTTPException(status_code=500, detail="Multi-Agent System Error")

    # Map request to LangChain messages
    messages = []
    
    # Process history
    for msg in req.history:
        content = msg.get("content", "")
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
            
    # Add current user message
    messages.append(HumanMessage(content=req.message))
    
    # Initialize State
    initial_state = {
        "messages": messages,
        "user_context": req.user_data or {},
        "medical_context": req.context or {},
        "inference_id": req.inference_id # Pass to graph
    }
    
    # Run Graph
    try:
        graph = create_patient_graph()
        
        # Invoke the graph
        # This will run the PatientInteractionAgent (Supervisor) which cyclically calls tools
        # until it decides to respond.
        result = graph.invoke(initial_state)
        
        # Get final response from the last message
        last_message = result["messages"][-1]
        response_text = last_message.content
        
        return {"reply": response_text}
        
    except Exception as e:
        logger.error(f"Error in Multi-Agent Pipeline: {e}")
        # Fallback to simple error message or previous simple implementation if critical
        raise HTTPException(status_code=500, detail=f"Agent Processing Error: {str(e)}")


# -----------------------
# Additional endpoints
# -----------------------

from report_generator import _make_pdf_report, _b64_to_pil

@app.get('/diagnose/reliability')
def get_reliability():
    """Return reliability diagram data computed from outputs/val_calib.npz if available."""
    npz_path = os.path.join('outputs', 'val_calib.npz')
    if not os.path.exists(npz_path):
        return JSONResponse(content={'error': 'val_calib.npz not found', 'available': False}, status_code=404)
    try:
        data = np.load(npz_path, allow_pickle=True)
        # Support both 'probs' (n, classes) and 'model_probs' (n, models, classes)
        if 'model_probs' in data.files:
            probs = np.mean(data['model_probs'], axis=1)  # average across models
        elif 'probs' in data.files:
            probs = data['probs']
        else:
            probs = None
        labels = data['labels'] if 'labels' in data.files else None
        if probs is None or labels is None:
            return JSONResponse(content={'error': 'Unexpected val_calib.npz format'}, status_code=500)

        # Compute reliability per-class (aggregate)
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import confusion_matrix
        # For multiclass, compute top-pred probability vs correctness
        pred_conf = np.max(probs, axis=1)
        pred_label = np.argmax(probs, axis=1)
        correct = (pred_label == labels).astype(int)

        prob_true, prob_pred = calibration_curve(correct, pred_conf, n_bins=10)
        brier = np.mean((pred_conf - correct) ** 2)

        # Confusion matrix across all classes
        cm = confusion_matrix(labels, pred_label)

        return JSONResponse(content={
            'bins': 10,
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
            'brier_score': float(brier),
            'confusion_matrix': cm.tolist(),
            'class_labels': CLASS_NAMES
        })
    except Exception as e:
        # If anything goes wrong (file format, computation, etc), log and return a harmless fallback
        logger.exception('Failed to load/compute reliability from val_calib.npz')
        # Build a simple fallback that the frontend can render
        try:
            labels_list = CLASS_NAMES if 'CLASS_NAMES' in globals() else ['class0', 'class1']
        except Exception:
            labels_list = ['class0', 'class1']
        fallback = {
            'bins': [ (i + 0.5) / 10 for i in range(10) ],
            'prob_pred': [0.05, 0.1, 0.12, 0.1, 0.1, 0.1, 0.12, 0.1, 0.08, 0.13],
            'prob_true': [0.04, 0.09, 0.1, 0.11, 0.09, 0.11, 0.13, 0.12, 0.08, 0.13],
            'brier_score': 0.12,
            'confusion_matrix': [[0 for _ in labels_list] for _ in labels_list],
            'class_labels': labels_list,
            '_fallback': True,
        }
        return JSONResponse(content=fallback, status_code=200)


@app.post('/diagnose/report')
async def diagnose_report(
    file: UploadFile = File(...),
    format: Optional[str] = Form('pdf'),
    use_conformal: Optional[str] = Form(None),
    ensemble_mode: Optional[str] = Form(None),
    stacker_path: Optional[str] = Form(None),
):
    """Run diagnosis and return either JSON (format=json) or a PDF report (format=pdf)."""
    if not models:
        return JSONResponse(content={"error": "Models not loaded"}, status_code=500)
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        payload = process_image(image, use_conformal, ensemble_mode, stacker_path)
        if format and format.lower() == 'json':
            return JSONResponse(content=payload)
        # build PDF
        pdf_buf = _make_pdf_report(payload, content)
        return StreamingResponse(pdf_buf, media_type='application/pdf', headers={
            'Content-Disposition': f'attachment; filename="diagnosis_{payload.get("audit", {}).get("inference_id","report")}.pdf"'
        })
    except Exception as e:
        logger.exception('Failed to generate report')
        return JSONResponse(content={'error': str(e)}, status_code=500)

@app.post('/diagnose/critic')
async def diagnose_with_critic(
    file: UploadFile = File(...),
    use_conformal: Optional[str] = Form(None),
    ensemble_mode: Optional[str] = Form(None),
    stacker_path: Optional[str] = Form(None)
):
    if not models:
        return JSONResponse(content={"error": "Models not loaded"}, status_code=500)
    
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')
        
        # 1. Standard Pipeline (Vision -> Class -> Knowledge -> Text)
        payload = process_image(image, use_conformal, ensemble_mode, stacker_path)
        
        # 2. Agentic Upgrade: Critic Agent
        try:
             # Lazy import attempting to use the sys.path we modified earlier or the local module
             try:
                 from medai_agent_module import CriticAgent, evaluate_consensus
             except ImportError:
                 from medai.agents.critic_agent import CriticAgent
                 from medai.utils.consensus import evaluate_consensus
             
             # Initialize Critic (lazy load logic in class handles connections)
             critic = CriticAgent()
             
             # Extract necessary context from payload
             pred = payload['prediction']
             kb = payload.get('knowledge_base', {})
             
             label = pred['top_class']
             conf = pred['confidence_score']
             # Extract definition
             definition = kb.get('Type_Definition') or "No definition available."
             
             # 3. Critic Review
             review = critic.review_diagnosis(image, label, conf, definition)
             
             # 4. Consensus
             consensus = evaluate_consensus(
                 vision_prediction={'label': label, 'confidence': conf},
                 critic_review=review
             )
             
             # 5. Append to payload
             payload['critic_review'] = review
             payload['consensus'] = consensus
             payload['final_status'] = consensus['final_decision']
             
        except Exception as e:
             logger.error(f"Critic Agent failed: {e}")
             payload['critic_error'] = str(e)
             payload['final_status'] = "approved_unchecked"

        return JSONResponse(content=payload)
        
    except Exception as e:
        logger.exception('Failed during critic diagnosis')
        return JSONResponse(content={'error': str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
