"""
MedAI - Integrated Multi-Agent Fracture Detection System
=========================================================
A Streamlit application integrating all six agents:
1. DiagnosticAgent - Single model inference
2. ModelEnsembleAgent - Cross-validation ensemble
3. ExplainabilityAgent - Grad-CAM explanations
4. EducationalAgent - Patient-friendly translations
5. KnowledgeAgent - RAG-based knowledge retrieval
6. PatientInteractionAgent - LLM-powered chat
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file immediately
load_dotenv()

# Configure Logging to Console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add project root and src to sys.path to allow imports from src.medai...
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir) # src
project_root = os.path.dirname(src_dir) # root

if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import io
import tempfile
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm
try:
    from uncertainty.conformal import predict_conformal_set
except Exception:
    from medai.uncertainty.conformal import predict_conformal_set

# Import Agentic Components
try:
    # Try importing with fully qualified name if root in path
    try:
        from src.medai.agents.critic_agent import CriticAgent
        from src.medai.utils.consensus import evaluate_consensus
    except ImportError:
        # Fallback to medai... if src in path
        from medai.agents.critic_agent import CriticAgent
        from medai.utils.consensus import evaluate_consensus
except ImportError as e:
    logger.warning(f"Could not import Agentic Components: {e}")
    CriticAgent = None
    evaluate_consensus = None

# Attempt to import optional dependencies
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# ============================================================================
# CUSTOM MODEL ARCHITECTURES
# ============================================================================

class _DenseLayer(nn.Module):
    """Single dense layer as used in DenseNet."""
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
    """Dense block containing multiple dense layers."""
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
    """Channel attention module for CBAM with shared MLP."""
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
    """Spatial attention module for CBAM."""
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
    """Convolutional Block Attention Module."""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class HypercolumnCBAMDenseNet(nn.Module):
    """
    Custom DenseNet169 with Hypercolumn fusion and CBAM attention.
    
    Architecture matches training checkpoint exactly:
    - features.*     : Full DenseNet169 backbone
    - init_conv.*    : Separate Conv2d(3,64) + BN (NOT a reference to features)
    - db1-4, t1-3    : References to features.denseblock*, features.transition*
    - norm_final     : Reference to features.norm5
    - Hypercolumn fusion upsamples to final feature map size (7x7)
    """
    def __init__(self, num_classes=8, growth_rate=32, bn_size=4, drop_rate=0.0):
        super(HypercolumnCBAMDenseNet, self).__init__()
        import torchvision.models as models
        
        # Use torchvision's DenseNet169 as backbone
        densenet = models.densenet169(weights=None)
        self.features = densenet.features
        
        # init_conv is SEPARATE from features (has its own trained weights)
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64)
        )
        
        # Dense blocks are REFERENCES to features (share weights)
        self.db1 = self.features.denseblock1
        self.db2 = self.features.denseblock2
        self.db3 = self.features.denseblock3
        self.db4 = self.features.denseblock4
        
        # Transitions are REFERENCES to features (share weights, include AvgPool2d)
        self.t1 = self.features.transition1
        self.t2 = self.features.transition2
        self.t3 = self.features.transition3
        
        # norm_final is a REFERENCE to features.norm5 (share weights)
        self.norm_final = self.features.norm5
        
        # Hypercolumn fusion: 1664 + 640 + 256 + 128 = 2688 -> 1024
        self.fusion_conv = nn.Conv2d(2688, 1024, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(1024)
        
        # CBAM attention
        self.cbam = CBAM(1024)
        
        # Classifier with dropout (matches training)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        # Use init_conv (separate trained weights)
        x = self.init_conv(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)  # 224->112->56
        
        # Dense block 1 -> Transition 1
        x = self.db1(x)       # 64->256 channels, 56x56
        t1_out = self.t1(x)   # 256->128 channels, 56->28 (includes AvgPool2d)
        
        # Dense block 2 -> Transition 2
        x = self.db2(t1_out)  # 128->512 channels, 28x28
        t2_out = self.t2(x)   # 512->256 channels, 28->14 (includes AvgPool2d)
        
        # Dense block 3 -> Transition 3
        x = self.db3(t2_out)  # 256->1280 channels, 14x14
        t3_out = self.t3(x)   # 1280->640 channels, 14->7 (includes AvgPool2d)
        
        # Dense block 4 -> Final norm
        x = self.db4(t3_out)  # 640->1664 channels, 7x7
        x_final = self.norm_final(x)  # 1664 channels, 7x7
        
        # Hypercolumn fusion - upsample all to match x_final size (7x7)
        target_size = x_final.shape[2:]
        t1_resized = nn.functional.interpolate(t1_out, size=target_size, mode='bilinear', align_corners=False)
        t2_resized = nn.functional.interpolate(t2_out, size=target_size, mode='bilinear', align_corners=False)
        t3_resized = nn.functional.interpolate(t3_out, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate: 1664 + 640 + 256 + 128 = 2688 (order matters!)
        hypercolumn = torch.cat([x_final, t3_resized, t2_resized, t1_resized], dim=1)
        
        # Fusion: 2688 -> 1024
        x = self.fusion_conv(hypercolumn)
        x = self.bn_fusion(x)
        x = nn.functional.relu(x)
        
        # Apply CBAM attention
        x = self.cbam(x)
        
        # Global average pooling
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)  # Flatten to [batch, 1024]
        
        # Classify (through dropout + linear)
        x = self.classifier(x)
        
        return x


# ============================================================================
# CONFIGURATION
# ============================================================================

CLASS_NAMES = [
    "Comminuted", "Greenstick", "Healthy", "Oblique", 
    "Oblique Displaced", "Spiral", "Transverse", "Transverse Displaced"
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

# Model configurations (must match training architectures)
# All hypercolumn variants use the custom HypercolumnCBAMDenseNet class
MODEL_CONFIGS = {
    # Standard timm models
    "swin": "swin_small_patch4_window7_224",
    "densenet169": "densenet169",
    "efficientnetv2": "efficientnet_b0",
    "mobilenetv2": "mobilenetv2_100",
    "maxvit": "maxvit_tiny_tf_224",
    # Hypercolumn variants (all use custom HypercolumnCBAMDenseNet architecture)
    "hypercolumn_cbam_densenet169": "custom",
    "hypercolumn_cbam_densenet169_focal": "custom",
    "hypercolumn_densenet169": "custom",
    "hypercolumn_densenet169_old": "custom",
    # RAD-DINO and YOLO (special loading paths)
    "rad_dino": "rad_dino",
    "yolo": "yolo",
}

# RAD-DINO constants
RAD_DINO_MODEL_NAME = "microsoft/rad-dino"

# YOLO model search paths
YOLO_SEARCH_PATHS = [
    "outputs/yolo_cls_finetune/yolo_cls_ft/weights/best.pt",
    "models/yolo_best.pt",
    "models/best.pt",
    "outputs/weights/best.pt",
    "weights/best.pt",
]

# OpenRouter configuration
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
# Access streamlit secrets defensively so importing app in non-streamlit contexts doesn't fail
try:
    OPENROUTER_API_KEY = st.secrets.get("openrouter_api_key", os.environ.get("OPENROUTER_API_KEY", ""))
    OPENROUTER_MODEL = st.secrets.get("openrouter_model", os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free"))
except Exception:
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free")

# ChromaDB configuration
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ============================================================================
# KNOWLEDGE AGENT CONFIGURATION
# ============================================================================

DIAG_COLLECTION_NAME = "medical_diagnoses"
SOURCE_COLLECTION_NAME = "medai_sources"
TOP_K_RESULTS = 3

# Gemini API Config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

# ============================================================================
# MEDICAL KNOWLEDGE BASE
# ============================================================================

MEDICAL_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "Comminuted": {
        "definition": "A fracture where the bone is broken into three or more fragments.",
        "icd_code": "S52.5",
        "severity": "High",
        "treatment_guidelines": [
            "Usually requires surgical intervention (open reduction internal fixation / ORIF).",
            "Long immobilization (8-12 weeks).",
            "Requires structured physical therapy after immobilization."
        ],
        "prognosis_notes": "Risk of non-union and malunion is higher. Full recovery may take 6+ months."
    },
    "Oblique Displaced": {
        "definition": "A diagonal break where the bone fragments are separated and misaligned.",
        "icd_code": "S52.9",
        "severity": "Medium-High",
        "treatment_guidelines": [
            "Requires reduction (closed or open) to restore alignment.",
            "Often treated with casting; unstable fractures may need internal fixation."
        ],
        "prognosis_notes": "Good prognosis if reduced early and adequately stabilized."
    },
    "Healthy": {
        "definition": "No radiographic evidence of fracture.",
        "icd_code": "Z00.0",
        "severity": "Low",
        "treatment_guidelines": [
            "No specific fracture treatment required.",
            "Advise routine follow-up and monitoring of symptoms."
        ],
        "prognosis_notes": "Normal bone health based on the available imaging."
    },
    "Transverse": {
        "definition": "A fracture line that is approximately perpendicular to the long axis of the bone.",
        "icd_code": "S52.0",
        "severity": "Medium",
        "treatment_guidelines": [
            "Closed reduction and casting are common for stable fractures.",
            "Unstable patterns may require pins, screws, or plates."
        ],
        "prognosis_notes": "Generally heals well with proper immobilization and alignment."
    },
    "Spiral": {
        "definition": "A fracture caused by a twisting force, with a spiral or helical fracture line.",
        "icd_code": "S52.7",
        "severity": "Medium-High",
        "treatment_guidelines": [
            "Often requires surgical fixation due to rotational instability.",
            "Longer recovery because of associated soft-tissue injury risk."
        ],
        "prognosis_notes": "Healing can be slow; higher risk of displacement during healing."
    },
    "Greenstick": {
        "definition": "An incomplete fracture where one cortex is broken and the other is bent, typically in children.",
        "icd_code": "S52.8",
        "severity": "Low",
        "treatment_guidelines": [
            "Usually treated with simple casting or splinting.",
            "Follow-up radiographs to ensure remodeling in growing bone."
        ],
        "prognosis_notes": "Excellent prognosis; children typically heal rapidly with complete remodeling."
    },
    "Impacted": {
        "definition": "A fracture where the ends of the bone are driven into each other, shortening the bone.",
        "icd_code": "S52.2",
        "severity": "Medium",
        "treatment_guidelines": [
            "May be stable enough for casting or functional bracing.",
            "Monitor for limb shortening or joint incongruity."
        ],
        "prognosis_notes": "Generally good stability and satisfactory healing if alignment is acceptable."
    },
    "Pathologic": {
        "definition": "A fracture occurring in bone weakened by disease (e.g., osteoporosis, tumor, metastasis).",
        "icd_code": "M84.4",
        "severity": "Often High due to the underlying pathology",
        "treatment_guidelines": [
            "Treat both the fracture and the underlying disease.",
            "May require specialized surgical fixation and oncology input."
        ],
        "prognosis_notes": "Highly dependent on the underlying condition and systemic disease control."
    }
}


# --------------------------------------------------------------------------
# RAG Knowledge Base: MedAI Domain + Technical Sources
# (Condensed from your table into documents that we can embed)
# --------------------------------------------------------------------------
RAG_SOURCE_DOCS: List[Dict[str, Any]] = [
    # ----------------- Domain Knowledge (Clinical & Radiology) -----------------
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
        "use_case": "Explain exact fracture code and terminology for model-predicted fracture classes."
    },
    {
        "id": "salter_harris_classification",
        "category": "Fracture Classification & Terminology",
        "title": "Salter-Harris Classification for Pediatric Physeal Injuries",
        "content": (
            "The Salter-Harris classification describes fractures involving the epiphyseal "
            "growth plate in children (Types I–V). It guides prognosis and treatment decisions "
            "in pediatric fractures. In MedAI, this knowledge is used when the pipeline detects "
            "a probable pediatric case, allowing LLaMA 3 to give age-appropriate explanations "
            "and warn about growth plate involvement."
        ),
        "use_case": "Provide pediatric-specific explanations when the patient is a child or adolescent."
    },
    {
        "id": "aaos_orthoinfo",
        "category": "Clinical Context & Management",
        "title": "OrthoInfo (AAOS) Patient-Friendly Fracture Articles",
        "content": (
            "OrthoInfo from the American Academy of Orthopaedic Surgeons (AAOS) provides "
            "patient-friendly explanations for fractures such as distal radius, tibial shaft, "
            "and ankle fractures. The content covers symptoms, mechanism of injury, typical "
            "treatment pathways, recovery timelines and self-care advice. In MedAI, these texts "
            "inform the patient-facing interface so that explanations are understandable and "
            "aligned with standard patient education material."
        ),
        "use_case": "Generate simple, patient-facing explanations about symptoms, treatment and recovery."
    },
    {
        "id": "rockwood_green_fractures_textbook",
        "category": "Clinical Context & Management",
        "title": "Rockwood and Green's Fractures in Adults and Children",
        "content": (
            "Rockwood and Green's is a standard orthopedic reference textbook that describes "
            "diagnosis, classification, indications for surgery and complications for fractures "
            "throughout the body. In MedAI, key diagnostic and management sections are used as "
            "high-authority clinical grounding to differentiate fracture types and to reason "
            "about complications such as non-union, malunion, and neurovascular injury."
        ),
        "use_case": "Deep clinical validation and high-authority grounding for clinician-level questions."
    },
    {
        "id": "radiopaedia_fracture_entries",
        "category": "Radiology & Interpretation",
        "title": "Radiopaedia Fracture Imaging Patterns",
        "content": (
            "Radiopaedia.org hosts detailed fracture entries with example radiographs and CT scans, "
            "describing typical imaging appearances, variants and pitfalls. It explains features such "
            "as butterfly fragments, wedge patterns, cortical step-offs and subtle trabecular changes. "
            "In MedAI, this material is used to contextualize Grad-CAM heatmaps and explain which visual "
            "features the vision transformers are expected to focus on for each fracture pattern."
        ),
        "use_case": "Explain Grad-CAM regions and image features underlying the model's decision."
    },
    {
        "id": "acr_appropriateness_criteria",
        "category": "Radiology & Interpretation",
        "title": "ACR Appropriateness Criteria for Musculoskeletal Imaging",
        "content": (
            "The American College of Radiology (ACR) Appropriateness Criteria provide evidence-based "
            "recommendations on when to order additional imaging such as CT, MRI or ultrasound. For "
            "fractures, they describe indications for follow-up imaging in occult injury, complex "
            "articular involvement and postoperative assessment. MedAI uses these guidelines to suggest "
            "standard next-step imaging options in an informational (non-prescriptive) manner."
        ),
        "use_case": "Inform non-binding recommendations about when additional imaging might be considered."
    },
    {
        "id": "ai_ethics_regulation",
        "category": "Ethical & Regulatory",
        "title": "FDA AI/ML Guidelines and Health Informatics Ethics (HIMSS/AMIA)",
        "content": (
            "Regulatory and ethics documents from bodies such as the FDA, HIMSS and AMIA emphasize "
            "transparency, bias mitigation, clinical oversight and safety for AI-based medical devices. "
            "Key themes include not replacing clinician judgment, providing understandable explanations, "
            "and clearly stating limitations. MedAI uses this knowledge to ensure that the LLaMA 3 "
            "interface gives appropriate disclaimers and avoids specific patient-tailored medical advice."
        ),
        "use_case": "Generate safety disclaimers and keep explanations informational rather than prescriptive."
    },

    # ----------------- Technical & Explainability Knowledge -----------------
    {
        "id": "swin_transformer_paper",
        "category": "Model Architecture & Vision Transformers",
        "title": "Swin Transformer Architecture",
        "content": (
            "The Swin Transformer is a hierarchical vision transformer that uses shifted windows to "
            "efficiently model local and global image context. It processes images as non-overlapping "
            "patches, applies self-attention within windows and gradually builds multi-scale feature maps. "
            "In MedAI, Swin-based models serve as core vision backbones, explaining how X-ray images are "
            "tokenized and how local fracture cues and global alignment are captured."
        ),
        "use_case": "Answer technical questions about why Swin was chosen and how it processes X-ray patches."
    },
    {
        "id": "convnext_paper",
        "category": "Model Architecture & Vision Transformers",
        "title": "ConvNeXt: Modernized CNN Architecture",
        "content": (
            "ConvNeXt is a convolutional neural network architecture that modernizes ResNet-style designs "
            "to achieve transformer-level performance while retaining convolutional inductive biases. "
            "It uses large kernels, depthwise convolutions and LayerNorm to improve accuracy and efficiency. "
            "In MedAI, ConvNeXt complements Swin as an alternative backbone in the ensemble, providing "
            "architectural diversity and robustness."
        ),
        "use_case": "Explain why a CNN-style backbone is included and how it differs from Swin."
    },
    {
        "id": "grad_cam_paper",
        "category": "Explainable AI",
        "title": "Grad-CAM: Visual Explanations from Deep Networks",
        "content": (
            "Grad-CAM (Gradient-weighted Class Activation Mapping) produces heatmaps by backpropagating "
            "gradients from a target class to convolutional feature maps, highlighting spatial regions that "
            "contribute most to the prediction. In MedAI, Grad-CAM is applied to vision transformer and "
            "ConvNeXt feature maps to produce clinically interpretable overlays on X-ray images, explaining "
            "which bone regions influenced the predicted fracture class. Limitations include coarse "
            "localization and dependence on the chosen layer."
        ),
        "use_case": "Explain how the heatmaps are generated and discuss strengths and limitations of Grad-CAM."
    },
    {
        "id": "ensemble_learning_review",
        "category": "Explainable AI",
        "title": "Ensemble Learning and Cross-Validation in MedAI",
        "content": (
            "Ensemble learning combines multiple models to improve robustness and generalization. Common "
            "strategies include majority voting, averaging of probabilities and stacking. Cross-validation "
            "quantifies performance stability across folds. In MedAI, five specialized diagnostic agents "
            "and cross-validated models are ensembled to achieve macro-F1 > 0.92, while still allowing "
            "interpretation at the level of individual agent predictions and Grad-CAM maps."
        ),
        "use_case": "Justify the ensemble agent design and answer questions about why multiple models are used."
    },
    {
        "id": "llama3_technical_report",
        "category": "Multi-Agent & RAG/LLM",
        "title": "LLaMA 3 Capabilities and Constraints",
        "content": (
            "LLaMA 3 is a large language model designed for instruction following and multi-turn dialogue. "
            "It is powerful at generating natural language explanations but may hallucinate if not grounded "
            "in external knowledge. In MedAI, LLaMA 3 is used strictly as a controlled natural language "
            "interface, grounded via retrieval-augmented generation (RAG) over curated medical and technical "
            "sources. Prompts emphasize not giving direct medical advice and staying within retrieved context."
        ),
        "use_case": "Explain how the language agent works, its limitations, and why RAG is necessary."
    },
    {
        "id": "rag_and_multi_agent_frameworks",
        "category": "Multi-Agent & RAG/LLM",
        "title": "RAG and Multi-Agent Framework Concepts",
        "content": (
            "RAG (retrieval-augmented generation) systems combine vector search over knowledge bases with "
            "LLM generation, passing retrieved documents as context to reduce hallucinations. Multi-agent "
            "frameworks such as LangChain or CrewAI decompose complex tasks into specialized agents for "
            "data retrieval, reasoning, explanation and tool use. MedAI adopts a multi-agent architecture "
            "with dedicated diagnostic, cross-validation, explanation, patient-facing and knowledge agents, "
            "each with clearly defined responsibilities."
        ),
        "use_case": "Describe the overall MedAI multi-agent architecture and how RAG fits into it."
    }
]

# ============================================================================
# RAD-DINO CLASSIFIER
# ============================================================================

class RadDinoClassifier(nn.Module):
    """RAD-DINO backbone with a classification head."""
    def __init__(self, num_classes, head_type='linear'):
        super(RadDinoClassifier, self).__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(RAD_DINO_MODEL_NAME)
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
        return self.classifier(cls_embedding)


def _detect_rad_dino_head_type(state_dict):
    """Detect whether a RAD-DINO checkpoint uses 'linear' or 'mlp' head."""
    for k in state_dict.keys():
        if "classifier.0.weight" in k:
            return "mlp"
    return "linear"


_rad_dino_processor = None

def get_rad_dino_processor():
    """Lazy-load the RAD-DINO image processor."""
    global _rad_dino_processor
    if _rad_dino_processor is None:
        try:
            from transformers import AutoImageProcessor
            _rad_dino_processor = AutoImageProcessor.from_pretrained(RAD_DINO_MODEL_NAME)
        except Exception:
            pass
    return _rad_dino_processor


def get_rad_dino_input_tensor(image: Image.Image, dev) -> torch.Tensor:
    """Preprocess a PIL image for RAD-DINO."""
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
    """Wraps a YOLO model to produce class probabilities aligned with CLASS_NAMES."""
    def __init__(self, yolo_model, class_names: List[str]):
        super().__init__()
        self.yolo_model = yolo_model
        self.class_names = class_names
        self._build_class_mapping()

    def _build_class_mapping(self):
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
        """Run YOLO prediction and return probabilities in CLASS_NAMES order."""
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

        s = probs.sum()
        if s > 0:
            probs = probs / s
        return probs

    def forward(self, x):
        raise NotImplementedError("Use predict_pil() for YOLO models.")


def is_yolo_model(model) -> bool:
    """Check if a model is a YOLO wrapper."""
    return isinstance(model, YOLOClassifierWrapper)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_device():
    """Detects and returns the appropriate torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_transforms(img_size: int = 224):
    """Returns standard image transforms for inference."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def read_threshold(path: str):
    """Read a numeric threshold (float) from a text file. Returns None on error."""
    try:
        if not path or not os.path.exists(path):
            return None
        with open(path, 'r') as fh:
            s = fh.read().strip()
            return float(s)
    except Exception:
        return None


def get_model(name: str, num_classes: int, pretrained: bool = False):
    """Loads a model architecture from timm and adapts the classifier head."""
    if not TIMM_AVAILABLE:
        return None
    
    # Handle custom HypercolumnCBAMDenseNet model
    if "hypercolumn" in name.lower() or "cbam" in name.lower():
        return HypercolumnCBAMDenseNet(num_classes=num_classes)
    
    # Skip RAD-DINO and YOLO — they have separate loading paths
    if name in ("rad_dino", "yolo"):
        return None
    
    model_name = MODEL_CONFIGS.get(name, name)
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception:
        # Fallback for some models where strict num_classes init might fail or is different
        model = timm.create_model(model_name, pretrained=pretrained)
        
    # Adjust classifier head based on common timm model types (must match training code)
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        if model.head.out_features != num_classes:
            model.head = nn.Linear(model.head.in_features, num_classes)
    elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
         if model.fc.out_features != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
         if model.classifier.out_features != num_classes:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        # Last resort generic reset
        if not (hasattr(model, 'get_classifier') and model.get_classifier().out_features == num_classes):
            try:
                model.reset_classifier(num_classes=num_classes)
            except Exception:
                # If we really can't set it, we might return None or warn
                # st.warning(f"Could not adapt classifier head for {name}")
                pass
    
    return model


def load_model_from_checkpoint(model_name: str, checkpoint_path: str, num_classes: int, device):
    """Loads a model with weights from a checkpoint."""
    model = get_model(model_name, num_classes, pretrained=False)
    if model is None:
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        # st.warning(f"Could not load {model_name}: {e}")
        return None


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


try:
    from medai.modules.diagnosis_module import DiagnosticModule
    from medai.modules.ensemble_module import EnsembleModule
    from medai.modules.explanation_module import ExplanationModule
except ImportError as e:
    logger.warning(f"Failed to import modules: {e}")
    # Fallback to local definitions or error out
    pass

# Alias for compatibility
DiagnosticAgent = DiagnosticModule
ModelEnsembleAgent = EnsembleModule
ExplainabilityAgent = ExplanationModule


# ============================================================================
# AGENT 4: EDUCATIONAL AGENT
# ============================================================================


class EducationalAgent:
    """Translates technical diagnoses into patient-friendly explanations."""
    
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
    
    def translate(self, diagnosis_result: Dict[str, Any], explanation_text: str, image: Optional[Image.Image] = None, gradcam_image: Optional[Image.Image] = None) -> Dict[str, str]:
        """Generates patient-friendly summary and action plan using Gemini Vision if available."""
        fracture_detected = diagnosis_result.get("fracture_detected", False)
        predicted_class = diagnosis_result.get("predicted_class", diagnosis_result.get("ensemble_prediction", "Unknown"))
        confidence = diagnosis_result.get("confidence_score", diagnosis_result.get("ensemble_confidence", 0.0))
        
        severity_layman = self.severity_map.get(predicted_class, "Unknown")
        
        # Fallback template-based generation
        if not fracture_detected:
            summary = (
                f"Great news! The AI analysis suggests your bone looks healthy. "
                f"The system is {confidence*100:.0f}% confident in this assessment."
            )
            action_plan = (
                "📋 **Next Steps / Action Plan:**\n"
                "1. If you're still experiencing pain, please discuss with your doctor.\n"
                "2. This AI result should be confirmed by a medical professional.\n"
                "3. No immediate treatment appears necessary based on this analysis."
            )
        else:
            summary = (
                f"The AI analysis has detected what appears to be a **{predicted_class}** fracture. "
                f"This is classified as **{severity_layman}**. "
                f"The system is {confidence*100:.0f}% confident in this finding."
            )
            
            kb_info = MEDICAL_KNOWLEDGE_BASE.get(predicted_class, {})
            guidelines = kb_info.get("treatment_guidelines", ["Consult with an orthopedic specialist."])
            
            action_plan = (
                "📋 **Next Steps / Action Plan:**\n"
                + "\n".join([f"{i+1}. {g}" for i, g in enumerate(guidelines)])
                + f"\n\n⚠️ **Important:** This is an AI-assisted analysis. "
                f"Please consult with {self.doctor_name} for definitive diagnosis and treatment."
            )
        
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
                
                context = f"Diagnosis: {predicted_class}\nConfidence: {confidence*100:.0f}%\n"
                
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
# AGENT 5: KNOWLEDGE AGENT
# ============================================================================

class KnowledgeAgent:
    """
    MedAI Knowledge Agent (Advanced):
    - Builds and manages ChromaDB collections.
    - Provides structured summaries for fracture diagnoses.
    - Supports RAG over MedAI clinical/technical sources.
    - Integrates LLaMA 3 for explanations (optional).
    """

    def __init__(self) -> None:
        self.client = None
        self.diag_collection = None
        self.source_collection = None

        if not CHROMADB_AVAILABLE:
            st.warning("ChromaDB not installed. Knowledge Agent features disabled.")
            return

        try:
            # Persistent Chroma client
            self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

            # Shared embedding function
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL_NAME
            )

            # Collections
            self.diag_collection = self._setup_diag_collection()
            self.source_collection = self._setup_source_collection()
        except Exception as e:
            st.warning(f"Knowledge Agent initialization failed (ChromaDB error): {e}")
            self.client = None

    # ----------------- Collection Setup -----------------
    def _setup_diag_collection(self):
        # logger.info("Checking/creating diagnosis collection...")
        collection = self.client.get_or_create_collection(
            name=DIAG_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

        diagnoses = list(MEDICAL_KNOWLEDGE_BASE.keys())
        ids = [d.lower().replace(" ", "-") for d in diagnoses]

        # If empty or count mismatch, repopulate
        if collection.count() != len(diagnoses):
            try:
                self.client.delete_collection(DIAG_COLLECTION_NAME)
            except:
                pass
            collection = self.client.get_or_create_collection(
                name=DIAG_COLLECTION_NAME,
                embedding_function=self.embedding_fn,
            )
            collection.add(documents=diagnoses, ids=ids)

        return collection

    def _setup_source_collection(self):
        # logger.info("Checking/creating RAG source collection...")
        collection = self.client.get_or_create_collection(
            name=SOURCE_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

        ids = [doc["id"] for doc in RAG_SOURCE_DOCS]
        docs = [
            f"Title: {doc['title']}\nCategory: {doc['category']}\n\n{doc['content']}\n\nUse case: {doc['use_case']}"
            for doc in RAG_SOURCE_DOCS
        ]
        metadatas = [
            {
                "title": doc["title"],
                "category": doc["category"],
                "use_case": doc["use_case"],
            }
            for doc in RAG_SOURCE_DOCS
        ]

        if collection.count() != len(docs):
            try:
                self.client.delete_collection(SOURCE_COLLECTION_NAME)
            except:
                pass
            collection = self.client.get_or_create_collection(
                name=SOURCE_COLLECTION_NAME,
                embedding_function=self.embedding_fn,
            )
            collection.add(ids=ids, documents=docs, metadatas=metadatas)

        return collection

    # ----------------- Structured Summary for Diagnoses -----------------
    def get_medical_summary(self, diagnosis: str, confidence: float) -> Dict[str, Any]:
        diagnosis = diagnosis.strip()
        
        # Guard clause if DB not init
        if not self.diag_collection:
            # Fallback to direct dict lookup if DB missing but KB exists
            if diagnosis in MEDICAL_KNOWLEDGE_BASE:
                 raw = MEDICAL_KNOWLEDGE_BASE[diagnosis]
                 return {
                    "Diagnosis": diagnosis,
                    "Ensemble_Confidence": f"{confidence:.2f}",
                    "Type_Definition": raw.get("definition"),
                    "ICD_Code": raw.get("icd_code", "N/A"),
                    "Severity_Rating": raw.get("severity"),
                    "Treatment_Guidelines": raw.get("treatment_guidelines"),
                    "Long_Term_Prognosis": raw.get("prognosis_notes"),
                }
            return {"error": "Knowledge Agent not initialized properly."}

        results = self.diag_collection.query(
            query_texts=[diagnosis],
            n_results=1,
            include=["documents", "distances"],
        )

        if not results or not results["documents"] or not results["documents"][0]:
            return {
                "error": f"Vector search failed to find a relevant diagnosis for '{diagnosis}'."
            }

        retrieved_name = results["documents"][0][0]
        raw = MEDICAL_KNOWLEDGE_BASE.get(retrieved_name)

        if not raw:
            return {
                "error": f"Retrieved diagnosis '{retrieved_name}' not present in knowledge base."
            }

        return {
            "Diagnosis": retrieved_name,
            "Ensemble_Confidence": f"{confidence:.2f}",
            "Type_Definition": raw.get("definition"),
            "ICD_Code": raw.get("icd_code", "N/A"),
            "Severity_Rating": raw.get("severity"),
            "Treatment_Guidelines": raw.get("treatment_guidelines"),
            "Long_Term_Prognosis": raw.get("prognosis_notes"),
        }

    # ----------------- Helper for Critic Agent -----------------
    def get_context_for_label(self, label: str) -> str:
        """
        Retrieves the definition context for the Critic Agent.
        """
        # We can reuse get_medical_summary with a dummy confidence
        summary = self.get_medical_summary(label, 1.0)
        if "error" in summary:
            # Fallback based on knowledge base keys slightly matching
            # Or generic definition
            return f"Condition '{label}' regarding bone integrity."
        
        return summary.get("Type_Definition", "No definition found.")

    # ----------------- RAG over MedAI Sources -----------------
    def retrieve_sources(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        print(f"[DEBUG] Retrieving sources for query: {query}")
        if not self.source_collection:
            print("[DEBUG] source_collection is None/Empty.")
            return []
            
        query = query.strip()
        results = self.source_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
        print(f"[DEBUG] Raw RAG results keys: {results.keys()}")

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        out: List[Dict[str, Any]] = []
        for doc_text, meta in zip(docs, metas):
            out.append(
                {
                    "title": meta.get("title"),
                    "category": meta.get("category"),
                    "use_case": meta.get("use_case"),
                    "content": doc_text,
                }
            )
        print(f"[DEBUG] Retrieved {len(out)} documents.")
        return out

    # ----------------- LLaMA 3 Integration (Optional) -----------------
    def gemini_available(self) -> bool:
        is_avail = bool(GEMINI_API_KEY)
        print(f"[DEBUG] gemini_available check: {is_avail} (Key present: {'Yes' if GEMINI_API_KEY else 'No'})")
        return is_avail

    def generate_explanation_with_gemini(
        self,
        summary: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        audience: str = "patient",
    ) -> Optional[str]:
        print("[DEBUG] Entering generate_explanation_with_gemini...")
        if not self.gemini_available():
            print("[DEBUG] optimize returning None because Gemini is not available.")
            return None
        
        # Check if Requests is available
        if 'requests' not in sys.modules and (not 'REQUESTS_AVAILABLE' in globals() or not REQUESTS_AVAILABLE):
             logger.warning("Requests not installed or imported. Cannot call Gemini.")
             print("[DEBUG] Requests library check failed.")
             return None

        print(f"[DEBUG] Preparing Gemini prompt for audience='{audience}' with {len(retrieved_docs)} docs.")
        
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
                "You are an expert orthopedic clinician. "
                "You are given:\n"
                "1) A structured fracture summary.\n"
                "2) Retrieved domain and technical documents from a curated knowledge base.\n\n"
                "Your job is to explain the diagnosis using ONLY this context. "
                "Do not invent new medical facts. Do not give direct medical advice or treatment plans. "
                "Emphasize that this is informational and does not replace a clinician."
            )
            user_instruction = (
                "Explain the diagnosis to a layperson patient. Use simple language to describe what "
                "the fracture means, roughly how it is treated and what recovery might involve. "
                "Avoid giving strict medical advice; encourage the patient to talk to their doctor."
            )

        docs_block = "\n\n---\n\n".join(
            f"[{d['category']}] {d['title']}\n\n{d['content']}" for d in retrieved_docs
        )

        # Remove the "produced by a diagnostic ensemble" part from the summary string to avoid leaking MedAI info
        clean_summary = str(summary).replace("MedAI", "").replace("ensemble", "").replace("Ensemble", "")
        clean_docs_block = docs_block.replace("MedAI", "").replace("ensemble", "").replace("Ensemble", "")

        context = (
            f"Structured summary:\n{clean_summary}\n\n"
            f"Retrieved RAG documents:\n\n{clean_docs_block}"
        )

        # Gemini REST API Format
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
        
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [{
                "role": "user",
                "parts": [{"text": user_instruction + "\n\nCONTEXT:\n" + context}]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1000
            }
        }

        try:
            print("[DEBUG] Sending request to Gemini API...")
            print(f"[DEBUG] URL: {url.split('?')[0]}...") # Log without key
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60.0
            )
            print(f"[DEBUG] Gemini Response Status: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            
            if 'candidates' in data and data['candidates']:
                print("[DEBUG] Successfully extracted candidate text.")
                return data['candidates'][0]['content']['parts'][0]['text']
            
            print("[DEBUG] No candidates found in Gemini response.")
            print(f"[DEBUG] Full response data: {data}")
            return None
        except Exception as e:
            print(f"[DEBUG] Gemini call failed with exception: {e}")
            if 'resp' in locals():
                print(f"[DEBUG] Response content: {resp.text}")
            st.warning(f"Gemini call failed: {e}")
            return None


# ============================================================================
# AGENT 6: PATIENT INTERACTION AGENT
# ============================================================================

class PatientInteractionAgent:
    """Handles patient chat using RAG and LLM."""
    
    def __init__(self, medical_summary: Dict[str, Any], patient_history: Dict[str, Any]):
        self.medical_summary = medical_summary
        self.patient_history = patient_history
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Creates the system prompt with medical context."""
        guidelines = "\n- ".join(self.medical_summary.get('Treatment_Guidelines', ["No specific guidelines available."]))
        
        return f"""
You are a knowledgeable and compassionate medical assistant specializing in fracture care. Your goal is to provide 
helpful, accurate information about fractures based on the context provided. 

IMPORTANT RULES:
1. ONLY use the information provided in the context below
2. Do NOT give specific medical advice or treatment plans
3. Always recommend consulting with a healthcare professional
4. Be empathetic and use clear, simple language
5. If unsure about something, acknowledge the limitation

--- DIAGNOSIS CONTEXT ---
Diagnosis: {self.medical_summary.get('Diagnosis')} (Confidence: {self.medical_summary.get('Ensemble_Confidence')})
ICD Code: {self.medical_summary.get('ICD_Code', 'N/A')}
Definition: {self.medical_summary.get('Type_Definition')}
Severity: {self.medical_summary.get('Severity_Rating')}
General Treatment Guidelines: 
- {guidelines}
Prognosis Note: {self.medical_summary.get('Long_Term_Prognosis', 'N/A')}

--- PATIENT INFORMATION ---
Age: {self.patient_history.get('age', 'Unknown')}
Gender: {self.patient_history.get('gender', 'Unknown')}
Medical History: {self.patient_history.get('history', 'None provided')}
"""
    
    def get_response(self, query: str) -> str:
        """Gets LLM response for a patient query via OpenRouter."""
        if not REQUESTS_AVAILABLE:
            return "Chat functionality requires the requests library."
        
        if not OPENROUTER_API_KEY:
            return "⚠️ OpenRouter API key not configured. Please set it in secrets.toml or as OPENROUTER_API_KEY environment variable."
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://medai-fracture-detection.streamlit.app",
            "X-Title": "MedAI Fracture Detection"
        }
        
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.1
        }
        
        # Retry logic with exponential backoff for rate limits
        max_retries = 3
        base_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "Could not get response from LLM.")
            except requests.exceptions.ConnectionError:
                return "⚠️ Cannot connect to OpenRouter API. Please check your internet connection."
            except requests.exceptions.HTTPError as e:
                if response.status_code == 401:
                    return "⚠️ Invalid OpenRouter API key. Please check your configuration."
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        import time
                        delay = base_delay * (2 ** attempt)  # Exponential backoff: 2, 4, 8 seconds
                        time.sleep(delay)
                        continue
                    return "⚠️ Rate limit exceeded. The free tier has limited requests. Please wait a moment and try again."
                return f"⚠️ API Error: {e}"
            except Exception as e:
                return f"⚠️ Error: {e}"
        
        return "⚠️ Failed to get response after multiple retries."


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def initialize_session_state():
    """Initializes session state variables."""
    defaults = {
        "diagnosis_result": None,
        "ensemble_result": None,
        "gradcam_image": None,
        "gradcam_images": {},
        "explanation_text": None,
        "educational_output": None,
        "medical_summary": None,
        "chat_messages": [],
        "patient_agent": None,
        "models_loaded": False,
        "uploaded_image": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_models(checkpoint_dir: str, selected_models: List[str], device):
    """Loads selected models from checkpoint directory.
    
    Supports standard timm models, hypercolumn, RAD-DINO, and YOLO.
    """
    models = {}
    
    for model_name in selected_models:
        # --- RAD-DINO ---
        if is_rad_dino_model(model_name):
            checkpoint_path = os.path.join(checkpoint_dir, "best_rad_dino_classifier.pth")
            if not os.path.exists(checkpoint_path):
                st.warning(f"RAD-DINO checkpoint not found: {checkpoint_path}")
                continue
            try:
                ck = torch.load(checkpoint_path, map_location=device)
                state_dict = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
                head_type = _detect_rad_dino_head_type(state_dict)
                model = RadDinoClassifier(NUM_CLASSES, head_type=head_type)
                model.load_state_dict(state_dict, strict=False)
                model.to(device)
                model.eval()
                models[model_name] = model
                print(f"  Loaded RAD-DINO ({head_type} head)")
            except Exception as e:
                st.warning(f"Failed to load RAD-DINO: {e}")
            continue

        # --- YOLO ---
        if "yolo" in model_name.lower():
            loaded = False
            for yp in YOLO_SEARCH_PATHS:
                if os.path.exists(yp):
                    try:
                        from ultralytics import YOLO
                        yolo_raw = YOLO(yp)
                        wrapper = YOLOClassifierWrapper(yolo_raw, CLASS_NAMES)
                        models[model_name] = wrapper
                        print(f"  Loaded YOLO from {yp}")
                        loaded = True
                        break
                    except ImportError:
                        st.warning("ultralytics not installed — cannot load YOLO model")
                        break
                    except Exception as e:
                        print(f"  Failed to load YOLO from {yp}: {e}")
            if not loaded:
                st.warning(f"Could not find valid YOLO checkpoint for {model_name}")
            continue

        # --- Standard timm / hypercolumn models ---
        checkpoint_path = os.path.join(checkpoint_dir, f"best_{model_name}.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.pth")
        
        if os.path.exists(checkpoint_path):
            model = load_model_from_checkpoint(model_name, checkpoint_path, NUM_CLASSES, device)
            if model is not None:
                models[model_name] = model
    
    return models


def render_sidebar():
    """Renders the sidebar configuration."""
    st.sidebar.title("⚙️ Configuration")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    checkpoint_dir = st.sidebar.text_input(
        "Checkpoint Directory",
        value="./models",
        help="Directory containing model checkpoint files"
    )
    
    available_models = list(MODEL_CONFIGS.keys())
    
    # Default ensemble: maxvit, yolo, hypercolumn_cbam_densenet169, rad_dino
    default_ensemble = ["maxvit", "yolo", "hypercolumn_cbam_densenet169", "rad_dino"]
    default_selection = [m for m in default_ensemble if m in available_models]
    
    selected_models = st.sidebar.multiselect(
        "Models to Load",
        options=available_models,
        default=default_selection,
        help="Select models for ensemble inference. Primary ensemble: maxvit, yolo, hypercolumn_cbam_densenet169, rad_dino"
    )
    
    # Patient info
    st.sidebar.subheader("Patient Information")
    patient_age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
    patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    patient_history = st.sidebar.text_area(
        "Medical History",
        value="No significant medical history.",
        height=100
    )
    # Conformal Prediction settings
    st.sidebar.subheader("Conformal Prediction")
    use_conformal = st.sidebar.checkbox("Enable conformal prediction", value=False,
                                        help="Include conformal prediction sets in outputs")
    conformal_threshold_path = st.sidebar.text_input(
        "Threshold file (optional)", value="./outputs/conformal/conformal_threshold.txt",
        help="Path to a text file containing a single float threshold value (nonconformity t)."
    )
    conformal_threshold_value = st.sidebar.number_input(
        "Manual threshold value (used if file missing)", value=0.10, format="%.6f"
    )

    # Agentic Reasoning Settings
    st.sidebar.subheader("Agentic Reasoning")
    enable_critic = st.sidebar.checkbox("Enable Critic Agent (Self-Correction)", value=True,
                                       help="Use MedGemma VLM to double-check the diagnosis against visual evidence.")
    
    return {
        "checkpoint_dir": checkpoint_dir,
        "selected_models": selected_models,
        "patient_info": {
            "age": patient_age,
            "gender": patient_gender,
            "history": patient_history
        }
        ,
        "use_conformal": use_conformal,
        "conformal_threshold_path": conformal_threshold_path,
        "conformal_threshold_value": float(conformal_threshold_value),
        "enable_critic": enable_critic
    }


def render_image_upload():
    """Renders the image upload section."""
    st.subheader("📤 Upload X-Ray Image")
    
    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Upload a bone X-ray image for analysis"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded X-Ray", width='stretch')
        
        return image
    
    return None


def render_diagnosis_results():
    """Renders the diagnosis results section."""
    if st.session_state.diagnosis_result is None and st.session_state.ensemble_result is None:
        return
    
    st.subheader("🔬 Diagnosis Results")
    
    col1, col2 = st.columns(2)
    
    # Single model result
    with col1:
        st.markdown("**Primary Model Diagnosis**")
        if st.session_state.diagnosis_result:
            result = st.session_state.diagnosis_result
            if "error" not in result:
                status = "🔴 Fracture Detected" if result["fracture_detected"] else "🟢 No Fracture"
                st.metric("Status", status)
                st.metric("Classification", result["predicted_class"])
                st.metric("Confidence", f"{result['confidence_score']:.2%}")
                # Show conformal set if present
                if "conformal_set" in result:
                    st.markdown("**Conformal Prediction Set (guaranteed coverage)**")
                    st.write(" ", ", ".join(result["conformal_set"]))
            else:
                st.error(result["error"])
    
    # Ensemble result
    with col2:
        st.markdown("**Ensemble Prediction**")
        if st.session_state.ensemble_result:
            result = st.session_state.ensemble_result
            if "error" not in result:
                status = "🔴 Fracture Detected" if result["fracture_detected"] else "🟢 No Fracture"
                st.metric("Status", status)
                st.metric("Classification", result["ensemble_prediction"])
                st.metric("Confidence", f"{result['ensemble_confidence']:.2%}")
                if "conformal_set" in result:
                    st.markdown("**Conformal Prediction Set (guaranteed coverage)**")
                    st.write(" ", ", ".join(result["conformal_set"]))
                
                # Show individual predictions
                with st.expander("Individual Model Predictions"):
                    for name, pred in result["individual_predictions"].items():
                        st.write(f"**{name}**: {pred['class']} ({pred['confidence']:.2%})")
            else:
                st.error(result["error"])
    
    # Probability distribution
    if st.session_state.ensemble_result:
        probs = None
        # Prefer dictionary format
        if "all_probabilities_dict" in st.session_state.ensemble_result:
             probs = st.session_state.ensemble_result["all_probabilities_dict"]
        elif "all_probabilities" in st.session_state.ensemble_result:
             probs = st.session_state.ensemble_result["all_probabilities"]
        
        # Ensure probs is a dict before plotting
        if probs and isinstance(probs, dict):
            st.markdown("**Class Probabilities**")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            classes = list(probs.keys())
            values = list(probs.values())
            colors = ['#2ecc71' if c == 'Healthy' else '#e74c3c' for c in classes]
            
            bars = ax.barh(classes, values, color=colors)
            ax.set_xlabel('Probability')
            ax.set_xlim(0, 1)
            
            for bar, val in zip(bars, values):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2%}', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            # show margin and uncertainty
            try:
                vals = list(probs.values())
                sorted_idxs = sorted(range(len(vals)), key=lambda k: vals[k], reverse=True)
                top1 = vals[sorted_idxs[0]]
                top2 = vals[sorted_idxs[1]] if len(vals) > 1 else 0.0
                margin = top1 - top2
                st.markdown(f"**Top-1 vs Top-2 margin:** {margin:.2%}")
                if margin < 0.15:
                    st.warning("Low margin between top classes — result may be ambiguous. See conformal set for alternatives.")
            except Exception:
                pass


def render_critic_review():
    """Renders the Critic Agent review section."""
    if "critic_review" not in st.session_state or not st.session_state.critic_review:
        return

    st.markdown("---")
    st.subheader("🕵️ Critic Agent Review (Self-Correction)")
    
    review = st.session_state.critic_review
    consensus = st.session_state.get("consensus")
    
    if "error" in review:
        st.error(f"Critic Agent Error: {review['error']}")
        return

    # Use columns to show Verdict vs Explanation
    c1, c2 = st.columns([1, 2])
    
    with c1:
        verdict = review.get("verdict", "uncertain").upper()
        if verdict == "YES":
             st.success("✅ **Critic Agrees**")
        elif verdict == "NO":
             st.error("❌ **Critic Disagrees**")
        else:
             st.warning("⚠️ **Critic Uncertain**")
        
        if consensus:
             decision = consensus.get("final_decision")
             if decision == "flagged":
                 st.error("🚩 **Flagged for Human Review**")
                 st.markdown(f"_Reason: {consensus.get('reason')}_")
             else:
                 st.info("System Consensus: Approved")
    
    with c2:
        st.markdown("**Critic's Analysis:**")
        st.info(review.get("explanation", "No explanation provided."))
        st.caption(f"Based on MedGemma analysis of visual features vs. '{st.session_state.ensemble_result.get('ensemble_prediction')}' definition.")

def render_explainability():
    """Renders the explainability section."""
    if (not st.session_state.gradcam_images) and st.session_state.gradcam_image is None and st.session_state.explanation_text is None:
        return

    st.subheader("🔍 AI Explanation")

    col1, col2 = st.columns(2)

    with col1:
        # If we have per-model gradcam images, show checkboxes to preview each
        if st.session_state.gradcam_images:
            st.markdown("**Per-model Grad-CAMs**")
            for m_name, pil_img in st.session_state.gradcam_images.items():
                key = f"gradcam_preview_{m_name}"
                if st.checkbox(f"Show {m_name}", key=key):
                    st.image(pil_img, caption=f"Grad-CAM: {m_name}")
        else:
            # Fallback to single gradcam image
            if st.session_state.gradcam_image:
                st.image(st.session_state.gradcam_image, caption="Grad-CAM Heatmap", width='stretch')
            else:
                st.info("Grad-CAM visualization not available.")

    with col2:
        if st.session_state.explanation_text:
            st.markdown("**Model Explanation:**")
            st.markdown(st.session_state.explanation_text)


def render_educational_output():
    """Renders the educational/patient-friendly section."""
    if st.session_state.educational_output is None:
        return
    
    st.subheader("📚 Simplified Explanation")
    
    output = st.session_state.educational_output
    
    st.info(output["patient_summary"])
    
    st.markdown(f"**Severity Level:** {output['severity_layman']}")
    
    st.markdown(f"**Next Steps / Action Plan:** {output['next_steps_action_plan']}")


def render_knowledge_base():
    """Renders the knowledge base section."""
    if st.session_state.medical_summary is None:
        return
    
    st.subheader("📖 Medical Knowledge Base")
    
    summary = st.session_state.medical_summary
    
    if "error" in summary:
        st.error(summary["error"])
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Diagnosis:** {summary.get('Diagnosis', 'N/A')}")
        st.markdown(f"**ICD Code:** {summary.get('ICD_Code', 'N/A')}")
        st.markdown(f"**Severity:** {summary.get('Severity_Rating', 'N/A')}")
    
    with col2:
        st.markdown(f"**Definition:** {summary.get('Type_Definition', 'N/A')}")
        st.markdown(f"**Prognosis:** {summary.get('Long_Term_Prognosis', 'N/A')}")
    
    with st.expander("Treatment Guidelines"):
        for guideline in summary.get("Treatment_Guidelines", []):
            st.markdown(f"• {guideline}")
    
    # Render Gemini Explanation
    gemini_expl = st.session_state.get("gemini_explanation")
    if gemini_expl:
        st.markdown("---")
        st.subheader("Detailed Clinical Analysis")
        st.info(gemini_expl)


def render_chat_interface():
    """Renders the patient chat interface."""
    st.subheader("💬 Ask Questions")
    
    if st.session_state.patient_agent is None:
        st.info("Complete the analysis above to enable the chat feature.")
        return
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your diagnosis, treatment, or recovery..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.patient_agent.get_response(prompt)
                st.markdown(response)
        
        st.session_state.chat_messages.append({"role": "assistant", "content": response})


def run_analysis(image: Image.Image, config: dict, device):
    """Runs the full analysis pipeline."""
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models(config["checkpoint_dir"], config["selected_models"], device)
        # keep models in session state for explainability UI
        st.session_state.loaded_models = models
        st.session_state.models_loaded = True
    
    if not models:
        st.error("No models could be loaded. Please check your checkpoint directory.")
        return
    
    # Get primary model for single diagnosis
    # Prefer a non-YOLO / non-RAD-DINO model as DiagnosticAgent primary because
    # those have different inference pipelines that DiagnosticModule doesn't support.
    primary_model_name = list(models.keys())[0]
    primary_model = models[primary_model_name]
    for _pname, _pmodel in models.items():
        if not is_yolo_model(_pmodel) and not is_rad_dino_model(_pname):
            primary_model_name = _pname
            primary_model = _pmodel
            break

    # Determine conformal threshold (file overrides manual value)
    conformal_threshold = None
    if config.get("use_conformal"):
        conformal_threshold = read_threshold(config.get("conformal_threshold_path"))
        if conformal_threshold is None:
            conformal_threshold = float(config.get("conformal_threshold_value", 0.10))
            st.info(f"Using manual conformal threshold: {conformal_threshold:.6f}")
        else:
            st.info(f"Loaded conformal threshold from file: {conformal_threshold:.6f}")
    
    # Agent 1: Diagnostic Agent
    with st.spinner("Running primary diagnosis..."):
        if is_yolo_model(primary_model):
            # YOLO has its own inference pipeline
            probs = primary_model.predict_pil(image)
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            pred_class = _swap_prediction_label(CLASS_NAMES[pred_idx])
            probs_np = probs.copy()
            try:
                if "Transverse" in CLASS_NAMES and "Transverse Displaced" in CLASS_NAMES:
                    it, itd = CLASS_NAMES.index("Transverse"), CLASS_NAMES.index("Transverse Displaced")
                    probs_np[it], probs_np[itd] = probs_np[itd], probs_np[it]
                if "Oblique" in CLASS_NAMES and "Oblique Displaced" in CLASS_NAMES:
                    io, iod = CLASS_NAMES.index("Oblique"), CLASS_NAMES.index("Oblique Displaced")
                    probs_np[io], probs_np[iod] = probs_np[iod], probs_np[io]
            except ValueError:
                pass
            st.session_state.diagnosis_result = {
                "image_path": "in-memory-image",
                "fracture_detected": pred_class != "Healthy",
                "predicted_class": pred_class,
                "severity_type": pred_class,
                "confidence_score": confidence,
                "uncertainty_score": 1.0 - confidence,
                "all_probabilities": probs_np.tolist(),
                "all_probabilities_dict": {_swap_prediction_label(CLASS_NAMES[i]): float(probs[i]) for i in range(len(probs))},
            }
        elif is_rad_dino_model(primary_model_name):
            # RAD-DINO has its own preprocessing
            rad_tensor = get_rad_dino_input_tensor(image, device)
            with torch.no_grad():
                logits = primary_model(rad_tensor)
            probs_t = torch.softmax(logits, dim=1).squeeze(0)
            probs = probs_t.cpu().numpy()
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            pred_class = _swap_prediction_label(CLASS_NAMES[pred_idx])
            probs_np = probs.copy()
            try:
                if "Transverse" in CLASS_NAMES and "Transverse Displaced" in CLASS_NAMES:
                    it, itd = CLASS_NAMES.index("Transverse"), CLASS_NAMES.index("Transverse Displaced")
                    probs_np[it], probs_np[itd] = probs_np[itd], probs_np[it]
                if "Oblique" in CLASS_NAMES and "Oblique Displaced" in CLASS_NAMES:
                    io, iod = CLASS_NAMES.index("Oblique"), CLASS_NAMES.index("Oblique Displaced")
                    probs_np[io], probs_np[iod] = probs_np[iod], probs_np[io]
            except ValueError:
                pass
            st.session_state.diagnosis_result = {
                "image_path": "in-memory-image",
                "fracture_detected": pred_class != "Healthy",
                "predicted_class": pred_class,
                "severity_type": pred_class,
                "confidence_score": confidence,
                "uncertainty_score": 1.0 - confidence,
                "all_probabilities": probs_np.tolist(),
                "all_probabilities_dict": {_swap_prediction_label(CLASS_NAMES[i]): float(probs[i]) for i in range(len(probs))},
            }
        else:
            diagnostic_agent = DiagnosticAgent(
                class_names=CLASS_NAMES, 
                model=primary_model, 
                device=device, 
                conformal_threshold=conformal_threshold
            )
            st.session_state.diagnosis_result = diagnostic_agent.diagnose(image)
    
    # Agent 2: Ensemble Agent
    if len(models) > 1:
        with st.spinner("Running ensemble analysis..."):
            # Use stacking if selected
            if config.get('ensemble_mode') == 'stacking' and os.path.exists(config.get('stacker_path', '')):
                import joblib
                stacker = joblib.load(config.get('stacker_path'))
                # create ensemble agent with stacking mode: pass stacker as additional attribute
                ensemble_agent = ModelEnsembleAgent(
                    class_names=CLASS_NAMES,
                    models=models,
                    device=device,
                    conformal_threshold=conformal_threshold
                )
                # monkey-patch stacker into agent for use
                ensemble_agent.stacker = stacker
                st.session_state.ensemble_result = ensemble_agent.run_ensemble(image, use_stacking=True)
            else:
                ensemble_agent = ModelEnsembleAgent(
                    class_names=CLASS_NAMES,
                    models=models,
                    device=device,
                    conformal_threshold=conformal_threshold
                )
                st.session_state.ensemble_result = ensemble_agent.run_ensemble(image)
    else:
        # Use single model result as ensemble result
        st.session_state.ensemble_result = {
            "ensemble_prediction": st.session_state.diagnosis_result["predicted_class"],
            "ensemble_confidence": st.session_state.diagnosis_result["confidence_score"],
            "individual_predictions": {primary_model_name: {
                "class": st.session_state.diagnosis_result["predicted_class"],
                "confidence": st.session_state.diagnosis_result["confidence_score"]
            }},
            "fracture_detected": st.session_state.diagnosis_result["fracture_detected"],
            "all_probabilities": st.session_state.diagnosis_result["all_probabilities"],
            "all_probabilities_dict": st.session_state.diagnosis_result.get("all_probabilities_dict")
        }
        # propagate conformal set from single-model diagnosis if present
        if "conformal_set" in st.session_state.diagnosis_result:
            st.session_state.ensemble_result["conformal_set"] = st.session_state.diagnosis_result["conformal_set"]
            st.session_state.ensemble_result["conformal_threshold"] = st.session_state.diagnosis_result.get("conformal_threshold")
    
    # Agent 3: Explainability Agent
    with st.spinner("Generating explanation..."):
        # Generate per-model Grad-CAM visualizations (store as PIL images in session state)
        gradcam_images = {}
        for m_name, m_model in models.items():
            # Skip YOLO and RAD-DINO – their architectures are incompatible with Grad-CAM
            if is_yolo_model(m_model) or is_rad_dino_model(m_name):
                continue
            try:
                explain_agent = ExplainabilityAgent(m_model, CLASS_NAMES, device, body_part="bone")
                pred_class = st.session_state.ensemble_result["ensemble_prediction"]
                pred_idx = CLASS_NAMES.index(pred_class) if pred_class in CLASS_NAMES else None
                cam_array = explain_agent.generate_gradcam(image, pred_idx)
                if cam_array is not None:
                    gradcam_images[m_name] = explain_agent.visualize_gradcam(image, cam_array)
            except Exception:
                # Skip models that fail explainability
                continue

        # Save per-model gradcam images (may be empty if not available)
        st.session_state.gradcam_images = gradcam_images

        # For backward compatibility, keep a single gradcam_image if at least one exists
        if gradcam_images:
            # pick primary model image if available else first
            st.session_state.gradcam_image = gradcam_images.get(primary_model_name, next(iter(gradcam_images.values())))
        else:
            st.session_state.gradcam_image = None

        # Generate textual explanation using primary model's cam if present
        primary_cam = None
        if primary_model_name in gradcam_images:
            # convert PIL to numpy array for explanation heuristics
            primary_cam = np.array(gradcam_images[primary_model_name].convert('L')) / 255.0

        # Find a standard model for ExplainabilityAgent text generation (not YOLO/RAD-DINO)
        explain_model = primary_model
        for _ename, _emodel in models.items():
            if not is_yolo_model(_emodel) and not is_rad_dino_model(_ename):
                explain_model = _emodel
                break
        explain_agent_primary = ExplainabilityAgent(explain_model, CLASS_NAMES, device, body_part="bone")
        st.session_state.explanation_text = explain_agent_primary.generate_explanation(
            st.session_state.ensemble_result, primary_cam
        )
    
    # Agent 4: Educational Agent
    with st.spinner("Preparing patient information..."):
        edu_agent = EducationalAgent(doctor_name="your healthcare provider")
        st.session_state.educational_output = edu_agent.translate(
            st.session_state.ensemble_result,
            st.session_state.explanation_text or "",
            image=image,
            gradcam_image=st.session_state.gradcam_image
        )
    
    # Agent 5: Knowledge Agent
    with st.spinner("Retrieving medical knowledge..."):
        knowledge_agent = KnowledgeAgent()
        st.session_state.medical_summary = knowledge_agent.get_medical_summary(
            st.session_state.ensemble_result["ensemble_prediction"],
            st.session_state.ensemble_result["ensemble_confidence"]
        )
        
        # 2. RAG + Gemini Explanation
        print("[DEBUG] Starting RAG + Gemini Explanation process...")
        label = st.session_state.ensemble_result["ensemble_prediction"]
        print(f"[DEBUG] Explanation Target Label: {label}")
        
        # Only proceed if we have a valid summary
        if "error" not in st.session_state.medical_summary:
            print("[DEBUG] Medical summary is valid.")
            try:
                # Retrieve context
                print("[DEBUG] Calling retrieve_sources...")
                relevant_docs = knowledge_agent.retrieve_sources(label) 
                print(f"[DEBUG] retrieve_sources returned {len(relevant_docs)} items.")
                
                # Generate explanation
                print("[DEBUG] Calling generate_explanation_with_gemini...")
                st.session_state.gemini_explanation = knowledge_agent.generate_explanation_with_gemini(
                    st.session_state.medical_summary,
                    relevant_docs,
                    audience="clinician"
                )
                print(f"[DEBUG] Gemini explanation result length: {len(st.session_state.gemini_explanation) if st.session_state.gemini_explanation else 'None'}")
            except Exception as e:
                # Log error but don't crash usage flow
                print(f"[DEBUG] ERROR in explanation pipeline: {e}")
                logger.error(f"Failed to generate Gemini explanation: {e}")
                st.session_state.gemini_explanation = None
        else:
            print(f"[DEBUG] Medical summary Error: {st.session_state.medical_summary.get('error')}")
            st.session_state.gemini_explanation = None
    
    # Agent 5.5: Critic Agent (if enabled)
    if config.get("enable_critic", True):
        # Only run if we have a valid result
        if st.session_state.ensemble_result and st.session_state.medical_summary and "error" not in st.session_state.medical_summary:
            with st.spinner("Critic Agent reviewing diagnosis..."):
                try:
                     # Remove lazy import since we handle it at top level with sys.path fix
                     # from medai.agents.critic_agent import CriticAgent
                     # from medai.utils.consensus import evaluate_consensus
                     
                     st.info("Critic Agent active: Consulting MedGemma regarding visual evidence...")
                     
                     if CriticAgent is None:
                        raise ImportError("CriticAgent module could not be imported. Check logs.")

                     # Check environment specifically for Streamlit context
                     mode = os.getenv("MEDGEMMA_MODE", "hf_spaces")
                     # Note: On simple Streamlit hosting, local mode usually fails memory.
                     
                     critic = CriticAgent(mode=mode)
                     
                     diagnosis = st.session_state.ensemble_result["ensemble_prediction"]
                     conf = st.session_state.ensemble_result["ensemble_confidence"]
                     definition = st.session_state.medical_summary.get("Type_Definition") or "No definition"
                     
                     review = critic.review_diagnosis(image, diagnosis, conf, definition)
                     consensus = evaluate_consensus(
                         {"label": diagnosis, "confidence": conf}, review
                     )
                     
                     st.session_state.critic_review = review
                     st.session_state.consensus = consensus
                     st.success("Critic Agent check complete.")
                except Exception as e:
                    st.error(f"Critic Agent failed: {e}")
                    # Don't block the rest of the flow
                    st.session_state.critic_review = {"error": str(e)}
                    st.session_state.consensus = None

    # Agent 6: Patient Interaction Agent
    if "error" not in st.session_state.medical_summary:
        st.session_state.patient_agent = PatientInteractionAgent(
            st.session_state.medical_summary,
            config["patient_info"]
        )
        st.session_state.chat_messages = [{
            "role": "assistant",
            "content": f"Hello! I've analyzed your X-ray and found: **{st.session_state.ensemble_result['ensemble_prediction']}** "
                       f"(Confidence: {st.session_state.ensemble_result['ensemble_confidence']:.1%}). "
                       f"How can I help answer your questions about this diagnosis?"
        }]


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="MedAI - Fracture Detection System",
        page_icon="🦴",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Header
    st.title("🦴 MedAI - Multi-Agent Fracture Detection System")
    st.markdown(
        "An AI-powered system for detecting and explaining bone fractures using "
        "ensemble deep learning and explainable AI techniques."
    )
    
    # Check dependencies
    missing_deps = []
    if not TIMM_AVAILABLE:
        missing_deps.append("timm")
    if not GRADCAM_AVAILABLE:
        missing_deps.append("pytorch-grad-cam")
    if not CHROMADB_AVAILABLE:
        missing_deps.append("chromadb")
    
    if missing_deps:
        st.warning(f"Some features may be limited. Missing optional dependencies: {', '.join(missing_deps)}")
    
    # Device info
    device = get_device()
    st.sidebar.info(f"🖥️ Device: {device}")
    
    # Sidebar configuration
    config = render_sidebar()
    # Ensemble mode selection
    ensemble_mode = st.sidebar.selectbox("Ensemble Mode", options=["weighted", "stacking"], index=0,
                                         help="Choose 'stacking' to use a trained meta-classifier saved at outputs/ensemble/stacker.joblib")
    stacker_path = st.sidebar.text_input("Stacker path", value="outputs/ensemble/stacker.joblib")
    config['ensemble_mode'] = ensemble_mode
    config['stacker_path'] = stacker_path
    
    st.markdown("---")
    
    # Main content
    col_upload, col_results = st.columns([1, 2])
    
    with col_upload:
        image = render_image_upload()
        
        if image is not None:
            if st.button("🔬 Analyze Image", type="primary", width='stretch'):
                run_analysis(image, config, device)
                st.rerun()
    
    with col_results:
        render_diagnosis_results()
        render_critic_review()
    
    st.markdown("---")
    
    # Explainability and Education
    col_explain, col_edu = st.columns(2)
    
    with col_explain:
        render_explainability()
    
    with col_edu:
        render_educational_output()
    
    st.markdown("---")
    
    # Knowledge Base
    render_knowledge_base()
    
    st.markdown("---")
    
    # Chat Interface
    render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This is an AI-assisted tool for educational purposes only. "
        "It is not intended to replace professional medical advice, diagnosis, or treatment. "
        "Always consult with a qualified healthcare provider for medical decisions."
    )


if __name__ == "__main__":
    main()
