"""
Model Downloader for MedAI Streamlit Deployment
Downloads model weights from Hugging Face Hub on first run.
"""

import os
import streamlit as st
import requests
from pathlib import Path

# Hugging Face Hub configuration
HF_REPO_ID = "ACM-Research-DJSCE/medai-fracture-models"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO_ID}/resolve/main"

# Model configuration with Hugging Face URLs
MODEL_REGISTRY = {
    "swin": {
        "filename": "best_swin.pth",
        "size_mb": 195,
        "url": f"{HF_BASE_URL}/best_swin.pth",
    },
    "densenet169": {
        "filename": "best_densenet169.pth",
        "size_mb": 151,
        "url": f"{HF_BASE_URL}/best_densenet169.pth",
    },
    "efficientnetv2": {
        "filename": "best_efficientnetv2.pth",
        "size_mb": 49,
        "url": f"{HF_BASE_URL}/best_efficientnetv2.pth",
    },
    "hypercolumn_cbam_densenet169": {
        "filename": "best_hypercolumn_cbam_densenet169.pth",
        "size_mb": 63,
        "url": f"{HF_BASE_URL}/best_hypercolumn_cbam_densenet169.pth",
    },
    "hypercolumn_cbam_densenet169_focal": {
        "filename": "best_hypercolumn_cbam_densenet169_focal.pth",
        "size_mb": 63,
        "url": f"{HF_BASE_URL}/best_hypercolumn_cbam_densenet169_focal.pth",
    },
    "hypercolumn_cbam_densenet169_old": {
        "filename": "best_hypercolumn_cbam_densenet169_old.pth",
        "size_mb": 63,
        "url": f"{HF_BASE_URL}/best_hypercolumn_cbam_densenet169_old.pth",
    },
    "hypercolumn_densenet169": {
        "filename": "best_hypercolumn_densenet169.pth",
        "size_mb": 63,
        "url": f"{HF_BASE_URL}/best_hypercolumn_densenet169.pth",
    },
    "hypercolumn_densenet169_old": {
        "filename": "best_hypercolumn_densenet169_old.pth",
        "size_mb": 63,
        "url": f"{HF_BASE_URL}/best_hypercolumn_densenet169_old.pth",
    },
    "mobilenetv2": {
        "filename": "best_mobilenetv2.pth",
        "size_mb": 27,
        "url": f"{HF_BASE_URL}/best_mobilenetv2.pth",
    },
    "maxvit": {
        "filename": "best_maxvit.pth",
        "size_mb": 366,
        "url": f"{HF_BASE_URL}/best_maxvit.pth",
    },
}


def get_model_path(model_name: str, models_dir: str = "./models") -> str:
    """Returns the path where a model should be stored."""
    if model_name not in MODEL_REGISTRY:
        return os.path.join(models_dir, f"best_{model_name}.pth")
    return os.path.join(models_dir, MODEL_REGISTRY[model_name]["filename"])


def is_model_downloaded(model_name: str, models_dir: str = "./models") -> bool:
    """Checks if a model is already downloaded."""
    path = get_model_path(model_name, models_dir)
    return os.path.exists(path)


def download_file_with_progress(url: str, destination: str, description: str = "Downloading"):
    """Downloads a file with a Streamlit progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    # Create directory if needed
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    
    progress_bar = st.progress(0, text=description)
    downloaded = 0
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = downloaded / total_size
                    progress_bar.progress(progress, text=f"{description} ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)")
    
    progress_bar.progress(1.0, text=f"{description} - Complete!")


def download_model(model_name: str, models_dir: str = "./models") -> bool:
    """Downloads a single model if not already present."""
    if model_name not in MODEL_REGISTRY:
        st.warning(f"Model '{model_name}' not found in registry.")
        return False
    
    model_info = MODEL_REGISTRY[model_name]
    
    if not model_info.get("url"):
        st.warning(f"No download URL configured for '{model_name}'.")
        return False
    
    destination = get_model_path(model_name, models_dir)
    
    if os.path.exists(destination):
        return True
    
    try:
        download_file_with_progress(
            model_info["url"],
            destination,
            f"Downloading {model_name} ({model_info['size_mb']}MB)"
        )
        return True
    except Exception as e:
        st.error(f"Failed to download {model_name}: {e}")
        return False


def ensure_models_available(model_names: list, models_dir: str = "./models") -> dict:
    """
    Ensures requested models are available, downloading if necessary.
    Returns dict of {model_name: path} for available models.
    """
    available = {}
    
    for model_name in model_names:
        path = get_model_path(model_name, models_dir)
        
        if os.path.exists(path):
            available[model_name] = path
        elif model_name in MODEL_REGISTRY and MODEL_REGISTRY[model_name].get("url"):
            if download_model(model_name, models_dir):
                available[model_name] = path
        else:
            # Model file not found and no download URL
            pass
    
    return available


def get_available_local_models(models_dir: str = "./models") -> list:
    """Returns list of models that are available locally."""
    available = []
    
    if not os.path.exists(models_dir):
        return available
    
    for model_name in MODEL_REGISTRY.keys():
        if is_model_downloaded(model_name, models_dir):
            available.append(model_name)
    
    # Also check for any .pth files that might be there
    for f in os.listdir(models_dir):
        if f.endswith('.pth'):
            # Extract model name from filename
            name = f.replace('best_', '').replace('.pth', '')
            if name not in available:
                available.append(name)
    
    return available
