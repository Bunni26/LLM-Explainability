"""Configuration settings for the LLM Explainability System."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Model settings
MODEL_CONFIG = {
    "default_model": os.getenv("DEFAULT_MODEL", "llama3-8b-8192"),
    "max_tokens": int(os.getenv("MAX_TOKENS", "4096")),
    "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    "embedding_model": "sentence-transformers/all-mpnet-base-v2"
}

# Feature attribution settings
ATTRIBUTION_CONFIG = {
    "methods": ["integrated_gradients", "shap"],
    "num_samples": 100,  # For integrated gradients
    "baseline": "zero"  # For integrated gradients
}

# Attention analysis settings
ATTENTION_CONFIG = {
    "max_sequence_length": 512,
    "include_layers": "all",  # or list of layer indices
    "aggregation": "mean"  # or "max", "min"
}

# Evaluation settings
EVALUATION_CONFIG = {
    "plausibility": {
        "aspects": ["logical_coherence", "common_sense", "internal_consistency"],
        "min_score": 0.0,
        "max_score": 1.0
    },
    "faithfulness": {
        "similarity_threshold": 0.7,
        "num_counterfactuals": 3
    }
}

# Visualization settings
VIZ_CONFIG = {
    "colorscales": {
        "attention": "Viridis",
        "attribution": "RdBu",
        "similarity": "Viridis"
    },
    "plot_dimensions": {
        "default_width": 800,
        "default_height": 600
    }
}

# Cache settings
CACHE_CONFIG = {
    "enabled": True,
    "max_size": "1GB",
    "ttl": 3600  # 1 hour
}

# Debug settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# API settings
API_CONFIG = {
    "groq": {
        "api_key": os.getenv("GROQ_API_KEY"),
        "timeout": 30,
        "max_retries": 3
    }
}

# Logging settings
LOGGING_CONFIG = {
    "level": "INFO" if not DEBUG else "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": BASE_DIR / "logs" / "app.log"
}

# ✅ Valid Groq-supported model list
VALID_GROQ_MODELS = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it"
]