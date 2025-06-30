"""Utility functions for the LLM Explainability System."""

import numpy as np
from typing import List, Dict, Union, Optional
import torch
from pathlib import Path
import json
import logging
from datetime import datetime
from ..config import CACHE_CONFIG, LOGGING_CONFIG

# Set up logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def setup_cache_dir(cache_dir: Union[str, Path]) -> Path:
    """Set up and validate cache directory."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def cache_result(
    key: str,
    data: Dict,
    cache_dir: Union[str, Path],
    ttl: Optional[int] = None
) -> None:
    """Cache results with timestamp."""
    if not CACHE_CONFIG["enabled"]:
        return
    
    cache_dir = setup_cache_dir(cache_dir)
    cache_file = cache_dir / f"{key}.json"
    
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    
    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
    except Exception as e:
        logger.error(f"Error caching result: {str(e)}")

def get_cached_result(
    key: str,
    cache_dir: Union[str, Path],
    ttl: Optional[int] = None
) -> Optional[Dict]:
    """Retrieve cached results if not expired."""
    if not CACHE_CONFIG["enabled"]:
        return None
    
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"{key}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
        
        # Check TTL
        if ttl is not None:
            timestamp = datetime.fromisoformat(cache_data["timestamp"])
            age = (datetime.now() - timestamp).total_seconds()
            if age > ttl:
                return None
        
        return cache_data["data"]
    except Exception as e:
        logger.error(f"Error reading cache: {str(e)}")
        return None

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range."""
    if len(scores) == 0:
        return scores
    min_val = np.min(scores)
    max_val = np.max(scores)
    if min_val == max_val:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)

def moving_average(
    data: np.ndarray,
    window_size: int = 3
) -> np.ndarray:
    """Calculate moving average of data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def cosine_similarity_matrix(
    embeddings: np.ndarray
) -> np.ndarray:
    """Calculate cosine similarity matrix between embeddings."""
    normalized = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    return normalized @ normalized.T

def batch_encode_text(
    texts: List[str],
    tokenizer,
    max_length: int = 512,
    batch_size: int = 32
) -> Dict:
    """Batch encode texts using tokenizer."""
    all_inputs = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        all_inputs.append(inputs)
    
    # Combine batches
    combined = {
        key: torch.cat([batch[key] for batch in all_inputs])
        for key in all_inputs[0].keys()
    }
    
    return combined

def format_time_delta(seconds: float) -> str:
    """Format time delta in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_memory_usage() -> str:
    """Get current memory usage."""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return f"{memory_mb:.1f}MB"
    except ImportError:
        return "N/A"

def log_execution_time(func):
    """Decorator to log function execution time."""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        duration = (datetime.now() - start_time).total_seconds()
        logger.debug(
            f"{func.__name__} executed in {format_time_delta(duration)} "
            f"(Memory: {get_memory_usage()})"
        )
        return result
    return wrapper 