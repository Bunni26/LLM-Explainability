import pytest
import numpy as np
from explanations import LocalExplanationGenerator
from data.sample_prompts import (
    FEATURE_ATTRIBUTION_SAMPLES,
    ATTENTION_ANALYSIS_SAMPLES
)

@pytest.fixture
def explanation_generator():
    return LocalExplanationGenerator()

def test_feature_attribution():
    generator = LocalExplanationGenerator()
    sample = FEATURE_ATTRIBUTION_SAMPLES["sentiment"]
    
    # Test integrated gradients
    result = generator.generate_feature_attribution(
        sample["text"],
        method="integrated_gradients"
    )
    
    assert "tokens" in result
    assert "scores" in result
    assert len(result["tokens"]) == len(result["scores"])
    assert all(isinstance(score, (int, float)) for score in result["scores"])

def test_attention_visualization():
    generator = LocalExplanationGenerator()
    sample = ATTENTION_ANALYSIS_SAMPLES["text"]
    
    result = generator.visualize_attention(sample["input"])
    
    assert "attention_weights" in result
    assert "tokens" in result
    assert "layer_idx" in result
    assert isinstance(result["attention_weights"], np.ndarray)
    assert len(result["tokens"]) > 0

def test_invalid_attribution_method():
    generator = LocalExplanationGenerator()
    sample = FEATURE_ATTRIBUTION_SAMPLES["sentiment"]
    
    with pytest.raises(ValueError):
        generator.generate_feature_attribution(
            sample["text"],
            method="invalid_method"
        ) 