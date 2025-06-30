import pytest
import numpy as np
from prompting import PromptAnalyzer
from data.sample_prompts import CHAIN_OF_THOUGHT_SAMPLES

@pytest.fixture
def prompt_analyzer():
    return PromptAnalyzer()

def test_chain_of_thought_analysis():
    analyzer = PromptAnalyzer()
    sample = CHAIN_OF_THOUGHT_SAMPLES["math"]
    
    result = analyzer.analyze_chain_of_thought(
        sample["question"],
        sample["examples"]
    )
    
    assert "standard_response" in result
    assert "cot_response" in result
    assert "analysis" in result
    assert isinstance(result["analysis"], dict)

def test_uncertainty_estimation():
    analyzer = PromptAnalyzer()
    text = "What is the capital of France?"
    
    result = analyzer.estimate_uncertainty(text, num_samples=3)
    
    assert "samples" in result
    assert "mean_similarity" in result
    assert "std_similarity" in result
    assert "uncertainty_score" in result
    assert len(result["samples"]) == 3
    assert 0 <= result["uncertainty_score"] <= 1

def test_hallucination_detection():
    analyzer = PromptAnalyzer()
    response = "The population of Paris is exactly 2,161,000."
    reference = "Paris has approximately 2.2 million inhabitants as of 2020."
    
    result = analyzer.detect_hallucination(response, reference)
    
    assert "response" in result
    assert "reference" in result
    assert "hallucination_analysis" in result
    assert isinstance(result["hallucination_analysis"], str) 