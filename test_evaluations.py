import pytest
import numpy as np
from evaluations import ExplanationEvaluator
from data.sample_prompts import EXPLANATION_EVALUATION_SAMPLES

@pytest.fixture
def evaluator():
    return ExplanationEvaluator()

def test_plausibility_evaluation():
    evaluator = ExplanationEvaluator()
    sample = EXPLANATION_EVALUATION_SAMPLES["plausibility"]
    
    result = evaluator.evaluate_plausibility(
        sample["explanation"],
        context=sample["context"]
    )
    
    assert "explanation" in result
    assert "plausibility_scores" in result
    assert "model_assessment" in result
    assert "overall_score" in result
    assert isinstance(result["plausibility_scores"], dict)
    assert 0 <= result["overall_score"] <= 1

def test_faithfulness_evaluation():
    evaluator = ExplanationEvaluator()
    sample = EXPLANATION_EVALUATION_SAMPLES["faithfulness"]
    
    result = evaluator.evaluate_faithfulness(
        sample["explanation"],
        sample["reference"],
        input_text=sample["input"]
    )
    
    assert "explanation" in result
    assert "reference" in result
    assert "semantic_similarity" in result
    assert "model_assessment" in result
    assert "faithfulness_score" in result
    assert isinstance(result["faithfulness_score"], float)
    assert 0 <= result["faithfulness_score"] <= 1

def test_counterfactual_generation():
    evaluator = ExplanationEvaluator()
    sample = EXPLANATION_EVALUATION_SAMPLES["plausibility"]
    
    result = evaluator._generate_counterfactuals(
        sample["explanation"],
        sample["context"],
        num_counterfactuals=2
    )
    
    assert isinstance(result, list)
    assert len(result) == 2
    for cf in result:
        assert "counterfactual" in cf
        assert "similarity_to_original" in cf
        assert isinstance(cf["similarity_to_original"], float)
        assert 0 <= cf["similarity_to_original"] <= 1

def test_semantic_similarity():
    evaluator = ExplanationEvaluator()
    text1 = "The model predicts high risk."
    text2 = "The system indicates elevated risk levels."
    
    similarity = evaluator._calculate_semantic_similarity(text1, text2)
    
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1 