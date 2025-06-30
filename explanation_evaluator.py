from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
from ..models.groq_helper import GroqHelper

class ExplanationEvaluator:
    def __init__(self):
        """Initialize the explanation evaluator."""
        self.groq_client = GroqHelper()
        self.tokenizer = None
        self.model = None
        
    def _ensure_model_loaded(self):
        """Ensure the model and tokenizer are loaded for embeddings."""
        if self.tokenizer is None or self.model is None:
            try:
                # Use a smaller model for embeddings
                model_name = "sentence-transformers/all-mpnet-base-v2"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
            except Exception as e:
                raise Exception(f"Error loading model: {str(e)}")
    
    def evaluate_plausibility(
        self,
        explanation: str,
        context: Optional[str] = None
    ) -> Dict:
        """Evaluate the plausibility of an explanation."""
        # Generate plausibility analysis
        plausibility_scores = self._analyze_plausibility_aspects(explanation)
        
        # Get model's assessment
        model_assessment = self.groq_client.evaluate_explanation(
            explanation,
            eval_type="plausibility"
        )
        
        return {
            "explanation": explanation,
            "context": context,
            "plausibility_scores": plausibility_scores,
            "model_assessment": model_assessment["text"],
            "overall_score": np.mean(list(plausibility_scores.values()))
        }
    
    def evaluate_faithfulness(
        self,
        explanation: str,
        reference: str,
        input_text: Optional[str] = None
    ) -> Dict:
        """Evaluate the faithfulness of an explanation to a reference."""
        # Calculate semantic similarity
        similarity_score = self._calculate_semantic_similarity(
            explanation,
            reference
        )
        
        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(
            explanation,
            input_text
        ) if input_text else None
        
        # Get model's assessment
        model_assessment = self.groq_client.evaluate_explanation(
            explanation,
            reference=reference,
            eval_type="faithfulness"
        )
        
        return {
            "explanation": explanation,
            "reference": reference,
            "semantic_similarity": similarity_score,
            "counterfactuals": counterfactuals,
            "model_assessment": model_assessment["text"],
            "faithfulness_score": similarity_score
        }
    
    def _analyze_plausibility_aspects(self, explanation: str) -> Dict[str, float]:
        """Analyze different aspects of plausibility."""
        aspects = {
            "logical_coherence": """
            Evaluate the logical coherence of this explanation.
            Consider:
            1. Are the arguments well-structured?
            2. Do the conclusions follow from the premises?
            3. Are there any logical fallacies?
            
            Explanation: {text}
            
            Rate the logical coherence from 0 to 1:
            """,
            
            "common_sense": """
            Evaluate how well this explanation aligns with common sense knowledge.
            Consider:
            1. Does it contradict well-known facts?
            2. Are the claims reasonable?
            3. Would a typical person find this believable?
            
            Explanation: {text}
            
            Rate the common sense alignment from 0 to 1:
            """,
            
            "internal_consistency": """
            Evaluate the internal consistency of this explanation.
            Consider:
            1. Are there any contradictions?
            2. Do all parts support each other?
            3. Is the reasoning consistent throughout?
            
            Explanation: {text}
            
            Rate the internal consistency from 0 to 1:
            """
        }
        
        scores = {}
        for aspect, prompt in aspects.items():
            response = self.groq_client.generate_completion(
                prompt.format(text=explanation),
                temperature=0.1
            )
            try:
                # Extract numerical score from response
                score = float(response["text"].strip().split()[-1])
                scores[aspect] = min(max(score, 0), 1)  # Ensure score is between 0 and 1
            except:
                scores[aspect] = 0.5  # Default score if parsing fails
        
        return scores
    
    def _calculate_semantic_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Calculate semantic similarity between two texts."""
        self._ensure_model_loaded()
        
        # Get embeddings
        embeddings = self._get_embeddings([text1, text2])
        similarity = cosine_similarity(embeddings)[0, 1]
        
        return float(similarity)
    
    def _generate_counterfactuals(
        self,
        explanation: str,
        input_text: str,
        num_counterfactuals: int = 3
    ) -> List[Dict]:
        """Generate counterfactual explanations."""
        prompt = f"""
        Generate {num_counterfactuals} alternative explanations that are slightly different
        from the original explanation but could also be plausible. Consider changing key
        aspects while maintaining overall coherence.
        
        Original input: {input_text}
        Original explanation: {explanation}
        
        Generate {num_counterfactuals} alternative explanations:
        """
        
        response = self.groq_client.generate_completion(prompt)
        
        # Split response into counterfactuals
        counterfactuals = [
            c.strip() for c in response["text"].split("\n")
            if c.strip() and c.strip() != explanation
        ][:num_counterfactuals]
        
        # Calculate similarity for each counterfactual
        similarities = [
            self._calculate_semantic_similarity(explanation, cf)
            for cf in counterfactuals
        ]
        
        return [
            {
                "counterfactual": cf,
                "similarity_to_original": sim
            }
            for cf, sim in zip(counterfactuals, similarities)
        ]
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts."""
        self._ensure_model_loaded()
        
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding.numpy())
        
        return np.vstack(embeddings) 