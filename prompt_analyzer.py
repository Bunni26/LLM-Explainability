from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from ..models.groq_helper import GroqHelper

class PromptAnalyzer:
    def __init__(self):
        """Initialize the prompt analyzer."""
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
    
    def analyze_in_context_learning(
        self,
        prompt: str,
        examples: List[str],
        variations: Optional[List[str]] = None
    ) -> Dict:
        """Analyze in-context learning behavior with examples."""
        # Generate response with examples
        response_with_examples = self.groq_client.generate_completion(
            "\n".join(examples + [prompt])
        )
        
        # Generate response without examples
        response_without_examples = self.groq_client.generate_completion(prompt)
        
        # If variations provided, analyze consistency
        variation_responses = []
        if variations:
            for variation in variations:
                variation_responses.append(
                    self.groq_client.generate_completion(variation)
                )
        
        return {
            "with_examples": response_with_examples,
            "without_examples": response_without_examples,
            "variation_responses": variation_responses if variations else None,
            "analysis": self._analyze_responses(
                response_with_examples["text"],
                response_without_examples["text"],
                [r["text"] for r in variation_responses] if variations else None
            )
        }
    
    def analyze_chain_of_thought(
        self,
        prompt: str,
        examples: Optional[List[str]] = None
    ) -> Dict:
        """Analyze chain-of-thought reasoning process."""
        # Generate standard response
        standard_response = self.groq_client.generate_completion(prompt)
        
        # Generate CoT response
        cot_response = self.groq_client.analyze_chain_of_thought(
            prompt,
            examples
        )
        
        return {
            "standard_response": standard_response,
            "cot_response": cot_response,
            "analysis": self._analyze_reasoning_steps(cot_response["text"])
        }
    
    def detect_hallucination(
        self,
        response: str,
        reference: Optional[str] = None
    ) -> Dict:
        """Detect potential hallucinations in model response."""
        # Generate explanation of the response
        explanation_prompt = f"""
        Analyze the following response for potential hallucinations or unsupported claims.
        Identify any statements that:
        1. Contain factual claims without clear support
        2. Make logical leaps without justification
        3. Introduce information not present in the context
        
        Response: {response}
        
        Reference (if available): {reference or 'No reference provided'}
        
        Provide a detailed analysis of potential hallucinations:
        """
        
        analysis = self.groq_client.generate_completion(
            explanation_prompt,
            temperature=0.2
        )
        
        return {
            "response": response,
            "reference": reference,
            "hallucination_analysis": analysis["text"]
        }
    
    def estimate_uncertainty(
        self,
        response: str,
        num_samples: int = 5
    ) -> Dict:
        """Estimate model uncertainty through multiple samples."""
        samples = []
        for _ in range(num_samples):
            sample = self.groq_client.generate_completion(
                response,
                temperature=0.7
            )
            samples.append(sample["text"])
        
        # Calculate similarity between samples
        embeddings = self._get_embeddings(samples)
        similarities = cosine_similarity(embeddings)
        
        return {
            "samples": samples,
            "mean_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
            "uncertainty_score": 1 - np.mean(similarities)
        }
    
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
    
    def _analyze_responses(
        self,
        with_examples: str,
        without_examples: str,
        variations: Optional[List[str]] = None
    ) -> Dict:
        """Analyze differences between responses."""
        # Get embeddings
        texts = [with_examples, without_examples]
        if variations:
            texts.extend(variations)
        
        embeddings = self._get_embeddings(texts)
        similarities = cosine_similarity(embeddings)
        
        return {
            "example_impact_score": 1 - similarities[0, 1],
            "variation_consistency": np.mean(similarities[2:, 2:]) if variations else None,
            "semantic_similarity": similarities.tolist()
        }
    
    def _analyze_reasoning_steps(self, cot_response: str) -> Dict:
        """Analyze the reasoning steps in a chain-of-thought response."""
        # Split into steps
        steps = [s.strip() for s in cot_response.split("\n") if s.strip()]
        
        # Analyze coherence between steps
        step_embeddings = self._get_embeddings(steps)
        step_similarities = cosine_similarity(step_embeddings)
        
        return {
            "num_steps": len(steps),
            "steps": steps,
            "step_coherence": np.mean(
                [step_similarities[i, i+1] for i in range(len(steps)-1)]
            ),
            "step_similarities": step_similarities.tolist()
        } 