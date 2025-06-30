"""Global explanation methods for LLM interpretability."""

import torch
import numpy as np
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from models.groq_helper import llama_infer
from config import MODEL_CONFIG

class GlobalExplanationGenerator:
    def __init__(self):
        """Initialize the global explanation generator."""
        self.groq_client = GroqHelper()
        self.tokenizer = None
        self.model = None
        
    def _ensure_model_loaded(self):
        """Ensure the model and tokenizer are loaded."""
        if self.tokenizer is None or self.model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["embedding_model"])
                self.model = AutoModelForCausalLM.from_pretrained(MODEL_CONFIG["embedding_model"])
            except Exception as e:
                raise Exception(f"Error loading model: {str(e)}")
    
    def analyze_mechanistic_patterns(
        self,
        texts: List[str],
        layer_idx: Optional[int] = None
    ) -> Dict:
        """Analyze mechanistic patterns in model activations."""
        self._ensure_model_loaded()
        
        # Get model activations
        activations = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Get activations from specified layer or last layer
                layer = layer_idx if layer_idx is not None else -1
                activation = outputs.hidden_states[layer].mean(dim=1)  # Average over sequence length
                activations.append(activation.numpy())
        
        # Stack activations
        activations = np.vstack(activations)
        
        # Perform PCA to find principal components
        pca = PCA(n_components=min(10, len(texts)))
        components = pca.fit_transform(activations)
        
        # Cluster activations
        kmeans = KMeans(n_clusters=min(5, len(texts)))
        clusters = kmeans.fit_predict(components)
        
        return {
            "components": components.tolist(),
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "clusters": clusters.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist()
        }
    
    def analyze_concept_attribution(
        self,
        texts: List[str],
        concepts: List[str]
    ) -> Dict:
        """Analyze attribution of text to high-level concepts."""
        # Get concept embeddings using Groq
        concept_embeddings = []
        for concept in concepts:
            prompt = f"""
            Explain the key characteristics and attributes of the concept: {concept}
            Focus on the fundamental aspects that define this concept.
            """
            response = self.groq_client.generate_completion(prompt)
            concept_embeddings.append(response["text"])
        
        # Get text embeddings
        text_embeddings = []
        for text in texts:
            prompt = f"""
            Analyze the following text and identify its key themes and concepts:
            {text}
            """
            response = self.groq_client.generate_completion(prompt)
            text_embeddings.append(response["text"])
        
        # Calculate attribution scores
        attribution_scores = []
        for text, text_emb in zip(texts, text_embeddings):
            scores = {}
            for concept, concept_emb in zip(concepts, concept_embeddings):
                # Calculate semantic similarity between text and concept
                prompt = f"""
                Rate how strongly the following text relates to the concept '{concept}'
                on a scale from 0 to 1, where 0 means no relation and 1 means strong relation.
                Provide only the numerical score.
                
                Text: {text}
                Concept explanation: {concept_emb}
                """
                response = self.groq_client.generate_completion(prompt)
                try:
                    score = float(response["text"].strip())
                    scores[concept] = min(max(score, 0), 1)  # Ensure score is between 0 and 1
                except:
                    scores[concept] = 0.0
            
            attribution_scores.append(scores)
        
        return {
            "texts": texts,
            "concepts": concepts,
            "attribution_scores": attribution_scores,
            "concept_explanations": dict(zip(concepts, concept_embeddings))
        }
    
    def identify_activation_patterns(
        self,
        texts: List[str],
        pattern_type: str = "sequential"
    ) -> Dict:
        """Identify common activation patterns across texts."""
        self._ensure_model_loaded()
        
        # Get model activations
        all_activations = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                activations = torch.stack(outputs.hidden_states).numpy()
                all_activations.append(activations)
        
        if pattern_type == "sequential":
            # Analyze sequential patterns
            patterns = self._analyze_sequential_patterns(all_activations)
        else:
            # Analyze global patterns
            patterns = self._analyze_global_patterns(all_activations)
        
        return patterns
    
    def _analyze_sequential_patterns(
        self,
        activations: List[np.ndarray]
    ) -> Dict:
        """Analyze sequential activation patterns."""
        # Average over batch dimension
        mean_activations = np.mean([act.mean(axis=1) for act in activations], axis=0)
        
        # Find peaks in activation
        peaks = []
        for layer_idx in range(mean_activations.shape[0]):
            layer_mean = mean_activations[layer_idx]
            peak_indices = np.where(layer_mean > np.mean(layer_mean) + np.std(layer_mean))[0]
            peaks.append({
                "layer": layer_idx,
                "indices": peak_indices.tolist(),
                "values": layer_mean[peak_indices].tolist()
            })
        
        return {
            "pattern_type": "sequential",
            "num_layers": mean_activations.shape[0],
            "activation_peaks": peaks,
            "layer_means": mean_activations.mean(axis=1).tolist()
        }
    
    def _analyze_global_patterns(
        self,
        activations: List[np.ndarray]
    ) -> Dict:
        """Analyze global activation patterns."""
        # Flatten activations
        flat_activations = [act.reshape(act.shape[0], -1) for act in activations]
        mean_activations = np.mean(flat_activations, axis=0)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(5, len(activations)))
        clusters = kmeans.fit_predict(mean_activations)
        
        return {
            "pattern_type": "global",
            "num_clusters": kmeans.n_clusters,
            "cluster_assignments": clusters.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "inertia": float(kmeans.inertia_)
        } 