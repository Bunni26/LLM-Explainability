"""Neuron activation analysis for LLM interpretability."""

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config import MODEL_CONFIG

class NeuronActivationTracer:
    def __init__(self):
        """Initialize the neuron activation tracer."""
        self.tokenizer = None
        self.model = None
        self.activation_cache = {}
        
    def _ensure_model_loaded(self):
        """Ensure the model and tokenizer are loaded."""
        if self.tokenizer is None or self.model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["embedding_model"])
                self.model = AutoModelForCausalLM.from_pretrained(MODEL_CONFIG["embedding_model"])
            except Exception as e:
                raise Exception(f"Error loading model: {str(e)}")
    
    def trace_neuron_activations(
        self,
        texts: List[str],
        layer_idx: Optional[int] = None,
        neuron_indices: Optional[List[int]] = None
    ) -> Dict:
        """Trace activations of specific neurons across inputs."""
        self._ensure_model_loaded()
        
        activations = []
        token_maps = []
        
        for text in texts:
            # Tokenize and get activations
            inputs = self.tokenizer(text, return_tensors="pt")
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Get activations from specified layer or last layer
                layer = layer_idx if layer_idx is not None else -1
                layer_activations = outputs.hidden_states[layer][0]  # Remove batch dimension
                
                if neuron_indices is not None:
                    layer_activations = layer_activations[:, neuron_indices]
                
                activations.append(layer_activations.numpy())
                token_maps.append(tokens)
        
        return {
            "activations": activations,
            "tokens": token_maps,
            "layer_idx": layer if layer_idx is not None else len(outputs.hidden_states) - 1,
            "neuron_indices": neuron_indices
        }
    
    def find_top_activating_tokens(
        self,
        trace_results: Dict,
        top_k: int = 10
    ) -> Dict:
        """Find tokens that most strongly activate specific neurons."""
        activations = trace_results["activations"]
        tokens = trace_results["tokens"]
        
        # Combine all activations and tokens
        all_activations = np.vstack([act for act in activations])
        all_tokens = [token for token_list in tokens for token in token_list]
        
        # Find top activating tokens for each neuron
        num_neurons = all_activations.shape[1]
        top_tokens = []
        
        for neuron_idx in range(num_neurons):
            neuron_activations = all_activations[:, neuron_idx]
            # Get indices of top k activating tokens
            top_indices = np.argsort(neuron_activations)[-top_k:][::-1]
            
            top_tokens.append({
                "neuron_idx": neuron_idx,
                "tokens": [all_tokens[i] for i in top_indices],
                "activations": neuron_activations[top_indices].tolist()
            })
        
        return {
            "layer_idx": trace_results["layer_idx"],
            "top_activating_tokens": top_tokens
        }
    
    def analyze_activation_patterns(
        self,
        trace_results: Dict,
        n_clusters: int = 5
    ) -> Dict:
        """Analyze patterns in neuron activations."""
        activations = trace_results["activations"]
        tokens = trace_results["tokens"]
        
        # Combine all activations
        all_activations = np.vstack([act for act in activations])
        
        # Standardize activations
        scaler = StandardScaler()
        scaled_activations = scaler.fit_transform(all_activations)
        
        # Cluster activation patterns
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_activations)
        
        # Analyze clusters
        cluster_analysis = []
        for cluster_idx in range(n_clusters):
            cluster_mask = clusters == cluster_idx
            cluster_activations = all_activations[cluster_mask]
            
            # Get mean activation pattern for cluster
            mean_pattern = cluster_activations.mean(axis=0)
            
            # Find most representative samples (closest to cluster center)
            distances = np.linalg.norm(
                cluster_activations - kmeans.cluster_centers_[cluster_idx],
                axis=1
            )
            top_sample_indices = np.argsort(distances)[:5]
            
            cluster_analysis.append({
                "cluster_idx": cluster_idx,
                "size": int(cluster_mask.sum()),
                "mean_activation": mean_pattern.tolist(),
                "std_activation": cluster_activations.std(axis=0).tolist(),
                "representative_samples": top_sample_indices.tolist()
            })
        
        return {
            "layer_idx": trace_results["layer_idx"],
            "n_clusters": n_clusters,
            "cluster_analysis": cluster_analysis,
            "cluster_centers": kmeans.cluster_centers_.tolist()
        }
    
    def trace_activation_flow(
        self,
        text: str,
        target_neurons: List[int]
    ) -> Dict:
        """Trace activation flow through layers for specific neurons."""
        self._ensure_model_loaded()
        
        inputs = self.tokenizer(text, return_tensors="pt")
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Collect activations across layers
        layer_activations = []
        for layer_idx, hidden_states in enumerate(outputs.hidden_states):
            layer_act = hidden_states[0, :, target_neurons].numpy()
            layer_activations.append({
                "layer_idx": layer_idx,
                "activations": layer_act.tolist()
            })
        
        return {
            "tokens": tokens,
            "target_neurons": target_neurons,
            "layer_activations": layer_activations,
            "num_layers": len(layer_activations)
        }
    
    def compute_neuron_similarity(
        self,
        trace_results: Dict
    ) -> Dict:
        """Compute similarity between neuron activation patterns."""
        activations = trace_results["activations"]
        
        # Combine all activations
        all_activations = np.vstack([act for act in activations])
        
        # Compute correlation matrix between neurons
        correlation_matrix = np.corrcoef(all_activations.T)
        
        # Find highly correlated neuron pairs
        num_neurons = correlation_matrix.shape[0]
        correlated_pairs = []
        
        for i in range(num_neurons):
            for j in range(i + 1, num_neurons):
                correlation = correlation_matrix[i, j]
                if abs(correlation) > 0.7:  # Threshold for strong correlation
                    correlated_pairs.append({
                        "neuron_1": i,
                        "neuron_2": j,
                        "correlation": float(correlation)
                    })
        
        return {
            "layer_idx": trace_results["layer_idx"],
            "correlation_matrix": correlation_matrix.tolist(),
            "correlated_pairs": sorted(
                correlated_pairs,
                key=lambda x: abs(x["correlation"]),
                reverse=True
            )
        } 