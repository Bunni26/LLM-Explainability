"""Linear probing analysis for LLM hidden states."""

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_CONFIG

class HiddenStateProber:
    def __init__(self):
        """Initialize the hidden state prober."""
        self.tokenizer = None
        self.model = None
        self.probes = {}
        
    def _ensure_model_loaded(self):
        """Ensure the model and tokenizer are loaded."""
        if self.tokenizer is None or self.model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["embedding_model"])
                self.model = AutoModelForCausalLM.from_pretrained(MODEL_CONFIG["embedding_model"])
            except Exception as e:
                raise Exception(f"Error loading model: {str(e)}")
    
    def extract_hidden_states(
        self,
        texts: List[str],
        layer_idx: Optional[int] = None
    ) -> np.ndarray:
        """Extract hidden states from specified layer."""
        self._ensure_model_loaded()
        
        hidden_states = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Get hidden states from specified layer or last layer
                layer = layer_idx if layer_idx is not None else -1
                states = outputs.hidden_states[layer].mean(dim=1)  # Average over sequence length
                hidden_states.append(states.numpy())
        
        return np.vstack(hidden_states)
    
    def train_probe(
        self,
        hidden_states: np.ndarray,
        labels: Union[List[int], List[float]],
        task_type: str = "classification",
        probe_name: str = "default"
    ) -> Dict:
        """Train a linear probe on hidden states."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            hidden_states,
            labels,
            test_size=0.2,
            random_state=42
        )
        
        # Select probe type
        if task_type == "classification":
            probe = LogisticRegression(max_iter=1000)
            metric = accuracy_score
        else:  # regression
            probe = LinearRegression()
            metric = r2_score
        
        # Train probe
        probe.fit(X_train, y_train)
        
        # Evaluate
        y_pred = probe.predict(X_test)
        score = metric(y_test, y_pred)
        
        # Store probe
        self.probes[probe_name] = {
            "probe": probe,
            "task_type": task_type,
            "score": score
        }
        
        # Get detailed metrics
        if task_type == "classification":
            details = classification_report(y_test, y_pred, output_dict=True)
        else:
            details = {
                "r2_score": score,
                "coefficients": probe.coef_.tolist(),
                "intercept": float(probe.intercept_)
            }
        
        return {
            "probe_name": probe_name,
            "task_type": task_type,
            "score": score,
            "details": details
        }
    
    def analyze_feature_importance(
        self,
        probe_name: str,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """Analyze feature importance in trained probe."""
        if probe_name not in self.probes:
            raise ValueError(f"Probe '{probe_name}' not found")
        
        probe_info = self.probes[probe_name]
        probe = probe_info["probe"]
        
        # Get feature importance
        if probe_info["task_type"] == "classification":
            importance = np.abs(probe.coef_).mean(axis=0)
        else:
            importance = np.abs(probe.coef_)
        
        # Sort features by importance
        sorted_idx = np.argsort(importance)[::-1]
        sorted_importance = importance[sorted_idx]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        sorted_features = [feature_names[i] for i in sorted_idx]
        
        return {
            "probe_name": probe_name,
            "task_type": probe_info["task_type"],
            "feature_importance": sorted_importance.tolist(),
            "feature_names": sorted_features,
            "top_features": list(zip(
                sorted_features[:10],
                sorted_importance[:10].tolist()
            ))
        }
    
    def probe_layer_dynamics(
        self,
        texts: List[str],
        labels: Union[List[int], List[float]],
        task_type: str = "classification"
    ) -> Dict:
        """Analyze how well each layer can be probed for a task."""
        self._ensure_model_loaded()
        
        layer_scores = []
        for layer_idx in range(len(self.model.config.hidden_size)):
            # Extract hidden states for this layer
            hidden_states = self.extract_hidden_states(texts, layer_idx)
            
            # Train probe
            probe_name = f"layer_{layer_idx}"
            result = self.train_probe(
                hidden_states,
                labels,
                task_type=task_type,
                probe_name=probe_name
            )
            
            layer_scores.append({
                "layer": layer_idx,
                "score": result["score"],
                "details": result["details"]
            })
        
        return {
            "task_type": task_type,
            "layer_scores": layer_scores,
            "best_layer": max(
                range(len(layer_scores)),
                key=lambda i: layer_scores[i]["score"]
            )
        }
    
    def compare_probes(
        self,
        probe_names: List[str]
    ) -> Dict:
        """Compare performance of different probes."""
        if not all(name in self.probes for name in probe_names):
            raise ValueError("One or more probe names not found")
        
        comparisons = []
        for name in probe_names:
            probe_info = self.probes[name]
            comparisons.append({
                "probe_name": name,
                "task_type": probe_info["task_type"],
                "score": probe_info["score"]
            })
        
        return {
            "comparisons": comparisons,
            "best_probe": max(
                comparisons,
                key=lambda x: x["score"]
            )
        } 