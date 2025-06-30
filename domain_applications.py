"""Domain-specific applications for LLM explainability."""

from typing import Dict, List, Optional
from ..models.groq_helper import GroqHelper
from ..evaluations.explanation_evaluator import ExplanationEvaluator
import numpy as np

class DomainExplainer:
    def __init__(self):
        """Initialize the domain explainer."""
        self.groq_client = GroqHelper()
        self.evaluator = ExplanationEvaluator()
        
    def explain_legal_decision(
        self,
        case_text: str,
        decision: str
    ) -> Dict:
        """Explain legal decision with relevant precedents and reasoning."""
        # Generate legal explanation
        prompt = f"""
        Analyze this legal case and decision, explaining the reasoning:
        
        Case: {case_text}
        Decision: {decision}
        
        Provide a detailed explanation including:
        1. Key legal principles applied
        2. Relevant precedents
        3. Chain of reasoning
        4. Potential implications
        """
        
        explanation = self.groq_client.generate_completion(prompt)
        
        # Evaluate explanation
        evaluation = self.evaluator.evaluate_plausibility(
            explanation["text"],
            context=case_text
        )
        
        return {
            "case_text": case_text,
            "decision": decision,
            "explanation": explanation["text"],
            "evaluation": evaluation,
            "domain": "legal"
        }
    
    def explain_financial_prediction(
        self,
        financial_data: str,
        prediction: str,
        metrics: Optional[Dict] = None
    ) -> Dict:
        """Explain financial predictions with market analysis."""
        # Generate financial explanation
        prompt = f"""
        Analyze this financial data and prediction:
        
        Data: {financial_data}
        Prediction: {prediction}
        Metrics: {metrics if metrics else 'Not provided'}
        
        Provide a detailed explanation including:
        1. Key market indicators
        2. Risk factors
        3. Historical patterns
        4. Confidence assessment
        """
        
        explanation = self.groq_client.generate_completion(prompt)
        
        # Evaluate explanation
        evaluation = self.evaluator.evaluate_faithfulness(
            explanation["text"],
            reference=financial_data
        )
        
        return {
            "financial_data": financial_data,
            "prediction": prediction,
            "metrics": metrics,
            "explanation": explanation["text"],
            "evaluation": evaluation,
            "domain": "finance"
        }
    
    def explain_medical_diagnosis(
        self,
        patient_data: str,
        diagnosis: str,
        confidence: Optional[float] = None
    ) -> Dict:
        """Explain medical diagnosis with clinical reasoning."""
        # Generate medical explanation
        prompt = f"""
        Analyze this medical case and diagnosis:
        
        Patient Data: {patient_data}
        Diagnosis: {diagnosis}
        Confidence: {confidence if confidence else 'Not provided'}
        
        Provide a detailed explanation including:
        1. Key symptoms and findings
        2. Diagnostic criteria met
        3. Differential diagnosis
        4. Evidence-based reasoning
        """
        
        explanation = self.groq_client.generate_completion(prompt)
        
        # Evaluate explanation
        evaluation = self.evaluator.evaluate_plausibility(
            explanation["text"],
            context=patient_data
        )
        
        # Check for medical consistency
        consistency_prompt = f"""
        Verify the medical consistency of this explanation:
        
        Patient Data: {patient_data}
        Diagnosis: {diagnosis}
        Explanation: {explanation["text"]}
        
        Check for:
        1. Alignment with standard medical practice
        2. Completeness of diagnostic reasoning
        3. Appropriate consideration of alternatives
        4. Evidence-based support
        
        Rate the medical consistency from 0 to 1:
        """
        
        consistency_check = self.groq_client.generate_completion(
            consistency_prompt,
            temperature=0.1
        )
        
        try:
            consistency_score = float(consistency_check["text"].strip())
        except:
            consistency_score = 0.5
        
        return {
            "patient_data": patient_data,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "explanation": explanation["text"],
            "evaluation": evaluation,
            "medical_consistency": consistency_score,
            "domain": "medical"
        }
    
    def analyze_domain_specific_patterns(
        self,
        explanations: List[Dict],
        domain: str
    ) -> Dict:
        """Analyze patterns in domain-specific explanations."""
        if not explanations:
            raise ValueError("No explanations provided")
        
        if domain not in ["legal", "finance", "medical"]:
            raise ValueError("Invalid domain")
        
        # Extract explanation texts
        texts = [exp["explanation"] for exp in explanations]
        
        # Analyze common patterns
        pattern_prompt = f"""
        Analyze these {domain} explanations and identify common patterns:
        
        Explanations:
        {' '.join(texts)}
        
        Identify:
        1. Common reasoning structures
        2. Domain-specific terminology
        3. Key decision factors
        4. Explanation strategies
        """
        
        pattern_analysis = self.groq_client.generate_completion(pattern_prompt)
        
        # Calculate average evaluation scores
        avg_plausibility = np.mean([
            exp["evaluation"]["overall_score"]
            for exp in explanations
        ])
        
        return {
            "domain": domain,
            "num_explanations": len(explanations),
            "pattern_analysis": pattern_analysis["text"],
            "avg_plausibility": float(avg_plausibility),
            "explanation_samples": texts[:3]  # Include first 3 as samples
        } 