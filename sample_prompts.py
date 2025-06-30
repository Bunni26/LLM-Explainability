"""Sample prompts and test cases for LLM explainability analysis."""

FEATURE_ATTRIBUTION_SAMPLES = {
    "sentiment": {
        "text": "The movie was absolutely fantastic, with brilliant acting and stunning visuals, although the pacing was a bit slow at times.",
        "task": "Analyze sentiment and key contributing words"
    },
    "classification": {
        "text": "Patient presents with high fever, persistent cough, and fatigue for the past three days.",
        "task": "Medical symptom classification"
    },
    "code": {
        "text": "def calculate_risk_score(income, credit_score, debt_ratio): return 0.4 * credit_score - 0.3 * debt_ratio + 0.3 * income",
        "task": "Code function purpose analysis"
    }
}

CHAIN_OF_THOUGHT_SAMPLES = {
    "math": {
        "question": "If a store offers a 20% discount on a $80 item and then applies a 10% tax, what is the final price?",
        "examples": [
            "Example 1: Calculate 15% tip on $40 bill\n1. Original amount: $40\n2. Tip calculation: $40 × 0.15 = $6\n3. Final amount: $40 + $6 = $46",
            "Example 2: 30% discount on $100 item\n1. Discount amount: $100 × 0.30 = $30\n2. Final price: $100 - $30 = $70"
        ]
    },
    "reasoning": {
        "question": "A student needs to read a 300-page book in 5 days. If they read the same number of pages each day and have already completed 2 days of reading with 140 pages done, will they finish on time?",
        "examples": [
            "Example: Can someone read 200 pages in 3 days at 60 pages per day?\n1. Daily rate: 60 pages\n2. Total possible in 3 days: 60 × 3 = 180 pages\n3. Required: 200 pages\n4. Conclusion: No, they cannot finish"
        ]
    }
}

EXPLANATION_EVALUATION_SAMPLES = {
    "plausibility": {
        "explanation": "The model predicts high customer churn risk because the customer has decreased their usage by 50% in the last month, has contacted support 3 times with complaints, and their contract is approaching renewal.",
        "context": "Customer churn prediction in telecommunications"
    },
    "faithfulness": {
        "explanation": "The image was classified as a dog because it shows pointed ears, a furry coat, and a wagging tail in a typical canine posture.",
        "reference": "The model detected key features including ear shape (0.8 confidence), fur texture (0.9 confidence), and body posture (0.85 confidence) consistent with canine characteristics.",
        "input": "dog_image.jpg"
    }
}

ATTENTION_ANALYSIS_SAMPLES = {
    "text": {
        "input": "The company's new AI product launch was successful, driving a 25% increase in quarterly revenue.",
        "focus": "Analyze attention patterns between business metrics and events"
    },
    "code": {
        "input": "def validate_user_input(data):\n    if not data:\n        raise ValueError('Empty input')\n    if len(data) > MAX_LENGTH:\n        raise ValueError('Input too long')\n    return sanitize_input(data)",
        "focus": "Analyze attention patterns in code validation logic"
    }
}

CONCEPT_ANALYSIS_SAMPLES = {
    "medical": {
        "text": "The patient exhibits symptoms of tachycardia with elevated blood pressure and irregular heart rhythm.",
        "concepts": ["cardiovascular", "vital signs", "arrhythmia"]
    },
    "finance": {
        "text": "The market volatility index increased sharply while bond yields decreased, indicating risk-off sentiment.",
        "concepts": ["market risk", "investor sentiment", "asset correlation"]
    }
} 