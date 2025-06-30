# LLM Explainability System 🔍

A comprehensive system for explaining and interpreting Large Language Models (LLMs) using Groq-hosted LLaMA models.

## 🎯 Features

### 1. Traditional Fine-tuning Paradigm
- Local Explanations (Feature Attribution, Attention Analysis)
- Global Explanations (Mechanistic Interpretability)
- Model Debugging & Analysis

### 2. Prompting Paradigm
- In-context Learning Analysis
- Chain-of-Thought Reasoning
- Representation Engineering
- Hallucination Detection

### 3. Explanation Evaluation
- Plausibility Assessment
- Faithfulness Metrics
- Counterfactual Testing

## 🚀 Getting Started

1. Clone the repository
```bash
git clone https://github.com/yourusername/llm-explainability.git
cd llm-explainability
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
cp .env.example .env
# Add your Groq API key to .env file
```

5. Run the application
```bash
streamlit run app.py
```

## 📁 Project Structure

```
.
├── app.py                 # Streamlit main application
├── explanations/          # Explanation implementations
├── prompting/             # Prompt engineering tools
├── evaluations/           # Explanation evaluation metrics
├── models/               # Groq inference helpers
├── ui/                   # Streamlit components
├── data/                # Sample data & prompts
└── tests/               # Unit tests
```

## 🛠️ Development

Run tests:
```bash
pytest
```

## 📄 License

MIT License - see LICENSE file for details 