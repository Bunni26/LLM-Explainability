# explanations/natural_language_explainer.py

from models.groq_helper import llama_infer

class NaturalLanguageExplainer:
    def __init__(self, model_name):
        self.model_name = model_name

    def explain_text(self, text):
        prompt = (
            f"Explain the following sentence in natural language as if you're interpreting why it is important or meaningful. "
            f"Be clear and concise.\nSentence: \"{text}\""
        )
        response = llama_infer(prompt, model=self.model_name)
        print("🗣️ Natural language explanation:", repr(response))
        return response.strip() if response else "No explanation generated."