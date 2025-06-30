# explanations/probing_analysis.py

from models.groq_helper import llama_infer

class ProbingAnalyzer:
    def __init__(self, model_name):
        self.model_name = model_name

    def probe_text(self, text):
        prompt = (
            f"Perform a probing analysis on the sentence below. "
            f"Identify the syntactic roles (subject, object, verb, etc.), "
            f"named entities (if any), and any implicit sentiment.\n"
            f"Sentence: \"{text}\""
        )
        response = llama_infer(prompt, model=self.model_name)
        print("🔬 Probing output:", repr(response))
        return response.strip() if response else "No probing info available."