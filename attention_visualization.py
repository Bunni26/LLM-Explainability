# explanations/attention_visualization.py

from models.groq_helper import llama_infer

class AttentionVisualizer:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate_attention_weights(self, text):
        prompt = (
            f"Analyze the attention distribution of this sentence and assign each word a score "
            f"between 0 and 1 based on how much attention it receives. "
            f"Format as 'word: score'.\nText: \"{text}\""
        )
        response = llama_infer(prompt, model=self.model_name)
        print("Attention raw response:", repr(response))

        tokens, scores = [], []
        for item in response.replace(',', '\n').split('\n'):
            if ':' in item:
                token, score = item.rsplit(':', 1)
                try:
                    tokens.append(token.strip())
                    scores.append(float(score.strip()))
                except ValueError:
                    continue

        if not tokens:
            tokens = ["No output"]
            scores = [0.0]

        return {"tokens": tokens, "scores": scores}