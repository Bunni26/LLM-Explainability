from models.groq_helper import llama_infer
import plotly.graph_objs as go

class LocalExplanationGenerator:
    def __init__(self, model_name="llama3-8b-8192"):
        self.model_name = model_name

    def generate_feature_attribution(self, text):
        prompt = (
            f"Analyze this text and provide feature attribution scores. "
            f"For each important word, give it a score between 0 and 1, where 1 means most important. "
            f"Format each word and score as 'word: score', separated by newlines or commas. Text: \"{text}\""
        )
        response = llama_infer(prompt, model=self.model_name)
        print('LLM raw response:', repr(response))  # Debug print

        tokens, scores = [], []
        for item in response.replace(',', '\n').split('\n'):
            item = item.strip()
            if not item or ':' not in item:
                continue
            parts = item.rsplit(':', 1)
            if len(parts) != 2:
                continue
            token, score = parts
            token = token.strip()
            score = score.strip()
            if token and score:
                tokens.append(token)
                try:
                    scores.append(float(score))
                except ValueError:
                    scores.append(0.0)
        if not tokens:
            tokens = ["No valid output"]
            scores = [0.0]
        return {"tokens": tokens, "scores": scores}

    def create_feature_attribution_plot(self, result):
        tokens = result["tokens"]
        scores = result["scores"]
        
        fig = go.Figure(data=[
            go.Bar(
                x=tokens,
                y=scores,
                marker_color='rgb(55, 83, 109)'
            )
        ])
        
        fig.update_layout(
            title="Feature Attribution Scores",
            xaxis_title="Tokens",
            yaxis_title="Importance Score",
            xaxis_tickangle=-45,
            bargap=0.15
        )
        
        return fig