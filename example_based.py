from models.groq_helper import llama_infer

class ExampleBasedExplainer:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_similar_examples(self, input_text):
        prompt = (
            f"Given the following sentence, generate 3 example sentences that are similar in meaning. "
            f"For each, provide a similarity score between 0 and 1.\n"
            f"Format each as: example: score\n"
            f"Input: \"{input_text}\""
        )

        response = llama_infer(prompt, model=self.model_name)
        print("🔍 Example-based LLM response:", repr(response))

        examples, scores = [], []

        for line in response.replace(",", "\n").split("\n"):
            if ':' in line:
                example, score = line.rsplit(":", 1)
                example = example.strip()
                try:
                    score = float(score.strip())
                    examples.append(example)
                    scores.append(score)
                except ValueError:
                    continue

        # ✅ Ensure matched lengths to avoid DataFrame/plotly errors
        min_len = min(len(examples), len(scores))
        examples = examples[:min_len]
        scores = scores[:min_len]

        # ✅ Fallback if no valid output
        if not examples:
            examples = ["No valid output"]
            scores = [0.0]

        return {"examples": examples, "scores": scores}