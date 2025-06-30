from models.groq_helper import llama_infer

class PromptingAnalyzer:
    def __init__(self, model_name):
        self.model_name = model_name

    def analyze_in_context_learning(self, examples: str, prompt: str) -> str:
        final_prompt = (
            "Use in-context learning to answer the final prompt using the following examples:\n\n"
            f"{examples.strip()}\n\nPrompt: {prompt.strip()}\nAnswer:"
        )
        return llama_infer(final_prompt, model=self.model_name)

    def analyze_chain_of_thought(self, prompt: str) -> str:
        """
        Analyze the chain-of-thought reasoning process.

        This method uses in-context learning to ask the model to think step-by-step
        before giving the final answer. The generated text will include the model's
        thought process.

        Parameters:
            prompt (str): The input prompt to analyze.

        Returns:
            str: Explanation output from the model
        """
        final_prompt = (
            f"Use a chain-of-thought approach to answer the following question. "
            f"Think step-by-step before giving the final answer.\n\nQuestion: {prompt}\nAnswer:"
        )
        return llama_infer(final_prompt, model=self.model_name)

    def analyze_representation_engineering(self, prompt: str) -> str:
        final_prompt = (
            f"Analyze the internal representation of this prompt. "
            f"Explain how the model might internally encode the meaning and structure.\n\nPrompt: {prompt}"
        )
        return llama_infer(final_prompt, model=self.model_name)

    def detect_hallucination(self, prompt: str) -> str:
        final_prompt = (
            f"Detect if the following output is hallucinated or grounded in factual data. "
            f"Explain why.\n\nPrompt: {prompt}"
        )
        return llama_infer(final_prompt, model=self.model_name)

    def estimate_uncertainty(self, prompt: str) -> str:
        final_prompt = (
            f"Estimate the uncertainty in this response. "
            f"Rate the confidence level from 0 to 1 and provide reasons.\n\nPrompt: {prompt}"
        )
        return llama_infer(final_prompt, model=self.model_name)

    def analyze_output_similarity(self, prompt: str) -> str:
        final_prompt = (
            f"Provide 3 different phrasings for the following prompt that produce similar outputs. "
            f"Also indicate how similar the outputs would be on a scale of 0 to 1.\n\nPrompt: {prompt}"
        )
        return llama_infer(final_prompt, model=self.model_name)

    def analyze_prompt(self, prompt: str, method: str = "cot", examples: str = "") -> str:
        """
        Unified interface for analyzing prompts based on different explanation techniques.

        Args:
            prompt (str): The input prompt to analyze.
            method (str): One of ['cot', 'icl', 'representation', 'hallucination', 'uncertainty', 'similarity']
            examples (str): Required only for ICL (in-context learning)

        Returns:
            str: Explanation output from the model
        """
        if method == "cot":
            return self.analyze_chain_of_thought(prompt)
        elif method == "icl":
            return self.analyze_in_context_learning(examples, prompt)
        elif method == "representation":
            return self.analyze_representation_engineering(prompt)
        elif method == "hallucination":
            return self.detect_hallucination(prompt)
        elif method == "uncertainty":
            return self.estimate_uncertainty(prompt)
        elif method == "similarity":
            return self.analyze_output_similarity(prompt)
        else:
            raise ValueError(f"Unknown prompt analysis method: {method}")