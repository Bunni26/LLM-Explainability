import streamlit as st 
import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

from config import VALID_GROQ_MODELS
from explanations.local_explanations import LocalExplanationGenerator
from explanations.example_based import ExampleBasedExplainer
from explanations.natural_language_explainer import NaturalLanguageExplainer
from explanations.probing_analysis import ProbingAnalyzer
from explanations.neuron_activation import NeuronActivationTracer
from explanations.prompting_analysis import PromptingAnalyzer  # ✅ FIXED

# ✅ Load .env
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

print("✅ Loaded Groq Key:", os.getenv("GROQ_API_KEY")[:10] + "..." if os.getenv("GROQ_API_KEY") else "None")

# Streamlit UI config
st.set_page_config(
    page_title="LLM Explainability System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("LLM Explainability System 🔍")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analysis Type",
        ["Traditional Fine-tuning", "Prompting Analysis", "Explanation Evaluation"]
    )

    # Model dropdown from config
    model = st.sidebar.selectbox(
        "Select Groq Model",
        VALID_GROQ_MODELS
    )

    if page == "Traditional Fine-tuning":
        st.header("Traditional Fine-tuning Paradigm")

        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Local Explanations", "Global Explanations", "Model Usage"]
        )

        if analysis_type == "Local Explanations":
            explanation_method = st.selectbox(
                "Select Explanation Method",
                [
                    "Feature Attribution",
                    "Example-based Explanations",
                    "Natural Language Explanations",
                    "Probing Analysis",
                    "Neuron Activation"
                ]
            )

            input_text = st.text_area("Enter text for analysis:", height=100)

            if st.button("Generate Explanation"):
                if not input_text:
                    st.warning("Please enter some text to analyze.")
                else:
                    with st.spinner("Generating explanation..."):
                        try:
                            if explanation_method == "Feature Attribution":
                                generator = LocalExplanationGenerator(model_name=model)
                                result = generator.generate_feature_attribution(input_text)
                                df = pd.DataFrame({"Token": result["tokens"], "Score": result["scores"]})
                                st.subheader("Feature Attribution Scores")
                                st.table(df)
                                fig = generator.create_feature_attribution_plot(result)
                                st.plotly_chart(fig, use_container_width=True)

                            elif explanation_method == "Example-based Explanations":
                                explainer = ExampleBasedExplainer(model_name=model)
                                result = explainer.get_similar_examples(input_text)
                                df = pd.DataFrame({
                                    "Similar Example": result["examples"],
                                    "Similarity Score": result["scores"]
                                })
                                st.subheader("Example-based Explanations")
                                st.table(df)

                            elif explanation_method == "Natural Language Explanations":
                                explainer = NaturalLanguageExplainer(model_name=model)
                                explanation = explainer.explain_text(input_text)
                                st.subheader("Natural Language Explanation")
                                st.write(f"🧠 {explanation}")

                            elif explanation_method == "Probing Analysis":
                                analyzer = ProbingAnalyzer(model_name=model)
                                probing_result = analyzer.probe_text(input_text)
                                st.subheader("Probing Analysis Result")
                                st.markdown(f"🧪 **Probing Output:**\n\n{probing_result}")

                            elif explanation_method == "Neuron Activation":
                                tracer = NeuronActivationTracer()
                                st.info("⚠️ This will load a transformer model (may take time).")
                                text_list = [input_text]
                                layer = st.number_input("Layer Index (e.g. 0-11 for BERT)", min_value=0, value=0)
                                neurons = st.text_input("Neuron Indices (comma-separated, e.g. 0,5,10)", "0,1,2")
                                neuron_list = [int(x.strip()) for x in neurons.split(",") if x.strip().isdigit()]

                                if st.button("Trace Neuron Activations"):
                                    trace = tracer.trace_neuron_activations(text_list, layer_idx=layer, neuron_indices=neuron_list)
                                    top_tokens = tracer.find_top_activating_tokens(trace)
                                    st.json(top_tokens)

                        except Exception as e:
                            st.error(f"❌ Error generating explanation:\n\n{e}")

    elif page == "Prompting Analysis":
        st.header("Prompting Paradigm Analysis")

        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "In-context Learning",
                "Chain-of-Thought Analysis",
                "Representation Engineering",
                "Hallucination Detection",
                "Uncertainty Estimation",
                "Output Similarity"
            ]
        )

        prompt = st.text_area("Enter your prompt:", height=100)
        examples = ""

        if analysis_type == "In-context Learning":
            examples_input = st.text_area("Enter few-shot examples (one per line):", height=150)
            examples = "\n".join(examples_input.split("\n"))

        method_map = {
            "In-context Learning": "icl",
            "Chain-of-Thought Analysis": "cot",
            "Representation Engineering": "representation",
            "Hallucination Detection": "hallucination",
            "Uncertainty Estimation": "uncertainty",
            "Output Similarity": "similarity"
        }

        if st.button("Analyze"):
            if not prompt:
                st.warning("Please enter a prompt to analyze.")
            else:
                try:
                    analyzer = PromptingAnalyzer(model_name=model)
                    with st.spinner("Analyzing..."):
                        method = method_map[analysis_type]
                        result = analyzer.analyze_prompt(prompt, method=method, examples=examples)
                        st.subheader(f"{analysis_type} Result")
                        st.write(result)
                except Exception as e:
                    st.error(f"❌ Error during prompt analysis:\n\n{e}")

    else:
        st.header("Explanation Evaluation")

        eval_type = st.selectbox(
            "Select Evaluation Type",
            ["Plausibility", "Faithfulness", "Combined Analysis"]
        )

        explanation = st.text_area("Enter explanation to evaluate:", height=100)
        reference = st.text_area("Enter reference/ground truth (if applicable):", height=100)

        if st.button("Evaluate"):
            if not explanation:
                st.warning("Please enter an explanation to evaluate.")
            else:
                with st.spinner("Evaluating..."):
                    st.info("Evaluation logic coming soon!")

if __name__ == "__main__":
    main()