import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Optional

def render_feature_attribution(
    tokens: List[str],
    scores: List[float],
    method: str = "integrated_gradients"
) -> None:
    """Render feature attribution visualization."""
    fig = go.Figure(data=go.Bar(
        x=tokens,
        y=scores,
        marker_color=np.where(np.array(scores) > 0, "green", "red")
    ))
    
    fig.update_layout(
        title=f"Feature Attribution Scores ({method})",
        xaxis_title="Tokens",
        yaxis_title="Attribution Score",
        width=1000,
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)

def render_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: List[str],
    layer_idx: int
) -> None:
    """Render attention weights heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        x=tokens,
        y=tokens,
        colorscale="Viridis"
    ))
    
    fig.update_layout(
        title=f"Attention Weights (Layer {layer_idx})",
        xaxis_title="Target Tokens",
        yaxis_title="Source Tokens",
        width=800,
        height=800
    )
    
    st.plotly_chart(fig)

def render_explanation_scores(
    plausibility_scores: Dict[str, float],
    faithfulness_score: Optional[float] = None
) -> None:
    """Render explanation evaluation scores."""
    # Create gauge charts for each score
    for aspect, score in plausibility_scores.items():
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            title={"text": aspect.replace("_", " ").title()},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 33], "color": "lightgray"},
                    {"range": [33, 66], "color": "gray"},
                    {"range": [66, 100], "color": "darkgray"}
                ]
            }
        ))
        
        fig.update_layout(width=400, height=300)
        st.plotly_chart(fig)
    
    if faithfulness_score is not None:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=faithfulness_score * 100,
            title={"text": "Faithfulness Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 33], "color": "lightgray"},
                    {"range": [33, 66], "color": "gray"},
                    {"range": [66, 100], "color": "darkgray"}
                ]
            }
        ))
        
        fig.update_layout(width=400, height=300)
        st.plotly_chart(fig)

def render_similarity_matrix(
    similarities: np.ndarray,
    labels: List[str]
) -> None:
    """Render similarity matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=similarities,
        x=labels,
        y=labels,
        colorscale="Viridis",
        zmin=0,
        zmax=1
    ))
    
    fig.update_layout(
        title="Similarity Matrix",
        width=800,
        height=800
    )
    
    st.plotly_chart(fig)

def render_uncertainty_plot(
    samples: List[str],
    similarities: np.ndarray
) -> None:
    """Render uncertainty visualization."""
    # Create similarity heatmap
    fig = go.Figure(data=go.Heatmap(
        z=similarities,
        colorscale="RdBu",
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title="Sample Similarity Heatmap",
        width=600,
        height=600
    )
    
    st.plotly_chart(fig)
    
    # Display uncertainty metrics
    st.write("Uncertainty Metrics:")
    st.write(f"- Mean Similarity: {np.mean(similarities):.3f}")
    st.write(f"- Std Similarity: {np.std(similarities):.3f}")
    st.write(f"- Uncertainty Score: {1 - np.mean(similarities):.3f}")

def render_counterfactual_analysis(
    original: str,
    counterfactuals: List[Dict]
) -> None:
    """Render counterfactual analysis visualization."""
    # Display original explanation
    st.write("Original Explanation:")
    st.write(original)
    
    # Display counterfactuals with similarity scores
    st.write("\nCounterfactual Explanations:")
    for i, cf in enumerate(counterfactuals, 1):
        st.write(f"\nCounterfactual {i}:")
        st.write(cf["counterfactual"])
        
        # Create similarity gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cf["similarity_to_original"] * 100,
            title={"text": f"Similarity to Original"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 33], "color": "lightgray"},
                    {"range": [33, 66], "color": "gray"},
                    {"range": [66, 100], "color": "darkgray"}
                ]
            }
        ))
        
        fig.update_layout(width=400, height=300)
        st.plotly_chart(fig)

def render_chain_of_thought_analysis(
    steps: List[str],
    step_similarities: np.ndarray
) -> None:
    """Render chain-of-thought analysis visualization."""
    # Display reasoning steps
    st.write("Reasoning Steps:")
    for i, step in enumerate(steps, 1):
        st.write(f"{i}. {step}")
    
    # Create step coherence heatmap
    fig = go.Figure(data=go.Heatmap(
        z=step_similarities,
        colorscale="Viridis",
        zmin=0,
        zmax=1
    ))
    
    fig.update_layout(
        title="Step Coherence Matrix",
        width=600,
        height=600
    )
    
    st.plotly_chart(fig)
    
    # Display coherence metrics
    coherence_scores = [
        step_similarities[i, i+1]
        for i in range(len(steps)-1)
    ]
    
    st.write("\nCoherence Metrics:")
    st.write(f"- Average Step Coherence: {np.mean(coherence_scores):.3f}")
    st.write(f"- Min Step Coherence: {np.min(coherence_scores):.3f}")
    st.write(f"- Max Step Coherence: {np.max(coherence_scores):.3f}") 