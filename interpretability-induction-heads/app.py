"""
Streamlit UI for Induction Head Analysis.

This application provides an interactive interface for:
1. Detecting and visualizing induction heads in GPT-2
2. Exploring attention patterns
3. Running activation patching experiments

Run with: streamlit run app.py
"""

import streamlit as st
import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import load_model, get_model_info, get_token_strs
from src.data_gen import (
    generate_repeated_sequence,
    generate_corrupted_pair,
    generate_simple_repeat,
    get_token_set,
    InductionPrompt,
)
from src.analysis import (
    analyze_induction_heads,
    get_top_induction_heads,
    get_attention_pattern,
)
from src.patching import run_patching_experiment, PatchingResult
from src.viz import (
    plot_attention_pattern,
    plot_induction_scores,
    plot_induction_heatmap,
    plot_patching_result,
)


# Page config
st.set_page_config(
    page_title="Induction Head Explorer",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def get_model(model_name: str):
    """Load and cache the model."""
    return load_model(model_name, device="cpu")


def main():
    st.title("🔍 Induction Head Explorer")
    st.markdown(
        """
    Explore induction heads in transformer models. Induction heads are attention heads
    that implement a simple pattern-completion algorithm: when they see a token that
    appeared earlier, they copy what came after it.
    """
    )

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Model selection
    model_name = st.sidebar.selectbox(
        "Model",
        ["gpt2-small", "attn-only-2l"],
        index=1,  # Default to smaller model
        help="Select the transformer model to analyze. 'attn-only-2l' is much smaller and faster.",
    )

    # Load model
    with st.spinner(f"Loading {model_name}..."):
        model = get_model(model_name)
    model_info = get_model_info(model)

    st.sidebar.success(f"✅ Model loaded: {model_info['n_layers']}L × {model_info['n_heads']}H")

    # Prompt configuration
    st.sidebar.header("📝 Prompt Settings")

    vocab_type = st.sidebar.selectbox(
        "Token Set",
        ["letters", "words", "numbers"],
        index=0,
        help="Type of tokens to use in generated prompts",
    )

    prefix_length = st.sidebar.slider(
        "Prefix Length",
        min_value=2,
        max_value=10,
        value=5,
        help="Number of tokens before the repeated pattern",
    )

    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=10000,
        value=42,
        help="Seed for reproducible prompt generation",
    )

    # Analysis settings
    st.sidebar.header("📊 Analysis Settings")

    top_k = st.sidebar.slider(
        "Top K Heads",
        min_value=5,
        max_value=30,
        value=15,
        help="Number of top induction heads to display",
    )

    n_prompts = st.sidebar.slider(
        "Prompts for Averaging",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of prompts to average scores over",
    )

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(
        ["🎯 Induction Scan", "🔬 Attention Viewer", "🔧 Activation Patching"]
    )

    # ==================== TAB 1: INDUCTION SCAN ====================
    with tab1:
        st.header("Induction Head Detection")
        st.markdown(
            """
        This section scans all attention heads to identify those with high induction scores.
        The induction score measures how much each head attends from a repeated token back
        to its first occurrence, enabling the model to copy the following token.
        """
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔍 Run Induction Scan", type="primary"):
                with st.spinner("Analyzing attention heads..."):
                    # Get top heads
                    top_heads = get_top_induction_heads(
                        model,
                        vocab_name=vocab_type,
                        n_prompts=n_prompts,
                        prefix_length=prefix_length,
                        top_k=top_k,
                        seed=seed,
                    )

                    # Store in session state
                    st.session_state["top_heads"] = top_heads
                    st.session_state["scan_complete"] = True

        # Display results if scan has been run
        if st.session_state.get("scan_complete", False):
            top_heads = st.session_state["top_heads"]

            with col1:
                st.subheader("Top Induction Heads")
                # Create a nice table
                table_data = []
                for layer, head, score in top_heads:
                    table_data.append(
                        {"Layer": layer, "Head": head, "Score": f"{score:.4f}"}
                    )
                st.table(table_data[:10])

            with col2:
                # Generate a sample prompt for visualization
                vocab = get_token_set(vocab_type)
                sample_prompt = generate_repeated_sequence(vocab, prefix_length, seed)

                st.subheader("Sample Prompt Structure")
                st.code(sample_prompt.text)
                st.markdown(f"**Expected next token:** `{sample_prompt.expected_next}`")
                st.markdown(
                    f"**Pattern:** Position {sample_prompt.first_occurrence_pos} → Position {sample_prompt.second_occurrence_pos}"
                )

            # Score plots
            st.subheader("Score Visualizations")
            col_a, col_b = st.columns(2)

            # Create HeadScore objects for plotting
            from src.analysis import HeadScore

            head_scores = [
                HeadScore(layer=l, head=h, score=s, attention_to_prev=0)
                for l, h, s in top_heads
            ]

            with col_a:
                fig = plot_induction_scores(head_scores, top_k=top_k)
                st.pyplot(fig)
                plt.close(fig)

            with col_b:
                fig = plot_induction_heatmap(
                    head_scores, model_info["n_layers"], model_info["n_heads"]
                )
                st.pyplot(fig)
                plt.close(fig)

    # ==================== TAB 2: ATTENTION VIEWER ====================
    with tab2:
        st.header("Attention Pattern Viewer")
        st.markdown(
            """
        Visualize the attention pattern for a specific head on a given prompt.
        Induction heads show a characteristic pattern of attending to earlier
        positions where similar patterns occurred.
        """
        )

        # Prompt input
        use_custom = st.checkbox("Use custom prompt", value=False)

        if use_custom:
            custom_text = st.text_input(
                "Enter prompt:",
                value="A B C D A",
                help="Enter space-separated tokens. Include a repeated token.",
            )
            prompt_text = custom_text
        else:
            vocab = get_token_set(vocab_type)
            prompt = generate_repeated_sequence(vocab, prefix_length, seed)
            prompt_text = prompt.text
            st.code(f"Generated prompt: {prompt_text}")

        # Head selection
        col1, col2 = st.columns(2)
        with col1:
            layer = st.selectbox(
                "Layer",
                range(model_info["n_layers"]),
                index=min(5, model_info["n_layers"] - 1),
            )
        with col2:
            head = st.selectbox(
                "Head",
                range(model_info["n_heads"]),
                index=0,
            )

        if st.button("📊 Show Attention Pattern", type="primary"):
            with st.spinner("Computing attention pattern..."):
                attn_pattern, tokens = get_attention_pattern(model, prompt_text, layer, head)

                st.subheader("Token Sequence")
                token_str = " | ".join([f"`{t}`" for t in tokens])
                st.markdown(token_str)

                st.subheader(f"Attention Pattern: Layer {layer}, Head {head}")
                fig = plot_attention_pattern(attn_pattern, tokens, layer, head)
                st.pyplot(fig)
                plt.close(fig)

                # Show some statistics
                st.subheader("Pattern Statistics")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Max Attention", f"{attn_pattern.max().item():.4f}")
                with col_b:
                    st.metric("Mean Attention", f"{attn_pattern.mean().item():.4f}")
                with col_c:
                    # Compute diagonal attention (attending to previous token)
                    diag_attn = torch.diagonal(attn_pattern, offset=-1).mean().item()
                    st.metric("Mean Prev-Token Attn", f"{diag_attn:.4f}")

    # ==================== TAB 3: ACTIVATION PATCHING ====================
    with tab3:
        st.header("Activation Patching Experiment")
        st.markdown(
            """
        **Activation patching** establishes causal relationships by:
        1. Running the model on a **clean** input (correct induction pattern)
        2. Running on a **corrupted** input (broken pattern)
        3. **Patching** activations from clean → corrupted to restore behavior

        If patching a head restores the model's ability to predict the correct next token,
        that head is causally important for induction.
        """
        )

        # Generate clean/corrupted pair
        vocab = get_token_set(vocab_type)

        if st.button("🎲 Generate New Prompt Pair"):
            st.session_state["patch_seed"] = seed + int(torch.rand(1).item() * 1000)

        patch_seed = st.session_state.get("patch_seed", seed)
        clean_prompt, corrupted_prompt = generate_corrupted_pair(
            vocab, prefix_length, patch_seed
        )

        # Display prompts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Clean Prompt")
            st.code(clean_prompt.text)
            st.markdown(f"**Expected next:** `{clean_prompt.expected_next}`")

        with col2:
            st.subheader("Corrupted Prompt")
            st.code(corrupted_prompt.text)
            st.markdown("*(First occurrence of key token changed)*")

        st.divider()

        # Patching configuration
        st.subheader("Patch Configuration")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            patch_type = st.selectbox(
                "Patch Type",
                ["attention_head", "residual_stream"],
                help="What to patch: specific attention head output or residual stream",
            )

        with col_b:
            patch_layer = st.selectbox(
                "Layer to Patch",
                range(model_info["n_layers"]),
                index=min(5, model_info["n_layers"] - 1),
            )

        with col_c:
            if patch_type == "attention_head":
                patch_head = st.selectbox(
                    "Head to Patch",
                    range(model_info["n_heads"]),
                    index=0,
                )
            else:
                patch_head = 0  # Not used for residual stream

        # Run patching
        if st.button("🔧 Run Patching Experiment", type="primary"):
            with st.spinner("Running patching experiment..."):
                result = run_patching_experiment(
                    model,
                    clean_prompt,
                    corrupted_prompt,
                    patch_type=patch_type,
                    layer=patch_layer,
                    head=patch_head,
                )

                st.session_state["patch_result"] = result

        # Display results
        if "patch_result" in st.session_state:
            result = st.session_state["patch_result"]

            st.subheader("Results")

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Clean P(target)",
                    f"{result.clean_prob:.4f}",
                    help="Probability of correct token with clean input",
                )
            with col2:
                st.metric(
                    "Corrupted P(target)",
                    f"{result.corrupted_prob:.4f}",
                    delta=f"{result.corrupted_prob - result.clean_prob:.4f}",
                    delta_color="inverse",
                )
            with col3:
                st.metric(
                    "Patched P(target)",
                    f"{result.patched_prob:.4f}",
                    delta=f"{result.patched_prob - result.corrupted_prob:.4f}",
                )
            with col4:
                recovery_pct = result.recovery_ratio * 100
                st.metric(
                    "Recovery",
                    f"{recovery_pct:.1f}%",
                    help="(patched - corrupted) / (clean - corrupted)",
                )

            # Plot
            fig = plot_patching_result(result)
            st.pyplot(fig)
            plt.close(fig)

            # Interpretation
            st.subheader("Interpretation")
            if result.recovery_ratio > 0.5:
                st.success(
                    f"✅ Patching **{result.patch_location}** recovers {recovery_pct:.1f}% of performance. "
                    f"This component is **causally important** for the induction behavior."
                )
            elif result.recovery_ratio > 0.1:
                st.warning(
                    f"⚠️ Patching **{result.patch_location}** shows partial recovery ({recovery_pct:.1f}%). "
                    f"This component contributes to but doesn't fully explain the behavior."
                )
            else:
                st.info(
                    f"ℹ️ Patching **{result.patch_location}** shows minimal recovery ({recovery_pct:.1f}%). "
                    f"This component may not be the primary mechanism for induction here."
                )

    # Footer
    st.divider()
    st.markdown(
        """
    ---
    **Induction Head Explorer** | Built with TransformerLens & Streamlit
    | [Learn more about mechanistic interpretability](https://www.neelnanda.io/mechanistic-interpretability/glossary)
    """
    )


if __name__ == "__main__":
    main()
