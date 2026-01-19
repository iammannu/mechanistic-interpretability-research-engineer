"""
Visualization utilities for induction head analysis.

This module provides plotting functions for:
- Attention pattern heatmaps
- Induction score bar charts
- Patching experiment results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Tuple, Optional, Dict
import io

from .analysis import HeadScore, InductionAnalysisResult
from .patching import PatchingResult, PatchingSweepResult


def plot_attention_pattern(
    attention: torch.Tensor,
    tokens: List[str],
    layer: int,
    head: int,
    title: Optional[str] = None,
    highlight_positions: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot attention pattern as a heatmap.

    Args:
        attention: Attention weights tensor [seq_len, seq_len]
        tokens: List of token strings
        layer: Layer index (for title)
        head: Head index (for title)
        title: Optional custom title
        highlight_positions: Optional (query_pos, key_pos) to highlight
        figsize: Figure size
        cmap: Colormap name

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()

    # Create heatmap
    im = ax.imshow(attention, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Set ticks and labels
    n_tokens = len(tokens)
    ax.set_xticks(range(n_tokens))
    ax.set_yticks(range(n_tokens))

    # Truncate long tokens for display
    display_tokens = [t[:10] + "..." if len(t) > 10 else t for t in tokens]
    ax.set_xticklabels(display_tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(display_tokens, fontsize=8)

    # Labels
    ax.set_xlabel("Key (Source)")
    ax.set_ylabel("Query (Destination)")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Attention Pattern: Layer {layer}, Head {head}")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Attention Weight")

    # Highlight specific positions if requested
    if highlight_positions:
        query_pos, key_pos = highlight_positions
        rect = plt.Rectangle(
            (key_pos - 0.5, query_pos - 0.5),
            1,
            1,
            fill=False,
            edgecolor="red",
            linewidth=3,
        )
        ax.add_patch(rect)

    plt.tight_layout()
    return fig


def plot_induction_scores(
    head_scores: List[HeadScore],
    top_k: int = 20,
    title: str = "Induction Head Scores",
    figsize: Tuple[int, int] = (12, 6),
    color: str = "#2ecc71",
) -> plt.Figure:
    """
    Plot bar chart of induction scores.

    Args:
        head_scores: List of HeadScore objects (should be pre-sorted)
        top_k: Number of top heads to display
        title: Plot title
        figsize: Figure size
        color: Bar color

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Take top k
    top_heads = head_scores[:top_k]

    # Create labels and values
    labels = [f"L{h.layer}H{h.head}" for h in top_heads]
    scores = [h.score for h in top_heads]

    # Create bars
    bars = ax.bar(range(len(labels)), scores, color=color, edgecolor="black", linewidth=0.5)

    # Customize
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Attention Head (Layer, Head)")
    ax.set_ylabel("Induction Score")
    ax.set_title(title)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(
            f"{score:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylim(0, max(scores) * 1.15 if scores else 1)
    plt.tight_layout()
    return fig


def plot_induction_heatmap(
    head_scores: List[HeadScore],
    n_layers: int,
    n_heads: int,
    title: str = "Induction Scores by Layer and Head",
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "YlOrRd",
) -> plt.Figure:
    """
    Plot induction scores as a layer x head heatmap.

    Args:
        head_scores: List of HeadScore objects
        n_layers: Number of layers in model
        n_heads: Number of heads per layer
        title: Plot title
        figsize: Figure size
        cmap: Colormap

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Build score matrix
    score_matrix = np.zeros((n_layers, n_heads))
    for hs in head_scores:
        score_matrix[hs.layer, hs.head] = hs.score

    # Create heatmap
    im = ax.imshow(score_matrix, cmap=cmap, aspect="auto")

    # Labels
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels([f"H{i}" for i in range(n_heads)])
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Induction Score")

    # Add text annotations for high scores
    for i in range(n_layers):
        for j in range(n_heads):
            score = score_matrix[i, j]
            if score > 0.1:  # Only annotate notable scores
                ax.text(
                    j,
                    i,
                    f"{score:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if score > 0.5 else "black",
                )

    plt.tight_layout()
    return fig


def plot_patching_result(
    result: PatchingResult,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Plot a single patching experiment result.

    Args:
        result: PatchingResult object
        title: Optional custom title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    categories = ["Clean", "Corrupted", "Patched"]
    probs = [result.clean_prob, result.corrupted_prob, result.patched_prob]
    colors = ["#27ae60", "#e74c3c", "#3498db"]

    bars = ax.bar(categories, probs, color=colors, edgecolor="black", linewidth=1)

    # Add value labels
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.annotate(
            f"{prob:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel(f"P('{result.target_token}')")
    ax.set_ylim(0, max(probs) * 1.2 if max(probs) > 0 else 0.1)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Patching Experiment: {result.patch_location}\nRecovery: {result.recovery_ratio:.2%}")

    # Add recovery ratio annotation
    ax.text(
        0.95,
        0.95,
        f"Recovery: {result.recovery_ratio:.2%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    return fig


def plot_patching_sweep(
    sweep_result: PatchingSweepResult,
    n_layers: int,
    n_heads: int,
    metric: str = "patched_prob",
    title: str = "Patching Sweep Results",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot results from a patching sweep as a heatmap.

    Args:
        sweep_result: PatchingSweepResult object
        n_layers: Number of layers
        n_heads: Number of heads
        metric: Which metric to plot ("patched_prob" or "recovery_ratio")
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Build matrix
    matrix = np.zeros((n_layers, n_heads))
    for (layer, head), result in sweep_result.results.items():
        if metric == "patched_prob":
            matrix[layer, head] = result.patched_prob
        else:
            matrix[layer, head] = result.recovery_ratio

    # Color normalization
    if metric == "recovery_ratio":
        # Center at 0 for recovery ratio
        vmax = max(abs(matrix.min()), abs(matrix.max()))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = "RdBu"
    else:
        norm = None
        cmap = "YlOrRd"

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", norm=norm)

    # Labels
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels([f"H{i}" for i in range(n_heads)])
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"{title}\nClean: {sweep_result.clean_prob:.4f}, Corrupted: {sweep_result.corrupted_prob:.4f}")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    label = "P(target)" if metric == "patched_prob" else "Recovery Ratio"
    cbar.set_label(label)

    plt.tight_layout()
    return fig


def plot_comparison_bar(
    values: Dict[str, float],
    title: str = "Comparison",
    ylabel: str = "Value",
    figsize: Tuple[int, int] = (8, 5),
    colors: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Generic comparison bar chart.

    Args:
        values: Dictionary of label -> value
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size
        colors: Optional list of colors

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(values.keys())
    vals = list(values.values())

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    bars = ax.bar(labels, vals, color=colors, edgecolor="black")

    for bar, val in zip(bars, vals):
        height = bar.get_height()
        ax.annotate(
            f"{val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 0.1)

    plt.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure) -> bytes:
    """
    Convert matplotlib figure to PNG bytes.

    Args:
        fig: Matplotlib figure

    Returns:
        PNG image as bytes
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """
    Save figure to file.

    Args:
        fig: Matplotlib figure
        path: Output file path
        dpi: Resolution
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved to {path}")
