"""
Induction head detection and scoring analysis.

This module implements the core induction head detection algorithm based on
attention pattern analysis. Induction heads are identified by their characteristic
attention pattern: attending from position i to position j-1, where the token at
position j matches the token at position i.

The induction score measures how much each attention head exhibits this pattern.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from transformer_lens import HookedTransformer, ActivationCache

from .data_gen import InductionPrompt, generate_batch, get_token_set


@dataclass
class HeadScore:
    """Container for a single head's induction score."""

    layer: int
    head: int
    score: float
    attention_to_prev: float  # Attention paid to position before repeated token


@dataclass
class InductionAnalysisResult:
    """Full results from induction head analysis."""

    head_scores: List[HeadScore]
    prompt: InductionPrompt
    tokens: List[str]
    attention_patterns: Dict[Tuple[int, int], torch.Tensor]  # (layer, head) -> pattern


def compute_induction_score_single(
    attention_pattern: torch.Tensor,
    first_pos: int,
    second_pos: int,
) -> float:
    """
    Compute induction score for a single attention pattern.

    The induction score measures attention from the second occurrence of a token
    to the position just before the first occurrence (which enables copying the
    token that followed the first occurrence).

    For pattern "... A B ... A", the induction head at position of second A
    should attend to position of B (to copy B as the next prediction).

    Actually, the standard definition is:
    - Position second_pos (second A) should attend to first_pos (first A's position)
      because the "previous token head" component attends to the previous token,
      and the induction head attends to where a similar pattern occurred.

    More precisely for copying: position second_pos attends to first_pos
    (where the pattern A...B occurred), enabling prediction of B.

    Args:
        attention_pattern: Attention weights [seq_len, seq_len] for one head
        first_pos: Position of first occurrence of key token
        second_pos: Position of second occurrence

    Returns:
        Induction score (attention from second_pos to first_pos)
    """
    # The induction score is the attention weight from the query at second_pos
    # to the key at first_pos (where the pattern to copy is)
    if second_pos >= attention_pattern.shape[0] or first_pos >= attention_pattern.shape[1]:
        return 0.0

    # Attention from second occurrence to first occurrence
    score = attention_pattern[second_pos, first_pos].item()

    return score


def compute_all_head_scores(
    model: HookedTransformer,
    cache: ActivationCache,
    prompt: InductionPrompt,
) -> List[HeadScore]:
    """
    Compute induction scores for all heads in the model.

    Args:
        model: The HookedTransformer model
        cache: Activation cache from forward pass
        prompt: The induction prompt used

    Returns:
        List of HeadScore objects for all heads, sorted by score descending
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    scores = []

    for layer in range(n_layers):
        # Get attention pattern for this layer: [batch, head, query_pos, key_pos]
        pattern_key = f"blocks.{layer}.attn.hook_pattern"
        if pattern_key not in cache:
            # Try alternative key format
            pattern_key = ("pattern", layer)

        try:
            attn_pattern = cache[pattern_key]
        except KeyError:
            continue

        # Remove batch dimension if present
        if attn_pattern.dim() == 4:
            attn_pattern = attn_pattern[0]  # [head, query, key]

        for head in range(n_heads):
            head_pattern = attn_pattern[head]  # [query, key]

            score = compute_induction_score_single(
                head_pattern,
                prompt.first_occurrence_pos,
                prompt.second_occurrence_pos,
            )

            # Also compute attention to the position before first occurrence
            # (this is where the value to copy is)
            attn_to_prev = 0.0
            if prompt.first_occurrence_pos > 0:
                attn_to_prev = head_pattern[
                    prompt.second_occurrence_pos, prompt.first_occurrence_pos + 1
                ].item()

            scores.append(
                HeadScore(
                    layer=layer,
                    head=head,
                    score=score,
                    attention_to_prev=attn_to_prev,
                )
            )

    # Sort by score descending
    scores.sort(key=lambda x: x.score, reverse=True)

    return scores


def analyze_induction_heads(
    model: HookedTransformer,
    prompt: InductionPrompt,
    top_k: int = 10,
) -> InductionAnalysisResult:
    """
    Full analysis of induction heads for a given prompt.

    Args:
        model: The HookedTransformer model
        prompt: The induction prompt to analyze
        top_k: Number of top heads to include detailed patterns for

    Returns:
        InductionAnalysisResult with scores and attention patterns
    """
    # Tokenize
    tokens = model.to_tokens(prompt.text, prepend_bos=True)
    token_strs = [model.tokenizer.decode(t.item()) for t in tokens[0]]

    # Run with cache, only caching attention patterns
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: "pattern" in name,
        )

    # Compute scores
    head_scores = compute_all_head_scores(model, cache, prompt)

    # Extract attention patterns for top heads
    attention_patterns = {}
    for hs in head_scores[:top_k]:
        pattern_key = f"blocks.{hs.layer}.attn.hook_pattern"
        try:
            attn = cache[pattern_key]
            if attn.dim() == 4:
                attn = attn[0]
            attention_patterns[(hs.layer, hs.head)] = attn[hs.head].cpu()
        except KeyError:
            pass

    return InductionAnalysisResult(
        head_scores=head_scores,
        prompt=prompt,
        tokens=token_strs,
        attention_patterns=attention_patterns,
    )


def batch_induction_scores(
    model: HookedTransformer,
    prompts: List[InductionPrompt],
) -> Dict[Tuple[int, int], float]:
    """
    Compute average induction scores across a batch of prompts.

    Args:
        model: The HookedTransformer model
        prompts: List of induction prompts

    Returns:
        Dictionary mapping (layer, head) to average score
    """
    all_scores: Dict[Tuple[int, int], List[float]] = {}

    for prompt in prompts:
        result = analyze_induction_heads(model, prompt)
        for hs in result.head_scores:
            key = (hs.layer, hs.head)
            if key not in all_scores:
                all_scores[key] = []
            all_scores[key].append(hs.score)

    # Average scores
    avg_scores = {key: np.mean(scores) for key, scores in all_scores.items()}

    return avg_scores


def get_top_induction_heads(
    model: HookedTransformer,
    vocab_name: str = "letters",
    n_prompts: int = 10,
    prefix_length: int = 5,
    top_k: int = 10,
    seed: int = 42,
) -> List[Tuple[int, int, float]]:
    """
    Identify top induction heads by averaging scores over multiple prompts.

    Args:
        model: The HookedTransformer model
        vocab_name: Token set to use ("letters", "words", "numbers")
        n_prompts: Number of prompts to average over
        prefix_length: Prefix length for generated prompts
        top_k: Number of top heads to return
        seed: Random seed

    Returns:
        List of (layer, head, avg_score) tuples, sorted by score
    """
    vocab = get_token_set(vocab_name)
    prompts = generate_batch(vocab, n_prompts, prefix_length, seed)

    avg_scores = batch_induction_scores(model, prompts)

    # Sort and return top k
    sorted_heads = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    return [(layer, head, score) for (layer, head), score in sorted_heads[:top_k]]


def get_attention_pattern(
    model: HookedTransformer,
    text: str,
    layer: int,
    head: int,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Get attention pattern for a specific head on given text.

    Args:
        model: The HookedTransformer model
        text: Input text
        layer: Layer index
        head: Head index

    Returns:
        Tuple of (attention_pattern [seq, seq], token_strings)
    """
    tokens = model.to_tokens(text, prepend_bos=True)
    token_strs = [model.tokenizer.decode(t.item()) for t in tokens[0]]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: f"blocks.{layer}.attn.hook_pattern" in name,
        )

    pattern_key = f"blocks.{layer}.attn.hook_pattern"
    attn = cache[pattern_key]
    if attn.dim() == 4:
        attn = attn[0]

    return attn[head].cpu(), token_strs


def compute_induction_stripe_score(
    attention_pattern: torch.Tensor,
    offset: int = 1,
) -> float:
    """
    Compute a more general induction score based on attention to offset positions.

    Induction heads characteristically attend to positions that are a fixed offset
    back from positions with matching tokens. This function computes the average
    attention along the "induction stripe" - the diagonal offset by the pattern length.

    Args:
        attention_pattern: Attention weights [seq_len, seq_len]
        offset: The offset to check (typically sequence repeat distance)

    Returns:
        Average attention along the offset diagonal
    """
    seq_len = attention_pattern.shape[0]
    if seq_len <= offset:
        return 0.0

    # Compute attention along the diagonal offset by `offset` positions
    stripe_attn = []
    for i in range(offset, seq_len):
        stripe_attn.append(attention_pattern[i, i - offset].item())

    return np.mean(stripe_attn) if stripe_attn else 0.0
