"""
Activation patching utilities for causal intervention experiments.

Activation patching is a technique for establishing causal relationships between
model components and behavior. We run the model on a "clean" input and a "corrupted"
input, then patch activations from the clean run into the corrupted run to see
if we can restore the clean behavior.

This module implements patching for:
- Attention head outputs
- Residual stream activations
- MLP outputs
"""

import torch
from typing import Callable, Dict, Tuple, Optional, List
from dataclasses import dataclass
from transformer_lens import HookedTransformer, ActivationCache

from .data_gen import InductionPrompt


@dataclass
class PatchingResult:
    """Results from an activation patching experiment."""

    clean_prob: float  # Probability of target token on clean input
    corrupted_prob: float  # Probability of target token on corrupted input
    patched_prob: float  # Probability after patching
    recovery_ratio: float  # (patched - corrupted) / (clean - corrupted)
    target_token: str
    patch_location: str  # Description of what was patched


@dataclass
class PatchingSweepResult:
    """Results from sweeping patches across all layers/heads."""

    results: Dict[Tuple[int, int], PatchingResult]  # (layer, head) -> result
    clean_prob: float
    corrupted_prob: float
    target_token: str


def get_target_token_prob(
    model: HookedTransformer,
    logits: torch.Tensor,
    target_token: str,
    position: int = -1,
) -> float:
    """
    Get probability of target token at specified position.

    Args:
        model: The HookedTransformer model
        logits: Model output logits
        target_token: Target token string
        position: Position to check (-1 for last)

    Returns:
        Probability of target token
    """
    probs = torch.softmax(logits[0, position, :], dim=-1)

    # Handle tokenization - the target might need a space prefix
    try:
        target_id = model.to_single_token(target_token)
    except Exception:
        # Try with space prefix
        try:
            target_id = model.to_single_token(" " + target_token.strip())
        except Exception:
            # Try without space
            target_id = model.to_single_token(target_token.strip())

    return probs[target_id].item()


def create_patch_hook(
    clean_cache: ActivationCache,
    cache_key: str,
    head_idx: Optional[int] = None,
    position: Optional[int] = None,
) -> Callable:
    """
    Create a hook function that patches in activations from clean cache.

    Args:
        clean_cache: Cache from clean forward pass
        cache_key: Key to access in cache (e.g., "blocks.0.attn.hook_result")
        head_idx: If patching a specific head, the head index
        position: If patching a specific position, the position index

    Returns:
        Hook function for use with model.run_with_hooks
    """
    clean_activation = clean_cache[cache_key]

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        """Replace activation with clean version."""
        if head_idx is not None and position is not None:
            # Patch specific head at specific position
            # activation shape: [batch, pos, head, d_head] or [batch, pos, d_model]
            if activation.dim() == 4:  # Has head dimension
                activation[:, position, head_idx, :] = clean_activation[:, position, head_idx, :]
            else:
                activation[:, position, :] = clean_activation[:, position, :]
        elif head_idx is not None:
            # Patch all positions for specific head
            if activation.dim() == 4:
                activation[:, :, head_idx, :] = clean_activation[:, :, head_idx, :]
            else:
                activation = clean_activation  # Full replacement
        elif position is not None:
            # Patch specific position, all heads
            activation[:, position, :] = clean_activation[:, position, :]
        else:
            # Full replacement
            activation = clean_activation

        return activation

    return hook_fn


def patch_attention_head(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    target_token: str,
    layer: int,
    head: int,
    position: Optional[int] = None,
) -> PatchingResult:
    """
    Patch attention head output from clean to corrupted run.

    Args:
        model: The HookedTransformer model
        clean_tokens: Tokenized clean input
        corrupted_tokens: Tokenized corrupted input
        target_token: Token we expect model to predict
        layer: Layer to patch
        head: Head to patch
        position: Specific position to patch (None for all)

    Returns:
        PatchingResult with probabilities and recovery metrics
    """
    # Get clean activations and probability
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        clean_prob = get_target_token_prob(model, clean_logits, target_token)

        # Get corrupted baseline
        corrupted_logits = model(corrupted_tokens)
        corrupted_prob = get_target_token_prob(model, corrupted_logits, target_token)

        # Create patching hook for attention head output
        # The hook point for attention output is "blocks.{layer}.attn.hook_result"
        hook_name = f"blocks.{layer}.attn.hook_result"

        # For head-specific patching, we need to handle the z (attention output) tensor
        # Shape is [batch, pos, head, d_head]
        def head_patch_hook(activation: torch.Tensor, hook) -> torch.Tensor:
            clean_act = clean_cache[hook_name]
            if position is not None:
                activation[:, position, head, :] = clean_act[:, position, head, :]
            else:
                activation[:, :, head, :] = clean_act[:, :, head, :]
            return activation

        # Run corrupted with patch
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, head_patch_hook)],
        )
        patched_prob = get_target_token_prob(model, patched_logits, target_token)

    # Calculate recovery ratio
    prob_diff = clean_prob - corrupted_prob
    if abs(prob_diff) > 1e-6:
        recovery = (patched_prob - corrupted_prob) / prob_diff
    else:
        recovery = 0.0

    location = f"L{layer}H{head}"
    if position is not None:
        location += f"@pos{position}"

    return PatchingResult(
        clean_prob=clean_prob,
        corrupted_prob=corrupted_prob,
        patched_prob=patched_prob,
        recovery_ratio=recovery,
        target_token=target_token,
        patch_location=location,
    )


def patch_residual_stream(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    target_token: str,
    layer: int,
    position: Optional[int] = None,
    stream_type: str = "post",
) -> PatchingResult:
    """
    Patch residual stream activations from clean to corrupted run.

    Args:
        model: The HookedTransformer model
        clean_tokens: Tokenized clean input
        corrupted_tokens: Tokenized corrupted input
        target_token: Token we expect model to predict
        layer: Layer to patch (residual stream after this layer)
        position: Specific position to patch (None for all)
        stream_type: "pre" (before layer), "mid" (after attn), or "post" (after MLP)

    Returns:
        PatchingResult with probabilities and recovery metrics
    """
    # Determine hook point name
    if stream_type == "pre":
        hook_name = f"blocks.{layer}.hook_resid_pre"
    elif stream_type == "mid":
        hook_name = f"blocks.{layer}.hook_resid_mid"
    else:  # post
        hook_name = f"blocks.{layer}.hook_resid_post"

    with torch.no_grad():
        # Get clean activations
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        clean_prob = get_target_token_prob(model, clean_logits, target_token)

        # Get corrupted baseline
        corrupted_logits = model(corrupted_tokens)
        corrupted_prob = get_target_token_prob(model, corrupted_logits, target_token)

        # Create patching hook
        def resid_patch_hook(activation: torch.Tensor, hook) -> torch.Tensor:
            clean_act = clean_cache[hook_name]
            if position is not None:
                activation[:, position, :] = clean_act[:, position, :]
            else:
                activation = clean_act
            return activation

        # Run with patch
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, resid_patch_hook)],
        )
        patched_prob = get_target_token_prob(model, patched_logits, target_token)

    # Calculate recovery
    prob_diff = clean_prob - corrupted_prob
    if abs(prob_diff) > 1e-6:
        recovery = (patched_prob - corrupted_prob) / prob_diff
    else:
        recovery = 0.0

    location = f"resid_{stream_type}_L{layer}"
    if position is not None:
        location += f"@pos{position}"

    return PatchingResult(
        clean_prob=clean_prob,
        corrupted_prob=corrupted_prob,
        patched_prob=patched_prob,
        recovery_ratio=recovery,
        target_token=target_token,
        patch_location=location,
    )


def sweep_attention_head_patches(
    model: HookedTransformer,
    clean_prompt: InductionPrompt,
    corrupted_prompt: InductionPrompt,
    layers: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
) -> PatchingSweepResult:
    """
    Sweep patching across multiple layers and heads.

    Args:
        model: The HookedTransformer model
        clean_prompt: Clean induction prompt
        corrupted_prompt: Corrupted prompt
        layers: Layers to sweep (None for all)
        heads: Heads to sweep (None for all)

    Returns:
        PatchingSweepResult with all results
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))
    if heads is None:
        heads = list(range(model.cfg.n_heads))

    clean_tokens = model.to_tokens(clean_prompt.text, prepend_bos=True)
    corrupted_tokens = model.to_tokens(corrupted_prompt.text, prepend_bos=True)

    # Get baseline probabilities
    with torch.no_grad():
        clean_logits = model(clean_tokens)
        clean_prob = get_target_token_prob(model, clean_logits, clean_prompt.expected_next)

        corrupted_logits = model(corrupted_tokens)
        corrupted_prob = get_target_token_prob(model, corrupted_logits, clean_prompt.expected_next)

    results = {}
    for layer in layers:
        for head in heads:
            result = patch_attention_head(
                model,
                clean_tokens,
                corrupted_tokens,
                clean_prompt.expected_next,
                layer,
                head,
            )
            results[(layer, head)] = result

    return PatchingSweepResult(
        results=results,
        clean_prob=clean_prob,
        corrupted_prob=corrupted_prob,
        target_token=clean_prompt.expected_next,
    )


def run_patching_experiment(
    model: HookedTransformer,
    clean_prompt: InductionPrompt,
    corrupted_prompt: InductionPrompt,
    patch_type: str = "attention_head",
    layer: int = 0,
    head: int = 0,
) -> PatchingResult:
    """
    High-level function to run a single patching experiment.

    Args:
        model: The HookedTransformer model
        clean_prompt: Clean input prompt
        corrupted_prompt: Corrupted input prompt
        patch_type: "attention_head" or "residual_stream"
        layer: Layer to patch
        head: Head to patch (only for attention_head type)

    Returns:
        PatchingResult with experiment results
    """
    clean_tokens = model.to_tokens(clean_prompt.text, prepend_bos=True)
    corrupted_tokens = model.to_tokens(corrupted_prompt.text, prepend_bos=True)

    if patch_type == "attention_head":
        return patch_attention_head(
            model,
            clean_tokens,
            corrupted_tokens,
            clean_prompt.expected_next,
            layer,
            head,
        )
    elif patch_type == "residual_stream":
        return patch_residual_stream(
            model,
            clean_tokens,
            corrupted_tokens,
            clean_prompt.expected_next,
            layer,
        )
    else:
        raise ValueError(f"Unknown patch_type: {patch_type}")
