"""
Model loading and tokenization utilities for mechanistic interpretability analysis.

This module provides helper functions to load transformer models using TransformerLens
and handle tokenization for induction head experiments.
"""

import torch
from transformer_lens import HookedTransformer
from typing import List, Tuple, Optional
import functools


@functools.lru_cache(maxsize=1)
def load_model(model_name: str = "gpt2-small", device: str = "cpu") -> HookedTransformer:
    """
    Load a HookedTransformer model with caching.

    Args:
        model_name: Name of the model to load (e.g., "gpt2-small", "gpt2-medium")
        device: Device to load model on ("cpu" or "cuda")

    Returns:
        HookedTransformer model instance
    """
    print(f"Loading model {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        default_prepend_bos=True,
    )
    model.eval()
    print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads per layer")
    return model


def get_model_info(model: HookedTransformer) -> dict:
    """
    Extract key model configuration info.

    Args:
        model: The HookedTransformer model

    Returns:
        Dictionary with model configuration details
    """
    return {
        "name": model.cfg.model_name,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "d_model": model.cfg.d_model,
        "d_head": model.cfg.d_head,
        "n_ctx": model.cfg.n_ctx,
        "d_vocab": model.cfg.d_vocab,
    }


def tokenize(model: HookedTransformer, text: str) -> torch.Tensor:
    """
    Tokenize text using the model's tokenizer.

    Args:
        model: The HookedTransformer model
        text: Text to tokenize

    Returns:
        Tensor of token IDs with shape (1, seq_len)
    """
    return model.to_tokens(text, prepend_bos=True)


def decode_tokens(model: HookedTransformer, tokens: torch.Tensor) -> List[str]:
    """
    Decode token IDs to strings.

    Args:
        model: The HookedTransformer model
        tokens: Tensor of token IDs

    Returns:
        List of decoded token strings
    """
    if tokens.dim() == 2:
        tokens = tokens[0]
    return [model.tokenizer.decode(t.item()) for t in tokens]


def get_token_strs(model: HookedTransformer, text: str) -> List[str]:
    """
    Get the string representation of each token in the text.

    Args:
        model: The HookedTransformer model
        text: Input text

    Returns:
        List of token strings (including BOS token)
    """
    tokens = tokenize(model, text)
    return decode_tokens(model, tokens)


def run_with_cache(
    model: HookedTransformer,
    tokens: torch.Tensor,
    names_filter: Optional[callable] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Run model forward pass and capture activations.

    Args:
        model: The HookedTransformer model
        tokens: Input token tensor
        names_filter: Optional filter for which activations to cache

    Returns:
        Tuple of (logits, cache_dict)
    """
    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=names_filter,
            remove_batch_dim=False,
        )
    return logits, cache


def get_logit_probs(
    model: HookedTransformer,
    logits: torch.Tensor,
    position: int = -1,
) -> torch.Tensor:
    """
    Get probability distribution at a specific position.

    Args:
        model: The HookedTransformer model
        logits: Logits tensor from model output
        position: Position to get probs for (-1 for last)

    Returns:
        Probability distribution tensor
    """
    return torch.softmax(logits[0, position, :], dim=-1)


def get_top_predictions(
    model: HookedTransformer,
    logits: torch.Tensor,
    position: int = -1,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Get top-k predictions at a position.

    Args:
        model: The HookedTransformer model
        logits: Logits tensor
        position: Position to get predictions for
        k: Number of top predictions

    Returns:
        List of (token_string, probability) tuples
    """
    probs = get_logit_probs(model, logits, position)
    top_probs, top_indices = torch.topk(probs, k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        token_str = model.tokenizer.decode(idx.item())
        results.append((token_str, prob.item()))

    return results


def get_token_probability(
    model: HookedTransformer,
    logits: torch.Tensor,
    target_token: str,
    position: int = -1,
) -> float:
    """
    Get probability of a specific token at a position.

    Args:
        model: The HookedTransformer model
        logits: Logits tensor
        target_token: Target token string
        position: Position to check

    Returns:
        Probability of the target token
    """
    probs = get_logit_probs(model, logits, position)
    target_id = model.to_single_token(target_token)
    return probs[target_id].item()
