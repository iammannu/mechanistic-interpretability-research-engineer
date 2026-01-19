"""
Data generation utilities for induction head experiments.

This module generates synthetic repeated-token sequences that trigger induction behavior.
Induction heads complete patterns like "A B ... A" -> "B" by copying from earlier context.
"""

import random
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class InductionPrompt:
    """Container for an induction head test prompt."""

    text: str  # Full prompt text
    tokens: List[str]  # Individual token strings
    first_occurrence_pos: int  # Position of first occurrence of repeated token
    second_occurrence_pos: int  # Position of second occurrence (where model predicts)
    expected_next: str  # The token that should follow (copied from after first occurrence)


# Default token sets for generating sequences
LETTER_TOKENS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
WORD_TOKENS = [
    " cat",
    " dog",
    " bird",
    " fish",
    " tree",
    " house",
    " car",
    " book",
    " phone",
    " chair",
    " table",
    " lamp",
    " cup",
    " pen",
    " key",
    " ball",
    " star",
    " moon",
    " sun",
    " rain",
]
NUMBER_TOKENS = [str(i) for i in range(10)]


def get_token_set(set_name: str) -> List[str]:
    """
    Get a predefined token set by name.

    Args:
        set_name: One of "letters", "words", or "numbers"

    Returns:
        List of token strings
    """
    sets = {
        "letters": LETTER_TOKENS,
        "words": WORD_TOKENS,
        "numbers": NUMBER_TOKENS,
    }
    return sets.get(set_name, LETTER_TOKENS)


def generate_repeated_sequence(
    vocab: List[str],
    prefix_length: int = 5,
    seed: Optional[int] = None,
) -> InductionPrompt:
    """
    Generate a sequence with a repeated pattern for induction head testing.

    Creates: [random prefix] [A] [B] [random middle] [A] -> expect [B]

    The induction pattern works as follows:
    - First occurrence: position i has token A, position i+1 has token B
    - Second occurrence: position j has token A
    - Induction heads should predict B at position j+1

    Args:
        vocab: List of tokens to sample from
        prefix_length: Number of tokens before the repeated pattern
        seed: Random seed for reproducibility

    Returns:
        InductionPrompt with the generated sequence and metadata
    """
    if seed is not None:
        random.seed(seed)

    # Ensure we have enough unique tokens
    if len(vocab) < prefix_length + 3:
        raise ValueError(f"Vocab size {len(vocab)} too small for prefix_length {prefix_length}")

    # Sample unique tokens for the sequence
    sampled = random.sample(vocab, min(len(vocab), prefix_length + 3))

    # Build the sequence
    # Structure: [prefix tokens] [key token A] [value token B] [middle] [key token A again]
    prefix = sampled[:prefix_length]
    key_token = sampled[prefix_length]  # The token that repeats
    value_token = sampled[prefix_length + 1]  # The token to copy after key

    # Build full sequence: prefix + key + value + key (repeated)
    sequence = prefix + [key_token, value_token, key_token]

    # Calculate positions (0-indexed, accounting for BOS token that model will add)
    # BOS is position 0, so our tokens start at position 1
    first_key_pos = prefix_length + 1  # +1 for BOS
    second_key_pos = prefix_length + 3  # +1 for BOS

    # Join tokens into text
    text = " ".join(sequence) if not any(t.startswith(" ") for t in vocab) else "".join(sequence)

    return InductionPrompt(
        text=text,
        tokens=sequence,
        first_occurrence_pos=first_key_pos,
        second_occurrence_pos=second_key_pos,
        expected_next=value_token,
    )


def generate_batch(
    vocab: List[str],
    batch_size: int = 10,
    prefix_length: int = 5,
    base_seed: Optional[int] = None,
) -> List[InductionPrompt]:
    """
    Generate a batch of induction prompts.

    Args:
        vocab: Token vocabulary to use
        batch_size: Number of prompts to generate
        prefix_length: Prefix length for each prompt
        base_seed: Base random seed (each prompt uses base_seed + i)

    Returns:
        List of InductionPrompt instances
    """
    prompts = []
    for i in range(batch_size):
        seed = (base_seed + i) if base_seed is not None else None
        prompt = generate_repeated_sequence(vocab, prefix_length, seed)
        prompts.append(prompt)
    return prompts


def generate_simple_repeat(
    token_a: str = "A",
    token_b: str = "B",
    prefix: str = "X Y Z",
) -> InductionPrompt:
    """
    Generate a simple, predictable induction prompt.

    Creates: [prefix] [A] [B] [A] -> expect [B]

    Args:
        token_a: The key token that repeats
        token_b: The value token to be copied
        prefix: Space-separated prefix tokens

    Returns:
        InductionPrompt with the simple pattern
    """
    prefix_tokens = prefix.split()
    sequence = prefix_tokens + [token_a, token_b, token_a]
    text = " ".join(sequence)

    first_key_pos = len(prefix_tokens) + 1  # +1 for BOS
    second_key_pos = len(prefix_tokens) + 3  # +1 for BOS

    return InductionPrompt(
        text=text,
        tokens=sequence,
        first_occurrence_pos=first_key_pos,
        second_occurrence_pos=second_key_pos,
        expected_next=token_b,
    )


def generate_corrupted_pair(
    vocab: List[str],
    prefix_length: int = 5,
    seed: Optional[int] = None,
) -> Tuple[InductionPrompt, InductionPrompt]:
    """
    Generate a clean/corrupted prompt pair for patching experiments.

    The corrupted prompt has the first key token replaced with a different token,
    which should break the induction pattern.

    Args:
        vocab: Token vocabulary
        prefix_length: Prefix length
        seed: Random seed

    Returns:
        Tuple of (clean_prompt, corrupted_prompt)
    """
    clean = generate_repeated_sequence(vocab, prefix_length, seed)

    # For corruption, replace the first key token with something else
    available = [t for t in vocab if t not in clean.tokens]
    if not available:
        # If all tokens are used, just pick one that's different from key
        available = [t for t in vocab if t != clean.tokens[clean.first_occurrence_pos - 1]]

    if seed is not None:
        random.seed(seed + 1000)  # Different seed for corruption

    corrupt_token = random.choice(available)

    # Build corrupted sequence (replace first occurrence of key token)
    corrupt_tokens = clean.tokens.copy()
    key_idx = clean.first_occurrence_pos - 1  # -1 to convert from model pos to list idx
    corrupt_tokens[key_idx] = corrupt_token

    corrupt_text = " ".join(corrupt_tokens) if " " not in clean.text else "".join(corrupt_tokens)

    corrupted = InductionPrompt(
        text=corrupt_text,
        tokens=corrupt_tokens,
        first_occurrence_pos=clean.first_occurrence_pos,
        second_occurrence_pos=clean.second_occurrence_pos,
        expected_next=clean.expected_next,  # Same expected token (but model shouldn't predict it)
    )

    return clean, corrupted


def generate_longer_sequence(
    vocab: List[str],
    total_length: int = 15,
    repeat_position: int = 10,
    seed: Optional[int] = None,
) -> InductionPrompt:
    """
    Generate a longer sequence with the repeated pattern at a specified position.

    Useful for testing if induction heads work at different distances.

    Args:
        vocab: Token vocabulary
        total_length: Total number of tokens in sequence
        repeat_position: Position where the key token repeats
        seed: Random seed

    Returns:
        InductionPrompt with longer sequence
    """
    if seed is not None:
        random.seed(seed)

    if repeat_position >= total_length:
        raise ValueError("repeat_position must be less than total_length")

    # We need: first_key, value, ..., second_key
    # first_key at position 0 (after any prefix)
    # value at position 1
    # second_key at repeat_position

    # For simplicity, put key+value near the start
    first_key_idx = 0
    value_idx = 1
    second_key_idx = repeat_position

    sampled = random.sample(vocab, min(len(vocab), total_length))

    # Ensure we have unique tokens
    if len(sampled) < total_length:
        sampled = sampled * (total_length // len(sampled) + 1)
        sampled = sampled[:total_length]

    key_token = sampled[0]
    value_token = sampled[1]

    # Make sure the second key position has the same token
    sampled[second_key_idx] = key_token

    text = " ".join(sampled)

    return InductionPrompt(
        text=text,
        tokens=sampled,
        first_occurrence_pos=first_key_idx + 1,  # +1 for BOS
        second_occurrence_pos=second_key_idx + 1,  # +1 for BOS
        expected_next=value_token,
    )
