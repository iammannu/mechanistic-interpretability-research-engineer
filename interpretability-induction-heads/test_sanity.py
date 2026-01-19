"""
Sanity check script for the induction head analysis project.

Run with: python test_sanity.py

This verifies that all components work correctly and produce expected results.
"""

import sys
import torch

print("=" * 60)
print("INDUCTION HEAD ANALYSIS - SANITY CHECK")
print("=" * 60)


def test_imports():
    """Test that all modules import correctly."""
    print("\n[1/6] Testing imports...")
    try:
        from src.model import load_model, get_model_info
        from src.data_gen import generate_repeated_sequence, get_token_set
        from src.analysis import analyze_induction_heads, get_top_induction_heads
        from src.patching import run_patching_experiment
        from src.viz import plot_induction_scores, plot_attention_pattern
        print("    ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"    ✗ Import failed: {e}")
        return False


def test_data_generation():
    """Test prompt generation."""
    print("\n[2/6] Testing data generation...")
    from src.data_gen import generate_repeated_sequence, get_token_set, generate_corrupted_pair

    vocab = get_token_set("letters")
    prompt = generate_repeated_sequence(vocab, prefix_length=5, seed=42)

    checks = [
        (len(prompt.tokens) == 8, "Prompt has correct token count (8)"),
        (prompt.expected_next in vocab, "Expected token is in vocabulary"),
        (prompt.first_occurrence_pos < prompt.second_occurrence_pos, "First occurrence before second"),
    ]

    all_passed = True
    for passed, msg in checks:
        status = "✓" if passed else "✗"
        print(f"    {status} {msg}")
        all_passed = all_passed and passed

    # Test corrupted pair
    clean, corrupted = generate_corrupted_pair(vocab, prefix_length=5, seed=42)
    if clean.text != corrupted.text:
        print("    ✓ Clean and corrupted prompts differ")
    else:
        print("    ✗ Clean and corrupted prompts should differ")
        all_passed = False

    return all_passed


def test_model_loading():
    """Test model loading (minimal)."""
    print("\n[3/6] Testing model loading...")
    from src.model import load_model, get_model_info

    try:
        model = load_model("gpt2-small", device="cpu")
        info = get_model_info(model)

        checks = [
            (info["n_layers"] == 12, f"12 layers (got {info['n_layers']})"),
            (info["n_heads"] == 12, f"12 heads (got {info['n_heads']})"),
            (info["d_model"] == 768, f"d_model=768 (got {info['d_model']})"),
        ]

        all_passed = True
        for passed, msg in checks:
            status = "✓" if passed else "✗"
            print(f"    {status} {msg}")
            all_passed = all_passed and passed

        return all_passed, model
    except Exception as e:
        print(f"    ✗ Model loading failed: {e}")
        return False, None


def test_induction_analysis(model):
    """Test induction head detection."""
    print("\n[4/6] Testing induction analysis...")
    from src.data_gen import generate_repeated_sequence, get_token_set
    from src.analysis import analyze_induction_heads, get_top_induction_heads

    vocab = get_token_set("letters")
    prompt = generate_repeated_sequence(vocab, prefix_length=5, seed=42)

    # Single prompt analysis
    result = analyze_induction_heads(model, prompt)

    checks = [
        (len(result.head_scores) == 144, f"144 head scores (12×12, got {len(result.head_scores)})"),
        (result.head_scores[0].score >= result.head_scores[-1].score, "Scores sorted descending"),
        (0 <= result.head_scores[0].score <= 1, "Scores in [0,1] range"),
    ]

    all_passed = True
    for passed, msg in checks:
        status = "✓" if passed else "✗"
        print(f"    {status} {msg}")
        all_passed = all_passed and passed

    # Multi-prompt analysis
    top_heads = get_top_induction_heads(model, n_prompts=3, top_k=5, seed=42)
    if len(top_heads) == 5:
        print(f"    ✓ Got top 5 heads")
        top_score = top_heads[0][2]
        print(f"    ℹ Top head: L{top_heads[0][0]}H{top_heads[0][1]} with score {top_score:.4f}")
    else:
        print(f"    ✗ Expected 5 top heads, got {len(top_heads)}")
        all_passed = False

    return all_passed


def test_patching(model):
    """Test activation patching."""
    print("\n[5/6] Testing activation patching...")
    from src.data_gen import generate_corrupted_pair, get_token_set
    from src.patching import run_patching_experiment

    vocab = get_token_set("letters")
    clean, corrupted = generate_corrupted_pair(vocab, prefix_length=5, seed=42)

    result = run_patching_experiment(
        model,
        clean,
        corrupted,
        patch_type="attention_head",
        layer=5,
        head=1,
    )

    checks = [
        (0 <= result.clean_prob <= 1, "Clean prob in valid range"),
        (0 <= result.corrupted_prob <= 1, "Corrupted prob in valid range"),
        (0 <= result.patched_prob <= 1, "Patched prob in valid range"),
        (result.clean_prob > 0, "Clean prob is non-zero"),
    ]

    all_passed = True
    for passed, msg in checks:
        status = "✓" if passed else "✗"
        print(f"    {status} {msg}")
        all_passed = all_passed and passed

    print(f"    ℹ Clean: {result.clean_prob:.4f}, Corrupted: {result.corrupted_prob:.4f}, Patched: {result.patched_prob:.4f}")
    print(f"    ℹ Recovery ratio: {result.recovery_ratio:.2%}")

    return all_passed


def test_visualization():
    """Test visualization functions (no display)."""
    print("\n[6/6] Testing visualization...")
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    from src.viz import plot_induction_scores, plot_attention_pattern
    from src.analysis import HeadScore

    try:
        # Test score plotting
        scores = [HeadScore(layer=i, head=j, score=0.5-i*0.03, attention_to_prev=0.1)
                  for i in range(5) for j in range(3)]
        fig = plot_induction_scores(scores[:10])
        plt.close(fig)
        print("    ✓ Score bar chart created")

        # Test attention pattern
        attn = torch.rand(8, 8)
        tokens = ["<BOS>", "A", "B", "C", "D", "A", "B", "C"]
        fig = plot_attention_pattern(attn, tokens, layer=0, head=0)
        plt.close(fig)
        print("    ✓ Attention heatmap created")

        return True
    except Exception as e:
        print(f"    ✗ Visualization failed: {e}")
        return False


def main():
    """Run all sanity checks."""
    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test data generation
    results.append(("Data Generation", test_data_generation()))

    # Test model loading
    model_ok, model = test_model_loading()
    results.append(("Model Loading", model_ok))

    if model is not None:
        # Test analysis
        results.append(("Induction Analysis", test_induction_analysis(model)))

        # Test patching
        results.append(("Activation Patching", test_patching(model)))

    # Test visualization
    results.append(("Visualization", test_visualization()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        all_passed = all_passed and passed

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED! The project is working correctly.")
    else:
        print("SOME CHECKS FAILED. Please review the errors above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
