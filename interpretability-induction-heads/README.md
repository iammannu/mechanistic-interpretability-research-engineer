# Induction Head Explorer

A mechanistic interpretability project for detecting, visualizing, and causally verifying **induction heads** in transformer language models (GPT-2).

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TransformerLens](https://img.shields.io/badge/TransformerLens-1.10+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## What are Induction Heads?

**Induction heads** are attention heads in transformers that implement a simple but powerful pattern-completion algorithm. When the model sees a sequence like `A B ... A`, induction heads attend from the second `A` back to the first `A`, enabling the model to predict `B` as the next token by "copying" what followed the first occurrence.

This mechanism is fundamental to in-context learning and was first characterized in the paper ["In-context Learning and Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) by Olsson et al.

Key properties of induction heads:
- They enable **copying** patterns from earlier in the context
- They typically emerge in **middle layers** of the transformer
- They work together with **previous token heads** (which attend to the prior token)
- They are causally important for in-context learning capabilities

## Features

- **Induction Head Detection**: Automatically scan all attention heads and rank them by induction score
- **Attention Visualization**: Interactive heatmaps showing attention patterns for any head
- **Activation Patching**: Causal intervention experiments demonstrating which heads are critical for pattern completion
- **Interactive UI**: Streamlit-based interface for exploring results without code

## Project Structure

```
interpretability-induction-heads/
├── app.py                    # Streamlit UI entry point
├── src/
│   ├── model.py              # Model loading & tokenization helpers
│   ├── data_gen.py           # Synthetic prompt generation
│   ├── analysis.py           # Induction scoring algorithms
│   ├── patching.py           # Activation patching utilities
│   └── viz.py                # Visualization functions
├── notebooks/
│   └── induction_heads_analysis.ipynb  # Detailed walkthrough
├── figures/                  # Saved plots
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/interpretability-induction-heads.git
cd interpretability-induction-heads

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run the Streamlit App

```bash
streamlit run app.py
```

This opens an interactive web interface where you can:
1. **Scan for induction heads** across all layers
2. **Visualize attention patterns** for specific heads
3. **Run patching experiments** to verify causal importance

### Run the Notebook

```bash
jupyter notebook notebooks/induction_heads_analysis.ipynb
```

The notebook provides a step-by-step walkthrough of the analysis with explanations.

### Use as a Library

```python
from src.model import load_model
from src.data_gen import generate_repeated_sequence, get_token_set
from src.analysis import get_top_induction_heads, analyze_induction_heads

# Load model
model = load_model("gpt2-small", device="cpu")

# Find top induction heads
top_heads = get_top_induction_heads(model, vocab_name="letters", n_prompts=10)
print("Top induction heads:", top_heads[:5])

# Analyze a specific prompt
vocab = get_token_set("letters")
prompt = generate_repeated_sequence(vocab, prefix_length=5, seed=42)
result = analyze_induction_heads(model, prompt)
```

## Methodology

### Induction Score

The **induction score** measures how much an attention head attends from a repeated token back to its first occurrence. For a prompt structured as:

```
[prefix tokens] [A] [B] ... [A]
```

We measure the attention weight from position `second_A` to position `first_A`. Heads with high scores consistently do this across many prompts.

### Activation Patching

**Activation patching** establishes causal relationships:

1. **Clean run**: Run model on correct induction prompt (e.g., `X Y Z A B A` → predicts `B`)
2. **Corrupted run**: Run on corrupted prompt where the pattern is broken (e.g., `X Y Z C B A` → doesn't predict `B`)
3. **Patched run**: Run corrupted prompt but replace specific activations with those from the clean run

If patching a head's output restores the correct prediction, that head is **causally important** for the induction behavior.

**Recovery ratio** = (patched_prob - corrupted_prob) / (clean_prob - corrupted_prob)

A recovery ratio near 1.0 indicates the patched component fully explains the behavior.

## Screenshots

*Coming soon - run the app and explore!*

### Induction Score Bar Chart
Shows top induction heads ranked by their scores.

### Attention Heatmap
Visualizes the attention pattern of a specific head, showing the characteristic "induction stripe" where repeated tokens attend to earlier occurrences.

### Patching Results
Bar chart comparing clean, corrupted, and patched probabilities.

## Expected Results

When running on GPT-2 small:

1. **Top induction heads** typically appear in layers 5-8
2. **Induction scores** for top heads are usually 0.2-0.5 (meaning 20-50% of attention goes to the key position)
3. **Patching recovery** for top heads should be 30-80%, confirming causal importance
4. **Attention patterns** show clear diagonal "stripe" patterns characteristic of induction

## Sanity Checks

To verify the code works correctly:

1. **Model Loading**: The model should report 12 layers and 12 heads for GPT-2 small
2. **Induction Scan**: Top heads should have scores > 0.1; random heads should score < 0.05
3. **Attention Pattern**: For a prompt like "A B C A", the induction head should show high attention from position 4 (second A) to position 1 (first A)
4. **Patching**: Clean probability should be noticeably higher than corrupted; patching top induction heads should partially restore the probability

## Performance Notes

- Runs on **CPU** (GPU not required)
- Model loading takes ~30 seconds on first run (cached afterwards)
- Full induction scan takes ~1-2 minutes for 10 prompts
- Patching experiments run in ~5 seconds each

## References

- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) - Olsson et al.
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) - Elhage et al.
- [TransformerLens Documentation](https://neelnanda-io.github.io/TransformerLens/)
- [Neel Nanda's Mechanistic Interpretability Glossary](https://www.neelnanda.io/mechanistic-interpretability/glossary)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or PR for:
- Bug fixes
- New analysis methods
- Additional visualizations
- Documentation improvements
