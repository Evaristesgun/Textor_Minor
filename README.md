# Textor Minor

> *Textor Minor* — Latin for "the little weaver."
> A *textor* weaves threads into cloth, and a language model weaves tokens into text (*textus*, "that which is woven").

**Textor Minor** is a small, **decoder-only GPT** built from scratch for learning. The goal is to understand every part of a modern language model by writing it myself, training it on modest hardware, and documenting what I learn along the way.

> **Status: early stage.** The architecture and training pipeline are still being built. Sections marked *TBD* will be filled in as the project progresses.

---

## Table of Contents

- [Goals](#goals)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training](#training)
- [Sampling](#sampling)
- [Results](#results)
- [Roadmap](#roadmap)
- [References](#references)
- [License](#license)

---

## Goals

- **Understand, don't import.** Implement the core pieces (tokenization, causal self-attention, the transformer block, the training loop) by hand instead of calling a ready-made model class.
- **Stay small.** Keep the model small enough to train on a single consumer GPU, or even a CPU, in a reasonable amount of time.
- **Keep it readable.** Clear code and comments matter more than squeezing out performance.
- **Record the process.** Keep notes on experiments, bugs and lessons learned.

## Architecture

Textor Minor follows the standard GPT-style, decoder-only transformer:

```
input tokens
    │
    ▼
Token Embedding + Positional Embedding
    │
    ▼
┌───────────────────────────────┐
│  Transformer Block  × N       │
│   ├─ LayerNorm                │
│   ├─ Masked Multi-Head        │
│   │   Self-Attention  (+res)  │
│   ├─ LayerNorm                │
│   └─ Feed-Forward MLP (+res)  │
└───────────────────────────────┘
    │
    ▼
Final LayerNorm
    │
    ▼
Linear (LM head) → softmax → next-token probabilities
```

- **Decoder-only:** a causal mask stops each position from attending to future tokens, so the model learns next-token prediction.
- **Pre-LayerNorm** blocks with residual connections, as in GPT-2.

### Default configuration (TBD)

| Hyperparameter        | Value |
|-----------------------|-------|
| Layers (`n_layer`)    | TBD   |
| Attention heads (`n_head`) | TBD |
| Embedding dim (`n_embd`) | TBD |
| Context length (`block_size`) | TBD |
| Vocabulary size       | TBD   |
| Parameters            | TBD   |
| Tokenizer             | TBD (character-level / BPE) |

## Project Structure

*(planned; will be updated as files are added)*

```
Textor_Minor/
├── data/            # dataset download & preprocessing scripts
├── textor/          # model and tokenizer source code
│   ├── model.py     # GPT model definition
│   └── tokenizer.py # tokenizer
├── train.py         # training loop
├── sample.py        # text generation
├── configs/         # hyperparameter configs
├── LICENSE
└── README.md
```

## Getting Started

### Requirements

- Python 3.10+
- PyTorch *(TBD: confirm version)*

### Installation

```bash
git clone https://github.com/Evaristesgun/Textor_Minor.git
cd Textor_Minor
pip install -r requirements.txt
```

## Training

*(TBD)*

```bash
# 1. Prepare the dataset
python data/prepare.py

# 2. Train
python train.py --config configs/default.yaml
```

**Dataset:** TBD

## Sampling

*(TBD)*

```bash
python sample.py --prompt "Once upon a time" --max_new_tokens 200 --temperature 0.8
```

## Results

*(Coming soon: loss curves, sample outputs and notes on what worked and what didn't.)*

## Roadmap

- [ ] Tokenizer (character-level first, then BPE)
- [ ] Data loading and train/val split
- [ ] Causal self-attention
- [ ] Transformer block and full GPT model
- [ ] Training loop with evaluation
- [ ] Text sampling (temperature, top-k)
- [ ] Checkpoint save/load
- [ ] Loss curves and experiment logs
- [ ] *Stretch:* RoPE, KV-cache, mixed-precision training

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Radford et al., [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019)
- Andrej Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) and *Let's build GPT: from scratch, in code, spelled out*

## License

This project is licensed under the [Apache License 2.0](LICENSE).
