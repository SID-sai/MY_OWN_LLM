# GPT From Scratch: Pretraining and Instruction-Tuning a 163M-Parameter Language Model on One Laptop GPU

[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-ee4c2c)](https://pytorch.org/)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-SAISIDD%2Fsai--gpt--from--scratch-yellow)](https://huggingface.co/SAISIDD/sai-gpt-from-scratch)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

A decoder-only Transformer (GPT-2 style) implemented in PyTorch and taken through the **full lifecycle** on consumer hardware (a single RTX 5060 Laptop GPU): data pipeline, pretraining on ~3B tokens of FineWeb-Edu, supervised instruction fine-tuning on ~877k examples, quantitative and qualitative evaluation, and an interactive chat loop.

The goal was not to beat anything. It was to understand every moving part of an LLM by building it, training it for real, and measuring honestly what a model of this size can and cannot do.

**Trained weights:** [huggingface.co/SAISIDD/sai-gpt-from-scratch](https://huggingface.co/SAISIDD/sai-gpt-from-scratch)

> **Status:** this is **v1**, complete end to end (pretraining, SFT, evaluation, chat, weights released). Newer versions trained on more data are planned (see [Roadmap](#roadmap)). The model is small and its factual reliability is poor (see [Limitations](#limitations-and-honest-notes)). Please read the results as a learning and engineering project, not a state-of-the-art system.

---

## Contents

- [Results at a glance](#results-at-a-glance)
- [What is implemented here](#what-is-implemented-here)
- [Model and training configuration](#model-and-training-configuration)
- [Pipeline](#pipeline)
- [Repository layout](#repository-layout)
- [Quickstart: use the released model](#quickstart-use-the-released-model)
- [Reproducing the training](#reproducing-the-training)
- [Theory notes](#theory-notes)
- [Limitations and honest notes](#limitations-and-honest-notes)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [Author](#author)
- [License](#license)

---

## Results at a glance

| Stage | Metric | Value |
|---|---|---|
| Model | Parameters | **163,009,536** (12 layers, 768-d, 12 heads, 1024 context) |
| Pretraining data | FineWeb-Edu, GPT-2 BPE tokens | 3.003B tokens (3.000B train / 3M held-out validation) |
| Pretraining | Tokens processed | ~2.9B in total, across three resumed runs |
| Pretraining | Loss at last logged evaluation (update 60,000) | train 3.284, validation **3.318** (perplexity ≈ 27.6; estimated on 50 validation batches) |
| Instruction tuning | Examples | 877,086 train / 51,594 validation / 103,186 test |
| Instruction tuning | Validation cross-entropy / perplexity | **2.037 / 7.67** (full validation set, bf16) |
| LLM-as-judge | Mean score, 100 held-out test examples | **37.6 / 100** (std 39.1; judge: Llama 3.1 8B via Ollama, greedy decoding) |

How to read these numbers:

- The pretrained model starts at a loss of ≈ 10.99 (roughly ln 50,257, i.e. a uniform guess over the vocabulary) and reaches ≈ 3.3 on held-out text.
- Before instruction tuning, the model does not follow the prompt format. It repeats the same answer and the `### Response:` marker in a loop (qualitative baseline in `results/`). After tuning, it produces fluent, on-format answers.
- The judge score is low and the spread is large (std 39), meaning the judge gave many near-0 and many high scores and few in between. Expect a 163M model trained on a few billion tokens to behave this way. The judge is itself a small model, so this number is a rough signal, not a benchmark.
- No standard benchmarks (HellaSwag, ARC, MMLU, ...) have been run yet. They are on the [roadmap](#planned-improvements).

Loss curve for the final SFT phase: `results/sft_loss_curve.png`.
Sample generations before and after SFT (unselected, first test examples): `results/instruction_before_sft_sample.json`, `results/instruction_after_sft_sample.json`.

---

## What is implemented here

**Written by hand in PyTorch (see `notebooks/my_own_llm.ipynb`):**

- Token + learned positional embeddings, embedding dropout
- Multi-head causal self-attention (Q/K/V projections, head splitting, output projection)
- Pre-LayerNorm Transformer blocks with residual connections and a 4x GELU feed-forward network
- The full GPT model and an autoregressive generator (greedy, temperature, top-k, EOS handling)
- A **memory-mapped token pipeline** (`uint16` token stream, ~6 GB on disk) so 3B tokens train without loading them into RAM
- A training loop with **gradient accumulation**, **linear warm-up + cosine learning-rate decay**, gradient clipping, bf16 autocast, periodic evaluation, and **checkpoint/resume** logic (the pretraining run survived several crashes this way)
- Instruction data preparation: conversation to (instruction, response) pairs, Alpaca-style prompt template, custom collate function with padding and `-100` label masking
- Evaluation: validation loss and perplexity, qualitative before/after comparison, and an automated LLM-as-judge pipeline against a local Ollama model
- A simple chat widget for interactive use

**Library components used (deliberately, for speed):**

- `tiktoken` for the GPT-2 BPE tokenizer (not re-implemented)
- `torch.nn.functional.scaled_dot_product_attention` (fused, causal) for the attention computation. The explicit attention-mask version is covered in the [theory notes](#theory-notes).
- `F.layer_norm` and `F.gelu(approximate="tanh")` for the final training runs. The hand-written LayerNorm and GELU are kept in the notebook as commented-out cells.

---

## Model and training configuration

| | |
|---|---|
| Architecture | Decoder-only Transformer, pre-LN, learned absolute positions |
| Layers / heads / width | 12 / 12 / 768 |
| Context length | 1024 |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Weight tying | **No** (token embedding and output head are separate, no output bias) |
| Dropout | 0.1 (pretraining), QKV bias off |
| Precision | bf16 autocast, fp32 loss |
| Hardware | 1x NVIDIA RTX 5060 Laptop GPU |

**Pretraining (final phase):** AdamW (peak LR 4e-4 to 4e-5 cosine, 200 warm-up updates, weight decay 0.1), gradient clipping at 1.0, micro-batch 2 x 1024 tokens x 16 accumulation steps = **32,768 tokens per update**, 60,000 updates (~1.97B tokens in this phase).

**Instruction tuning:** AdamW (LR 5e-5, weight decay 0.1), batch size 1, gradient clipping at 1.0, bf16 autocast. Run in three resumed phases (152,500 + 160,937 + 563,649 examples, about 877k, i.e. approximately one epoch; see the note under [Limitations](#limitations-and-honest-notes)). The last phase took about 11.8 hours.

**Datasets:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) for pretraining, [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) for instruction tuning.

---

## Pipeline

```
FineWeb-Edu parquet --> GPT-2 BPE --> fineweb_tokens.bin (uint16 memmap, 3.003B tokens)
                                            |
                                            v
                         pretraining (grad-accum, cosine LR, bf16, resumable)
                                            |
                                            v
SmolTalk (parquet) --> prompt template --> SFT (loss on full sequence, -100 on padding)
                                            |
                                            v
           validation loss / perplexity  +  LLM-as-judge (Llama 3.1 8B, Ollama)  +  chat widget
```

---

## Repository layout

```
.
├── notebooks/
│   └── my_own_llm.ipynb          # the whole project: model, data, training, SFT, evaluation, chat
├── docs/
│   └── transformer_math_flow.pdf # handwritten step-by-step math of one forward pass
├── results/                      # loss curve, before/after sample generations, SFT loss log
├── requirements.txt
├── .gitignore                    # weights, token files and datasets are NOT tracked
├── LICENSE
└── README.md
```

Model weights (~700 MB) are not stored in Git. They are hosted on Hugging Face: [SAISIDD/sai-gpt-from-scratch](https://huggingface.co/SAISIDD/sai-gpt-from-scratch).

---

## Quickstart: use the released model

You do not need to retrain anything to try the model.

**1. Install and download the weights**

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# RTX 50-series GPUs need a CUDA 12.8 PyTorch build:
# pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download("SAISIDD/sai-gpt-from-scratch", "my-finetuned-gpt.pth")
```

**2. Load it**

The model class (`GPTModel`) is defined in `notebooks/my_own_llm.ipynb`. Run the model-definition cells, then:

```python
import torch

GPT_CONFIG_163M = {
    "vocab_size": 50257, "context_length": 1024, "emb_dim": 768,
    "n_heads": 12, "n_layers": 12, "drop_rate": 0.1, "qkv_bias": False,
}

model = GPTModel(GPT_CONFIG_163M)
model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
model.eval()
```

The notebook's chat cell (`generate_reply`) wraps this with the same prompt format used in training (Alpaca-style, ending in `### Response:`), temperature 0.7, top-k 40 and a light repetition penalty.

---

## Reproducing the training

**1. Environment:** as above.

**2. Data**

- Pretraining: download a FineWeb-Edu parquet subset (`text` column). Set `DATA_DIR` in the notebook, then run the (commented) tokenization cell to build `fineweb_tokens.bin`.
- Instruction tuning: SmolTalk parquet files with a `messages` column (list of `{role, content}`) and a `source` column. Set `FT_DATA_DIR`.

The notebook currently contains hard-coded Windows paths (`D:\...`); edit these two variables for your machine.

**3. Run**

Open `notebooks/my_own_llm.ipynb` and run top to bottom. Expect **multiple days** for pretraining and roughly **half a day to a day** for SFT on a laptop GPU. Checkpoints are written periodically and the training cells can resume from them.

**4. Optional: LLM-as-judge**

```bash
ollama pull llama3.1:8b     # then keep the Ollama server running
```

---

## Theory notes

`docs/transformer_math_flow.pdf` is a handwritten walkthrough (28 numbered steps) of one forward pass and one generation step, using the prompt "I Love AI" as a worked example: tokenization, embeddings, LayerNorm, Q/K/V projections, head splitting, scaled attention scores, causal mask, softmax, attention dropout, value mixing, output projection, residuals, feed-forward (768 to 3072 to 768, GELU), final LayerNorm, vocabulary logits, softmax, next-token selection. Every tensor shape is tracked.

Two simplifications are used for hand-calculation and differ from the trained model: the notes use **2 heads and 1 layer** (the trained model uses 12 and 12), and they write the output projection with a bias term `b_out` for generality (the trained model's output head has **no bias**, which is what gives exactly 163,009,536 parameters).

---

## Limitations and honest notes

- **Small model, small data.** 163M parameters and a few billion tokens is far below what is needed for reliable knowledge. Answers are fluent but frequently factually wrong, and long generations can become repetitive. Judge it as a systems and learning project.
- **Parameter count.** The file and folder names mention "124M" (GPT-2 small). Because input embeddings and the output head are *not* tied, this model has 163M parameters. The 163M figure is the accurate one.
- **Evaluation is thin.** Single run, single seed, no confidence intervals, no standard benchmark suite yet. The LLM-judge uses a small local model and 100 examples, so it is noisy.
- **SFT loss includes prompt tokens.** Labels are masked for padding only, not for the instruction part, so the reported SFT loss/perplexity is over the whole sequence and is not directly comparable to response-only numbers.
- **Data split.** The SFT train/validation/test split is contiguous in file order, not shuffled by source, so the held-out sets may differ in source mix from the training set.
- **"One epoch" is approximate.** SFT was interrupted and resumed on re-shuffled subsets of the remaining examples (documented in the notebook), so examples were not partitioned exactly once.
- **Pretraining validation loss** is estimated on 50 validation batches per evaluation, not the full 3M-token validation set.
- **Notebook, not a package.** Everything lives in one notebook with local paths; it is meant to be read as a log of the work. Refactoring into a script-based codebase is planned.

---

## Roadmap

**v1 (this release):** 163M parameters, ~3B pretraining tokens (FineWeb-Edu), ~877k SFT examples (SmolTalk).

**Next versions:** I plan to train further versions on more data to improve the model, and to release each one with its own weights and results so versions can be compared directly. Planned changes:

- Pretrain on more tokens than v1's ~3B and report how loss and downstream scores change with data size
- Evaluate on standard benchmarks with `lm-evaluation-harness` and report against GPT-2 small as a reference
- Mask instruction tokens during SFT and compare against the current setup
- Shuffled, source-stratified train/validation/test splits
- Ablation: weight tying (124M) vs. untied (163M)
- KV-cache for faster generation; refactor the notebook into a small installable package with a CLI chat script

Timelines depend on compute (a single laptop GPU), so v1 numbers above are the reference point for all later versions.

---

## Acknowledgements

- The architecture and the overall structure of the training and fine-tuning code follow Sebastian Raschka's book *Build a Large Language Model (From Scratch)* (Manning, 2024). This repository scales that blueprint up and extends it with the memory-mapped data pipeline, gradient accumulation, cosine LR schedule, bf16 training, crash-safe resume logic, large-scale SFT, perplexity and LLM-judge evaluation, and the theory notes above.
- Pretraining corpus: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (Hugging Face).
- Instruction data: [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (Hugging Face).
- Tokenizer: OpenAI's GPT-2 BPE via [`tiktoken`](https://github.com/openai/tiktoken).
- Judge model: Llama 3.1 8B, served locally with [Ollama](https://ollama.com).

---

## Author

**Sai Siddardha** · [Hugging Face](https://huggingface.co/SAISIDD) · [GitHub](https://github.com/SID-sai) · [saisiddharth22@gmail.com](mailto:saisiddharth22@gmail.com)

---

## License

Apache-2.0, see [`LICENSE`](LICENSE). Datasets and third-party models keep their own licenses.
