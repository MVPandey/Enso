# Enso: A Complete Technical Walkthrough

A deep dive into the theory, architecture, and implementation of an Energy-Based Model that solves Sudoku through latent reasoning — replicating and exceeding [Kona 1.0's 96.2% benchmark](https://logicalintelligence.com/blog/energy-based-model-sudoku-demo) with 96.6% puzzle accuracy.

---

## Table of Contents

1. [Background: Why Energy-Based Models?](#1-background-why-energy-based-models)
   - [The LLM Failure Mode](#the-llm-failure-mode)
   - [Self-Supervised Learning](#self-supervised-learning)
   - [Energy-Based Models](#energy-based-models)
   - [Joint Embedding Predictive Architecture (JEPA)](#joint-embedding-predictive-architecture-jepa)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Pipeline](#3-data-pipeline)
   - [Dataset Source and Loading](#dataset-source-and-loading)
   - [Tensor Encoding](#tensor-encoding)
   - [Train/Val/Test Splitting](#trainvaltest-splitting)
4. [The Encoder: Why Transformers?](#4-the-encoder-why-transformers)
   - [Sudoku-Aware Positional Encoding](#sudoku-aware-positional-encoding)
   - [Transformer Encoder Architecture](#transformer-encoder-architecture)
   - [Context Encoder vs Target Encoder](#context-encoder-vs-target-encoder)
   - [EMA: Why a Separate Target Encoder?](#ema-why-a-separate-target-encoder)
5. [The Latent Space](#5-the-latent-space)
   - [z_encoder: Projecting to Latent Space](#z_encoder-projecting-to-latent-space)
   - [The Noise Injection Problem (and Solution)](#the-noise-injection-problem-and-solution)
   - [The Predictor: Why Limited Capacity?](#the-predictor-why-limited-capacity)
6. [The Decoder](#6-the-decoder)
   - [From Latent to Logits](#from-latent-to-logits)
   - [Hard Clue Enforcement](#hard-clue-enforcement)
7. [Loss Functions](#7-loss-functions)
   - [Energy Loss](#energy-loss)
   - [VICReg: Preventing Representation Collapse](#vicreg-preventing-representation-collapse)
   - [Decode Loss](#decode-loss)
   - [Constraint Loss](#constraint-loss)
   - [The Combined Loss](#the-combined-loss)
8. [Langevin Dynamics: Thinking at Inference Time](#8-langevin-dynamics-thinking-at-inference-time)
   - [The Core Idea](#the-core-idea)
   - [Self-Consistency Energy](#self-consistency-energy)
   - [Multi-Chain Inference](#multi-chain-inference)
   - [Temperature Annealing](#temperature-annealing)
9. [Training Infrastructure](#9-training-infrastructure)
   - [Learning Rate Schedule](#learning-rate-schedule)
   - [EMA Momentum Schedule](#ema-momentum-schedule)
   - [GPU Auto-Scaling](#gpu-auto-scaling)
   - [Checkpoint Management](#checkpoint-management)
   - [Experiment Tracking](#experiment-tracking)
10. [Training Runs and Lessons Learned](#10-training-runs-and-lessons-learned)
    - [Run 1: MVP Validation](#run-1-mvp-validation)
    - [Run 2: Full Scale](#run-2-full-scale)
    - [Run 3: Larger Batch + Scaled LR](#run-3-larger-batch--scaled-lr)
    - [Run 4: Langevin Fixes](#run-4-langevin-fixes)
    - [Run 5: Scaled Architecture](#run-5-scaled-architecture)
    - [Key Takeaways](#key-takeaways)
11. [Code Walkthrough](#11-code-walkthrough)
    - [Project Structure](#project-structure)
    - [File-by-File Guide](#file-by-file-guide)
    - [End-to-End Data Flow](#end-to-end-data-flow)

---

## 1. Background: Why Energy-Based Models?

### The LLM Failure Mode

Large Language Models generate text token-by-token, left to right. Each token is committed to permanently — there's no mechanism to revise an earlier decision when later context reveals a conflict. For many tasks (summarization, translation, conversation), this is fine. But for constraint satisfaction problems like Sudoku, it's fatal.

Consider a Sudoku puzzle: placing a "5" in row 1, column 3 constrains what can appear in row 1, column 3's box, row 1, and column 3 — potentially 20 other cells. An LLM filling cells sequentially has no way to "go back" when a constraint violation emerges 30 tokens later. The result: frontier LLMs (GPT-5.2, Claude Opus 4.5, Gemini 3 Pro, DeepSeek V3.2) achieve roughly **2% combined accuracy** on hard Sudoku puzzles.

Energy-Based Models take a fundamentally different approach: they produce a **complete candidate solution** and evaluate it against **all constraints simultaneously**. Instead of committing to decisions sequentially, they navigate a continuous energy landscape where valid solutions correspond to low-energy states, using gradient information to move toward correctness.

### Self-Supervised Learning

Traditional supervised learning requires explicit labels: "this image is a cat", "this sentence is positive." Self-supervised learning instead creates learning signal from the **structure of the data itself**. The model learns to predict part of the input from other parts — for example, predicting masked words in a sentence (BERT) or predicting the next patch in an image (I-JEPA).

The key insight is that to make good predictions, the model must develop rich internal representations that capture the underlying structure of the data. You don't need to tell a model "these Sudoku digits satisfy row/column/box constraints" — if it learns to predict masked cells from given clues, it must implicitly learn those constraints.

In Enso, the self-supervised signal is: **given a partial puzzle (the clues), predict the complete solution.** The model never sees explicit constraint rules during training — it discovers them through the prediction objective.

### Energy-Based Models

An Energy-Based Model (EBM) assigns a scalar energy value to every possible configuration of its inputs. Low energy = compatible/correct configuration. High energy = incompatible/incorrect. The model learns an energy function $E(x, y)$ such that:

$$E(\text{puzzle}, \text{correct\_solution}) \to \textbf{low energy}$$
$$E(\text{puzzle}, \text{wrong\_solution}) \to \textbf{high energy}$$

The elegance of EBMs is that at inference time, you can **search for the lowest-energy configuration** using gradient descent. Given an input $x$, the model finds:

$$y^* = \arg\min_y E(x, y)$$

Unlike a feedforward model that produces a single answer in one pass, an EBM can iteratively refine a candidate solution by following the energy gradient $\nabla_y E(x, y)$ downhill.

This is exactly what the brain seems to do: rather than computing an answer in a single pass, we "think about it" — iteratively evaluating and adjusting candidate solutions until we reach one that satisfies all constraints.

### Joint Embedding Predictive Architecture (JEPA)

JEPA, introduced by Yann LeCun and formalized in the [I-JEPA paper](https://arxiv.org/abs/2301.08243) (CVPR 2023), is a specific flavor of self-supervised energy-based learning. The key idea: **make predictions in representation space, not pixel/token space.**

Why not predict in data space directly? Because data space is enormous and full of irrelevant variation. Two correct Sudoku solutions that differ only in the ordering of equivalent digits are fundamentally identical — but in raw data space, they look completely different. By mapping inputs to a learned representation space and making predictions there, the model can focus on the abstract structure that matters.

JEPA has three main components:
1. **Context encoder** $f_\theta$: Encodes the observable input (puzzle) to a representation $\mathbf{z}_{\text{context}} = f_\theta(\text{puzzle})$
2. **Target encoder** $\bar{f}_\theta$: Encodes the target (solution) to a representation $\mathbf{z}_{\text{target}} = \bar{f}_\theta(\text{solution})$
3. **Predictor** $g_\phi$: Maps $(\mathbf{z}_{\text{context}}, \mathbf{z}) \to \mathbf{z}_{\text{pred}}$, trained so that $\mathbf{z}_{\text{pred}} \approx \mathbf{z}_{\text{target}}$

The energy is simply the squared $L^2$ distance between the predicted and target representations:

$$E(\mathbf{z}_{\text{pred}}, \mathbf{z}_{\text{target}}) = \|\mathbf{z}_{\text{pred}} - \mathbf{z}_{\text{target}}\|_2^2$$

At inference time, we don't have $\mathbf{z}_{\text{target}}$ (we don't know the solution), so we use Langevin dynamics to search for a latent variable $\mathbf{z}$ that minimizes the energy.

Enso draws from three key papers:
- **I-JEPA** (CVPR 2023): The foundational JEPA framework for self-supervised learning
- **IRED** (2024): Iterative Reasoning with Energy Diffusion — applying EBMs to reasoning tasks with Langevin dynamics
- **JEPA-Reasoner** (2025): Applying JEPA specifically to logical reasoning tasks like Sudoku

---

## 2. Architecture Overview

Enso's architecture has two distinct modes: **training** and **inference**.

**During training**, the model sees both the puzzle and the solution:
1. The **context encoder** $f_\theta$ processes the puzzle $\to \mathbf{z}_{\text{context}} \in \mathbb{R}^{512}$
2. The **target encoder** $\bar{f}_\theta$ (EMA copy) processes the solution $\to \mathbf{z}_{\text{target}} \in \mathbb{R}^{512}$
3. $\mathbf{z}_{\text{target}}$ is projected to latent space via **z_encoder** and noise is added $\to \mathbf{z} \in \mathbb{R}^{256}$
4. The **predictor** $g_\phi$ maps $(\mathbf{z}_{\text{context}}, \mathbf{z}) \to \mathbf{z}_{\text{pred}} \in \mathbb{R}^{512}$
5. The **decoder** $D_\psi$ maps $(\mathbf{z}_{\text{context}}, \mathbf{z}) \to$ cell logits $\in \mathbb{R}^{9 \times 9 \times 9}$
6. Loss combines energy ($\mathbf{z}_{\text{pred}}$ vs $\mathbf{z}_{\text{target}}$), VICReg, cross-entropy on decode, and constraint penalty

**During inference**, the model sees only the puzzle:
1. The context encoder processes the puzzle $\to \mathbf{z}_{\text{context}}$
2. Multiple $\mathbf{z}$ vectors are initialized randomly from $\mathcal{N}(\mathbf{0}, \mathbf{I})$
3. For $T$ steps, $\mathbf{z}$ is optimized via Langevin dynamics to minimize self-consistency energy + constraint penalty
4. The lowest-energy $\mathbf{z}$ is decoded to a $9 \times 9$ solution grid

The total model has **36.5M trainable parameters** (in the final Run 5 configuration).

---

## 3. Data Pipeline

### Dataset Source and Loading

**File:** `src/ebm/dataset/loader.py`

The dataset is the [9 Million Sudoku Puzzles and Solutions](https://www.kaggle.com/datasets/rohanrao/sudoku) from Kaggle. Each row contains two 81-character strings: the puzzle (0 = empty cell, 1-9 = given clue) and the solution (all digits 1-9).

```
puzzle:   004300209005009001070060043006205800...
solution: 864371259325849761971562843436295817...
```

The `SudokuDataset` class handles automatic download via `kagglehub` on first use. It provides three loading methods:
- `load_all()` — the full 9M dataset
- `load_head(k)` — first k rows (for quick experiments)
- `load_fraction(frac)` — random sample without reading the full file

### Tensor Encoding

**File:** `src/ebm/dataset/torch_dataset.py`

Raw 81-character strings are converted to tensors suitable for the neural network:

**Puzzle encoding — (10, 9, 9) one-hot:**
- Channel 0: binary mask indicating empty cells (1 = empty, 0 = given clue)
- Channels 1-9: one-hot encoding for digits 1-9

The extra "empty" channel (channel 0) is critical — it explicitly tells the encoder which cells are unknown. Without it, the encoder would see zeros for empty cells, which is ambiguous (is it an empty cell, or missing information?).

**Solution encoding — (9, 9, 9) one-hot:**
- Dimension 0-1: spatial position (row, column)
- Dimension 2: one-hot over digits 1-9 (channel 0 = digit 1, ..., channel 8 = digit 9)

**Mask — (9, 9) binary:**
- 1 = given clue, 0 = empty cell
- Used to compute decode loss only on empty cells and to hard-enforce given clues in the decoder

The implementation pre-parses all strings into `int8` numpy arrays at initialization time (in `_parse_strings`), so `__getitem__` only needs to do fast numpy indexing rather than string parsing.

### Train/Val/Test Splitting

**File:** `src/ebm/dataset/splits.py`

The 9M dataset is split deterministically using scikit-learn's `train_test_split` with a fixed seed (42):
- **Training:** 8M puzzles
- **Validation:** 500K puzzles
- **Test:** 500K puzzles

For smaller experiments (via `--n-samples`), the split sizes are dynamically adjusted to 5%/5% of the total to avoid the holdout exceeding the dataset size.

---

## 4. The Encoder: Why Transformers?

### Sudoku-Aware Positional Encoding

**File:** `src/ebm/model/encoder.py` (lines 15-55)

A Sudoku puzzle isn't just a 9x9 grid — it has rich structural relationships. Every cell belongs to three overlapping constraint groups: its **row**, its **column**, and its **3x3 box**. Two cells that share any of these groups must contain different digits.

Standard positional encodings (sinusoidal or learned 1D/2D) would treat the grid as a flat sequence or a plain 2D image, missing these constraint relationships entirely. Instead, Enso uses **Sudoku-aware positional encoding**: three separate learned embedding tables for row (0-8), column (0-8), and box (0-8), summed together:

```python
pos = self.row_embed(self.row_ids) + self.col_embed(self.col_ids) + self.box_embed(self.box_ids)
```

This means two cells in the same row share a row embedding, two cells in the same box share a box embedding, etc. The attention mechanism can use these shared components to "know" which cells constrain each other, without having to learn this structure from scratch.

The box indices are computed statically:
```python
BOX_INDICES = [(r // 3) * 3 + c // 3 for r in range(9) for c in range(9)]
```

This maps each of the 81 cells to one of 9 boxes (0-8), arranged left-to-right, top-to-bottom.

### Transformer Encoder Architecture

**File:** `src/ebm/model/encoder.py` (lines 58-113)

Why Transformers for Sudoku encoding? Three reasons:

1. **Global attention over all 81 cells.** Every cell can directly attend to every other cell. This is essential for Sudoku: a digit in cell (0,0) constrains cells across the entire board — in row 0, column 0, and box 0. CNNs with local receptive fields would need many layers to propagate this information; Transformers get it in a single layer via self-attention.

2. **Permutation-aware but position-informed.** The Transformer's self-attention is inherently permutation-equivariant — it treats the input as a set. Combined with the Sudoku-aware positional encoding, it can reason about constraint groups without hardcoding specific interaction patterns.

3. **Scalable capacity.** Transformers scale predictably with depth and width, which proved critical: scaling from 6 to 8 layers and 256 to 512 dimensions drove a +13.1% puzzle accuracy improvement in Run 5.

The encoder processes a (B, C, 9, 9) grid as follows:
1. **Reshape** to 81 tokens: (B, C, 9, 9) → (B, C, 81) → (B, 81, C)
2. **Project** each token from C channels to d_model dimensions via a linear layer
3. **Add** Sudoku-aware positional encoding
4. **Process** through N pre-norm Transformer layers (GELU activation, dropout)
5. **Pool** via mean over the 81 tokens → (B, d_model)

The choice of **pre-norm** (LayerNorm before attention/FFN, `norm_first=True`) over post-norm is a practical one: pre-norm Transformers are more stable during training, especially at larger depths, and don't require careful learning rate tuning.

The final **mean pooling** compresses the 81-token sequence into a single d_model-dimensional vector. This forces the representation to be a holistic summary of the entire grid, not a collection of per-cell features. The decoder handles the reverse — expanding back to per-cell predictions.

### Context Encoder vs Target Encoder

The model uses two encoder instances with the **same architecture** but different roles:

| | Context Encoder | Target Encoder |
|---|---|---|
| **Input** | Puzzle (10 channels: 1 empty + 9 digits) | Solution (9 channels: 9 digits) |
| **Gradients** | Yes (trained normally) | No (EMA updated) |
| **Purpose** | Encode what we know (partial info) | Encode what we want to predict (full info) |

The context encoder's `input_proj` has 10 input channels (to accommodate the empty-cell channel), while the target encoder's has 9. All other layers are architecturally identical.

### EMA: Why a Separate Target Encoder?

The target encoder is **not** trained via backpropagation. Instead, its weights are an Exponential Moving Average (EMA) of the context encoder:

$$\bar{\theta}_t \leftarrow m \cdot \bar{\theta}_{t-1} + (1 - m) \cdot \theta_t$$

where $\theta$ are the context encoder weights, $\bar{\theta}$ are the target encoder weights, and $m$ is the momentum.

**File:** `src/ebm/model/jepa.py` (lines 98-112)

Why EMA instead of sharing weights or training independently?

**Shared weights** cause a degenerate solution: the model can trivially minimize energy by mapping everything to the same point (representation collapse). If the same encoder processes both puzzle and solution, it can learn to ignore the input entirely and produce a constant output — energy zero for free.

**Independent training** doesn't provide a stable target. If both encoders change rapidly every step, the energy landscape shifts under the predictor's feet, making optimization chaotic.

**EMA** provides the best of both worlds: the target encoder **slowly tracks** the context encoder, providing a stable but gradually improving target. Early in training ($m = 0.996$), the target updates relatively quickly to keep up with rapid learning. Late in training ($m \to 1.0$), the target barely moves, providing a nearly fixed reference for fine-tuning.

The `_init_target_encoder` method copies matching weights from the context encoder at initialization, but skips `input_proj` since the channel dimensions differ (10 vs 9). EMA updates similarly skip mismatched parameters.

---

## 5. The Latent Space

### z_encoder: Projecting to Latent Space

**File:** `src/ebm/model/jepa.py` (line 81)

The target encoder produces a $d_{\text{model}}$-dimensional (512) representation of the solution. The `z_encoder` is a single linear layer that projects this down to $d_{\text{latent}}$ dimensions (256):

```python
self.z_encoder = nn.Linear(arch_cfg.d_model, arch_cfg.d_latent)
```

This projection serves two purposes:
1. **Information bottleneck:** The latent variable $\mathbf{z}$ should carry a compressed, essential summary of the solution — not a lossless copy. A smaller dimension forces the model to distill the most important information.
2. **Latent space for Langevin dynamics:** At inference time, we optimize $\mathbf{z}$ via gradient descent. A lower-dimensional space is easier to search.

The z_encoder output is **L2-normalized** before noise is added:
```python
z_target_latent = F.normalize(self.z_encoder(z_target), dim=-1)
```

This maps the output to the unit hypersphere $\|\hat{\mathbf{z}}\|_2 = 1$. This normalization is critical — see the next section for why.

### The Noise Injection Problem (and Solution)

During training, $\mathbf{z}$ is constructed as:

$$\mathbf{z} = \hat{\mathbf{z}}_{\text{target}} + \sigma \cdot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

where $\hat{\mathbf{z}}_{\text{target}}$ is the normalized z_encoder output and $\sigma$ is the noise scale.

The noise injection simulates the inference scenario: at test time, $\mathbf{z}$ is initialized randomly and iteratively refined. During training, the model sees a **noisy version** of the true $\mathbf{z}$, so it must learn to work with imprecise latent variables.

**The problem discovered in Runs 2-3:** Without normalization, the z_encoder output had $\|\mathbf{z}_{\text{enc}}\|_2 \approx 144$, while the noise (from `randn` in $\mathbb{R}^{128}$) had $\|\boldsymbol{\epsilon}\|_2 \approx 11$. The signal-to-noise ratio was $\sim$13:1 — $\mathbf{z}$ was essentially a deterministic copy of the solution encoding, not a noisy variable. The decoder learned to treat $\mathbf{z}$ as a lookup table, making Langevin dynamics useless: optimizing a random $\mathbf{z}$ couldn't recover the high-magnitude $\mathbf{z}$ the decoder expected.

**The fix in Run 4:** L2-normalize the z_encoder output to unit norm ($\|\hat{\mathbf{z}}\|_2 = 1$), and reduce $\sigma$ from 1.0 to 0.1. For a random vector $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ in $\mathbb{R}^{256}$, the expected norm is $\mathbb{E}[\|\boldsymbol{\epsilon}\|_2] = \sqrt{d} \approx 16$, so the noise term has expected norm $\sigma \cdot \sqrt{d} = 0.1 \times 16 = 1.6$. This gives an SNR of:

$$\text{SNR} = \frac{\|\hat{\mathbf{z}}\|_2}{\sigma \cdot \sqrt{d_{\text{latent}}}} = \frac{1.0}{1.6} \approx 0.6$$

Now $\mathbf{z}$ carries solution information but is genuinely noisy — the model must learn to reason under uncertainty, which is exactly what Langevin dynamics requires.

### The Predictor: Why Limited Capacity?

**File:** `src/ebm/model/predictor.py`

The predictor is a small 3-layer MLP with residual connections:

```
Input: concat(z_context [512], z [256]) = 768
→ Linear(768, 1024) → GELU
→ Linear(1024, 1024) → GELU → + residual → LayerNorm
→ Linear(1024, 512) → z_pred
```

The predictor's job: given the puzzle encoding (z_context) and the latent variable (z), predict the target encoder's output (z_target) in 512-dimensional space.

**Why is it intentionally small?** If the predictor had unlimited capacity (e.g., a deep Transformer), it could learn to produce z_pred ≈ z_target entirely from z_context alone, ignoring z completely. This is the JEPA "shortcut" failure mode: the predictor bypasses the latent variable, making it carry no information, which again breaks Langevin dynamics.

By keeping the predictor to a shallow MLP, it **cannot** solve the prediction problem from z_context alone. It must rely on information from z to bridge the gap. This forces z to carry meaningful solution information, which is exactly what we need for inference-time optimization.

The residual connection (`x = self.norm(x + residual)`) helps with gradient flow through the three layers, and GELU activation provides smooth, non-saturating nonlinearity.

---

## 6. The Decoder

### From Latent to Logits

**File:** `src/ebm/model/decoder.py`

The decoder takes z_context (512-dim) and z (256-dim) and produces per-cell digit logits — a (B, 9, 9, 9) tensor where the last dimension represents scores for digits 1-9.

The architecture:

1. **Concatenate** z_context and z → 768-dim vector
2. **Project** to 81 * d_cell features via a single linear layer: `Linear(768, 81 * 128)` → (B, 81, 128)
3. **Add** Sudoku-aware positional encoding (same structure as the encoder's)
4. **Process** through 4 pre-norm Transformer layers (refine per-cell features with inter-cell attention)
5. **Head:** Linear(128, 9) → per-cell logits for digits 1-9
6. **Reshape** to (B, 9, 9, 9)

The decoder is lighter than the encoder (4 vs 8 layers, 128 vs 512 per-cell dimension) because its job is different: it doesn't need to build a holistic understanding of the grid from scratch. It receives a compressed representation (z_context + z) and needs to "unpack" it into per-cell predictions, guided by inter-cell attention for consistency.

### Hard Clue Enforcement

**File:** `src/ebm/model/decoder.py` (lines 74-76)

Given clues in the puzzle are **hard-enforced** — the decoder's predictions for clue cells are overwritten with the known values:

```python
clue_logits = puzzle[:, 1:].permute(0, 2, 3, 1)    # one-hot clues
clue_mask = mask.unsqueeze(-1)                        # (B, 9, 9, 1)
logits = logits * (1 - clue_mask) + clue_logits * clue_mask * 1e6
```

For given clue cells (mask=1): the correct digit gets logit 1e6 (effectively infinite after softmax), all others get 0. For empty cells (mask=0): the decoder's predictions pass through unchanged.

This is preferable to letting the decoder learn to copy clues, because:
1. It guarantees clue cells are always correct (no capacity wasted learning an identity function)
2. It focuses the decoder's capacity entirely on the hard problem: predicting empty cells
3. It provides clean gradients — decode loss is computed only on empty cells

---

## 7. Loss Functions

**File:** `src/ebm/training/losses.py`

The total training loss combines four terms, each serving a distinct purpose:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{energy}} + \mathcal{L}_{\text{VICReg}} + w_{\text{decode}} \cdot \mathcal{L}_{\text{decode}} + w_{\text{constraint}} \cdot \mathcal{L}_{\text{constraint}}$$

### Energy Loss

**File:** `src/ebm/model/energy.py`

$$\mathcal{L}_{\text{energy}} = \frac{1}{B} \sum_{i=1}^{B} \|\mathbf{z}_{\text{pred}}^{(i)} - \mathbf{z}_{\text{target}}^{(i)}\|_2^2$$

This is the core JEPA objective: minimize the squared $L^2$ distance between the predicted representation $\mathbf{z}_{\text{pred}}$ and the target representation $\mathbf{z}_{\text{target}}$. Lower energy means the predictor successfully anticipated the target encoder's output given the puzzle encoding and the latent variable.

The energy function is deliberately simple — just $\|\mathbf{z}_{\text{pred}} - \mathbf{z}_{\text{target}}\|_2^2$. Complex energy functions (e.g., with learned temperature parameters or cosine similarity) are harder to optimize and can introduce training instabilities. The simplicity here is a feature: it provides clean, well-scaled gradients and makes the energy landscape directly interpretable as Euclidean distance in representation space.

### VICReg: Preventing Representation Collapse

#### The Collapse Problem

Without regularization, JEPA-style models have a trivial solution to the energy objective: the context encoder maps every puzzle to the same constant vector $\mathbf{c}$, the target encoder maps every solution to $\mathbf{c}$, and the predictor learns the identity function. Energy $= \|\mathbf{c} - \mathbf{c}\|^2 = 0$ for free — but the representation carries zero information.

This is **representation collapse**, and it's the central failure mode of joint-embedding architectures. Contrastive methods (SimCLR, MoCo) prevent collapse by explicitly pushing apart representations of different inputs using negative pairs. But contrastive learning requires careful negative sampling, large batch sizes, and memory banks — adding significant complexity.

#### VICReg: A Non-Contrastive Alternative

**VICReg** (Variance-Invariance-Covariance Regularization, [Bardes et al., 2022](https://arxiv.org/abs/2105.04906)) prevents collapse **without negative pairs** by directly regularizing the statistical properties of the representation distribution. The key insight: a collapsed representation has zero variance across the batch — if we force the representations to maintain high variance and low inter-dimensional correlation, collapse becomes impossible.

The full VICReg loss has three terms (though Enso uses two):

$$\mathcal{L}_{\text{VICReg}} = \lambda_{\text{var}} \cdot \mathcal{L}_{\text{variance}} + \lambda_{\text{cov}} \cdot \mathcal{L}_{\text{covariance}}$$

#### Variance Term: Preventing Dimensional Collapse

Given a batch of representations $\mathbf{Z} \in \mathbb{R}^{B \times D}$, compute the standard deviation along each dimension $j$ across the batch:

$$\sigma_j = \sqrt{\text{Var}(\mathbf{Z}_{:,j}) + \epsilon}$$

The variance loss penalizes any dimension whose standard deviation falls below 1:

$$\mathcal{L}_{\text{variance}} = \frac{1}{D} \sum_{j=1}^{D} \max(0, \; 1 - \sigma_j)$$

This is a hinge loss — it doesn't push variance above 1 (that would fight against the energy loss), but it prevents any dimension from collapsing to a constant. If all $D$ dimensions have $\sigma_j \geq 1$, the variance loss is exactly zero.

**Why $\gamma = 1$ as the target?** Setting the target standard deviation to 1 keeps representations on a natural scale. A much larger target would inflate representations and destabilize training; a smaller target would provide insufficient anti-collapse pressure.

#### Covariance Term: Preventing Redundancy

The variance term alone isn't sufficient. The model could satisfy it by repeating the same signal in every dimension — all dimensions vary, but they carry identical information. The covariance term decorrelates dimensions:

$$\mathbf{C} = \frac{1}{B-1} (\mathbf{Z} - \bar{\mathbf{Z}})^\top (\mathbf{Z} - \bar{\mathbf{Z}})$$

$$\mathcal{L}_{\text{covariance}} = \frac{1}{D} \sum_{i \neq j} C_{ij}^2$$

This penalizes the squared off-diagonal entries of the covariance matrix $\mathbf{C}$, pushing them toward zero. When the covariance loss is minimized, each dimension of $\mathbf{z}_{\text{context}}$ carries **independent** information — no redundancy, maximum information capacity for a given dimensionality.

Together, the variance and covariance terms ensure the representation uses the full $D$-dimensional space with each dimension carrying unique, spread-out information — exactly the opposite of collapse.

#### Why Apply VICReg to $\mathbf{z}_{\text{context}}$?

**Critical implementation detail:** VICReg is applied to $\mathbf{z}_{\text{context}}$ (the context encoder's output), **not** $\mathbf{z}_{\text{pred}}$. Run 1 discovered that applying VICReg to $\mathbf{z}_{\text{pred}}$ allowed the target encoder to collapse freely — the predictor's output had high variance (VICReg forced this), but the target $\mathbf{z}_{\text{target}}$ it was predicting had collapsed to a constant. The predictor simply learned to produce varied outputs that all mapped to the same degenerate target. Applying VICReg to $\mathbf{z}_{\text{context}}$ ensures the context encoder maintains rich, spread-out representations, which transitively prevents the target encoder (its EMA copy) from collapsing.

#### VICReg Weight Choices

In the implementation:

```python
vreg = vicreg_loss(out.z_context, var_weight=1.0, cov_weight=0.01)
```

- $\lambda_{\text{var}} = 1.0$: Equal weight with the energy loss. The variance term is the primary anti-collapse mechanism and needs to be strong enough to counteract the energy loss's pull toward degenerate solutions.
- $\lambda_{\text{cov}} = 0.01$: Much smaller because the covariance loss has a different magnitude scale. For $D = 512$ dimensions, there are $D(D-1)/2 \approx 131{,}000$ off-diagonal entries summed. A weight of 0.01 keeps this term from dominating while still providing meaningful decorrelation pressure. Empirically, higher values (e.g., 1.0) caused training instability, while lower values (e.g., 0.001) left noticeable dimension correlation in the learned representations.

### Decode Loss

$$\mathcal{L}_{\text{decode}} = -\frac{1}{|\mathcal{E}|} \sum_{(r,c) \in \mathcal{E}} \log p_{\text{model}}(y_{r,c} \mid \mathbf{z}_{\text{context}}, \mathbf{z})$$

where $\mathcal{E} = \{(r, c) : \text{mask}_{r,c} = 0\}$ is the set of empty cells.

Standard cross-entropy loss on the decoder's digit predictions, computed **only on empty cells**. This is an auxiliary loss that provides direct supervision: the decoder should predict the correct digit for each unknown cell.

Computing loss only on empty cells is essential — including given clue cells would let the model achieve low loss by simply copying the input, without learning to solve anything.

### Constraint Loss

**File:** `src/ebm/model/constraints.py`

Let $\mathbf{p} \in \mathbb{R}^{9 \times 9 \times 9}$ be the softmax probabilities from the decoder, where $p_{r,c,d}$ is the predicted probability that cell $(r, c)$ contains digit $d$. Let $\mathcal{G} = \{G_1, \ldots, G_{27}\}$ be the 27 constraint groups (9 rows + 9 columns + 9 boxes), each containing 9 cell positions.

For a valid Sudoku solution, each digit $d \in \{1, \ldots, 9\}$ appears exactly once in each group $G_k$, meaning:

$$\sum_{(r,c) \in G_k} p_{r,c,d} = 1.0 \quad \forall \; k \in \{1, \ldots, 27\}, \; d \in \{1, \ldots, 9\}$$

The constraint penalty measures the squared deviation from this ideal:

$$\mathcal{L}_{\text{constraint}} = \frac{1}{B} \sum_{i=1}^{B} \sum_{k=1}^{27} \sum_{d=1}^{9} \left( \sum_{(r,c) \in G_k} p_{r,c,d}^{(i)} - 1 \right)^2$$

For example, if digit 5 has probability 0.3 in three cells of the same row (and 0 elsewhere), the sum is 0.9, contributing $(0.9 - 1.0)^2 = 0.01$ to the penalty. A perfectly valid solution (each digit appearing exactly once per group with probability 1.0) gives penalty $= 0$.

**Why this was added in Run 4:** Without explicit constraint signals during training, the model only learns Sudoku rules implicitly through cross-entropy loss. It discovers "each digit appears once per row" by observing enough examples, but never sees this stated as a rule. The constraint loss provides a direct gradient signal: "this configuration violates Sudoku rules, adjust these cell probabilities." This was one of three fixes that made Langevin dynamics work.

The `GROUP_INDICES` tensor $(27 \times 9)$ is precomputed at module load time, mapping each of the 27 groups to 9 flat cell indices (0-80).

### The Combined Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{energy}} + \mathcal{L}_{\text{VICReg}} + w_{\text{decode}} \cdot \mathcal{L}_{\text{decode}} + w_{\text{constraint}} \cdot \mathcal{L}_{\text{constraint}}$$

With $w_{\text{decode}} = 1.0$ and $w_{\text{constraint}} = 0.1$.

The energy and VICReg terms are unweighted (weight 1.0) because they are the core representational learning objectives and naturally counterbalance each other — energy pulls representations together while VICReg pushes them apart. The decode loss also at 1.0 provides strong auxiliary supervision that directly improves cell accuracy. The constraint loss at 0.1 is deliberately conservative — too much constraint weight fights the cross-entropy signal during early training, when predictions are poor and the soft constraint penalty disagrees with the hard digit labels. At 0.1, it provides gentle structural guidance without destabilizing optimization.

---

## 8. Langevin Dynamics: Thinking at Inference Time

### The Core Idea

**File:** `src/ebm/model/jepa.py` (lines 147-212)

At inference time, we don't have the solution — we can't compute $\mathbf{z}_{\text{target}}$. The entire inference strategy is: **find a latent variable $\mathbf{z}$ that the model believes is consistent with the puzzle.**

Langevin dynamics is a gradient-based MCMC method that samples from an energy landscape by iteratively following the energy gradient with added noise:

$$\mathbf{z}_{t+1} = \mathbf{z}_t - \eta \, \nabla_{\mathbf{z}} E(\mathbf{z}_t) + \sigma \, \boldsymbol{\epsilon}_t, \quad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

In physics terms: a particle ($\mathbf{z}$) rolls downhill on the energy surface (gradient term $-\eta \, \nabla_{\mathbf{z}} E$) while being jostled by random thermal fluctuations (noise term $\sigma \, \boldsymbol{\epsilon}$). The gradient term drives the particle toward low-energy regions (valid solutions); the noise term prevents it from getting stuck in local minima.

The implementation:

```python
z = torch.randn(batch_size * n_chains, d_latent, device=device)  # random init

for step in range(n_steps):
    z_pred = self.predictor(z_context_exp, z)
    logits = self.decoder(z_context_exp, z, puzzle_exp, mask_exp)
    probs = torch.softmax(logits, dim=-1)

    # Self-consistency energy
    z_target_est = self.target_encoder(probs.permute(0, 3, 1, 2))
    self_consistency = ((z_pred - z_target_est) ** 2).sum(dim=-1)

    # Constraint penalty
    c_penalty = constraint_penalty(probs)

    # Temperature annealing
    temp = 1.0 - step / max(n_steps, 1)
    total_energy = self_consistency + c_penalty * (1.0 + 2.0 * (1.0 - temp))

    # Langevin update
    grad_z = torch.autograd.grad(total_energy.sum(), z)[0]
    noise = noise_scale * temp * torch.randn_like(z)
    z = (z - lr * grad_z + noise).detach().requires_grad_(True)
```

### Self-Consistency Energy

The inference energy is **not** the same as the training energy. During training, the energy is $\|\mathbf{z}_{\text{pred}} - \mathbf{z}_{\text{target}}\|^2$ where $\mathbf{z}_{\text{target}}$ comes from encoding the **true solution**. At inference, we don't have the solution.

**Runs 2-3** used $\|\mathbf{z}_{\text{pred}}\|^2$ as a proxy (assuming $\mathbf{z}_{\text{target}} \approx \mathbf{0}$), which is wrong — $\mathbf{z}_{\text{target}}$ is not zero. This produced misleading gradients that actively hurt accuracy.

**Run 4** introduced **self-consistency energy**: decode the current $\mathbf{z}$ to a candidate solution (soft probabilities $\hat{\mathbf{p}}$), re-encode that candidate through the target encoder, and measure how well the predictor's output matches:

$$E_{\text{self-consistency}} = \left\| g_\phi(\mathbf{z}_{\text{context}}, \mathbf{z}) - \bar{f}_\theta(\text{softmax}(D_\psi(\mathbf{z}_{\text{context}}, \mathbf{z}))) \right\|_2^2$$

In code:
```python
z_target_est = self.target_encoder(probs.permute(0, 3, 1, 2))
self_consistency = ((z_pred - z_target_est) ** 2).sum(dim=-1)
```

The intuition: a consistent $\mathbf{z}$ should produce a decoded solution that, when re-encoded, matches the predictor's expectation. If $\mathbf{z}$ leads to a garbled solution, re-encoding it produces a different representation than what the predictor expects, resulting in high self-consistency energy.

### Multi-Chain Inference

Instead of running a single z trajectory, the solver runs **n_chains** parallel trajectories (default: 8) per puzzle, each starting from a different random initialization:

```python
z = torch.randn(batch_size * n_chains, d_latent, device=device)
```

After all steps, the chain with the lowest final energy is selected:

```python
chain_energy = total_energy.detach().reshape(batch_size, n_chains)
best_chain = chain_energy.argmin(dim=1)
best_logits = final_logits[torch.arange(batch_size, device=device), best_chain]
```

Multiple chains hedge against local minima: even if some chains get stuck in bad energy basins, at least one is likely to find a good solution. This is computationally parallel on the GPU, so the wall-clock cost is minimal.

### Temperature Annealing

The noise scale and constraint weight both change over the course of the Langevin trajectory via a linear temperature schedule $\tau_t$:

$$\tau_t = 1 - \frac{t}{T}$$

The full Langevin update at step $t$ becomes:

$$\mathbf{z}_{t+1} = \mathbf{z}_t - \eta \, \nabla_{\mathbf{z}} \Big[ E_{\text{sc}}(\mathbf{z}_t) + (1 + 2(1 - \tau_t)) \cdot \mathcal{L}_{\text{constraint}}(\mathbf{z}_t) \Big] + \sigma \, \tau_t \, \boldsymbol{\epsilon}_t$$

**Early steps ($\tau \approx 1$):** High noise ($\sigma \, \tau$) encourages exploration of the energy landscape. Constraint weight is low ($1.0$), allowing the model to focus on energy minimization.

**Late steps ($\tau \approx 0$):** Noise vanishes, letting $\mathbf{z}$ settle into the nearest energy minimum. Constraint weight increases to $3.0$, heavily penalizing Sudoku violations in the final solution.

This is analogous to simulated annealing: start with high "temperature" for exploration, gradually "cool" to exploitation.

---

## 9. Training Infrastructure

### Learning Rate Schedule

**File:** `src/ebm/training/scheduler.py`

The learning rate follows **linear warmup + cosine decay**:

1. **Warmup** (first 2000 steps, capped at total_steps/5): LR linearly increases from ~0 to the peak LR. This prevents training instability from large gradients in early random-weight stages.

2. **Cosine decay** (remaining steps): LR smoothly decreases following a cosine curve to ~0 at the end of training. Cosine decay provides a more gradual reduction than step decay, allowing finer adjustments near convergence.

```python
warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
decay = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])
```

### EMA Momentum Schedule

$$m_t = m_0 + \frac{t}{T}(m_T - m_0), \quad m_0 = 0.996, \; m_T = 1.0$$

The target encoder's EMA momentum increases linearly from $0.996$ to $1.0$. At momentum $0.996$, roughly $0.4\%$ of the context encoder's weights bleed through per step — fast enough to track early rapid learning. At momentum $\approx 1.0$ near the end, the target is nearly frozen, providing a stable reference for the final fine-tuning phase.

### GPU Auto-Scaling

**File:** `src/ebm/utils/device.py`

The system detects available GPU VRAM and automatically scales the batch size and learning rate:

**Batch size tiers:**
| VRAM | Batch Size |
|------|-----------|
| < 6 GB | 256 |
| 6-12 GB | 512 |
| 12-24 GB | 1024 |
| 24+ GB | 2048 |

**Learning rate scaling** uses the square-root rule (Hoffer et al., 2017):

$$\eta_{\text{scaled}} = \eta_{\text{base}} \cdot \sqrt{\frac{B_{\text{actual}}}{B_{\text{base}}}}, \quad \eta_{\text{base}} = 3 \times 10^{-4}, \; B_{\text{base}} = 512$$

For example, $B_{\text{actual}} = 2048 \implies \eta_{\text{scaled}} = 3 \times 10^{-4} \cdot \sqrt{4} = 6 \times 10^{-4}$.

The sqrt rule is more conservative than linear scaling ($\eta \propto B$) and works well with adaptive optimizers like AdamW. The intuition: with $k$ times more samples per gradient step, the gradient estimate's variance decreases by $1/k$, so its magnitude (standard deviation) decreases by $1/\sqrt{k}$, justifying a $\sqrt{k}$ increase in step size.

### Checkpoint Management

**File:** `src/ebm/training/checkpoint.py`

The `CheckpointManager` keeps only the top-K checkpoints (default: 3) ranked by **cell accuracy**. A min-heap tracks the K best checkpoints; when a new checkpoint exceeds the worst of the K, it replaces it and the old file is deleted from disk.

**Critical lesson from Run 1:** Early versions checkpointed by lowest energy. But the energy objective has a degenerate minimum at zero (representation collapse), so the "best energy" checkpoint was often a collapsed model with 11.1% accuracy (random chance). Switching to cell accuracy as the checkpoint metric fixed this.

Each checkpoint stores: model state dict, optimizer state dict, epoch, step, validation energy, and cell accuracy.

### Experiment Tracking

**File:** `src/ebm/training/metrics.py`

All training is logged to [Weights & Biases](https://wandb.ai) when configured:

**Per training step:**
- Total loss, energy loss, VICReg loss, decode loss, constraint loss
- Current learning rate and EMA momentum

**Per validation epoch:**
- Mean validation energy
- Cell accuracy (fraction of empty cells correct)
- Puzzle accuracy (fraction of fully solved puzzles)
- z-variance (collapse detector — should stay well above zero)

The z-variance metric is particularly important: it's an early warning system for representation collapse. If z_variance drops near zero, all inputs are mapping to the same point and training has failed, even if the total loss looks reasonable.

Run names include timestamps for easy identification: `sudoku-jepa-20250215-143022`.

---

## 10. Training Runs and Lessons Learned

### Run 1: MVP Validation
**Config:** 100K samples, 20 epochs, RTX 5090, bs=512, lr=3e-4

Before any meaningful training happened, five critical bugs were discovered and fixed:

1. **Solution tensor encoding mismatch** — The encoding produced (digit, row, col) but the model expected (row, col, digit). Every solution was scrambled.

2. **Decode loss included given cells** — The model achieved low loss by copying the input clues rather than predicting empty cells. Masking to empty-only cells exposed the true difficulty.

3. **VICReg applied to z_pred instead of z_context** — VICReg on the predictor's output didn't prevent the target encoder from collapsing. The collapse happened upstream; the predictor just learned to produce varied outputs from a degenerate target.

4. **LR warmup exceeded total steps** — The warmup period was longer than the entire training run for small experiments, so the learning rate never reached its peak. Fixed by capping warmup at total_steps/5.

5. **Checkpointing on lowest energy** — Representation collapse gives energy ≈ 0 (a degenerate "perfect" score). The best-energy checkpoint was the worst model. Fixed by checkpointing on cell accuracy.

**Result:** 13.9% cell accuracy — barely above the 11.1% random baseline, but representations weren't collapsing. Limited by data and epochs.

### Run 2: Full Scale
**Config:** 9M samples, 20 epochs, RTX 5090, bs=512, lr=3e-4, ~7.5h

**Key fix:** Added `z_encoder` — a linear projection from d_model to d_latent. Previously, z was pure random noise with no connection to the solution. The model learned to ignore z completely, using only z_context for predictions. Now `z = z_encoder(z_target) + noise`, so z carries noisy solution information.

**Results:** 97.2% cell / 74.7% puzzle accuracy. Strong forward-pass performance.

**Langevin evaluation:** 96.5% cell / 70.7% puzzle — **solver made things worse** (-4.0% puzzle accuracy). The z_encoder output had L2 norm ~144 vs noise norm ~11. The decoder treated z as a deterministic lookup table, not a variable to reason over.

### Run 3: Larger Batch + Scaled LR
**Config:** 9M samples, 20 epochs, RTX 5090, bs=2048, lr=6e-4, ~7h

Auto-scaled batch size from GPU VRAM with sqrt-rule LR scaling.

**Results:** 98.3% cell / 83.8% puzzle. Significant improvement (+1.1% cell, +9.1% puzzle) from better gradient estimates with larger batches.

**Langevin evaluation:** 97.5% cell / 81.0% puzzle — solver still hurt (-2.8% puzzle). Same structural issues as Run 2.

**Root cause analysis** identified three problems:
1. z_encoder magnitude too high → z is a lookup table, not a noisy variable
2. Inference energy disconnected from training (||z_pred||² assumes z_target ≈ 0)
3. No Sudoku rules during training (constraint penalty only at inference)

### Run 4: Langevin Fixes
**Config:** 9M samples, 20 epochs, RTX 5090, bs=2048, lr=6e-4, ~7h

Three targeted fixes from the Run 3 analysis:

1. **L2-normalize z_encoder output** — Forces unit-norm z before noise, giving calibrated SNR. `z_noise_scale` reduced from 1.0 to 0.1 for ~0.6:1 SNR.

2. **Constraint loss during training** — Added `constraint_penalty(softmax(logits))` to the loss (weight=0.1). The model now explicitly learns Sudoku rules, not just implicitly through cross-entropy.

3. **Self-consistency inference energy** — Replaced ||z_pred||² with the decode → re-encode → compare loop.

**Results:** 97.6% cell / 82.5% puzzle. Slightly lower forward-pass accuracy than Run 3 (the constraint loss trades some cell accuracy for structural correctness).

**Langevin evaluation:** 97.8% cell / **83.5%** puzzle — **solver improved results for the first time** (+1.0% puzzle accuracy). In Runs 2-3, Langevin always decreased accuracy. This validated all three fixes.

### Run 5: Scaled Architecture
**Config:** 9M samples, 20 epochs, H200 (144GB), bs=2048, lr=6e-4, ~22h

Scaled from 7.4M to 36.5M trainable parameters:

| Parameter | Run 4 | Run 5 |
|-----------|-------|-------|
| d_model | 256 | 512 |
| Encoder layers | 6 | 8 |
| Decoder layers | 2 | 4 |
| d_latent | 128 | 256 |
| Predictor hidden | 512 | 1024 |
| Decoder d_cell | 64 | 128 |
| **Total params** | **7.4M** | **36.5M** |

**Results:** 99.3% cell / 95.6% puzzle — massive improvement from capacity alone (+1.7% cell, +13.1% puzzle). The model plateaued around epoch 17.

**Langevin evaluation:** 99.4% cell / **96.6%** puzzle / 99.4% constraint satisfaction. **Exceeds Kona 1.0's 96.2% benchmark.** Langevin added +1.0% puzzle accuracy, consistent with Run 4.

### Key Takeaways

1. **Representation collapse is the primary failure mode.** Without VICReg on the correct tensor (z_context), encoders collapse to a constant. Always monitor z_variance.

2. **z must carry information but not too much.** Random z → model ignores it. High-norm z → deterministic lookup. L2-normalize + calibrated noise gives a useful signal the decoder must reason over.

3. **Loss decomposition matters.** A decreasing total loss can mask complete failure in individual components. Always check each loss term independently.

4. **Checkpoint what you care about.** Checkpointing by energy saved collapsed models (energy 0 = degenerate). Use the actual target metric (cell accuracy).

5. **Training and inference must be aligned.** The inference energy must correspond to what the model learned. Constraint rules must be present during training if they're used during inference.

6. **Scale matters, but only after correctness.** Runs 2-3 had the same architecture as Run 4 but with broken Langevin dynamics. 5x parameter increase (Run 5) produced massive gains only because the foundational issues were resolved first.

---

## 11. Code Walkthrough

### Project Structure

```
src/ebm/
    main.py                 # CLI entry point: train and eval subcommands
    dataset/
        loader.py           # Kaggle dataset download + loading
        torch_dataset.py    # Tensor encoding (one-hot, masks)
        splits.py           # Deterministic train/val/test split
    model/
        encoder.py          # SudokuPositionalEncoding + SudokuEncoder
        predictor.py        # LatentPredictor (3-layer MLP)
        decoder.py          # SudokuDecoder (Transformer + clue enforcement)
        energy.py           # Energy function (L2 distance)
        constraints.py      # Differentiable Sudoku constraint penalty
        jepa.py             # SudokuJEPA orchestrator (forward + solve)
    training/
        trainer.py          # Training loop (AdamW, EMA, validation)
        losses.py           # VICReg + combined loss computation
        scheduler.py        # LR warmup + cosine decay, EMA schedule
        checkpoint.py       # Best-K checkpoint management
        metrics.py          # W&B logging, accuracy/variance metrics
    evaluation/
        solver.py           # Batch Langevin solving
        metrics.py          # EvalMetrics (cell/puzzle acc, constraints)
    utils/
        config.py           # Pydantic configs (arch, training, env)
        device.py           # GPU detection + auto-scaling
tests/                      # 101 unit tests, >96% coverage
scripts/
    eval_checkpoint.py      # Standalone evaluation + visualization
```

### File-by-File Guide

**`src/ebm/utils/config.py`** — Three Pydantic configuration classes. `Config` reads API keys from `.env`. `ArchitectureConfig` holds model hyperparameters (d_model, n_layers, etc.). `TrainingConfig` holds training hyperparameters (batch size, LR, loss weights, Langevin parameters). All have sensible defaults for the Run 5 configuration.

**`src/ebm/utils/device.py`** — Detects GPU VRAM and auto-scales batch size (via tier lookup) and learning rate (via sqrt rule). Falls back to config defaults on CPU.

**`src/ebm/dataset/loader.py`** — Wraps `kagglehub.dataset_download` with a file-existence check. Provides `load_all()`, `load_head(k)`, and `load_fraction(frac)` for flexible data loading without reading the full 9M-row CSV when not needed.

**`src/ebm/dataset/torch_dataset.py`** — Converts 81-character puzzle/solution strings to PyTorch tensors. Puzzles are (10, 9, 9) with an explicit empty-cell channel; solutions are (9, 9, 9) one-hot; masks are (9, 9) binary. Pre-parses all strings to `int8` numpy arrays at init for fast `__getitem__`.

**`src/ebm/dataset/splits.py`** — Deterministic train/val/test splitting via scikit-learn with seed 42. Default: 8M/500K/500K.

**`src/ebm/model/encoder.py`** — `SudokuPositionalEncoding` adds learned row + column + box embeddings. `SudokuEncoder` reshapes (B, C, 9, 9) to 81 tokens, projects to d_model, adds positional encoding, processes through pre-norm Transformer layers, and mean-pools to (B, d_model).

**`src/ebm/model/predictor.py`** — `LatentPredictor`: concat(z_context, z) → Linear → GELU → Linear → GELU → LayerNorm(+residual) → Linear → z_pred. Intentionally shallow to force reliance on z.

**`src/ebm/model/decoder.py`** — `SudokuDecoder`: concat(z_context, z) → project to 81 × d_cell → positional encoding → Transformer layers → Linear(d_cell, 9) → reshape to (B, 9, 9, 9). Hard-enforces given clues by overwriting clue cell logits.

**`src/ebm/model/energy.py`** — Single function: `energy_fn(z_pred, z_target) = ((z_pred - z_target) ** 2).sum(dim=-1)`. Per-sample L2 energy.

**`src/ebm/model/constraints.py`** — Precomputes 27 constraint groups (9 rows + 9 columns + 9 boxes), each listing 9 cell indices. `constraint_penalty` sums soft probabilities per group per digit and penalizes deviation from 1.0.

**`src/ebm/model/jepa.py`** — The central orchestrator. `SudokuJEPA.__init__` wires together context encoder, target encoder, z_encoder, predictor, decoder. `forward()` is the training pass: encode puzzle and solution, add noise to z, predict, decode, return all outputs. `solve()` is Langevin inference: initialize random z chains, iterate gradient descent with noise and temperature annealing, select lowest-energy chain, decode to integer grid.

**`src/ebm/training/losses.py`** — `vicreg_loss` computes variance + covariance penalty on z_context. `compute_loss` combines energy, VICReg, cross-entropy (empty cells only), and constraint penalty into the total training loss.

**`src/ebm/training/scheduler.py`** — `create_lr_scheduler` builds a SequentialLR: LinearLR warmup (2000 steps) + CosineAnnealingLR decay. `get_ema_momentum` linearly interpolates momentum from 0.996 to 1.0.

**`src/ebm/training/checkpoint.py`** — `CheckpointManager` uses a min-heap to track top-K checkpoints by cell accuracy. New checkpoints only saved if they exceed the current worst. Old checkpoints are deleted from disk. Loading restores model + optional optimizer state.

**`src/ebm/training/metrics.py`** — W&B integration (init, step logging, validation logging, artifact upload, finish). Also includes `compute_cell_accuracy` (empty cells only), `compute_puzzle_accuracy` (all-correct puzzles), and `compute_z_variance` (collapse detector).

**`src/ebm/training/trainer.py`** — `Trainer` runs the full loop: per-epoch `_train_epoch()` (forward, loss, backward, optimizer step, scheduler step, EMA update, logging) and `_validate()` (compute accuracy/energy/z-variance). Manages checkpoint saving and W&B lifecycle.

**`src/ebm/evaluation/solver.py`** — `solve_batch` wraps `model.solve()`. `solve_dataset` iterates a DataLoader, collecting predictions/solutions/masks for metric computation.

**`src/ebm/evaluation/metrics.py`** — `evaluate()` aggregates cell accuracy, puzzle accuracy, and constraint satisfaction across batches. `_constraint_satisfaction` sorts each group's predicted digits and checks against [1,2,...,9].

**`src/ebm/main.py`** — CLI entry point with `train` and `eval` subcommands. Orchestrates dataset loading, splitting, model creation, training/evaluation, and device management.

**`scripts/eval_checkpoint.py`** — Standalone script that loads a specific checkpoint, runs forward-pass evaluation on ~1K test puzzles, runs Langevin evaluation on 100 puzzles, prints example grids with error highlighting, and summarizes the Langevin delta.

### End-to-End Data Flow

**Training — one step:**
```
81-char string → SudokuTorchDataset.__getitem__()
  → puzzle: (10, 9, 9), solution: (9, 9, 9), mask: (9, 9)

DataLoader batches → puzzle: (B, 10, 9, 9), solution: (B, 9, 9, 9), mask: (B, 9, 9)

SudokuJEPA.forward(puzzle, solution, mask):
  context_encoder(puzzle)                    → z_context: (B, 512)
  target_encoder(solution)                   → z_target:  (B, 512)  [no grad]
  z_encoder(z_target) → normalize → + noise  → z:         (B, 256)
  predictor(z_context, z)                    → z_pred:    (B, 512)
  energy_fn(z_pred, z_target)                → energy:    (B,)
  decoder(z_context, z, puzzle, mask)        → logits:    (B, 9, 9, 9)

compute_loss(output, solution, mask, cfg):
  energy.mean()                              → energy_loss
  vicreg_loss(z_context)                     → vicreg_loss
  cross_entropy(logits[empty], targets)      → decode_loss
  constraint_penalty(softmax(logits))        → constraint_loss
  sum with weights                           → total_loss

total_loss.backward()
optimizer.step()
scheduler.step()
model.update_target_encoder(momentum)
```

**Inference — one puzzle:**
```
puzzle: (1, 10, 9, 9), mask: (1, 9, 9)

SudokuJEPA.solve(puzzle, mask):
  context_encoder(puzzle)                         → z_context: (1, 512)
  repeat for n_chains                             → z_context: (8, 512)
  z = randn(8, 256)                               → z: (8, 256)

  for step in range(50):
    predictor(z_context, z)                       → z_pred: (8, 512)
    decoder(z_context, z, puzzle, mask)            → logits: (8, 9, 9, 9)
    softmax(logits)                               → probs: (8, 9, 9, 9)
    target_encoder(probs)                         → z_target_est: (8, 512)
    (z_pred - z_target_est)²                      → self_consistency: (8,)
    constraint_penalty(probs)                     → c_penalty: (8,)
    total_energy = self_consistency + weighted c_penalty
    grad_z = autograd.grad(total_energy, z)
    z = z - lr * grad_z + noise * temp

  select min-energy chain                         → best_logits: (1, 9, 9, 9)
  argmax + 1                                      → solution: (1, 9, 9)
```

---

## Appendix: Hyperparameter Justifications

### Architecture (Run 5)

**$d_{\text{model}} = 512$** — The Transformer hidden dimension. Run 4 used $d_{\text{model}} = 256$ (7.4M params) and plateaued at 82.5% puzzle accuracy; doubling to 512 (36.5M params) reached 95.6%. The capacity increase was justified by the observation that the model's forward-pass accuracy was still improving at epoch 20 in Run 4 — it was underfitting, not overfitting. 512 is a standard Transformer width (matching BERT-base, ViT-Small) that balances capacity with training speed. 1024 was not tested due to memory constraints, but the accuracy curve's near-plateau at epoch 17-20 of Run 5 suggests diminishing returns.

**$n_{\text{layers}} = 8$** — Encoder Transformer depth. Sudoku requires multi-hop reasoning (placing digit X in row 1 constrains row 1, which constrains column 3, which constrains box 2). Each Transformer layer can propagate information one "hop" via attention. 8 layers provide sufficient depth for the longest constraint chains in a 9x9 grid. Run 4 used 6 layers; the increase to 8 was part of the general capacity scaling. Deeper models (12+) risk gradient degradation with the pre-norm architecture and the relatively short 81-token sequence.

**$n_{\text{heads}} = 8$** — Attention heads, giving head dimension $d_{\text{model}} / n_{\text{heads}} = 64$. This is the standard head dimension used in most Transformer architectures (GPT-2, BERT, ViT). Each head can specialize in different constraint patterns (row relationships, column relationships, box relationships, etc.). Fewer heads (4) would reduce this specialization capacity; more (16) would halve the head dimension to 32, reducing per-head expressiveness.

**$d_{\text{ffn}} = 2048$** — Feed-forward network inner dimension, following the standard $4 \times d_{\text{model}}$ ratio from the original Transformer (Vaswani et al., 2017). This expansion ratio provides a nonlinear capacity bottleneck that has been empirically validated across hundreds of Transformer applications.

**$\text{dropout} = 0.1$** — Standard Transformer dropout. With 8M training samples and 36.5M parameters, the model is in a data-rich regime (data/param ratio $\approx 220$), so heavy regularization is unnecessary. 0.1 provides mild regularization against co-adaptation without significantly hurting training signal.

**$d_{\text{latent}} = 256$** — Latent variable $\mathbf{z}$ dimension, set to $d_{\text{model}} / 2$. This is the information bottleneck between the target encoder and the predictor/decoder. Too small ($< 64$) would lose critical solution information; too large ($= d_{\text{model}}$) would eliminate the bottleneck, making $\mathbf{z}$ a near-lossless copy. $d_{\text{model}} / 2$ halves the information while keeping the Langevin search space tractable — a random search in $\mathbb{R}^{256}$ is substantially easier than $\mathbb{R}^{512}$.

**$\text{predictor\_hidden} = 1024$** — Predictor MLP hidden dimension, set to $2 \times d_{\text{model}}$. The predictor must be capacity-limited (to force reliance on $\mathbf{z}$), but not so small that it can't learn the prediction mapping. A single hidden layer of $2 \times d_{\text{model}}$ provides a $768 \to 1024 \to 512$ bottleneck that can approximate the mapping while remaining too shallow to ignore $\mathbf{z}$. The 3-layer depth (input → hidden → output) was chosen as the minimum that supports a residual connection.

**$\text{decoder\_layers} = 4$** — Decoder depth, half the encoder. The decoder's job is simpler: unpack a compressed representation into per-cell predictions, guided by inter-cell attention for consistency. It doesn't need to build a holistic understanding from scratch. 4 layers provide enough refinement while keeping inference (Langevin dynamics, which calls the decoder every step) computationally tractable.

**$\text{decoder\_d\_cell} = 128$** — Per-cell feature dimension in the decoder, $d_{\text{model}} / 4$. Each cell needs to represent a distribution over 9 digits — 128 dimensions is generous for this. The decoder processes 81 tokens of dimension 128, so Transformer attention cost is $O(81^2 \times 128)$, kept modest for the per-step Langevin calls.

### Training

**$B = 2048$** — Batch size, auto-scaled from GPU VRAM. With the RTX 5090 (32GB), 2048 was the maximum stable batch size for the 7.4M model (Run 4). The H200 (144GB) could fit larger batches but 2048 was kept for comparability. Larger batches provide better gradient estimates (lower variance) and improve GPU utilization. The jump from $B = 512$ (Run 2) to $B = 2048$ (Run 3) contributed +9.1% puzzle accuracy, demonstrating the value of gradient quality.

**$\eta = 6 \times 10^{-4}$** — Peak learning rate, auto-scaled from the base pair $(\eta_{\text{base}} = 3 \times 10^{-4}, B_{\text{base}} = 512)$ via the sqrt rule. $3 \times 10^{-4}$ is the standard AdamW learning rate (established by BERT, widely adopted). The sqrt scaling to $6 \times 10^{-4}$ for $B = 2048$ is conservative — linear scaling would give $1.2 \times 10^{-3}$, which risks instability with adaptive optimizers.

**$\text{weight\_decay} = 0.01$** — AdamW L2 regularization. The standard default from the AdamW paper (Loshchilov & Hutter, 2019). With a large dataset (8M samples), the model isn't prone to overfitting, so weight decay serves primarily as a stability mechanism rather than a regularizer.

**$\text{warmup\_steps} = 2000$** — Linear LR warmup. Prevents catastrophic updates in early training when weights are random and gradients are large/noisy. 2000 steps corresponds to ~1 epoch at $B = 2048$ with 8M samples ($8M / 2048 \approx 3900$ steps/epoch), giving roughly half an epoch of warmup. This is in the standard range (0.5-2 epochs) for Transformer training.

**$\text{epochs} = 20$** — Training duration. Empirically determined: Run 5 plateaued at epoch 17 (99.3% cell / 95.6% puzzle for epochs 17-19), indicating convergence. More epochs would provide negligible improvement with cosine-decayed LR near zero.

**$\text{grad\_clip\_norm} = 1.0$** — Gradient clipping. Standard safety measure for Transformer training that prevents occasional large gradients (from unlucky batches or numerical instabilities) from causing catastrophic parameter updates. 1.0 is the most common choice (used by GPT-2, T5, etc.).

**$m \in [0.996, 1.0]$** — EMA momentum schedule. $m_0 = 0.996$ corresponds to an effective averaging window of $1/(1-m) = 250$ steps. This is fast enough to track early learning while providing smoothing. The linear increase to $1.0$ gradually freezes the target encoder, following the BYOL (Grill et al., 2020) and I-JEPA convention. Alternative schedules (cosine) showed no significant difference in preliminary experiments.

**$\sigma_z = 0.1$** — Training noise scale on $\mathbf{z}$. Calibrated in Run 4 to give SNR $\approx 0.6$ after L2 normalization. This is in the "genuinely noisy" regime: the model can extract useful signal from $\mathbf{z}$ but cannot treat it as deterministic. Values much lower ($0.01$) make $\mathbf{z}$ too clean (reverting to the lookup-table problem from Runs 2-3); values much higher ($1.0$) overwhelm the signal and the model ignores $\mathbf{z}$ entirely.

**$w_{\text{decode}} = 1.0$** — Decode loss weight. Equal weight with energy and VICReg ensures the decoder receives strong supervision. The decode loss provides the most direct learning signal (correct digit predictions), so underweighting it would slow convergence significantly.

**$w_{\text{constraint}} = 0.1$** — Constraint loss weight. Deliberately conservative: the constraint penalty and cross-entropy loss can conflict during early training (soft probability sums may deviate from 1.0 even for correct argmax predictions). At $0.1$, the constraint signal provides gentle structural guidance without destabilizing the dominant cross-entropy signal. Higher values ($0.5$, $1.0$) were tested and caused slower initial convergence due to the conflicting gradients.

**$\lambda_{\text{var}} = 1.0$, $\lambda_{\text{cov}} = 0.01$** — VICReg weights. The variance term is the primary anti-collapse mechanism and needs full weight. The covariance term operates on a different scale ($D(D-1)/2 \approx 131K$ off-diagonal entries for $D = 512$) and must be downweighted to avoid dominating. $0.01$ was chosen empirically; $1.0$ caused training instability, $0.001$ left noticeable dimension correlation.

### Inference (Langevin Dynamics)

**$T = 50$ steps** — Langevin iteration count. Empirically, accuracy improvements plateau after $\sim$30-50 steps. More steps add latency without meaningful gains; fewer steps don't allow sufficient convergence from random initialization. At 50 steps, the solver adds +1.0% puzzle accuracy over the forward pass.

**$K = 8$ chains** — Parallel Langevin trajectories per puzzle. Multiple chains hedge against local minima in the energy landscape. The improvement from 1 to 4 chains is significant; from 4 to 8 is modest; from 8 to 16 is marginal. 8 chains provide a good balance between accuracy and compute — all chains run in parallel on the GPU, so the wall-clock cost scales with the number of chains only through memory bandwidth, not latency.

**$\eta_{\text{Langevin}} = 0.01$** — Langevin step size. An order of magnitude smaller than the training LR ($6 \times 10^{-4}$), appropriate because Langevin optimization must be conservative — overshooting in latent space can produce solutions that are worse than the starting point. The energy gradients in the learned landscape are sharper than training loss gradients, requiring smaller steps for stability.

**$\sigma_{\text{Langevin}} = 0.005$** — Base noise magnitude for Langevin dynamics. Half the step size, following the Langevin MCMC convention that noise should be smaller than the gradient step to ensure net progress toward low-energy regions. The noise is further annealed by the temperature schedule $\tau_t$, so effective noise is $0.005 \cdot \tau_t$, vanishing to zero by the final step.
