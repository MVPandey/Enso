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

An Energy-Based Model (EBM) assigns a scalar energy value to every possible configuration of its inputs. Low energy = compatible/correct configuration. High energy = incompatible/incorrect. The model learns an energy function E(x, y) such that:

- E(puzzle, correct_solution) → **low energy**
- E(puzzle, wrong_solution) → **high energy**

The elegance of EBMs is that at inference time, you can **search for the lowest-energy configuration** using gradient descent. Unlike a feedforward model that produces a single answer, an EBM can iteratively refine a candidate solution by following the energy gradient downhill.

This is exactly what the brain seems to do: rather than computing an answer in a single pass, we "think about it" — iteratively evaluating and adjusting candidate solutions until we reach one that satisfies all constraints.

### Joint Embedding Predictive Architecture (JEPA)

JEPA, introduced by Yann LeCun and formalized in the [I-JEPA paper](https://arxiv.org/abs/2301.08243) (CVPR 2023), is a specific flavor of self-supervised energy-based learning. The key idea: **make predictions in representation space, not pixel/token space.**

Why not predict in data space directly? Because data space is enormous and full of irrelevant variation. Two correct Sudoku solutions that differ only in the ordering of equivalent digits are fundamentally identical — but in raw data space, they look completely different. By mapping inputs to a learned representation space and making predictions there, the model can focus on the abstract structure that matters.

JEPA has three main components:
1. **Context encoder**: Encodes the observable input (puzzle) to a representation z_context
2. **Target encoder**: Encodes the target (solution) to a representation z_target
3. **Predictor**: Maps z_context → z_pred, trained so that z_pred ≈ z_target

The energy is simply ||z_pred - z_target||² — how far the predicted representation is from the target representation. At inference time, we don't have z_target (we don't know the solution), so we use Langevin dynamics to search for a latent variable z that minimizes the energy.

Enso draws from three key papers:
- **I-JEPA** (CVPR 2023): The foundational JEPA framework for self-supervised learning
- **IRED** (2024): Iterative Reasoning with Energy Diffusion — applying EBMs to reasoning tasks with Langevin dynamics
- **JEPA-Reasoner** (2025): Applying JEPA specifically to logical reasoning tasks like Sudoku

---

## 2. Architecture Overview

Enso's architecture has two distinct modes: **training** and **inference**.

**During training**, the model sees both the puzzle and the solution:
1. The **context encoder** processes the puzzle → z_context (512-dim vector)
2. The **target encoder** (EMA copy) processes the solution → z_target (512-dim vector)
3. z_target is projected to latent space via **z_encoder** and noise is added → z (256-dim vector)
4. The **predictor** maps (z_context, z) → z_pred (512-dim vector)
5. The **decoder** maps (z_context, z) → cell logits (9x9x9 tensor)
6. Loss combines energy (z_pred vs z_target), VICReg, cross-entropy on decode, and constraint penalty

**During inference**, the model sees only the puzzle:
1. The context encoder processes the puzzle → z_context
2. Multiple z vectors are initialized randomly from N(0, I)
3. For T steps, z is optimized via Langevin dynamics to minimize self-consistency energy + constraint penalty
4. The lowest-energy z is decoded to a 9x9 solution grid

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

```python
target_param = momentum * target_param + (1 - momentum) * context_param
```

**File:** `src/ebm/model/jepa.py` (lines 98-112)

Why EMA instead of sharing weights or training independently?

**Shared weights** cause a degenerate solution: the model can trivially minimize energy by mapping everything to the same point (representation collapse). If the same encoder processes both puzzle and solution, it can learn to ignore the input entirely and produce a constant output — energy zero for free.

**Independent training** doesn't provide a stable target. If both encoders change rapidly every step, the energy landscape shifts under the predictor's feet, making optimization chaotic.

**EMA** provides the best of both worlds: the target encoder **slowly tracks** the context encoder, providing a stable but gradually improving target. Early in training (momentum=0.996), the target updates relatively quickly to keep up with rapid learning. Late in training (momentum→1.0), the target barely moves, providing a nearly fixed reference for fine-tuning.

The momentum schedule is linear interpolation:
```python
momentum = 0.996 + progress * (1.0 - 0.996)  # 0.996 → 1.0 over training
```

The `_init_target_encoder` method copies matching weights from the context encoder at initialization, but skips `input_proj` since the channel dimensions differ (10 vs 9). EMA updates similarly skip mismatched parameters.

---

## 5. The Latent Space

### z_encoder: Projecting to Latent Space

**File:** `src/ebm/model/jepa.py` (line 81)

The target encoder produces a d_model-dimensional (512) representation of the solution. The `z_encoder` is a single linear layer that projects this down to d_latent dimensions (256):

```python
self.z_encoder = nn.Linear(arch_cfg.d_model, arch_cfg.d_latent)
```

This projection serves two purposes:
1. **Information bottleneck:** The latent variable z should carry a compressed, essential summary of the solution — not a lossless copy. A smaller dimension forces the model to distill the most important information.
2. **Latent space for Langevin dynamics:** At inference time, we optimize z via gradient descent. A lower-dimensional space is easier to search.

The z_encoder output is **L2-normalized** before noise is added:
```python
z_target_latent = F.normalize(self.z_encoder(z_target), dim=-1)
```

This normalization is critical — see the next section for why.

### The Noise Injection Problem (and Solution)

During training, z is constructed as:
```python
z = z_target_latent + noise_scale * torch.randn_like(z_target_latent)
```

The noise injection simulates the inference scenario: at test time, z is initialized randomly and iteratively refined. During training, the model sees a **noisy version** of the true z, so it must learn to work with imprecise latent variables.

**The problem discovered in Runs 2-3:** Without normalization, the z_encoder output had L2 norm ~144, while the noise (from `randn`) had norm ~11. The signal-to-noise ratio was 13:1 — z was essentially a deterministic copy of the solution encoding, not a noisy variable. The decoder learned to treat z as a lookup table, making Langevin dynamics useless: optimizing a random z couldn't recover the high-magnitude z the decoder expected.

**The fix in Run 4:** L2-normalize the z_encoder output to unit norm, and reduce `z_noise_scale` from 1.0 to 0.1. With d_latent=256, random noise of scale 0.1 has L2 norm ≈ 0.1 * sqrt(256) ≈ 1.6, giving an SNR of roughly 0.6:1. Now z carries solution information but is genuinely noisy — the model must learn to reason under uncertainty, which is exactly what Langevin dynamics requires.

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

```python
energy_loss = out.energy.mean()  # = ((z_pred - z_target) ** 2).sum(dim=-1).mean()
```

**File:** `src/ebm/model/energy.py`

This is the core JEPA objective: minimize the squared L2 distance between the predicted representation (z_pred) and the target representation (z_target). Lower energy means the predictor successfully anticipated the target encoder's output given the puzzle encoding and the latent variable.

The energy function is deliberately simple — just ||z_pred - z_target||². Complex energy functions (e.g., with learned temperature parameters) are harder to optimize and can introduce training instabilities. The simplicity here is a feature: it provides clean gradients and makes the energy landscape interpretable.

### VICReg: Preventing Representation Collapse

```python
vreg = vicreg_loss(out.z_context, var_weight=1.0, cov_weight=0.01)
```

**VICReg** (Variance-Invariance-Covariance Regularization, [Bardes et al., 2022](https://arxiv.org/abs/2105.04906)) is a regularization technique that prevents **representation collapse** — the failure mode where the encoder maps all inputs to the same point in representation space.

Without VICReg, there's a trivial solution to the energy objective: map every puzzle to the same constant vector, map every solution to the same constant vector, predict that constant. Energy = 0, but the representation carries no information.

VICReg has two components applied to z_context:

**Variance term** — push the standard deviation of each representation dimension toward 1:
```python
var_loss = F.relu(1.0 - std).mean()  # penalize dimensions with std < 1
```
This ensures the representation uses the full space, not collapsing to a low-dimensional manifold.

**Covariance term** — decorrelate dimensions:
```python
cov = (z.T @ z) / (num_samples - 1)
off_diag = cov - torch.diag(cov.diag())
cov_loss = (off_diag**2).sum() / num_dims
```
This pushes the off-diagonal entries of the covariance matrix toward zero, ensuring each dimension carries independent information. Without this, the model could satisfy the variance constraint by simply repeating the same signal across many dimensions.

**Critical implementation detail:** VICReg is applied to **z_context** (the context encoder's output), not z_pred. Run 1 discovered that applying VICReg to z_pred allowed the target encoder to collapse freely — the predictor's output had high variance, but the target it was predicting was degenerate.

### Decode Loss

```python
targets = solution.argmax(dim=-1)           # (B, 9, 9) digit indices 0-8
empty = mask == 0                           # boolean mask
decode_loss = F.cross_entropy(out.decode_logits[empty], targets[empty])
```

Standard cross-entropy loss on the decoder's digit predictions, computed **only on empty cells** (mask == 0). This is an auxiliary loss that provides direct supervision: the decoder should predict the correct digit for each unknown cell.

Computing loss only on empty cells is essential — including given clue cells would let the model achieve low loss by simply copying the input, without learning to solve anything.

### Constraint Loss

**File:** `src/ebm/model/constraints.py`

```python
decode_probs = torch.softmax(out.decode_logits, dim=-1)
constraint_loss = constraint_penalty(decode_probs).mean()
```

A differentiable Sudoku constraint penalty that explicitly teaches the model about row/column/box uniqueness rules.

The constraint groups are the 27 sets of 9 cells that must each contain all digits 1-9: 9 rows + 9 columns + 9 boxes.

For each group and each digit, the penalty measures how far the sum of probabilities deviates from 1.0:

```python
digit_sums = group_probs.sum(dim=2)   # sum of P(digit d) across 9 cells in group
penalty = ((digit_sums - 1.0) ** 2).sum(dim=(1, 2))  # sum over 27 groups x 9 digits
```

If digit 5 has probability 0.3 in three cells of the same row, the sum is 0.9, contributing (0.9 - 1.0)² = 0.01 to the penalty. A perfectly valid solution (each digit appearing exactly once per group) gives penalty = 0.

**Why this was added in Run 4:** Without explicit constraint signals during training, the model only learns Sudoku rules implicitly through cross-entropy loss. It discovers "each digit appears once per row" by observing enough examples, but never sees this stated as a rule. The constraint loss provides a direct gradient signal: "this configuration violates Sudoku rules, adjust these cell probabilities." This was one of three fixes that made Langevin dynamics work.

The `GROUP_INDICES` tensor (27 x 9) is precomputed at module load time, mapping each of the 27 groups to 9 flat cell indices (0-80).

### The Combined Loss

```python
total = energy_loss + vreg + 1.0 * decode_loss + 0.1 * constraint_loss
```

The decode loss weight (1.0) and constraint loss weight (0.1) were chosen empirically. The constraint loss at 0.1 is conservative — too much constraint weight can fight against the cross-entropy signal, since the soft constraint penalty and the hard digit labels may disagree during early training when predictions are poor.

---

## 8. Langevin Dynamics: Thinking at Inference Time

### The Core Idea

**File:** `src/ebm/model/jepa.py` (lines 147-212)

At inference time, we don't have the solution — we can't compute z_target. The entire inference strategy is: **find a latent variable z that the model believes is consistent with the puzzle.**

Langevin dynamics is a gradient-based MCMC method that samples from an energy landscape by iteratively following the energy gradient with added noise:

```
z_{t+1} = z_t - η * ∇_z E(z_t) + σ * noise_t
```

In physics terms: a particle (z) rolls downhill on the energy surface (gradient term) while being jostled by random thermal fluctuations (noise term). The gradient term drives the particle toward low-energy regions (valid solutions); the noise term prevents it from getting stuck in local minima.

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

The inference energy is **not** the same as the training energy. During training, the energy is ||z_pred - z_target||² where z_target comes from encoding the **true solution**. At inference, we don't have the solution.

**Runs 2-3** used ||z_pred||² as a proxy (assuming z_target ≈ 0), which is wrong — z_target is not zero. This produced misleading gradients that actively hurt accuracy.

**Run 4** introduced **self-consistency energy**: decode the current z to a candidate solution (soft probabilities), re-encode that candidate through the target encoder, and measure how well the predictor's output matches:

```python
z_target_est = self.target_encoder(probs.permute(0, 3, 1, 2))
self_consistency = ((z_pred - z_target_est) ** 2).sum(dim=-1)
```

The intuition: a consistent z should produce a decoded solution that, when re-encoded, matches the predictor's expectation. If z leads to a garbled solution, re-encoding it produces a different representation than what the predictor expects, resulting in high self-consistency energy.

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

The noise scale and constraint weight both change over the course of the Langevin trajectory:

```python
temp = 1.0 - step / max(n_steps, 1)  # 1.0 → 0.0 linearly

noise = noise_scale * temp * torch.randn_like(z)                    # decreasing noise
total_energy = self_consistency + c_penalty * (1.0 + 2.0 * (1.0 - temp))  # increasing constraint weight
```

**Early steps (temp ≈ 1.0):** High noise encourages exploration of the energy landscape. Constraint weight is low (1.0), allowing the model to focus on energy minimization.

**Late steps (temp ≈ 0.0):** Noise vanishes, letting z settle into the nearest energy minimum. Constraint weight increases to 3.0, heavily penalizing Sudoku violations in the final solution.

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

```python
momentum = ema_momentum_start + progress * (ema_momentum_end - ema_momentum_start)
# 0.996 → 1.0 over training
```

The target encoder's EMA momentum increases linearly from 0.996 to 1.0. At momentum 0.996, roughly 0.4% of the context encoder's weights bleed through per step — fast enough to track early rapid learning. At momentum ~1.0 near the end, the target is nearly frozen, providing a stable reference for the final fine-tuning phase.

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
```python
scaled_lr = base_lr * sqrt(actual_batch / base_batch)
# base: lr=3e-4 at batch_size=512
# e.g., batch_size=2048 → lr = 3e-4 * sqrt(4) = 6e-4
```

The sqrt rule is more conservative than linear scaling and works well with adaptive optimizers like AdamW. The intuition: with k times more samples per gradient step, the gradient estimate's variance decreases by 1/k, so its magnitude (standard deviation) decreases by 1/sqrt(k), justifying a sqrt(k) increase in step size.

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

## Appendix: Hyperparameter Reference

### Architecture (Run 5)

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 512 | Transformer hidden dimension |
| n_layers | 8 | Encoder Transformer layers |
| n_heads | 8 | Attention heads |
| d_ffn | 2048 | Feed-forward inner dimension |
| dropout | 0.1 | Dropout rate |
| d_latent | 256 | Latent variable z dimension |
| predictor_hidden | 1024 | Predictor MLP hidden size |
| decoder_layers | 4 | Decoder Transformer layers |
| decoder_heads | 8 | Decoder attention heads |
| decoder_d_cell | 128 | Per-cell decoder dimension |

### Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| batch_size | 2048 | Training batch size (auto-scaled) |
| lr | 6e-4 | Peak learning rate (auto-scaled) |
| weight_decay | 0.01 | AdamW weight decay |
| warmup_steps | 2000 | LR warmup steps |
| epochs | 20 | Training epochs |
| grad_clip_norm | 1.0 | Gradient clipping max norm |
| ema_momentum | 0.996 → 1.0 | Target encoder EMA momentum |
| z_noise_scale | 0.1 | Training noise on z |
| decode_loss_weight | 1.0 | Cross-entropy loss weight |
| constraint_loss_weight | 0.1 | Sudoku constraint loss weight |
| vicreg_var_weight | 1.0 | VICReg variance weight |
| vicreg_cov_weight | 0.01 | VICReg covariance weight |

### Inference (Langevin Dynamics)

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_steps | 50 | Langevin iterations |
| n_chains | 8 | Parallel z chains per puzzle |
| lr | 0.01 | Langevin step size |
| noise_scale | 0.005 | Base noise magnitude |
