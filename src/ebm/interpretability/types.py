"""Shared data structures for mechanistic interpretability analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from torch import Tensor


class StrategyLabel(Enum):
    """Human Sudoku solving strategies."""

    NAKED_SINGLE = 'naked_single'
    HIDDEN_SINGLE = 'hidden_single'
    POINTING_PAIR = 'pointing_pair'
    BOX_LINE_REDUCTION = 'box_line_reduction'
    NAKED_PAIR = 'naked_pair'
    HIDDEN_PAIR = 'hidden_pair'
    NAKED_TRIPLE = 'naked_triple'
    HIDDEN_TRIPLE = 'hidden_triple'
    X_WING = 'x_wing'
    UNKNOWN = 'unknown'


@dataclass
class StepSnapshot:
    """One Langevin step's complete state."""

    step: int
    z: Tensor
    logits: Tensor
    probs: Tensor
    board: Tensor
    energy: Tensor
    self_consistency: Tensor
    constraint_penalty: Tensor
    grad_norm: Tensor
    encoder_attention: dict[str, Tensor] | None = None
    decoder_attention: dict[str, Tensor] | None = None


@dataclass
class Trajectory:
    """Full trajectory for a batch of puzzles."""

    puzzle: Tensor
    solution: Tensor
    mask: Tensor
    steps: list[StepSnapshot]
    final_board: Tensor


@dataclass
class CellEvent:
    """A detected cell-fill event during the Langevin trajectory."""

    step: int
    row: int
    col: int
    digit: int
    strategy: StrategyLabel | None = None
    confidence: float = 0.0


@dataclass
class HeadProfile:
    """Structural specialization scores for one attention head."""

    layer: str
    head_idx: int
    row_score: float
    col_score: float
    box_score: float
    specialization: str  # 'row', 'column', 'box', or 'mixed'


@dataclass
class LockInEvent:
    """When a cell locks in to its final correct digit."""

    row: int
    col: int
    digit: int
    lock_in_step: int
    strategy: StrategyLabel
    confidence_at_lock: float


@dataclass
class TrajectoryMetrics:
    """Quantitative interpretability metrics for one trajectory."""

    strategy_coverage: float
    lock_in_events: list[LockInEvent]
    step_strategy_correlation: float
    phase_boundaries: list[int]
    energy_decomposition: list[tuple[int, float, float]]


@dataclass
class AblationResult:
    """Results from a head ablation experiment."""

    ablated_heads: list[tuple[str, int]]
    overall_accuracy: float
    strategy_accuracy: dict[str, float]
    baseline_accuracy: float
    baseline_strategy_accuracy: dict[str, float]


@dataclass
class EnergyProfile:
    """Energy and its components at a specific z value."""

    z: Tensor  # (B, d_latent)
    energy: Tensor  # (B,)
    self_consistency: Tensor  # (B,)
    constraint_penalty: Tensor  # (B,)
    logits: Tensor  # (B, 9, 9, 9)
    probs: Tensor  # (B, 9, 9, 9)


@dataclass
class MultiChainTrajectory:
    """Trajectory data for multiple Langevin chains on the same puzzle."""

    puzzle: Tensor  # (B, 10, 9, 9)
    solution: Tensor  # (B, 9, 9, 9)
    mask: Tensor  # (B, 9, 9)
    n_chains: int
    n_steps: int
    chain_z: Tensor  # (n_steps, B, n_chains, d_latent)
    chain_energy: Tensor  # (n_steps, B, n_chains)
    final_boards: Tensor  # (B, n_chains, 9, 9)
    final_z: Tensor  # (B, n_chains, d_latent)
    final_energies: Tensor  # (B, n_chains)


@dataclass
class AnalysisResult:
    """Aggregated analysis of a trajectory."""

    events: list[CellEvent]
    strategy_counts: dict[str, int] = field(default_factory=dict)
    strategy_by_step: dict[int, list[CellEvent]] = field(default_factory=dict)
    attention_by_strategy: dict[str, list[Tensor]] = field(default_factory=dict)
