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
class AnalysisResult:
    """Aggregated analysis of a trajectory."""

    events: list[CellEvent]
    strategy_counts: dict[str, int] = field(default_factory=dict)
    strategy_by_step: dict[int, list[CellEvent]] = field(default_factory=dict)
    attention_by_strategy: dict[str, list[Tensor]] = field(default_factory=dict)
