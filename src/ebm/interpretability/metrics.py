"""Quantitative interpretability metrics for trajectory analysis."""

from __future__ import annotations

import math
from collections import Counter

from scipy.stats import spearmanr
from torch import Tensor

from ebm.interpretability.strategies import STRATEGY_DIFFICULTY
from ebm.interpretability.types import CellEvent, LockInEvent, StrategyLabel, Trajectory, TrajectoryMetrics


class MetricsComputer:
    """Compute quantitative interpretability metrics from trajectories."""

    def __init__(self, lock_in_threshold: float = 0.9) -> None:
        """Initialize with lock-in probability threshold."""
        self._threshold = lock_in_threshold

    def compute_lock_in(
        self,
        trajectory: Trajectory,
        solution: Tensor,
        batch_idx: int,
    ) -> list[LockInEvent]:
        """
        Detect when each cell locks in to its final correct digit.

        Lock-in step = first step where prob >= threshold AND it never drops
        below threshold * 0.95 for the rest of the trajectory.

        Args:
            trajectory: Full trajectory from TrajectoryRecorder.
            solution: (B, 9, 9, 9) one-hot ground truth.
            batch_idx: Index into the batch dimension.

        Returns:
            List of LockInEvent for each non-clue cell that locks in.

        """
        mask = trajectory.mask[batch_idx]
        sol = solution[batch_idx]  # (9, 9, 9)
        n_steps = len(trajectory.steps)
        events: list[LockInEvent] = []
        sustain_threshold = self._threshold * 0.95

        for r in range(9):
            for c in range(9):
                if mask[r, c] > 0:
                    continue
                correct_digit = int(sol[r, c].argmax().item()) + 1
                d_idx = correct_digit - 1

                # Collect probabilities for the correct digit across all steps
                probs_series = []
                for step in trajectory.steps:
                    p = float(step.probs[batch_idx, r, c, d_idx].item())
                    probs_series.append(p)

                # Find lock-in step
                lock_in_step = None
                for s in range(n_steps):
                    if probs_series[s] >= self._threshold:
                        sustained = all(probs_series[t] >= sustain_threshold for t in range(s, n_steps))
                        if sustained:
                            lock_in_step = s
                            break

                if lock_in_step is not None:
                    events.append(
                        LockInEvent(
                            row=r,
                            col=c,
                            digit=correct_digit,
                            lock_in_step=lock_in_step,
                            strategy=StrategyLabel.UNKNOWN,
                            confidence_at_lock=probs_series[lock_in_step],
                        )
                    )

        return events

    def compute_strategy_step_correlation(self, events: list[CellEvent]) -> float:
        """
        Compute Spearman rank correlation between strategy difficulty and step number.

        Args:
            events: Classified cell events.

        Returns:
            Spearman rho value. Returns 0.0 if insufficient data.

        """
        if len(events) < 3:
            return 0.0

        difficulties = []
        steps = []
        for event in events:
            if event.strategy is not None:
                difficulties.append(STRATEGY_DIFFICULTY.get(event.strategy, 6))
                steps.append(event.step)

        if len(difficulties) < 3:
            return 0.0

        rho, _ = spearmanr(difficulties, steps)
        # spearmanr can return nan for constant input
        if math.isnan(rho):
            return 0.0
        return float(rho)

    def detect_phase_transitions(self, events: list[CellEvent], window: int = 5) -> list[int]:
        """
        Detect steps where the dominant strategy type changes.

        Args:
            events: Classified cell events.
            window: Bin size for grouping events by step range.

        Returns:
            List of step numbers where phase transitions occur.

        """
        if not events:
            return []

        bins: dict[int, list[CellEvent]] = {}
        for event in events:
            bin_idx = event.step // window
            bins.setdefault(bin_idx, []).append(event)

        # Find dominant strategy per bin
        dominant: dict[int, str] = {}
        for bin_idx in sorted(bins):
            counter = Counter(e.strategy.value if e.strategy else 'unknown' for e in bins[bin_idx])
            dominant[bin_idx] = counter.most_common(1)[0][0]

        # Find transitions
        boundaries: list[int] = []
        sorted_bins = sorted(dominant)
        for i in range(1, len(sorted_bins)):
            if dominant[sorted_bins[i]] != dominant[sorted_bins[i - 1]]:
                boundaries.append(sorted_bins[i] * window)

        return boundaries

    def compute_strategy_coverage(self, events: list[CellEvent]) -> float:
        """
        Compute fraction of events with known strategy (not UNKNOWN).

        Args:
            events: Classified cell events.

        Returns:
            Coverage ratio in [0, 1].

        """
        if not events:
            return 1.0
        unknown_count = sum(1 for e in events if e.strategy is None or e.strategy == StrategyLabel.UNKNOWN)
        return 1.0 - unknown_count / len(events)

    def compute_energy_decomposition(
        self,
        trajectory: Trajectory,
        batch_idx: int,
    ) -> list[tuple[int, float, float]]:
        """
        Extract per-step energy decomposition.

        Args:
            trajectory: Full trajectory.
            batch_idx: Index into the batch.

        Returns:
            List of (step, self_consistency, constraint_penalty) tuples.

        """
        result: list[tuple[int, float, float]] = []
        for step in trajectory.steps:
            sc = (
                float(step.self_consistency[batch_idx].item())
                if step.self_consistency.dim() > 0
                else float(step.self_consistency.item())
            )
            cp = (
                float(step.constraint_penalty[batch_idx].item())
                if step.constraint_penalty.dim() > 0
                else float(step.constraint_penalty.item())
            )
            result.append((step.step, sc, cp))
        return result

    def compute_all(
        self,
        trajectory: Trajectory,
        events: list[CellEvent],
        solution: Tensor,
        batch_idx: int,
    ) -> TrajectoryMetrics:
        """
        Compute all metrics at once.

        Args:
            trajectory: Full trajectory.
            events: Classified cell events.
            solution: (B, 9, 9, 9) one-hot ground truth.
            batch_idx: Index into the batch.

        Returns:
            TrajectoryMetrics with all computed values.

        """
        lock_in_events = self.compute_lock_in(trajectory, solution, batch_idx)
        correlation = self.compute_strategy_step_correlation(events)
        phases = self.detect_phase_transitions(events)
        coverage = self.compute_strategy_coverage(events)
        energy = self.compute_energy_decomposition(trajectory, batch_idx)

        return TrajectoryMetrics(
            strategy_coverage=coverage,
            lock_in_events=lock_in_events,
            step_strategy_correlation=correlation,
            phase_boundaries=phases,
            energy_decomposition=energy,
        )
