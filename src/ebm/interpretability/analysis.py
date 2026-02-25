"""Trajectory analysis pipeline correlating strategies with attention patterns."""

from __future__ import annotations

from collections import defaultdict

from torch import Tensor

from ebm.interpretability.attention_analysis import AttentionAnalyzer
from ebm.interpretability.metrics import MetricsComputer
from ebm.interpretability.strategies import StrategyDetector
from ebm.interpretability.types import (
    AnalysisResult,
    CellEvent,
    HeadProfile,
    Trajectory,
    TrajectoryMetrics,
)


class TrajectoryAnalyzer:
    """
    Analyze Langevin trajectories for strategy-attention correlations.

    Detects cell-fill events across the trajectory by comparing consecutive
    board snapshots, classifies each event via the strategy detector, and
    collects the corresponding attention maps.
    """

    def __init__(self, strategy_detector: StrategyDetector | None = None) -> None:
        """Initialize with an optional custom strategy detector."""
        self._detector = strategy_detector or StrategyDetector()
        self._attention_analyzer = AttentionAnalyzer()
        self._metrics = MetricsComputer()

    def analyze_trajectory(self, trajectory: Trajectory, batch_idx: int = 0) -> AnalysisResult:
        """
        Analyze a single trajectory for one sample in the batch.

        Args:
            trajectory: Full trajectory from TrajectoryRecorder.
            batch_idx: Index into the batch dimension to analyze.

        Returns:
            AnalysisResult with classified events and attention groupings.

        """
        events: list[CellEvent] = []
        mask = trajectory.mask[batch_idx]

        for i in range(1, len(trajectory.steps)):
            prev_step = trajectory.steps[i - 1]
            curr_step = trajectory.steps[i]

            board_before = prev_step.board[batch_idx]
            board_after = curr_step.board[batch_idx]
            probs = curr_step.probs[batch_idx]

            step_events = self._detector.classify(board_before, board_after, mask, probs=probs)
            for event in step_events:
                event.step = curr_step.step
            events.extend(step_events)

        # Aggregate results
        strategy_counts: dict[str, int] = defaultdict(int)
        strategy_by_step: dict[int, list[CellEvent]] = defaultdict(list)
        attention_by_strategy: dict[str, list[Tensor]] = defaultdict(list)

        for event in events:
            label = event.strategy.value if event.strategy else 'unknown'
            strategy_counts[label] += 1
            strategy_by_step[event.step].append(event)

            # Collect attention maps from the step of this event
            attn = self._get_attention_for_step(trajectory, event.step)
            if attn is not None:
                attention_by_strategy[label].append(attn)

        return AnalysisResult(
            events=events,
            strategy_counts=dict(strategy_counts),
            strategy_by_step=dict(strategy_by_step),
            attention_by_strategy=dict(attention_by_strategy),
        )

    def full_analysis(
        self,
        trajectory: Trajectory,
        solution: Tensor,
        batch_idx: int = 0,
    ) -> tuple[AnalysisResult, TrajectoryMetrics, list[HeadProfile]]:
        """
        Run complete analysis: strategies + metrics + attention head profiling.

        Args:
            trajectory: Full trajectory from TrajectoryRecorder.
            solution: (B, 9, 9, 9) one-hot ground truth.
            batch_idx: Index into the batch dimension to analyze.

        Returns:
            Tuple of (AnalysisResult, TrajectoryMetrics, list[HeadProfile]).

        """
        result = self.analyze_trajectory(trajectory, batch_idx)
        metrics = self._metrics.compute_all(trajectory, result.events, solution, batch_idx)

        # Collect all attention maps from captured steps
        all_attention: dict[str, Tensor] = {}
        for step in trajectory.steps:
            if step.encoder_attention:
                all_attention.update(step.encoder_attention)
            if step.decoder_attention:
                all_attention.update(step.decoder_attention)

        profiles = self._attention_analyzer.compute_head_profiles(all_attention) if all_attention else []

        return result, metrics, profiles

    @staticmethod
    def _get_attention_for_step(trajectory: Trajectory, step: int) -> Tensor | None:
        """
        Return attention tensor for the given step, if captured.

        Prefers encoder attention; falls back to decoder attention.
        """
        for snap in trajectory.steps:
            if snap.step != step:
                continue
            if snap.encoder_attention:
                first_key = next(iter(snap.encoder_attention))
                return snap.encoder_attention[first_key]
            if snap.decoder_attention:
                first_key = next(iter(snap.decoder_attention))
                return snap.decoder_attention[first_key]
        return None
