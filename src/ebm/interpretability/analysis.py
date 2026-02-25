"""Trajectory analysis pipeline correlating strategies with attention patterns."""

from __future__ import annotations

from collections import defaultdict

from torch import Tensor

from ebm.interpretability.strategies import StrategyDetector
from ebm.interpretability.types import AnalysisResult, CellEvent, Trajectory


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
