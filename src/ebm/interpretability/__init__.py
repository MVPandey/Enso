"""Mechanistic interpretability infrastructure for ENSO EBM."""

from ebm.interpretability.analysis import TrajectoryAnalyzer
from ebm.interpretability.attention import AttentionExtractor
from ebm.interpretability.recorder import TrajectoryRecorder
from ebm.interpretability.strategies import StrategyDetector
from ebm.interpretability.types import AnalysisResult, CellEvent, StepSnapshot, StrategyLabel, Trajectory

__all__ = [
    'AnalysisResult',
    'AttentionExtractor',
    'CellEvent',
    'StepSnapshot',
    'StrategyDetector',
    'StrategyLabel',
    'Trajectory',
    'TrajectoryAnalyzer',
    'TrajectoryRecorder',
]
