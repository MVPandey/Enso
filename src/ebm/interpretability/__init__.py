"""Mechanistic interpretability infrastructure for ENSO EBM."""

from ebm.interpretability.ablation import HeadAblator
from ebm.interpretability.analysis import TrajectoryAnalyzer
from ebm.interpretability.attention import AttentionExtractor
from ebm.interpretability.attention_analysis import AttentionAnalyzer
from ebm.interpretability.difficulty import DifficultyBucket, classify_difficulty, count_givens, stratify_dataframe
from ebm.interpretability.landscape import EnergyEvaluator
from ebm.interpretability.metrics import MetricsComputer
from ebm.interpretability.recorder import TrajectoryRecorder
from ebm.interpretability.strategies import STRATEGY_DIFFICULTY, StrategyDetector
from ebm.interpretability.types import (
    AblationResult,
    AnalysisResult,
    CellEvent,
    EnergyProfile,
    HeadProfile,
    LockInEvent,
    MultiChainTrajectory,
    StepSnapshot,
    StrategyLabel,
    Trajectory,
    TrajectoryMetrics,
)

__all__ = [
    'STRATEGY_DIFFICULTY',
    'AblationResult',
    'AnalysisResult',
    'AttentionAnalyzer',
    'AttentionExtractor',
    'CellEvent',
    'DifficultyBucket',
    'EnergyEvaluator',
    'EnergyProfile',
    'HeadAblator',
    'HeadProfile',
    'LockInEvent',
    'MetricsComputer',
    'MultiChainTrajectory',
    'StepSnapshot',
    'StrategyDetector',
    'StrategyLabel',
    'Trajectory',
    'TrajectoryAnalyzer',
    'TrajectoryMetrics',
    'TrajectoryRecorder',
    'classify_difficulty',
    'count_givens',
    'stratify_dataframe',
]
