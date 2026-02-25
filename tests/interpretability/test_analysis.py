import torch

from ebm.interpretability.analysis import TrajectoryAnalyzer
from ebm.interpretability.recorder import TrajectoryRecorder
from ebm.interpretability.strategies import StrategyDetector
from ebm.interpretability.types import AnalysisResult, HeadProfile, StepSnapshot, Trajectory, TrajectoryMetrics
from ebm.model.jepa import InferenceConfig, SudokuJEPA
from ebm.utils.config import ArchitectureConfig, TrainingConfig

SMALL_ARCH = ArchitectureConfig(
    d_model=32,
    n_layers=1,
    n_heads=4,
    d_ffn=64,
    d_latent=16,
    predictor_hidden=32,
    decoder_layers=1,
    decoder_heads=2,
    decoder_d_cell=16,
)
SMALL_TRAIN = TrainingConfig(langevin_steps=3, n_chains=2)
SMALL_INFERENCE = InferenceConfig(n_steps=6, n_chains=1)


def _make_batch(b: int = 2):
    puzzle = torch.zeros(b, 10, 9, 9)
    puzzle[:, 0] = 1.0
    solution = torch.zeros(b, 9, 9, 9)
    solution[:, :, :, 0] = 1.0
    mask = torch.zeros(b, 9, 9)
    return puzzle, solution, mask


def test_analyze_returns_analysis_result():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    analyzer = TrajectoryAnalyzer()
    result = analyzer.analyze_trajectory(traj, batch_idx=0)

    assert isinstance(result, AnalysisResult)


def test_events_have_valid_fields():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    analyzer = TrajectoryAnalyzer()
    result = analyzer.analyze_trajectory(traj, batch_idx=0)

    for event in result.events:
        assert 0 <= event.row < 9
        assert 0 <= event.col < 9
        assert 1 <= event.digit <= 9
        assert event.strategy is not None
        assert event.step >= 0


def test_strategy_counts_sum_to_events():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    analyzer = TrajectoryAnalyzer()
    result = analyzer.analyze_trajectory(traj, batch_idx=0)

    total_counts = sum(result.strategy_counts.values())
    assert total_counts == len(result.events)


def test_strategy_by_step_groups_correctly():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    analyzer = TrajectoryAnalyzer()
    result = analyzer.analyze_trajectory(traj, batch_idx=0)

    total_by_step = sum(len(evts) for evts in result.strategy_by_step.values())
    assert total_by_step == len(result.events)

    for step, evts in result.strategy_by_step.items():
        for e in evts:
            assert e.step == step


def test_attention_by_strategy_with_attention():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=True, attention_stride=1)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    analyzer = TrajectoryAnalyzer()
    result = analyzer.analyze_trajectory(traj, batch_idx=0)

    if result.events:
        assert len(result.attention_by_strategy) > 0


def test_custom_strategy_detector():
    detector = StrategyDetector()
    analyzer = TrajectoryAnalyzer(strategy_detector=detector)
    assert analyzer._detector is detector


def test_analyze_with_synthetic_trajectory():
    """Test analysis on a hand-crafted trajectory with known board transitions."""
    b = 1
    puzzle = torch.zeros(b, 10, 9, 9)
    puzzle[:, 0] = 1.0
    solution = torch.zeros(b, 9, 9, 9)
    solution[:, :, :, 0] = 1.0
    mask = torch.zeros(b, 9, 9)

    d_latent = 16

    # Step 0: empty board (all argmax to same digit)
    logits_0 = torch.zeros(b, 9, 9, 9)
    logits_0[:, :, :, 0] = 10.0  # all cells predict digit 1
    probs_0 = torch.softmax(logits_0, dim=-1)
    board_0 = logits_0.argmax(dim=-1) + 1  # all 1s

    # Step 1: some cells change
    logits_1 = logits_0.clone()
    logits_1[0, 0, 1, 1] = 20.0  # cell (0,1) now predicts digit 2
    logits_1[0, 0, 1, 0] = -10.0
    probs_1 = torch.softmax(logits_1, dim=-1)
    board_1 = logits_1.argmax(dim=-1) + 1

    snap_0 = StepSnapshot(
        step=0,
        z=torch.randn(b, d_latent),
        logits=logits_0,
        probs=probs_0,
        board=board_0,
        energy=torch.tensor([1.0]),
        self_consistency=torch.tensor([0.5]),
        constraint_penalty=torch.tensor([0.5]),
        grad_norm=torch.tensor([0.1]),
    )
    snap_1 = StepSnapshot(
        step=1,
        z=torch.randn(b, d_latent),
        logits=logits_1,
        probs=probs_1,
        board=board_1,
        energy=torch.tensor([0.8]),
        self_consistency=torch.tensor([0.4]),
        constraint_penalty=torch.tensor([0.4]),
        grad_norm=torch.tensor([0.08]),
    )

    traj = Trajectory(puzzle=puzzle, solution=solution, mask=mask, steps=[snap_0, snap_1], final_board=board_1)

    analyzer = TrajectoryAnalyzer()
    result = analyzer.analyze_trajectory(traj, batch_idx=0)

    # Cell (0,1) changed from 1 to 2 — should be detected
    changed = [e for e in result.events if e.row == 0 and e.col == 1]
    assert len(changed) == 1
    assert changed[0].digit == 2


def test_empty_trajectory_no_crash():
    """Trajectory with only one step should produce no events."""
    b = 1
    puzzle = torch.zeros(b, 10, 9, 9)
    solution = torch.zeros(b, 9, 9, 9)
    mask = torch.zeros(b, 9, 9)
    board = torch.ones(b, 9, 9, dtype=torch.long)

    snap = StepSnapshot(
        step=0,
        z=torch.randn(b, 16),
        logits=torch.zeros(b, 9, 9, 9),
        probs=torch.ones(b, 9, 9, 9) / 9,
        board=board,
        energy=torch.tensor([0.0]),
        self_consistency=torch.tensor([0.0]),
        constraint_penalty=torch.tensor([0.0]),
        grad_norm=torch.tensor([0.0]),
    )

    traj = Trajectory(puzzle=puzzle, solution=solution, mask=mask, steps=[snap], final_board=board)

    analyzer = TrajectoryAnalyzer()
    result = analyzer.analyze_trajectory(traj, batch_idx=0)

    assert len(result.events) == 0


def test_full_analysis_returns_three_outputs():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch(b=1)
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    analyzer = TrajectoryAnalyzer()
    result, metrics, profiles = analyzer.full_analysis(traj, solution, batch_idx=0)

    assert isinstance(result, AnalysisResult)
    assert isinstance(metrics, TrajectoryMetrics)
    assert isinstance(profiles, list)


def test_full_analysis_with_attention():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=True, attention_stride=1)
    puzzle, solution, mask = _make_batch(b=1)
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    analyzer = TrajectoryAnalyzer()
    result, metrics, profiles = analyzer.full_analysis(traj, solution, batch_idx=0)

    assert isinstance(result, AnalysisResult)
    assert isinstance(metrics, TrajectoryMetrics)
    # With attention captured, profiles should be non-empty
    assert isinstance(profiles, list)
    if profiles:
        assert all(isinstance(p, HeadProfile) for p in profiles)


def test_full_analysis_metrics_coverage():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch(b=1)
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    analyzer = TrajectoryAnalyzer()
    _, metrics, _ = analyzer.full_analysis(traj, solution, batch_idx=0)

    assert 0 <= metrics.strategy_coverage <= 1.0
    assert isinstance(metrics.lock_in_events, list)
    assert isinstance(metrics.energy_decomposition, list)
    assert len(metrics.energy_decomposition) == len(traj.steps)
