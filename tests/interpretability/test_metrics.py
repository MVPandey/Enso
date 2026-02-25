import torch

from ebm.interpretability.metrics import MetricsComputer
from ebm.interpretability.types import (
    CellEvent,
    LockInEvent,
    StepSnapshot,
    StrategyLabel,
    Trajectory,
    TrajectoryMetrics,
)


def _make_trajectory(n_steps=10, batch_size=1, d_latent=16):
    """Create a synthetic trajectory for testing."""
    puzzle = torch.zeros(batch_size, 10, 9, 9)
    puzzle[:, 0] = 1.0
    solution = torch.zeros(batch_size, 9, 9, 9)
    solution[:, :, :, 0] = 1.0  # all cells answer digit 1
    mask = torch.zeros(batch_size, 9, 9)

    steps = []
    for s in range(n_steps):
        # Probability for digit 1 increases over steps
        prob = min(0.1 + s * 0.1, 1.0)
        probs = torch.ones(batch_size, 9, 9, 9) * ((1 - prob) / 8)
        probs[:, :, :, 0] = prob

        board = torch.ones(batch_size, 9, 9, dtype=torch.long)  # all 1s
        energy = torch.tensor([1.0 - s * 0.1])
        sc = torch.tensor([0.5 - s * 0.05])
        cp = torch.tensor([0.5 - s * 0.05])

        steps.append(
            StepSnapshot(
                step=s,
                z=torch.randn(batch_size, d_latent),
                logits=torch.zeros(batch_size, 9, 9, 9),
                probs=probs,
                board=board,
                energy=energy,
                self_consistency=sc,
                constraint_penalty=cp,
                grad_norm=torch.tensor([0.1]),
            )
        )

    return Trajectory(
        puzzle=puzzle,
        solution=solution,
        mask=mask,
        steps=steps,
        final_board=steps[-1].board,
    )


def _make_events():
    """Create sample cell events."""
    return [
        CellEvent(step=0, row=0, col=0, digit=1, strategy=StrategyLabel.NAKED_SINGLE, confidence=0.9),
        CellEvent(step=1, row=0, col=1, digit=2, strategy=StrategyLabel.NAKED_SINGLE, confidence=0.8),
        CellEvent(step=3, row=1, col=0, digit=3, strategy=StrategyLabel.HIDDEN_SINGLE, confidence=0.7),
        CellEvent(step=5, row=2, col=0, digit=4, strategy=StrategyLabel.POINTING_PAIR, confidence=0.6),
        CellEvent(step=8, row=3, col=0, digit=5, strategy=StrategyLabel.X_WING, confidence=0.5),
    ]


def test_compute_lock_in():
    traj = _make_trajectory(n_steps=12)
    solution = traj.solution
    mc = MetricsComputer(lock_in_threshold=0.9)
    events = mc.compute_lock_in(traj, solution, batch_idx=0)

    # All cells are non-clue and the correct digit is 1 (index 0)
    # Prob reaches 0.9 at step 8 (0.1 + 8*0.1 = 0.9)
    # and stays >= 0.855 (0.9 * 0.95) for remaining steps
    assert len(events) > 0
    for e in events:
        assert isinstance(e, LockInEvent)
        assert e.digit == 1
        assert e.lock_in_step >= 0


def test_compute_lock_in_with_clues():
    traj = _make_trajectory(n_steps=12)
    traj.mask[0, 0, 0] = 1  # Mark (0,0) as clue
    mc = MetricsComputer(lock_in_threshold=0.9)
    events = mc.compute_lock_in(traj, traj.solution, batch_idx=0)

    # (0,0) should be skipped
    for e in events:
        assert not (e.row == 0 and e.col == 0)


def test_compute_lock_in_no_lock():
    """Low threshold probs never lock in."""
    traj = _make_trajectory(n_steps=5)
    mc = MetricsComputer(lock_in_threshold=0.99)
    events = mc.compute_lock_in(traj, traj.solution, batch_idx=0)

    # With only 5 steps, prob maxes at 0.5 — no lock-in at 0.99 threshold
    assert len(events) == 0


def test_strategy_step_correlation():
    mc = MetricsComputer()
    events = _make_events()
    rho = mc.compute_strategy_step_correlation(events)

    # Difficulty increases with step — should be positive correlation
    assert rho > 0


def test_strategy_step_correlation_insufficient_data():
    mc = MetricsComputer()
    events = [CellEvent(step=0, row=0, col=0, digit=1, strategy=StrategyLabel.NAKED_SINGLE)]
    rho = mc.compute_strategy_step_correlation(events)
    assert rho == 0.0


def test_strategy_step_correlation_empty():
    mc = MetricsComputer()
    rho = mc.compute_strategy_step_correlation([])
    assert rho == 0.0


def test_detect_phase_transitions():
    mc = MetricsComputer()
    # Events with clear phase transition at step 5
    events = [
        CellEvent(step=0, row=0, col=0, digit=1, strategy=StrategyLabel.NAKED_SINGLE),
        CellEvent(step=1, row=0, col=1, digit=2, strategy=StrategyLabel.NAKED_SINGLE),
        CellEvent(step=2, row=0, col=2, digit=3, strategy=StrategyLabel.NAKED_SINGLE),
        CellEvent(step=5, row=1, col=0, digit=4, strategy=StrategyLabel.HIDDEN_SINGLE),
        CellEvent(step=6, row=1, col=1, digit=5, strategy=StrategyLabel.HIDDEN_SINGLE),
        CellEvent(step=7, row=1, col=2, digit=6, strategy=StrategyLabel.HIDDEN_SINGLE),
    ]
    boundaries = mc.detect_phase_transitions(events, window=5)
    assert len(boundaries) > 0


def test_detect_phase_transitions_empty():
    mc = MetricsComputer()
    boundaries = mc.detect_phase_transitions([])
    assert boundaries == []


def test_detect_phase_transitions_single_phase():
    mc = MetricsComputer()
    events = [CellEvent(step=i, row=0, col=i, digit=i + 1, strategy=StrategyLabel.NAKED_SINGLE) for i in range(5)]
    boundaries = mc.detect_phase_transitions(events, window=10)
    assert boundaries == []


def test_compute_strategy_coverage():
    mc = MetricsComputer()
    events = [
        CellEvent(step=0, row=0, col=0, digit=1, strategy=StrategyLabel.NAKED_SINGLE),
        CellEvent(step=1, row=0, col=1, digit=2, strategy=StrategyLabel.UNKNOWN),
        CellEvent(step=2, row=0, col=2, digit=3, strategy=StrategyLabel.HIDDEN_SINGLE),
        CellEvent(step=3, row=0, col=3, digit=4, strategy=StrategyLabel.UNKNOWN),
    ]
    coverage = mc.compute_strategy_coverage(events)
    assert abs(coverage - 0.5) < 1e-6


def test_compute_strategy_coverage_all_known():
    mc = MetricsComputer()
    events = [CellEvent(step=0, row=0, col=0, digit=1, strategy=StrategyLabel.NAKED_SINGLE)]
    coverage = mc.compute_strategy_coverage(events)
    assert coverage == 1.0


def test_compute_strategy_coverage_empty():
    mc = MetricsComputer()
    coverage = mc.compute_strategy_coverage([])
    assert coverage == 1.0


def test_compute_energy_decomposition():
    traj = _make_trajectory(n_steps=5)
    mc = MetricsComputer()
    decomp = mc.compute_energy_decomposition(traj, batch_idx=0)

    assert len(decomp) == 5
    for step_num, sc, cp in decomp:
        assert isinstance(step_num, int)
        assert isinstance(sc, float)
        assert isinstance(cp, float)


def test_compute_all():
    traj = _make_trajectory(n_steps=12)
    events = _make_events()
    mc = MetricsComputer(lock_in_threshold=0.9)
    metrics = mc.compute_all(traj, events, traj.solution, batch_idx=0)

    assert isinstance(metrics, TrajectoryMetrics)
    assert 0 <= metrics.strategy_coverage <= 1.0
    assert isinstance(metrics.lock_in_events, list)
    assert isinstance(metrics.step_strategy_correlation, float)
    assert isinstance(metrics.phase_boundaries, list)
    assert isinstance(metrics.energy_decomposition, list)
