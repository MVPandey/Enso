import torch

from ebm.interpretability.recorder import TrajectoryRecorder
from ebm.interpretability.types import MultiChainTrajectory, StepSnapshot, Trajectory
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


def test_record_returns_trajectory():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    assert isinstance(traj, Trajectory)


def test_trajectory_step_count():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    assert len(traj.steps) == SMALL_INFERENCE.n_steps


def test_step_snapshot_shapes():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    b = 2
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch(b)
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    for snap in traj.steps:
        assert isinstance(snap, StepSnapshot)
        assert snap.z.shape == (b, SMALL_ARCH.d_latent)
        assert snap.logits.shape == (b, 9, 9, 9)
        assert snap.probs.shape == (b, 9, 9, 9)
        assert snap.board.shape == (b, 9, 9)
        assert snap.energy.shape == (b,)
        assert snap.self_consistency.shape == (b,)
        assert snap.constraint_penalty.shape == (b,)
        assert snap.grad_norm.shape == (b,)


def test_board_values_in_range():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    for snap in traj.steps:
        assert (snap.board >= 1).all()
        assert (snap.board <= 9).all()


def test_final_board_matches_last_step():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    assert torch.equal(traj.final_board, traj.steps[-1].board)


def test_attention_recording_stride():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    stride = 3
    recorder = TrajectoryRecorder(model, record_attention=True, attention_stride=stride)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    for snap in traj.steps:
        if snap.step % stride == 0:
            assert snap.encoder_attention is not None
            assert snap.decoder_attention is not None
            for attn in snap.encoder_attention.values():
                assert attn.shape[2] == 81
        else:
            assert snap.encoder_attention is None
            assert snap.decoder_attention is None


def test_no_attention_when_disabled():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    for snap in traj.steps:
        assert snap.encoder_attention is None
        assert snap.decoder_attention is None


def test_trajectory_stores_inputs():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    assert traj.puzzle.shape == puzzle.shape
    assert traj.solution.shape == solution.shape
    assert traj.mask.shape == mask.shape


def test_probs_sum_to_one():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    for snap in traj.steps:
        sums = snap.probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_energy_nonnegative():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)

    for snap in traj.steps:
        assert (snap.self_consistency >= 0).all()
        assert (snap.constraint_penalty >= 0).all()


def test_works_inside_no_grad():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch(1)
    with torch.no_grad():
        traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE)
    assert len(traj.steps) == SMALL_INFERENCE.n_steps


# --- step_stride tests ---


def test_step_stride_reduces_stored_steps():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    # 6 steps with stride=3 -> stored at steps 0, 3, 5 (last always stored)
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE, step_stride=3)

    assert len(traj.steps) < SMALL_INFERENCE.n_steps
    # Steps 0, 3 are divisible by 3; step 5 is the last step
    stored_steps = [s.step for s in traj.steps]
    assert 0 in stored_steps
    assert 3 in stored_steps
    assert 5 in stored_steps  # last step always stored


def test_step_stride_one_stores_all():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE, step_stride=1)

    assert len(traj.steps) == SMALL_INFERENCE.n_steps


def test_step_stride_final_board_matches_last_stored():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE, step_stride=2)

    assert torch.equal(traj.final_board, traj.steps[-1].board)


def test_step_stride_shapes_unchanged():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    b = 2
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch(b)
    traj = recorder.record(puzzle, mask, solution, SMALL_INFERENCE, step_stride=2)

    for snap in traj.steps:
        assert snap.z.shape == (b, SMALL_ARCH.d_latent)
        assert snap.logits.shape == (b, 9, 9, 9)
        assert snap.energy.shape == (b,)


# --- record_multi_chain tests ---


def test_multi_chain_returns_type():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    result = recorder.record_multi_chain(puzzle, mask, solution, SMALL_INFERENCE, n_chains=4)

    assert isinstance(result, MultiChainTrajectory)


def test_multi_chain_shapes():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    b = 2
    n_chains = 4
    puzzle, solution, mask = _make_batch(b)
    result = recorder.record_multi_chain(puzzle, mask, solution, SMALL_INFERENCE, n_chains=n_chains)

    assert result.chain_z.shape == (SMALL_INFERENCE.n_steps, b, n_chains, SMALL_ARCH.d_latent)
    assert result.chain_energy.shape == (SMALL_INFERENCE.n_steps, b, n_chains)
    assert result.final_boards.shape == (b, n_chains, 9, 9)
    assert result.final_z.shape == (b, n_chains, SMALL_ARCH.d_latent)
    assert result.final_energies.shape == (b, n_chains)


def test_multi_chain_metadata():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    n_chains = 4
    result = recorder.record_multi_chain(puzzle, mask, solution, SMALL_INFERENCE, n_chains=n_chains)

    assert result.n_chains == n_chains
    assert result.n_steps == SMALL_INFERENCE.n_steps
    assert result.puzzle.shape == puzzle.shape
    assert result.solution.shape == solution.shape
    assert result.mask.shape == mask.shape


def test_multi_chain_boards_in_range():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    result = recorder.record_multi_chain(puzzle, mask, solution, SMALL_INFERENCE, n_chains=4)

    assert (result.final_boards >= 1).all()
    assert (result.final_boards <= 9).all()


def test_multi_chain_energy_nonneg():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch()
    result = recorder.record_multi_chain(puzzle, mask, solution, SMALL_INFERENCE, n_chains=4)

    # Self-consistency + constraint penalty are both nonneg, so energy should be >= 0
    assert (result.chain_energy >= 0).all()


def test_multi_chain_works_inside_no_grad():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    recorder = TrajectoryRecorder(model, record_attention=False)
    puzzle, solution, mask = _make_batch(1)
    with torch.no_grad():
        result = recorder.record_multi_chain(puzzle, mask, solution, SMALL_INFERENCE, n_chains=2)
    assert result.chain_z.shape[0] == SMALL_INFERENCE.n_steps
