import torch

from ebm.interpretability.landscape import EnergyEvaluator
from ebm.interpretability.types import EnergyProfile
from ebm.model.jepa import SudokuJEPA
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


def _make_batch(b: int = 2):
    puzzle = torch.zeros(b, 10, 9, 9)
    puzzle[:, 0] = 1.0
    solution = torch.zeros(b, 9, 9, 9)
    solution[:, :, :, 0] = 1.0
    mask = torch.zeros(b, 9, 9)
    return puzzle, solution, mask


def test_evaluate_returns_energy_profile():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    evaluator = EnergyEvaluator(model)
    puzzle, _solution, mask = _make_batch()
    z = torch.randn(2, SMALL_ARCH.d_latent)

    profile = evaluator.evaluate(puzzle, mask, z)

    assert isinstance(profile, EnergyProfile)


def test_evaluate_shapes():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    evaluator = EnergyEvaluator(model)
    b = 3
    puzzle, _solution, mask = _make_batch(b)
    z = torch.randn(b, SMALL_ARCH.d_latent)

    profile = evaluator.evaluate(puzzle, mask, z)

    assert profile.z.shape == (b, SMALL_ARCH.d_latent)
    assert profile.energy.shape == (b,)
    assert profile.self_consistency.shape == (b,)
    assert profile.constraint_penalty.shape == (b,)
    assert profile.logits.shape == (b, 9, 9, 9)
    assert profile.probs.shape == (b, 9, 9, 9)


def test_evaluate_energy_nonnegative():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    evaluator = EnergyEvaluator(model)
    puzzle, _solution, mask = _make_batch()
    z = torch.randn(2, SMALL_ARCH.d_latent)

    profile = evaluator.evaluate(puzzle, mask, z)

    assert (profile.self_consistency >= 0).all()
    assert (profile.constraint_penalty >= 0).all()


def test_evaluate_probs_sum_to_one():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    evaluator = EnergyEvaluator(model)
    puzzle, _solution, mask = _make_batch()
    z = torch.randn(2, SMALL_ARCH.d_latent)

    profile = evaluator.evaluate(puzzle, mask, z)
    sums = profile.probs.sum(dim=-1)

    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_evaluate_with_precomputed_z_context():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    evaluator = EnergyEvaluator(model)
    puzzle, _solution, mask = _make_batch()
    z = torch.randn(2, SMALL_ARCH.d_latent)

    z_context = evaluator.compute_z_context(puzzle)
    profile1 = evaluator.evaluate(puzzle, mask, z, z_context=z_context)
    profile2 = evaluator.evaluate(puzzle, mask, z)

    assert torch.allclose(profile1.energy, profile2.energy, atol=1e-5)


def test_interpolate_length():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    evaluator = EnergyEvaluator(model)
    puzzle, _solution, mask = _make_batch()
    z_start = torch.randn(2, SMALL_ARCH.d_latent)
    z_end = torch.randn(2, SMALL_ARCH.d_latent)

    profiles = evaluator.interpolate(puzzle, mask, z_start, z_end, n_points=10)

    assert len(profiles) == 10
    for p in profiles:
        assert isinstance(p, EnergyProfile)
        assert p.energy.shape == (2,)


def test_interpolate_endpoints():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    evaluator = EnergyEvaluator(model)
    puzzle, _solution, mask = _make_batch()
    z_start = torch.randn(2, SMALL_ARCH.d_latent)
    z_end = torch.randn(2, SMALL_ARCH.d_latent)

    profiles = evaluator.interpolate(puzzle, mask, z_start, z_end, n_points=5)

    # First profile should be at z_start
    assert torch.allclose(profiles[0].z, z_start, atol=1e-5)
    # Last profile should be at z_end
    assert torch.allclose(profiles[-1].z, z_end, atol=1e-5)


def test_compute_oracle_z_shape():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    evaluator = EnergyEvaluator(model)
    _, solution, _ = _make_batch(3)

    z_oracle = evaluator.compute_oracle_z(solution)

    assert z_oracle.shape == (3, SMALL_ARCH.d_latent)


def test_compute_oracle_z_normalized():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    evaluator = EnergyEvaluator(model)
    _, solution, _ = _make_batch()

    z_oracle = evaluator.compute_oracle_z(solution)
    norms = z_oracle.norm(dim=-1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_compute_z_context_shape():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    evaluator = EnergyEvaluator(model)
    puzzle, _, _ = _make_batch(3)

    z_context = evaluator.compute_z_context(puzzle)

    assert z_context.shape == (3, SMALL_ARCH.d_model)


def test_evaluate_different_temps():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    evaluator = EnergyEvaluator(model)
    puzzle, _solution, mask = _make_batch()
    z = torch.randn(2, SMALL_ARCH.d_latent)

    profile_hot = evaluator.evaluate(puzzle, mask, z, temp=1.0)
    profile_cold = evaluator.evaluate(puzzle, mask, z, temp=0.0)

    # Energy should differ due to different constraint penalty weighting
    # At temp=1.0: weight = 1+2*(1-1) = 1.0
    # At temp=0.0: weight = 1+2*(1-0) = 3.0
    # Self-consistency stays the same
    assert torch.allclose(profile_hot.self_consistency, profile_cold.self_consistency, atol=1e-5)
    # Unless constraint_penalty is exactly 0, energies should differ
    if (profile_hot.constraint_penalty > 1e-6).any():
        assert not torch.allclose(profile_hot.energy, profile_cold.energy)
