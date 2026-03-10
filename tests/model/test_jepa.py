import torch

from ebm.model.jepa import InferenceConfig, JEPAOutput, SudokuJEPA
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
SMALL_INFERENCE = InferenceConfig(n_steps=3, n_chains=2)
SMALL_SVGD_INFERENCE = InferenceConfig(method='svgd', n_steps=3, n_chains=4, lr=0.1)


def _make_batch(b: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    puzzle = torch.zeros(b, 10, 9, 9)
    puzzle[:, 0] = 1.0
    solution = torch.zeros(b, 9, 9, 9)
    solution[:, :, :, 0] = 1.0
    mask = torch.zeros(b, 9, 9)
    return puzzle, solution, mask


def test_forward_returns_jepa_output():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    puzzle, solution, mask = _make_batch()
    out = model(puzzle, solution, mask)
    assert isinstance(out, JEPAOutput)


def test_forward_shapes():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    puzzle, solution, mask = _make_batch(4)
    out = model(puzzle, solution, mask)
    assert out.energy.shape == (4,)
    assert out.z_pred.shape == (4, 32)
    assert out.z_target.shape == (4, 32)
    assert out.decode_logits.shape == (4, 9, 9, 9)


def test_energy_nonnegative():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    puzzle, solution, mask = _make_batch()
    out = model(puzzle, solution, mask)
    assert (out.energy >= 0).all()


def test_z_target_no_grad():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    puzzle, solution, mask = _make_batch()
    out = model(puzzle, solution, mask)
    assert not out.z_target.requires_grad


def test_target_encoder_frozen():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    for p in model.target_encoder.parameters():
        assert not p.requires_grad


def test_ema_update():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    target_before = {k: v.clone() for k, v in model.target_encoder.state_dict().items()}

    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    model.update_target_encoder(momentum=0.5)

    changed = False
    for k, v in model.target_encoder.state_dict().items():
        if not torch.allclose(v, target_before[k]):
            changed = True
            break
    assert changed


def test_solve_shape():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    puzzle, _, mask = _make_batch(2)
    solution = model.solve(puzzle, mask, SMALL_INFERENCE)
    assert solution.shape == (2, 9, 9)
    assert solution.dtype == torch.int64


def test_solve_values_in_range():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    puzzle, _, mask = _make_batch(2)
    solution = model.solve(puzzle, mask, SMALL_INFERENCE)
    assert (solution >= 1).all()
    assert (solution <= 9).all()


def test_solve_works_inside_no_grad():
    """Langevin dynamics must work even inside a torch.no_grad() context."""
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    puzzle, _, mask = _make_batch(1)
    with torch.no_grad():
        solution = model.solve(puzzle, mask, SMALL_INFERENCE)
    assert solution.shape == (1, 9, 9)


def test_inference_config_from_training_config():
    cfg = InferenceConfig.from_training_config(SMALL_TRAIN)
    assert cfg.n_steps == SMALL_TRAIN.langevin_steps
    assert cfg.n_chains == SMALL_TRAIN.n_chains


def test_backward_through_forward():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    puzzle, solution, mask = _make_batch()
    out = model(puzzle, solution, mask)
    loss = out.energy.mean() + out.decode_logits.sum() * 0.001
    loss.backward()
    for name, p in model.context_encoder.named_parameters():
        assert p.grad is not None, f'No gradient for context_encoder.{name}'
    for name, p in model.predictor.named_parameters():
        assert p.grad is not None, f'No gradient for predictor.{name}'
    for name, p in model.z_encoder.named_parameters():
        assert p.grad is not None, f'No gradient for z_encoder.{name}'


def test_solve_svgd_shape():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    puzzle, _, mask = _make_batch(2)
    solution = model.solve(puzzle, mask, SMALL_SVGD_INFERENCE)
    assert solution.shape == (2, 9, 9)
    assert solution.dtype == torch.int64


def test_solve_svgd_values_in_range():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    puzzle, _, mask = _make_batch(2)
    solution = model.solve(puzzle, mask, SMALL_SVGD_INFERENCE)
    assert (solution >= 1).all()
    assert (solution <= 9).all()


def test_solve_svgd_works_inside_no_grad():
    """SVGD must work even inside a torch.no_grad() context."""
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    puzzle, _, mask = _make_batch(1)
    with torch.no_grad():
        solution = model.solve(puzzle, mask, SMALL_SVGD_INFERENCE)
    assert solution.shape == (1, 9, 9)


def test_inference_config_default_is_langevin():
    cfg = InferenceConfig()
    assert cfg.method == 'langevin'


def test_inference_config_svgd_from_training_config():
    train_cfg = TrainingConfig(inference_method='svgd', kernel_bandwidth=0.5)
    cfg = InferenceConfig.from_training_config(train_cfg)
    assert cfg.method == 'svgd'
    assert cfg.kernel_bandwidth == 0.5
