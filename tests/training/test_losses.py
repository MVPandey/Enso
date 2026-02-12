import torch

from ebm.model.jepa import JEPAOutput
from ebm.training.losses import LossOutput, compute_loss, vicreg_loss
from ebm.utils.config import TrainingConfig


def _make_jepa_output(b: int = 4, d: int = 32) -> JEPAOutput:
    z_context = torch.randn(b, d, requires_grad=True)
    z_pred = torch.randn(b, d, requires_grad=True)
    z_target = torch.randn(b, d)
    energy = ((z_pred - z_target) ** 2).sum(dim=-1)
    logits = torch.randn(b, 9, 9, 9, requires_grad=True)
    return JEPAOutput(energy=energy, z_context=z_context, z_pred=z_pred, z_target=z_target, decode_logits=logits)


def _make_solution(b: int = 4) -> torch.Tensor:
    solution = torch.zeros(b, 9, 9, 9)
    solution[:, :, :, 0] = 1.0
    return solution


def _make_mask(b: int = 4) -> torch.Tensor:
    return torch.zeros(b, 9, 9)


def test_vicreg_loss_scalar():
    z = torch.randn(16, 32)
    loss = vicreg_loss(z)
    assert loss.shape == ()


def test_vicreg_collapsed_has_high_loss():
    z_collapsed = torch.ones(16, 32)
    z_normal = torch.randn(16, 32)
    assert vicreg_loss(z_collapsed) > vicreg_loss(z_normal)


def test_compute_loss_returns_loss_output():
    out = _make_jepa_output()
    result = compute_loss(out, _make_solution(), _make_mask(), TrainingConfig())
    assert isinstance(result, LossOutput)
    assert result.total.shape == ()


def test_compute_loss_components_positive():
    out = _make_jepa_output(b=8)
    result = compute_loss(out, _make_solution(b=8), _make_mask(b=8), TrainingConfig())
    assert result.energy > 0
    assert result.decode > 0


def test_compute_loss_gradient_flows():
    out = _make_jepa_output()
    result = compute_loss(out, _make_solution(), _make_mask(), TrainingConfig())
    result.total.backward()
    assert out.z_pred.grad is not None
    assert out.decode_logits.grad is not None
