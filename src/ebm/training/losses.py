"""Loss functions for Sudoku JEPA training."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ebm.model.constraints import constraint_penalty
from ebm.model.jepa import JEPAOutput
from ebm.utils.config import TrainingConfig


@dataclass
class LossOutput:
    """Container for individual loss components and total."""

    total: Tensor
    energy: Tensor
    vicreg: Tensor
    decode: Tensor
    constraint: Tensor


def vicreg_loss(z: Tensor, var_weight: float = 25.0, cov_weight: float = 1.0) -> Tensor:
    """
    Variance-Invariance-Covariance regularization to prevent representation collapse.

    Args:
        z: (B, D) representations to regularize.
        var_weight: Weight for variance term (push variance toward 1).
        cov_weight: Weight for covariance term (decorrelate dimensions).

    Returns:
        Scalar VICReg loss.

    """
    z = z - z.mean(dim=0)
    std = z.std(dim=0)
    var_loss = F.relu(1.0 - std).mean()

    num_samples, num_dims = z.shape
    cov = (z.T @ z) / (num_samples - 1)
    off_diag = cov - torch.diag(cov.diag())
    cov_loss = (off_diag**2).sum() / num_dims

    return var_weight * var_loss + cov_weight * cov_loss


def compute_loss(out: JEPAOutput, solution: Tensor, mask: Tensor, cfg: TrainingConfig) -> LossOutput:
    """
    Compute combined training loss.

    L_total = L_energy + L_vicreg + decode_loss_weight * L_decode

    Decode loss is computed only on empty cells (mask == 0) so the model
    learns to predict the solution rather than just memorizing given clues.

    Args:
        out: Forward pass output from SudokuJEPA.
        solution: (B, 9, 9, 9) one-hot encoded solution.
        mask: (B, 9, 9) binary mask, 1 = given clue.
        cfg: Training config with loss weights.

    Returns:
        LossOutput with total and component losses.

    """
    energy_loss = out.energy.mean()

    vreg = vicreg_loss(
        out.z_context,
        var_weight=cfg.vicreg_var_weight,
        cov_weight=cfg.vicreg_cov_weight,
    )

    targets = solution.argmax(dim=-1)
    empty = mask == 0
    decode_loss = F.cross_entropy(out.decode_logits[empty], targets[empty])

    decode_probs = torch.softmax(out.decode_logits, dim=-1)
    constraint_loss = constraint_penalty(decode_probs).mean()

    total = energy_loss + vreg + cfg.decode_loss_weight * decode_loss + cfg.constraint_loss_weight * constraint_loss

    return LossOutput(total=total, energy=energy_loss, vicreg=vreg, decode=decode_loss, constraint=constraint_loss)
