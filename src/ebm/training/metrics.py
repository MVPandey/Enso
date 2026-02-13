"""Experiment tracking integration with Weights & Biases."""

import logging
from datetime import datetime, timezone
from pathlib import Path

from torch import Tensor

from ebm.training.losses import LossOutput
from ebm.utils.config import Config

logger = logging.getLogger(__name__)

_wandb_available = False
try:
    import wandb

    _wandb_available = True
except ImportError:
    pass


def init_wandb(app_cfg: Config, run_name: str | None = None, extra_config: dict | None = None) -> bool:
    """
    Initialize a W&B run if credentials are configured.

    Args:
        app_cfg: Application config with wandb_api_key, wandb_project, wandb_entity.
        run_name: Optional name for the run.
        extra_config: Optional dict of hyperparameters to log.

    Returns:
        True if wandb was initialized, False otherwise.

    """
    if not _wandb_available or not app_cfg.wandb_api_key:
        logger.info('W&B not configured, skipping initialization')
        return False

    timestamp = datetime.now(tz=timezone.utc).strftime('%Y%m%d-%H%M%S')
    name = f'{run_name}-{timestamp}' if run_name else timestamp

    wandb.login(key=app_cfg.wandb_api_key)
    wandb.init(
        project=app_cfg.wandb_project,
        entity=app_cfg.wandb_entity,
        name=name,
        config=extra_config or {},
    )
    return True


def log_train_step(loss_out: LossOutput, lr: float, ema_momentum: float, step: int) -> None:
    """
    Log training step metrics to W&B.

    Args:
        loss_out: Loss components from compute_loss.
        lr: Current learning rate.
        ema_momentum: Current EMA momentum.
        step: Global step number.

    """
    if not _wandb_available or wandb.run is None:
        return

    wandb.log(
        {
            'train/loss_total': loss_out.total.item(),
            'train/loss_energy': loss_out.energy.item(),
            'train/loss_vicreg': loss_out.vicreg.item(),
            'train/loss_decode': loss_out.decode.item(),
            'train/lr': lr,
            'train/ema_momentum': ema_momentum,
        },
        step=step,
    )


def log_validation(
    val_energy: float,
    cell_accuracy: float,
    puzzle_accuracy: float,
    z_variance: float,
    step: int,
) -> None:
    """
    Log validation metrics to W&B.

    Args:
        val_energy: Mean validation energy.
        cell_accuracy: Fraction of empty cells correctly predicted.
        puzzle_accuracy: Fraction of puzzles fully solved.
        z_variance: Mean variance of encoder representations.
        step: Global step number.

    """
    if not _wandb_available or wandb.run is None:
        return

    wandb.log(
        {
            'val/energy': val_energy,
            'val/cell_accuracy': cell_accuracy,
            'val/puzzle_accuracy': puzzle_accuracy,
            'val/z_variance': z_variance,
        },
        step=step,
    )


def upload_checkpoint_to_wandb(checkpoint_path: Path) -> None:
    """
    Upload a checkpoint file as a W&B model artifact.

    Args:
        checkpoint_path: Path to the checkpoint .pt file.

    """
    if not _wandb_available or wandb.run is None:
        return

    artifact = wandb.Artifact(
        name=f'model-{wandb.run.id}',
        type='model',
    )
    artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(artifact)
    logger.info('Uploaded checkpoint to W&B: %s', checkpoint_path.name)


def finish_wandb() -> None:
    """Finalize the W&B run, ensuring all artifacts are uploaded."""
    if not _wandb_available or wandb.run is None:
        return
    wandb.finish()


def compute_cell_accuracy(pred: Tensor, target: Tensor, mask: Tensor) -> float:
    """
    Compute fraction of empty cells correctly predicted.

    Args:
        pred: (B, 9, 9) predicted digit grid (1-9).
        target: (B, 9, 9, 9) one-hot solution.
        mask: (B, 9, 9) binary mask (1 = given clue).

    Returns:
        Cell accuracy as a float in [0, 1].

    """
    target_digits = target.argmax(dim=-1) + 1
    empty = mask == 0
    if empty.sum() == 0:
        return 1.0
    return (pred[empty] == target_digits[empty]).float().mean().item()


def compute_puzzle_accuracy(pred: Tensor, target: Tensor) -> float:
    """
    Compute fraction of puzzles fully solved correctly.

    Args:
        pred: (B, 9, 9) predicted digit grid (1-9).
        target: (B, 9, 9, 9) one-hot solution.

    Returns:
        Puzzle accuracy as a float in [0, 1].

    """
    target_digits = target.argmax(dim=-1) + 1
    correct = (pred == target_digits).all(dim=(1, 2))
    return correct.float().mean().item()


def compute_z_variance(z: Tensor) -> float:
    """
    Compute mean per-dimension variance of representations.

    A collapse detector — if this drops near zero, all inputs map to the
    same point.

    Args:
        z: (B, D) representations.

    Returns:
        Mean variance across dimensions.

    """
    return z.var(dim=0).mean().item()
