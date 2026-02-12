"""Learning rate and EMA momentum schedules."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from ebm.utils.config import TrainingConfig


def create_lr_scheduler(optimizer: Optimizer, cfg: TrainingConfig, total_steps: int) -> LRScheduler:
    """
    Create a linear warmup + cosine decay LR scheduler.

    Args:
        optimizer: The optimizer to schedule.
        cfg: Training config with warmup_steps.
        total_steps: Total number of training steps.

    Returns:
        SequentialLR combining warmup and cosine decay.

    """
    warmup_steps = min(cfg.warmup_steps, total_steps // 5)
    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    decay = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])


def get_ema_momentum(step: int, total_steps: int, cfg: TrainingConfig) -> float:
    """
    Linearly interpolate EMA momentum from start to end over training.

    Args:
        step: Current training step.
        total_steps: Total number of training steps.
        cfg: Training config with ema_momentum_start and ema_momentum_end.

    Returns:
        Current EMA momentum value.

    """
    progress = min(step / max(total_steps, 1), 1.0)
    return cfg.ema_momentum_start + progress * (cfg.ema_momentum_end - cfg.ema_momentum_start)
