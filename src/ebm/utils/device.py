"""GPU device detection and training parameter auto-scaling."""

import logging
import math

import torch

from ebm.utils.config import TrainingConfig

logger = logging.getLogger(__name__)

_BASE_BATCH_SIZE = 512
_BASE_LR = 3e-4

_VRAM_BATCH_TIERS: list[tuple[float, int]] = [
    (6.0, 256),
    (12.0, 512),
    (24.0, 1024),
    (40.0, 2048),
    (80.0, 4096),
]


def get_gpu_vram_gb() -> float | None:
    """
    Return total VRAM of the current CUDA device in GB.

    Returns:
        VRAM in gigabytes, or None if no CUDA device is available.

    """
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def resolve_batch_size(vram_gb: float) -> int:
    """
    Select the largest batch size appropriate for the given VRAM.

    Uses conservative tiers tuned for the 7.4M-parameter SudokuJEPA model.

    Args:
        vram_gb: Available GPU memory in gigabytes.

    Returns:
        Optimal batch size (power of 2).

    """
    batch_size = _VRAM_BATCH_TIERS[0][1]
    for threshold, bs in _VRAM_BATCH_TIERS:
        if vram_gb >= threshold:
            batch_size = bs
    return batch_size


def scale_lr(base_lr: float, base_batch: int, actual_batch: int) -> float:
    """
    Scale learning rate using the square-root rule for batch size changes.

    The sqrt rule (Hoffer et al., 2017) is more conservative than linear
    scaling and works well with AdamW.

    Args:
        base_lr: Learning rate tuned for base_batch.
        base_batch: Reference batch size the LR was tuned for.
        actual_batch: Batch size being used.

    Returns:
        Scaled learning rate.

    """
    return base_lr * math.sqrt(actual_batch / base_batch)


def auto_scale_config(cfg: TrainingConfig, batch_size_override: int | None = None) -> TrainingConfig:
    """
    Auto-detect GPU VRAM and scale batch size + learning rate.

    If batch_size_override is provided, uses that instead of auto-detecting
    from VRAM. LR is always scaled relative to the base pair (lr=3e-4,
    batch_size=512) using the square-root rule.

    Args:
        cfg: Training configuration with base hyperparameters.
        batch_size_override: Explicit batch size (skips VRAM auto-detection).

    Returns:
        New TrainingConfig with scaled batch_size and lr.

    """
    if batch_size_override:
        batch_size = batch_size_override
        logger.info('Using explicit batch size: %d', batch_size)
    else:
        vram_gb = get_gpu_vram_gb()
        if vram_gb is None:
            logger.info('No GPU detected, using defaults (batch_size=%d, lr=%g)', cfg.batch_size, cfg.lr)
            return cfg

        gpu_name = torch.cuda.get_device_name(0)
        batch_size = resolve_batch_size(vram_gb)
        logger.info('Detected %s (%.1f GB VRAM) → batch_size=%d', gpu_name, vram_gb, batch_size)

    scaled_lr = scale_lr(_BASE_LR, _BASE_BATCH_SIZE, batch_size)
    logger.info('LR scaled: %g → %g (sqrt rule, base_batch=%d)', _BASE_LR, scaled_lr, _BASE_BATCH_SIZE)

    return cfg.model_copy(update={'batch_size': batch_size, 'lr': scaled_lr})
