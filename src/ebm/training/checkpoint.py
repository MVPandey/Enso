"""Checkpoint save/load with best-K tracking."""

import heapq
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _CheckpointEntry:
    """Tracks a checkpoint by its metric value and path."""

    metric: float
    path: Path = field(compare=False)


@dataclass
class _CheckpointData:
    """Data to persist in a checkpoint file."""

    model: nn.Module
    optimizer: Optimizer
    epoch: int
    step: int
    val_energy: float
    cell_accuracy: float


class CheckpointManager:
    """
    Manages saving/loading checkpoints, keeping only the best K by cell accuracy.

    Higher cell accuracy is better. When a new checkpoint is saved and the limit
    is exceeded, the worst (lowest accuracy) checkpoint is deleted.
    """

    def __init__(self, checkpoint_dir: Path, keep_top_k: int = 3) -> None:
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints in.
            keep_top_k: Number of best checkpoints to retain.

        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_top_k = keep_top_k
        self._heap: list[_CheckpointEntry] = []

    def save(self, data: _CheckpointData) -> Path | None:
        """
        Save a checkpoint if it's among the best K.

        Args:
            data: Checkpoint data to save.

        Returns:
            Path to saved checkpoint, or None if not in top K.

        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f'checkpoint_epoch{data.epoch:03d}_acc{data.cell_accuracy:.4f}.pt'
        entry = _CheckpointEntry(metric=data.cell_accuracy, path=path)

        if len(self._heap) < self.keep_top_k:
            self._write(data, path)
            heapq.heappush(self._heap, entry)
            return path

        worst = self._heap[0]
        if data.cell_accuracy > worst.metric:
            self._write(data, path)
            evicted = heapq.heapreplace(self._heap, entry)
            if evicted.path.exists():
                evicted.path.unlink()
                logger.info('Removed checkpoint %s', evicted.path.name)
            return path

        return None

    @staticmethod
    def _write(data: _CheckpointData, path: Path) -> None:
        """Write checkpoint dict to disk."""
        torch.save(
            {
                'model_state_dict': data.model.state_dict(),
                'optimizer_state_dict': data.optimizer.state_dict(),
                'epoch': data.epoch,
                'step': data.step,
                'val_energy': data.val_energy,
                'cell_accuracy': data.cell_accuracy,
            },
            path,
        )
        logger.info('Saved checkpoint %s (cell_acc=%.4f)', path.name, data.cell_accuracy)

    @staticmethod
    def load(path: Path, model: nn.Module, optimizer: Optimizer | None = None) -> dict:
        """
        Load a checkpoint from disk.

        Args:
            path: Path to checkpoint file.
            model: Model to load state into.
            optimizer: Optional optimizer to load state into.

        Returns:
            The full checkpoint dict (with epoch, step, val_energy).

        """
        checkpoint = torch.load(path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
