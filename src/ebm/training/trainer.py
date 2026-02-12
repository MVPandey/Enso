"""Training loop for Sudoku JEPA."""

import logging
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ebm.model.jepa import SudokuJEPA
from ebm.training.checkpoint import CheckpointManager, _CheckpointData
from ebm.training.losses import compute_loss
from ebm.training.metrics import (
    compute_cell_accuracy,
    compute_puzzle_accuracy,
    compute_z_variance,
    init_wandb,
    log_train_step,
    log_validation,
)
from ebm.training.scheduler import create_lr_scheduler, get_ema_momentum
from ebm.utils.config import ArchitectureConfig, Config, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Grouped configuration for the Trainer."""

    app: Config
    arch: ArchitectureConfig
    train: TrainingConfig


class Trainer:
    """
    Trains a SudokuJEPA model with AdamW, LR scheduling, EMA, and checkpointing.

    Handles the full training loop including validation, metric logging,
    and checkpoint management.
    """

    def __init__(
        self,
        model: SudokuJEPA,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainerConfig,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model: SudokuJEPA model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            cfg: Grouped configuration (app, arch, train).

        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_cfg = cfg.train

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        self.total_steps = len(train_loader) * cfg.train.epochs
        self.scheduler = create_lr_scheduler(self.optimizer, cfg.train, self.total_steps)
        self.checkpoint_mgr = CheckpointManager(cfg.train.checkpoint_dir, cfg.train.keep_top_k)

        self.global_step = 0

        init_wandb(
            cfg.app,
            run_name='sudoku-jepa',
            extra_config={**cfg.arch.model_dump(), **cfg.train.model_dump()},
        )

    def train(self) -> None:
        """Run the full training loop for all epochs."""
        for epoch in range(self.train_cfg.epochs):
            train_loss = self._train_epoch()
            val_energy, cell_acc, puzzle_acc, z_var = self._validate()

            log_validation(val_energy, cell_acc, puzzle_acc, z_var, self.global_step)
            self.checkpoint_mgr.save(_CheckpointData(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                step=self.global_step,
                val_energy=val_energy,
                cell_accuracy=cell_acc,
            ))

            logger.info(
                'Epoch %d | train_loss=%.4f | val_energy=%.4f | cell_acc=%.4f | puzzle_acc=%.4f',
                epoch, train_loss, val_energy, cell_acc, puzzle_acc,
            )

    def _train_epoch(self) -> float:
        """
        Run one training epoch.

        Returns:
            Mean training loss for the epoch.

        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            puzzle = batch['puzzle'].to(self.device)
            solution = batch['solution'].to(self.device)
            mask = batch['mask'].to(self.device)

            out = self.model(puzzle, solution, mask)
            loss_out = compute_loss(out, solution, mask, self.train_cfg)

            self.optimizer.zero_grad()
            loss_out.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip_norm)
            self.optimizer.step()
            self.scheduler.step()

            momentum = get_ema_momentum(self.global_step, self.total_steps, self.train_cfg)
            self.model.update_target_encoder(momentum)

            lr = float(self.scheduler.get_last_lr()[0])
            log_train_step(loss_out, lr, momentum, self.global_step)

            total_loss += loss_out.total.item()
            n_batches += 1
            self.global_step += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> tuple[float, float, float, float]:
        """
        Run validation and compute metrics.

        Returns:
            Tuple of (mean_energy, cell_accuracy, puzzle_accuracy, z_variance).

        """
        self.model.eval()
        total_energy = 0.0
        all_cell_acc = 0.0
        all_puzzle_acc = 0.0
        all_z_var = 0.0
        n_batches = 0

        for batch in self.val_loader:
            puzzle = batch['puzzle'].to(self.device)
            solution = batch['solution'].to(self.device)
            mask = batch['mask'].to(self.device)

            out = self.model(puzzle, solution, mask)
            total_energy += out.energy.mean().item()

            pred_digits = out.decode_logits.argmax(dim=-1) + 1
            all_cell_acc += compute_cell_accuracy(pred_digits, solution, mask)
            all_puzzle_acc += compute_puzzle_accuracy(pred_digits, solution)
            all_z_var += compute_z_variance(out.z_pred)
            n_batches += 1

        divisor = max(n_batches, 1)
        return total_energy / divisor, all_cell_acc / divisor, all_puzzle_acc / divisor, all_z_var / divisor
