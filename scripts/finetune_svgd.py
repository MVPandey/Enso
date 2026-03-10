"""
Fine-tune SudokuJEPA with SVGD-in-the-loop to shape the energy landscape for SVGD.

Loads a pre-trained checkpoint and adds an auxiliary SVGD decode loss: at each
training step, a few differentiable SVGD steps are unrolled (create_graph=True)
and the decoded output is compared against the ground truth. This teaches the
model to produce energy surfaces that SVGD particles can navigate to correct
solutions.

Usage:
    uv run python scripts/finetune_svgd.py
    uv run python scripts/finetune_svgd.py --checkpoint path/to/model.pt --epochs 5
    uv run python scripts/finetune_svgd.py --svgd-steps 5 --svgd-particles 8 --lr 3e-5
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ebm.dataset.loader import SudokuDataset
from ebm.dataset.splits import split_dataset
from ebm.dataset.torch_dataset import SudokuTorchDataset
from ebm.model.constraints import constraint_penalty
from ebm.model.jepa import InferenceConfig, SudokuJEPA
from ebm.training.checkpoint import CheckpointManager, _CheckpointData
from ebm.training.losses import compute_loss
from ebm.training.metrics import compute_cell_accuracy, compute_puzzle_accuracy, finish_wandb, init_wandb
from ebm.utils.config import ArchitectureConfig, Config, TrainingConfig

try:
    import wandb

    _wandb_available = True
except ImportError:
    _wandb_available = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path('checkpoints/checkpoint_epoch019_acc0.9926.pt')


def _log_step(total_loss: float, jepa_loss: float, svgd_loss: float, lr: float, step: int) -> None:
    """Log per-step metrics to W&B."""
    if not _wandb_available or wandb.run is None:
        return
    wandb.log(
        {
            'finetune/loss_total': total_loss,
            'finetune/loss_jepa': jepa_loss,
            'finetune/loss_svgd': svgd_loss,
            'finetune/lr': lr,
        },
        step=step,
    )


def _log_epoch(val_energy: float, cell_acc: float, puzzle_acc: float, step: int) -> None:
    """Log per-epoch validation metrics to W&B."""
    if not _wandb_available or wandb.run is None:
        return
    wandb.log(
        {
            'val/energy': val_energy,
            'val/cell_accuracy': cell_acc,
            'val/puzzle_accuracy': puzzle_acc,
        },
        step=step,
    )


def _create_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Load dataset and create train/val DataLoaders."""
    ds = SudokuDataset()
    df = ds.load_head(args.n_samples) if args.n_samples else ds.load_all()
    logger.info('Dataset size: %d', len(df))

    val_size = min(50_000, max(1, int(len(df) * 0.05)))
    test_size = min(50_000, max(1, int(len(df) * 0.05)))
    train_df, val_df, _ = split_dataset(df, val_size=val_size, test_size=test_size)
    logger.info('Train: %d | Val: %d', len(train_df), len(val_df))

    train_loader = DataLoader(
        SudokuTorchDataset(train_df),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        SudokuTorchDataset(val_df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[SudokuJEPA, TrainingConfig]:
    """Load model from checkpoint."""
    arch_cfg = ArchitectureConfig()
    train_cfg = TrainingConfig()
    model = SudokuJEPA(arch_cfg, train_cfg)

    logger.info('Loading checkpoint: %s', checkpoint_path)
    CheckpointManager.load(checkpoint_path, model)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Trainable parameters: %s', f'{param_count:,}')
    return model, train_cfg


@torch.no_grad()
def _validate(model: SudokuJEPA, val_loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    """Run validation and return (energy, cell_acc, puzzle_acc)."""
    model.eval()
    total_energy = 0.0
    total_cell_acc = 0.0
    total_puzzle_acc = 0.0
    n_batches = 0

    for batch in val_loader:
        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)

        out = model(puzzle, solution, mask)
        total_energy += out.energy.mean().item()

        pred_digits = out.decode_logits.argmax(dim=-1) + 1
        total_cell_acc += compute_cell_accuracy(pred_digits, solution, mask)
        total_puzzle_acc += compute_puzzle_accuracy(pred_digits, solution)
        n_batches += 1

    d = max(n_batches, 1)
    return total_energy / d, total_cell_acc / d, total_puzzle_acc / d


@dataclass
class _StepConfig:
    """Bundled config for a single training step."""

    train_cfg: TrainingConfig
    svgd_cfg: InferenceConfig
    svgd_weight: float
    device: torch.device


def _train_step(
    model: SudokuJEPA,
    batch: dict[str, torch.Tensor],
    cfg: _StepConfig,
) -> tuple[torch.Tensor, float, float]:
    """Run one training step. Returns (total_loss, jepa_loss_value, svgd_loss_value)."""
    puzzle = batch['puzzle'].to(cfg.device)
    solution = batch['solution'].to(cfg.device)
    mask = batch['mask'].to(cfg.device)

    # Standard JEPA forward + loss
    out = model(puzzle, solution, mask)
    jepa_loss = compute_loss(out, solution, mask, cfg.train_cfg)

    # SVGD-in-the-loop: differentiable unrolled SVGD
    svgd_logits = model.forward_svgd(puzzle, mask, cfg.svgd_cfg)

    # SVGD decode loss on empty cells
    targets = solution.argmax(dim=-1)
    empty = mask == 0
    svgd_decode_loss = F.cross_entropy(svgd_logits[empty], targets[empty])

    # SVGD constraint loss
    svgd_probs = torch.softmax(svgd_logits, dim=-1)
    svgd_constraint = constraint_penalty(svgd_probs).mean()

    svgd_loss = svgd_decode_loss + cfg.train_cfg.constraint_loss_weight * svgd_constraint
    total_loss = jepa_loss.total + cfg.svgd_weight * svgd_loss

    return total_loss, jepa_loss.total.item(), svgd_loss.item()


def finetune(args: argparse.Namespace) -> None:
    """Run SVGD-in-the-loop fine-tuning."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device: %s', device)
    if device.type == 'cuda':
        logger.info(
            'GPU: %s (%.1f GB)',
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / (1024**3),
        )

    # Initialize W&B
    app_cfg = Config()
    init_wandb(
        app_cfg,
        run_name='svgd-finetune',
        extra_config={
            'finetune_lr': args.lr,
            'finetune_epochs': args.epochs,
            'finetune_batch_size': args.batch_size,
            'svgd_steps': args.svgd_steps,
            'svgd_particles': args.svgd_particles,
            'svgd_lr': args.svgd_lr,
            'svgd_weight': args.svgd_weight,
            'ema_momentum': args.ema_momentum,
        },
    )

    train_loader, val_loader = _create_loaders(args)
    model, train_cfg = _load_model(Path(args.checkpoint), device)

    svgd_cfg = InferenceConfig(
        method='svgd',
        n_steps=args.svgd_steps,
        n_chains=args.svgd_particles,
        lr=args.svgd_lr,
    )
    logger.info(
        'SVGD config: steps=%d, particles=%d, lr=%.4f, weight=%.2f',
        svgd_cfg.n_steps,
        svgd_cfg.n_chains,
        svgd_cfg.lr,
        args.svgd_weight,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_mgr = CheckpointManager(checkpoint_dir, keep_top_k=3)

    # Baseline validation before fine-tuning
    val_energy, cell_acc, puzzle_acc = _validate(model, val_loader, device)
    logger.info('Baseline | val_energy=%.4f | cell_acc=%.4f | puzzle_acc=%.4f', val_energy, cell_acc, puzzle_acc)
    _log_epoch(val_energy, cell_acc, puzzle_acc, step=0)

    step_cfg = _StepConfig(train_cfg=train_cfg, svgd_cfg=svgd_cfg, svgd_weight=args.svgd_weight, device=device)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_jepa = 0.0
        epoch_svgd = 0.0
        n_batches = 0

        for batch in train_loader:
            total_loss, jepa_val, svgd_val = _train_step(
                model,
                batch,
                step_cfg,
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # EMA update with high momentum (model is already well-trained)
            model.update_target_encoder(args.ema_momentum)

            epoch_loss += total_loss.item()
            epoch_jepa += jepa_val
            epoch_svgd += svgd_val
            n_batches += 1
            global_step += 1

            lr = scheduler.get_last_lr()[0]
            _log_step(total_loss.item(), jepa_val, svgd_val, lr, global_step)

            if global_step % args.log_every == 0:
                logger.info(
                    'Step %d | total=%.4f | jepa=%.4f | svgd=%.4f | lr=%.2e',
                    global_step,
                    total_loss.item(),
                    jepa_val,
                    svgd_val,
                    lr,
                )

        # Validate
        val_energy, cell_acc, puzzle_acc = _validate(model, val_loader, device)
        _log_epoch(val_energy, cell_acc, puzzle_acc, global_step)

        d = max(n_batches, 1)
        logger.info(
            'Epoch %d/%d | loss=%.4f (jepa=%.4f + svgd=%.4f) | val_energy=%.4f | cell=%.4f | puzzle=%.4f',
            epoch + 1,
            args.epochs,
            epoch_loss / d,
            epoch_jepa / d,
            epoch_svgd / d,
            val_energy,
            cell_acc,
            puzzle_acc,
        )

        checkpoint_mgr.save(
            _CheckpointData(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                val_energy=val_energy,
                cell_accuracy=cell_acc,
            )
        )

    logger.info('Fine-tuning complete. Checkpoints saved to %s', checkpoint_dir)
    finish_wandb()


def main() -> None:
    """Parse arguments and run fine-tuning."""
    parser = argparse.ArgumentParser(description='Fine-tune SudokuJEPA with SVGD-in-the-loop')

    # Data
    parser.add_argument('--n-samples', type=int, default=None, help='Limit dataset size (default: all)')
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size (default: 256)')

    # Checkpoint
    parser.add_argument(
        '--checkpoint', type=str, default=str(DEFAULT_CHECKPOINT), help='Path to pre-trained checkpoint'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints/svgd_finetune', help='Output checkpoint directory'
    )

    # Training
    parser.add_argument('--epochs', type=int, default=5, help='Fine-tuning epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (default: 1e-5)')
    parser.add_argument('--ema-momentum', type=float, default=0.999, help='EMA momentum (default: 0.999)')

    # SVGD-in-the-loop
    parser.add_argument('--svgd-steps', type=int, default=3, help='Unrolled SVGD steps per training step (default: 3)')
    parser.add_argument('--svgd-particles', type=int, default=4, help='SVGD particles per puzzle (default: 4)')
    parser.add_argument('--svgd-lr', type=float, default=0.01, help='SVGD step size (default: 0.01)')
    parser.add_argument('--svgd-weight', type=float, default=1.0, help='Weight for SVGD loss (default: 1.0)')

    # Logging
    parser.add_argument('--log-every', type=int, default=100, help='Log every N steps (default: 100)')

    args = parser.parse_args()
    finetune(args)


if __name__ == '__main__':
    main()
