"""CLI entry point for training and evaluation."""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ebm.dataset.loader import SudokuDataset
from ebm.dataset.splits import split_dataset
from ebm.dataset.torch_dataset import SudokuTorchDataset
from ebm.evaluation.metrics import evaluate
from ebm.evaluation.solver import solve_dataset
from ebm.model.jepa import InferenceConfig, SudokuJEPA
from ebm.training.checkpoint import CheckpointManager
from ebm.training.trainer import Trainer, TrainerConfig
from ebm.utils.config import ArchitectureConfig, Config, TrainingConfig
from ebm.utils.device import auto_scale_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def train(args: argparse.Namespace) -> None:
    """
    Run training.

    Args:
        args: Parsed CLI arguments with optional epochs, batch_size, n_samples.

    """
    app_cfg = Config()
    arch_cfg = ArchitectureConfig()
    train_cfg = TrainingConfig()

    if args.epochs:
        train_cfg = train_cfg.model_copy(update={'epochs': args.epochs})

    train_cfg = auto_scale_config(train_cfg, batch_size_override=args.batch_size)

    logger.info('Loading dataset...')
    ds = SudokuDataset()
    if args.n_samples:
        df = ds.load_head(args.n_samples)
    else:
        df = ds.load_all()
    logger.info('Dataset size: %d', len(df))

    val_size = train_cfg.val_size
    test_size = train_cfg.test_size
    if val_size + test_size >= len(df):
        val_size = max(1, int(len(df) * 0.05))
        test_size = max(1, int(len(df) * 0.05))

    train_df, val_df, _ = split_dataset(df, val_size=val_size, test_size=test_size)
    logger.info('Train: %d | Val: %d', len(train_df), len(val_df))

    train_ds = SudokuTorchDataset(train_df)
    val_ds = SudokuTorchDataset(val_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
    )

    model = SudokuJEPA(arch_cfg, train_cfg)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Trainable parameters: %s', f'{param_count:,}')

    trainer = Trainer(model, train_loader, val_loader, TrainerConfig(app=app_cfg, arch=arch_cfg, train=train_cfg))
    trainer.train()


def eval_model(args: argparse.Namespace) -> None:
    """
    Run evaluation on test set.

    Args:
        args: Parsed CLI arguments with checkpoint, n_samples, langevin_steps, n_chains.

    """
    arch_cfg = ArchitectureConfig()
    train_cfg = TrainingConfig()

    logger.info('Loading dataset...')
    ds = SudokuDataset()
    if args.n_samples:
        df = ds.load_head(args.n_samples)
    else:
        df = ds.load_all()

    val_size = train_cfg.val_size
    test_size = train_cfg.test_size
    if val_size + test_size >= len(df):
        val_size = max(1, int(len(df) * 0.05))
        test_size = max(1, int(len(df) * 0.05))

    _, _, test_df = split_dataset(df, val_size=val_size, test_size=test_size)
    logger.info('Test set: %d puzzles', len(test_df))

    test_ds = SudokuTorchDataset(test_df)
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SudokuJEPA(arch_cfg, train_cfg)

    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        logger.error('--checkpoint is required for evaluation')
        return

    logger.info('Loading checkpoint: %s', checkpoint_path)
    CheckpointManager.load(Path(checkpoint_path), model)
    model.to(device)

    inference_cfg = InferenceConfig.from_training_config(train_cfg)
    if args.langevin_steps:
        inference_cfg.n_steps = args.langevin_steps
    if args.n_chains:
        inference_cfg.n_chains = args.n_chains

    logger.info('Running inference (steps=%d, chains=%d)...', inference_cfg.n_steps, inference_cfg.n_chains)
    preds, solutions, masks = solve_dataset(model, test_loader, inference_cfg, device)

    metrics = evaluate(preds, solutions, masks)
    logger.info('Results on %d puzzles:', metrics.n_puzzles)
    logger.info('  Cell accuracy:          %.4f', metrics.cell_accuracy)
    logger.info('  Puzzle accuracy:        %.4f', metrics.puzzle_accuracy)
    logger.info('  Constraint satisfaction: %.4f', metrics.constraint_satisfaction)


def main() -> None:
    """Parse arguments and dispatch to train or eval."""
    parser = argparse.ArgumentParser(description='Sudoku JEPA — Energy-Based Model')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Training batch size')
    train_parser.add_argument('--n-samples', type=int, help='Limit dataset to first N samples (for debugging)')

    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained model')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--n-samples', type=int, help='Limit dataset to first N samples')
    eval_parser.add_argument('--langevin-steps', type=int, help='Override Langevin dynamics steps')
    eval_parser.add_argument('--n-chains', type=int, help='Override number of parallel chains')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'eval':
        eval_model(args)


if __name__ == '__main__':
    main()
