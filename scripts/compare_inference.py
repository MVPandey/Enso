"""
Compare Langevin dynamics vs SVGD inference on a trained checkpoint.

Runs a grid of (method x n_chains) configurations, with an optional
bandwidth sweep for SVGD. Saves metrics to JSON and generates plots.

Usage:
    uv run python scripts/compare_inference.py
    uv run python scripts/compare_inference.py --checkpoint path/to/model.pt
    uv run python scripts/compare_inference.py --n-puzzles 500 --chains 4 8 16 32
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ebm.dataset.loader import SudokuDataset
from ebm.dataset.splits import split_dataset
from ebm.dataset.torch_dataset import SudokuTorchDataset
from ebm.evaluation.metrics import evaluate
from ebm.evaluation.plotting import (
    plot_accuracy_vs_chains,
    plot_bandwidth_sweep,
    plot_constraint_satisfaction,
    plot_timing,
)
from ebm.model.jepa import InferenceConfig, SudokuJEPA
from ebm.training.checkpoint import CheckpointManager
from ebm.utils.config import ArchitectureConfig, TrainingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# -- Defaults -----------------------------------------------------------------

DEFAULT_CHECKPOINT = Path('checkpoints/checkpoint_epoch019_acc0.9926.pt')
DEFAULT_N_PUZZLES = 200
DEFAULT_N_STEPS = 50
DEFAULT_CHAIN_COUNTS = [4, 8, 16, 32]
DEFAULT_BANDWIDTHS = [0.01, 0.1, 1.0, 10.0]
DEFAULT_BATCH_SIZE = 50


def _load_test_data(n_puzzles: int) -> SudokuTorchDataset:
    """Load the test split, capped to n_puzzles."""
    ds = SudokuDataset()
    # Load enough rows to get a test split of the desired size
    n_load = max(n_puzzles * 20, 20_000)
    df = ds.load_head(n_load)

    val_size = max(1, int(len(df) * 0.05))
    test_size = max(1, int(len(df) * 0.05))
    _, _, test_df = split_dataset(df, val_size=val_size, test_size=test_size)

    test_df = test_df.head(n_puzzles)
    logger.info('Test puzzles: %d', len(test_df))
    return SudokuTorchDataset(test_df)


def _run_config(
    model: SudokuJEPA,
    loader: DataLoader,
    cfg: InferenceConfig,
    device: torch.device,
) -> dict:
    """Run inference with a single config and return metrics + timing."""
    model.eval()
    all_preds, all_solutions, all_masks = [], [], []

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.perf_counter()

    for batch in loader:
        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)
        pred = model.solve(puzzle, mask, cfg)
        all_preds.append(pred.cpu())
        all_solutions.append(solution.cpu())
        all_masks.append(mask.cpu())

    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = time.perf_counter() - t0

    metrics = evaluate(all_preds, all_solutions, all_masks)
    return {
        'method': cfg.method,
        'n_chains': cfg.n_chains,
        'n_steps': cfg.n_steps,
        'kernel_bandwidth': cfg.kernel_bandwidth,
        'n_puzzles': metrics.n_puzzles,
        'cell_accuracy': metrics.cell_accuracy,
        'puzzle_accuracy': metrics.puzzle_accuracy,
        'constraint_satisfaction': metrics.constraint_satisfaction,
        'time_s': elapsed,
    }


def _run_bandwidth_sweep(
    model: SudokuJEPA,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict]:
    """Run SVGD bandwidth sensitivity sweep."""
    bw_results: list[dict] = []
    bw_chain_count = args.bandwidth_chains
    for bw in args.bandwidths:
        cfg = InferenceConfig(
            method='svgd',
            n_steps=args.n_steps,
            n_chains=bw_chain_count,
            lr=0.01,
            kernel_bandwidth=bw,
        )
        logger.info('Running: SVGD bandwidth=%.3f, chains=%d', bw, bw_chain_count)
        result = _run_config(model, loader, cfg, device)
        bw_results.append(result)
        logger.info('  bandwidth=%.3f -> puzzle_acc=%.2f%%', bw, result['puzzle_accuracy'] * 100)
    return bw_results


def _save_and_plot(results: list[dict], bw_results: list[dict], output_dir: Path) -> None:
    """Save results JSON and generate all plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {'main_sweep': results, 'bandwidth_sweep': bw_results}
    results_path = output_dir / 'results.json'
    results_path.write_text(json.dumps(all_results, indent=2))
    logger.info('Results saved to %s', results_path)

    plot_accuracy_vs_chains(results, output_dir / 'accuracy_vs_chains.png')
    plot_accuracy_vs_chains(results, output_dir / 'cell_accuracy_vs_chains.png', metric='cell_accuracy')
    plot_timing(results, output_dir / 'timing.png')
    plot_constraint_satisfaction(results, output_dir / 'constraint_satisfaction.png')
    if bw_results:
        plot_bandwidth_sweep(bw_results, output_dir / 'bandwidth_sweep.png')
    logger.info('Plots saved to %s', output_dir)

    # Print summary table
    print('\n' + '=' * 80)
    print('RESULTS SUMMARY')
    print('=' * 80)
    print(f'{"Method":<10} {"Chains":>6} {"Puzzle %":>9} {"Cell %":>9} {"Constr %":>9} {"Time (s)":>9}')
    print('-' * 80)
    for r in results:
        print(
            f'{r["method"]:<10} {r["n_chains"]:>6d} '
            f'{r["puzzle_accuracy"] * 100:>8.2f}% '
            f'{r["cell_accuracy"] * 100:>8.2f}% '
            f'{r["constraint_satisfaction"] * 100:>8.2f}% '
            f'{r["time_s"]:>8.1f}s'
        )
    print('=' * 80)


def run_experiment(args: argparse.Namespace) -> None:
    """Run the full comparison experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device: %s', device)
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info('GPU: %s (%.1f GB)', gpu_name, vram_gb)

    # Load model
    arch_cfg = ArchitectureConfig()
    train_cfg = TrainingConfig()
    model = SudokuJEPA(arch_cfg, train_cfg)
    checkpoint_path = Path(args.checkpoint)
    logger.info('Loading checkpoint: %s', checkpoint_path)
    CheckpointManager.load(checkpoint_path, model)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info('Model parameters: %s', f'{param_count:,}')

    # Load data
    test_ds = _load_test_data(args.n_puzzles)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # -- Main sweep: method x n_chains ----------------------------------------
    results: list[dict] = []
    for n_chains in args.chains:
        for method in ('langevin', 'svgd'):
            cfg = InferenceConfig(
                method=method,
                n_steps=args.n_steps,
                n_chains=n_chains,
                lr=0.01,
                noise_scale=0.005,
            )
            label = f'{method:>8s} | chains={n_chains:>3d}'
            logger.info('Running: %s', label)
            result = _run_config(model, loader, cfg, device)
            results.append(result)
            logger.info(
                '  %s -> puzzle_acc=%.2f%% | cell_acc=%.2f%% | constraints=%.2f%% | %.1fs',
                label,
                result['puzzle_accuracy'] * 100,
                result['cell_accuracy'] * 100,
                result['constraint_satisfaction'] * 100,
                result['time_s'],
            )

    # -- Bandwidth sweep (SVGD only) ------------------------------------------
    bw_results = _run_bandwidth_sweep(model, loader, device, args) if args.bandwidth_sweep else []

    _save_and_plot(results, bw_results, Path(args.output_dir))


def main() -> None:
    """Parse arguments and run the experiment."""
    parser = argparse.ArgumentParser(description='Compare Langevin vs SVGD inference')
    parser.add_argument('--checkpoint', type=str, default=str(DEFAULT_CHECKPOINT), help='Path to model checkpoint')
    parser.add_argument('--n-puzzles', type=int, default=DEFAULT_N_PUZZLES, help='Number of test puzzles')
    parser.add_argument('--n-steps', type=int, default=DEFAULT_N_STEPS, help='Inference steps per run')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for inference')
    parser.add_argument('--chains', type=int, nargs='+', default=DEFAULT_CHAIN_COUNTS, help='Chain counts to sweep')
    parser.add_argument('--output-dir', type=str, default='results/svgd_comparison', help='Output directory')
    parser.add_argument('--bandwidth-sweep', action='store_true', help='Run SVGD bandwidth sensitivity sweep')
    parser.add_argument('--bandwidth-chains', type=int, default=16, help='Chain count for bandwidth sweep')
    parser.add_argument(
        '--bandwidths', type=float, nargs='+', default=DEFAULT_BANDWIDTHS, help='Bandwidth values to sweep'
    )
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
