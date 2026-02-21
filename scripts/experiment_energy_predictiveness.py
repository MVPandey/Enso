"""Experiment 0: Energy predictiveness — does intermediate energy predict final solution quality."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from scipy import stats
from torch.utils.data import DataLoader

from ebm.dataset.loader import SudokuDataset
from ebm.dataset.splits import split_dataset
from ebm.dataset.torch_dataset import SudokuTorchDataset
from ebm.model.jepa import InferenceConfig, SudokuJEPA
from ebm.training.checkpoint import CheckpointManager
from ebm.utils.config import ArchitectureConfig, TrainingConfig

_STD_EPSILON = 1e-10


@dataclass
class DiagnosticConfig:
    """Parameters for the diagnostic run."""

    n_chains: int
    n_steps: int
    batch_size: int
    device: torch.device


@dataclass
class ExperimentResults:
    """Aggregated experiment metrics for W&B logging."""

    diagnostics: dict[str, np.ndarray]
    rank_corr: np.ndarray
    early_sel_acc: np.ndarray
    pearson_r: float
    spearman_rho: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Experiment 0: Energy predictiveness analysis')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to model checkpoint')
    parser.add_argument('--n-puzzles', type=int, default=1000, help='Number of test puzzles to evaluate')
    parser.add_argument('--n-chains', type=int, default=64, help='Number of parallel Langevin chains')
    parser.add_argument('--n-steps', type=int, default=50, help='Number of Langevin steps')
    parser.add_argument(
        '--batch-size', type=int, default=2, help='Puzzles per batch (limited by VRAM with many chains)'
    )
    parser.add_argument('--output-dir', type=Path, default=Path('results/experiment_0'), help='Output directory')
    parser.add_argument('--wandb-project', type=str, default='enso', help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default='exp0-energy-predictiveness', help='W&B run name')
    return parser.parse_args()


def load_test_data(n_puzzles: int) -> SudokuTorchDataset:
    """
    Load the deterministic test split.

    Args:
        n_puzzles: Approximate number of test puzzles desired.

    Returns:
        SudokuTorchDataset for the test split.

    """
    load_size = n_puzzles * 20
    ds = SudokuDataset()
    df = ds.load_head(load_size)
    _, _, test_df = split_dataset(df, val_size=max(1, int(len(df) * 0.05)), test_size=max(1, int(len(df) * 0.05)))

    if len(test_df) > n_puzzles:
        test_df = test_df.head(n_puzzles)

    return SudokuTorchDataset(test_df)


def run_diagnostic(
    model: SudokuJEPA,
    dataset: SudokuTorchDataset,
    cfg: DiagnosticConfig,
) -> dict[str, np.ndarray]:
    """
    Run solve_diagnostic on the full dataset and aggregate results.

    Args:
        model: Trained SudokuJEPA model.
        dataset: Test dataset.
        cfg: Diagnostic run configuration.

    Returns:
        Dict of aggregated numpy arrays keyed by metric name.

    """
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    inference_cfg = InferenceConfig(n_steps=cfg.n_steps, n_chains=1, lr=0.01, noise_scale=0.005)

    all_diagnostics: dict[str, list[torch.Tensor]] = {
        'energy': [],
        'self_consistency': [],
        'constraint_penalty': [],
        'cell_accuracy': [],
    }

    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader):
        puzzle = batch['puzzle'].to(cfg.device)
        solution = batch['solution'].to(cfg.device)
        mask = batch['mask'].to(cfg.device)

        _, diag = model.solve_diagnostic(
            puzzle,
            mask,
            solution,
            inference_cfg=inference_cfg,
            n_chains_override=cfg.n_chains,
        )
        for key, tensors in all_diagnostics.items():
            tensors.append(diag[key])

        n_done = (batch_idx + 1) * cfg.batch_size
        print(f'  Batch {batch_idx + 1}/{total_batches} ({min(n_done, len(dataset))}/{len(dataset)} puzzles)')

    return {key: torch.cat(tensors, dim=0).numpy() for key, tensors in all_diagnostics.items()}


def compute_rank_correlation(energy: np.ndarray) -> np.ndarray:
    """
    Compute Spearman rank correlation of chain energy at each step vs final step.

    Args:
        energy: (N, n_chains, n_steps) energy array.

    Returns:
        (n_steps,) array of mean Spearman correlations.

    """
    n_puzzles, _, n_steps = energy.shape
    final_energy = energy[:, :, -1]
    correlations = np.zeros(n_steps)

    for t in range(n_steps):
        step_corrs = []
        for p in range(n_puzzles):
            if np.std(energy[p, :, t]) > _STD_EPSILON and np.std(final_energy[p]) > _STD_EPSILON:
                corr, _ = stats.spearmanr(energy[p, :, t], final_energy[p])
                step_corrs.append(corr)
        correlations[t] = np.mean(step_corrs) if step_corrs else 0.0

    return correlations


def compute_early_selection_accuracy(
    energy: np.ndarray,
    cell_accuracy: np.ndarray,
) -> np.ndarray:
    """
    Compute puzzle accuracy if we selected the best-energy chain at each step.

    Args:
        energy: (N, n_chains, n_steps) energy array.
        cell_accuracy: (N, n_chains, n_steps) cell accuracy array.

    Returns:
        (n_steps,) array of puzzle accuracy at each step.

    """
    n_puzzles, _, n_steps = energy.shape
    puzzle_acc = np.zeros(n_steps)

    for t in range(n_steps):
        best_chain = energy[:, :, t].argmin(axis=1)
        final_acc = cell_accuracy[np.arange(n_puzzles), best_chain, -1]
        puzzle_acc[t] = (final_acc == 1.0).mean()

    return puzzle_acc


def compute_energy_accuracy_correlation(
    energy: np.ndarray,
    cell_accuracy: np.ndarray,
) -> tuple[float, float]:
    """
    Compute correlation between final energy and final cell accuracy.

    Args:
        energy: (N, n_chains, n_steps) energy array.
        cell_accuracy: (N, n_chains, n_steps) cell accuracy array.

    Returns:
        Tuple of (Pearson r, Spearman rho).

    """
    final_energy = energy[:, :, -1].ravel()
    final_acc = cell_accuracy[:, :, -1].ravel()
    pearson_r, _ = stats.pearsonr(final_energy, final_acc)
    spearman_rho, _ = stats.spearmanr(final_energy, final_acc)
    return float(pearson_r), float(spearman_rho)


def plot_rank_correlation(correlations: np.ndarray, output_dir: Path) -> Path:
    """
    Plot A: Spearman rank correlation over Langevin steps.

    Args:
        correlations: (n_steps,) correlation values.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved plot.

    """
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = np.arange(len(correlations))
    ax.plot(steps, correlations, 'b-', linewidth=2)

    for threshold in [0.5, 0.7, 0.9]:
        crossings = np.where(correlations >= threshold)[0]
        if len(crossings) > 0:
            ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=crossings[0], color='gray', linestyle=':', alpha=0.5)
            ax.annotate(f'r={threshold} at step {crossings[0]}', xy=(crossings[0], threshold), fontsize=9)

    ax.set_xlabel('Langevin Step')
    ax.set_ylabel('Spearman Rank Correlation (vs Final Step)')
    ax.set_title('Rank Stability: Energy Rankings Over Time')
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / 'plot_a_rank_correlation.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_early_selection(puzzle_acc: np.ndarray, baseline_acc: float, output_dir: Path) -> Path:
    """
    Plot B: Early-selection puzzle accuracy over Langevin steps.

    Args:
        puzzle_acc: (n_steps,) puzzle accuracy at each step.
        baseline_acc: Baseline accuracy (8 chains x 50 steps).
        output_dir: Directory to save the plot.

    Returns:
        Path to saved plot.

    """
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = np.arange(len(puzzle_acc))
    ax.plot(steps, puzzle_acc * 100, 'b-', linewidth=2, label='64 chains, select at step t')
    ax.axhline(y=baseline_acc * 100, color='r', linestyle='--', linewidth=2, label='Baseline: 8 chains (96.6%)')
    ax.set_xlabel('Langevin Step (selection point)')
    ax.set_ylabel('Puzzle Accuracy (%)')
    ax.set_title('Early-Selection: Accuracy vs Selection Step')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / 'plot_b_early_selection.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_energy_vs_accuracy(energy: np.ndarray, cell_accuracy: np.ndarray, output_dir: Path) -> Path:
    """
    Plot C: Scatter of final energy vs final cell accuracy.

    Args:
        energy: (N, n_chains, n_steps) energy array.
        cell_accuracy: (N, n_chains, n_steps) cell accuracy array.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved plot.

    """
    final_energy = energy[:, :, -1].ravel()
    final_acc = cell_accuracy[:, :, -1].ravel()

    n_samples = min(5000, len(final_energy))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(final_energy), size=n_samples, replace=False)

    pearson_r, _ = stats.pearsonr(final_energy, final_acc)
    spearman_rho, _ = stats.spearmanr(final_energy, final_acc)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(final_energy[idx], final_acc[idx], alpha=0.3, s=10, c='steelblue')
    ax.set_xlabel('Final Energy (Step 50)')
    ax.set_ylabel('Final Cell Accuracy (Step 50)')
    ax.set_title(f'Energy vs Accuracy  |  Pearson r={pearson_r:.3f}, Spearman rho={spearman_rho:.3f}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / 'plot_c_energy_vs_accuracy.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_energy_trajectories(
    energy: np.ndarray,
    cell_accuracy: np.ndarray,
    output_dir: Path,
    n_examples: int = 5,
) -> Path:
    """
    Plot D: Energy trajectories for example puzzles, colored by final accuracy.

    Args:
        energy: (N, n_chains, n_steps) energy array.
        cell_accuracy: (N, n_chains, n_steps) cell accuracy array.
        output_dir: Directory to save the plot.
        n_examples: Number of example puzzles to plot.

    Returns:
        Path to saved plot.

    """
    n_examples = min(n_examples, energy.shape[0])
    fig, axes = plt.subplots(1, n_examples, figsize=(5 * n_examples, 5), squeeze=False)

    for i in range(n_examples):
        ax = axes[0, i]
        n_chains = energy.shape[1]
        n_steps = energy.shape[2]
        final_acc = cell_accuracy[i, :, -1]

        for c in range(n_chains):
            color = plt.cm.RdYlGn(final_acc[c])
            ax.plot(range(n_steps), energy[i, c, :], color=color, alpha=0.5, linewidth=0.8)

        ax.set_xlabel('Langevin Step')
        ax.set_ylabel('Energy')
        ax.set_title(f'Puzzle {i + 1}')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Energy Trajectories (green=high acc, red=low acc)', fontsize=14)
    fig.tight_layout()

    path = output_dir / 'plot_d_trajectories.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def log_to_wandb(results: ExperimentResults, plot_paths: dict[str, Path]) -> None:
    """
    Log all metrics, raw data, and plots to W&B.

    Args:
        results: Aggregated experiment results.
        plot_paths: Dict mapping plot name to file path.

    """
    wandb.summary['pearson_r_energy_accuracy'] = results.pearson_r
    wandb.summary['spearman_rho_energy_accuracy'] = results.spearman_rho

    for step_idx in range(len(results.rank_corr)):
        wandb.log(
            {
                'step': step_idx,
                'rank_correlation': results.rank_corr[step_idx],
                'early_selection_accuracy': results.early_sel_acc[step_idx],
                'mean_energy': results.diagnostics['energy'][:, :, step_idx].mean(),
                'mean_cell_accuracy': results.diagnostics['cell_accuracy'][:, :, step_idx].mean(),
            }
        )

    for name, path in plot_paths.items():
        wandb.log({name: wandb.Image(str(path))})

    artifact = wandb.Artifact('experiment_0_diagnostics', type='diagnostics')
    for key, arr in results.diagnostics.items():
        np.save(f'/tmp/exp0_{key}.npy', arr)
        artifact.add_file(f'/tmp/exp0_{key}.npy', name=f'{key}.npy')

    np.save('/tmp/exp0_rank_correlation.npy', results.rank_corr)
    artifact.add_file('/tmp/exp0_rank_correlation.npy', name='rank_correlation.npy')

    np.save('/tmp/exp0_early_selection_accuracy.npy', results.early_sel_acc)
    artifact.add_file('/tmp/exp0_early_selection_accuracy.npy', name='early_selection_accuracy.npy')

    wandb.log_artifact(artifact)


def main() -> None:
    """Run the energy predictiveness experiment."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            'n_puzzles': args.n_puzzles,
            'n_chains': args.n_chains,
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'checkpoint': str(args.checkpoint),
        },
    )

    print('Loading model...')
    arch_cfg = ArchitectureConfig()
    train_cfg = TrainingConfig()
    model = SudokuJEPA(arch_cfg, train_cfg)
    CheckpointManager.load(args.checkpoint, model)
    model.to(device)
    model.eval()
    print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    print(f'Loading {args.n_puzzles} test puzzles...')
    test_ds = load_test_data(args.n_puzzles)
    print(f'Loaded {len(test_ds)} test puzzles')

    diag_cfg = DiagnosticConfig(n_chains=args.n_chains, n_steps=args.n_steps, batch_size=args.batch_size, device=device)
    print(f'Running diagnostics: {args.n_chains} chains x {args.n_steps} steps...')
    diagnostics = run_diagnostic(model, test_ds, diag_cfg)

    energy = diagnostics['energy']
    cell_accuracy = diagnostics['cell_accuracy']
    print(f'Collected diagnostics for {energy.shape[0]} puzzles')

    print('Computing rank correlation...')
    rank_corr = compute_rank_correlation(energy)

    print('Computing early-selection accuracy...')
    early_sel_acc = compute_early_selection_accuracy(energy, cell_accuracy)

    print('Computing energy-accuracy correlation...')
    pearson_r, spearman_rho = compute_energy_accuracy_correlation(energy, cell_accuracy)

    print('\n=== Results ===')
    print(f'Energy-Accuracy Pearson r:  {pearson_r:.4f}')
    print(f'Energy-Accuracy Spearman:   {spearman_rho:.4f}')

    for threshold, label in [(0.5, '0.5'), (0.7, '0.7'), (0.9, '0.9')]:
        crossings = np.where(rank_corr >= threshold)[0]
        step_str = str(crossings[0]) if len(crossings) > 0 else 'never'
        print(f'Rank correlation >= {label}:    step {step_str}')

    print(f'Final puzzle accuracy (64 chains): {early_sel_acc[-1] * 100:.1f}%')

    print('Generating plots...')
    plot_paths = {
        'rank_correlation': plot_rank_correlation(rank_corr, args.output_dir),
        'early_selection': plot_early_selection(early_sel_acc, 0.966, args.output_dir),
        'energy_vs_accuracy': plot_energy_vs_accuracy(energy, cell_accuracy, args.output_dir),
        'energy_trajectories': plot_energy_trajectories(energy, cell_accuracy, args.output_dir),
    }

    np.savez(
        args.output_dir / 'diagnostics.npz',
        **diagnostics,
        rank_correlation=rank_corr,
        early_selection_accuracy=early_sel_acc,
    )
    print(f'Raw data saved to {args.output_dir / "diagnostics.npz"}')

    print('Logging to W&B...')
    results = ExperimentResults(
        diagnostics=diagnostics,
        rank_corr=rank_corr,
        early_sel_acc=early_sel_acc,
        pearson_r=pearson_r,
        spearman_rho=spearman_rho,
    )
    log_to_wandb(results, plot_paths)

    wandb.finish()
    print('Done!')


if __name__ == '__main__':
    main()
