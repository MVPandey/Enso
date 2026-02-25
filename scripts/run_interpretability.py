#!/usr/bin/env python3
"""
Run mechanistic interpretability experiments on trained ENSO EBM checkpoints.

Usage:
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment trajectory-decomposition
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment strategy-progression
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment attention-specialization
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment causal-ablation
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment forward-vs-langevin

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import torch
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ebm.dataset import SudokuDataset, SudokuTorchDataset, split_dataset
from ebm.interpretability import (
    STRATEGY_DIFFICULTY,
    AttentionAnalyzer,
    HeadAblator,
    StrategyDetector,
    TrajectoryAnalyzer,
    TrajectoryRecorder,
)
from ebm.model.jepa import InferenceConfig, SudokuJEPA
from ebm.utils.config import ArchitectureConfig

EXPERIMENTS = [
    'trajectory-decomposition',
    'strategy-progression',
    'attention-specialization',
    'causal-ablation',
    'forward-vs-langevin',
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='ENSO Mechanistic Interpretability Experiments')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to model checkpoint.')
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        choices=EXPERIMENTS,
        help='Which experiment to run.',
    )
    parser.add_argument('--num-puzzles', type=int, default=100, help='Number of puzzles to analyze.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for data loading.')
    parser.add_argument('--langevin-steps', type=int, default=50, help='Number of Langevin dynamics steps.')
    parser.add_argument('--attention-stride', type=int, default=5, help='Record attention every N steps.')
    parser.add_argument('--output-dir', type=Path, default=Path('results/interpretability'), help='Output directory.')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda).')
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> SudokuJEPA:
    """Load a SudokuJEPA model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = SudokuJEPA(arch_cfg=ArchitectureConfig())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_test_data(num_puzzles: int, batch_size: int) -> DataLoader:
    """Load test puzzles from the Kaggle dataset."""
    ds = SudokuDataset()
    df = ds.load_head(num_puzzles * 20)
    _, _, test_df = split_dataset(df)
    test_df = test_df.head(num_puzzles)
    test_ds = SudokuTorchDataset(test_df)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def run_trajectory_decomposition(args: argparse.Namespace, model: SudokuJEPA, loader: DataLoader) -> None:
    """Experiment 1: Record trajectories and analyze lock-in times."""
    out_dir = args.output_dir / 'trajectory-decomposition'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    inference_cfg = InferenceConfig(n_steps=args.langevin_steps, n_chains=1)
    recorder = TrajectoryRecorder(model, record_attention=False)
    analyzer = TrajectoryAnalyzer()

    all_lock_in_steps: list[int] = []
    all_difficulties: list[int] = []
    puzzle_results: list[dict] = []

    for batch_idx, batch in enumerate(loader):
        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)

        traj = recorder.record(puzzle, mask, solution, inference_cfg)
        b_size = puzzle.shape[0]

        for b in range(b_size):
            result, metrics, _ = analyzer.full_analysis(traj, solution, batch_idx=b)
            for lock_ev in metrics.lock_in_events:
                all_lock_in_steps.append(lock_ev.lock_in_step)
                all_difficulties.append(STRATEGY_DIFFICULTY.get(lock_ev.strategy, 6))

            puzzle_results.append(
                {
                    'batch': batch_idx,
                    'sample': b,
                    'num_events': len(result.events),
                    'strategy_coverage': metrics.strategy_coverage,
                    'num_lock_ins': len(metrics.lock_in_events),
                    'correlation': metrics.step_strategy_correlation,
                    'phase_boundaries': metrics.phase_boundaries,
                }
            )

        print(f'  Batch {batch_idx}: processed {b_size} puzzles')

    # Save results
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(puzzle_results, f, indent=2)

    # Plot lock-in step distribution
    if all_lock_in_steps:
        _fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(all_lock_in_steps, bins=20, edgecolor='black')
        axes[0].set_xlabel('Lock-in Step')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Distribution of Lock-in Steps')

        if all_difficulties and all_lock_in_steps:
            axes[1].scatter(all_difficulties, all_lock_in_steps, alpha=0.3)
            axes[1].set_xlabel('Strategy Difficulty')
            axes[1].set_ylabel('Lock-in Step')
            axes[1].set_title('Lock-in Step vs Strategy Difficulty')

        plt.tight_layout()
        plt.savefig(out_dir / 'lock_in_analysis.png', dpi=150)
        plt.close()

    print(f'Results saved to {out_dir}')


def run_strategy_progression(args: argparse.Namespace, model: SudokuJEPA, loader: DataLoader) -> None:
    """Experiment 2: Classify all events and plot strategy type vs step."""
    out_dir = args.output_dir / 'strategy-progression'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    inference_cfg = InferenceConfig(n_steps=args.langevin_steps, n_chains=1)
    recorder = TrajectoryRecorder(model, record_attention=False)
    analyzer = TrajectoryAnalyzer()

    all_events: list[dict] = []
    total_counts: dict[str, int] = {}

    for batch_idx, batch in enumerate(loader):
        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)

        traj = recorder.record(puzzle, mask, solution, inference_cfg)

        for b in range(puzzle.shape[0]):
            result = analyzer.analyze_trajectory(traj, batch_idx=b)
            for event in result.events:
                label = event.strategy.value if event.strategy else 'unknown'
                all_events.append(
                    {
                        'step': event.step,
                        'strategy': label,
                        'row': event.row,
                        'col': event.col,
                        'digit': event.digit,
                        'confidence': event.confidence,
                    }
                )
                total_counts[label] = total_counts.get(label, 0) + 1

        print(f'  Batch {batch_idx}: {len(all_events)} events so far')

    # Save events
    with open(out_dir / 'events.json', 'w') as f:
        json.dump(all_events, f, indent=2)
    with open(out_dir / 'strategy_counts.json', 'w') as f:
        json.dump(total_counts, f, indent=2)

    # Plot strategy distribution by step
    if all_events:
        strategies = sorted(set(e['strategy'] for e in all_events))
        max_step = max(e['step'] for e in all_events)
        bin_size = max(1, max_step // 10)
        bins = range(0, max_step + bin_size, bin_size)

        _fig, ax = plt.subplots(figsize=(12, 6))
        bottom = [0] * (len(bins) - 1)
        for strategy in strategies:
            counts = []
            for i in range(len(bins) - 1):
                c = sum(1 for e in all_events if bins[i] <= e['step'] < bins[i + 1] and e['strategy'] == strategy)
                counts.append(c)
            ax.bar(range(len(counts)), counts, bottom=bottom, label=strategy)
            bottom = [b + c for b, c in zip(bottom, counts)]

        ax.set_xlabel('Step Bin')
        ax.set_ylabel('Event Count')
        ax.set_title('Strategy Distribution by Langevin Step')
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'strategy_progression.png', dpi=150)
        plt.close()

    print(f'Results saved to {out_dir}')


def run_attention_specialization(args: argparse.Namespace, model: SudokuJEPA, loader: DataLoader) -> None:
    """Experiment 3: Profile attention heads for structural specialization."""
    out_dir = args.output_dir / 'attention-specialization'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    inference_cfg = InferenceConfig(n_steps=args.langevin_steps, n_chains=1)
    recorder = TrajectoryRecorder(model, record_attention=True, attention_stride=args.attention_stride)
    attention_analyzer = AttentionAnalyzer()

    all_profiles: list[dict] = []

    for batch_idx, batch in enumerate(loader):
        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)

        traj = recorder.record(puzzle, mask, solution, inference_cfg)

        # Collect attention from all captured steps
        all_attention: dict[str, torch.Tensor] = {}
        for step in traj.steps:
            if step.encoder_attention:
                all_attention.update(step.encoder_attention)
            if step.decoder_attention:
                all_attention.update(step.decoder_attention)

        if all_attention:
            profiles = attention_analyzer.compute_head_profiles(all_attention)
            for p in profiles:
                all_profiles.append(
                    {
                        'batch': batch_idx,
                        'layer': p.layer,
                        'head_idx': p.head_idx,
                        'row_score': p.row_score,
                        'col_score': p.col_score,
                        'box_score': p.box_score,
                        'specialization': p.specialization,
                    }
                )

        print(f'  Batch {batch_idx}: {len(all_profiles)} head profiles')
        break  # One batch is sufficient for attention profiling

    # Save results
    with open(out_dir / 'head_profiles.json', 'w') as f:
        json.dump(all_profiles, f, indent=2)

    # Plot specialization heatmap
    if all_profiles:
        layers = sorted(set(p['layer'] for p in all_profiles))
        _fig, ax = plt.subplots(figsize=(12, max(4, len(layers) * 0.5)))

        for i, layer in enumerate(layers):
            layer_profiles = [p for p in all_profiles if p['layer'] == layer]
            for p in layer_profiles:
                color = {'row': 'red', 'column': 'blue', 'box': 'green'}.get(p['specialization'], 'gray')
                ax.barh(
                    i,
                    max(p['row_score'], p['col_score'], p['box_score']),
                    left=p['head_idx'] * 2,
                    height=0.8,
                    color=color,
                    alpha=0.7,
                )

        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=7)
        ax.set_xlabel('Head Index (spaced) x Score')
        ax.set_title('Attention Head Specialization')
        plt.tight_layout()
        plt.savefig(out_dir / 'head_specialization.png', dpi=150)
        plt.close()

    print(f'Results saved to {out_dir}')


def run_causal_ablation(args: argparse.Namespace, model: SudokuJEPA, loader: DataLoader) -> None:
    """Experiment 4: Ablate individual heads and measure accuracy impact."""
    out_dir = args.output_dir / 'causal-ablation'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    inference_cfg = InferenceConfig(n_steps=args.langevin_steps, n_chains=1)
    ablator = HeadAblator(model)

    # First, get attention profiles to know which heads to ablate
    attention_analyzer = AttentionAnalyzer()
    recorder = TrajectoryRecorder(model, record_attention=True, attention_stride=args.attention_stride)

    # Get first batch for profiling
    batch = next(iter(loader))
    puzzle = batch['puzzle'].to(device)
    solution = batch['solution'].to(device)
    mask = batch['mask'].to(device)

    traj = recorder.record(puzzle, mask, solution, inference_cfg)
    all_attention: dict[str, torch.Tensor] = {}
    for step in traj.steps:
        if step.encoder_attention:
            all_attention.update(step.encoder_attention)
        if step.decoder_attention:
            all_attention.update(step.decoder_attention)

    profiles = attention_analyzer.compute_head_profiles(all_attention) if all_attention else []

    # Ablate top-N most specialized heads
    specialized = [p for p in profiles if p.specialization != 'mixed']
    targets = specialized[: min(10, len(specialized))] if specialized else profiles[: min(5, len(profiles))]

    ablation_results: list[dict] = []
    if targets:
        print(f'  Ablating {len(targets)} heads...')
        results = ablator.run_ablation_sweep(puzzle, mask, solution, inference_cfg, targets)
        for r in results:
            ablation_results.append(
                {
                    'ablated_heads': r.ablated_heads,
                    'overall_accuracy': r.overall_accuracy,
                    'strategy_accuracy': r.strategy_accuracy,
                    'baseline_accuracy': r.baseline_accuracy,
                    'accuracy_drop': r.baseline_accuracy - r.overall_accuracy,
                }
            )

    with open(out_dir / 'ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)

    # Plot accuracy drops
    if ablation_results:
        _fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f'{r["ablated_heads"][0][0].split(".")[-1]}:h{r["ablated_heads"][0][1]}' for r in ablation_results]
        drops = [r['accuracy_drop'] for r in ablation_results]
        ax.bar(range(len(labels)), drops)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Accuracy Drop')
        ax.set_title('Impact of Single-Head Ablation')
        plt.tight_layout()
        plt.savefig(out_dir / 'ablation_impact.png', dpi=150)
        plt.close()

    print(f'Results saved to {out_dir}')


def run_forward_vs_langevin(args: argparse.Namespace, model: SudokuJEPA, loader: DataLoader) -> None:
    """Experiment 5: Compare single forward pass vs full Langevin dynamics."""
    out_dir = args.output_dir / 'forward-vs-langevin'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    inference_cfg = InferenceConfig(n_steps=args.langevin_steps, n_chains=1)
    recorder = TrajectoryRecorder(model, record_attention=False)
    detector = StrategyDetector()

    forward_counts: dict[str, int] = {}
    langevin_counts: dict[str, int] = {}
    comparison: list[dict] = []

    for batch_idx, batch in enumerate(loader):
        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)

        traj = recorder.record(puzzle, mask, solution, inference_cfg)

        for b in range(puzzle.shape[0]):
            sol_board = solution[b].argmax(dim=-1) + 1

            # Forward pass: step 0 board
            board_0 = traj.steps[0].board[b]
            events_0 = detector.classify(torch.zeros_like(board_0), board_0, mask[b])
            forward_correct = int(((board_0 == sol_board) & (mask[b] == 0)).sum().item())

            # Langevin: final board
            board_final = traj.final_board[b]
            events_final = detector.classify(torch.zeros_like(board_final), board_final, mask[b])
            langevin_correct = int(((board_final == sol_board) & (mask[b] == 0)).sum().item())

            total_cells = int((mask[b] == 0).sum().item())

            for e in events_0:
                label = e.strategy.value if e.strategy else 'unknown'
                forward_counts[label] = forward_counts.get(label, 0) + 1
            for e in events_final:
                label = e.strategy.value if e.strategy else 'unknown'
                langevin_counts[label] = langevin_counts.get(label, 0) + 1

            comparison.append(
                {
                    'forward_accuracy': forward_correct / max(total_cells, 1),
                    'langevin_accuracy': langevin_correct / max(total_cells, 1),
                    'forward_events': len(events_0),
                    'langevin_events': len(events_final),
                }
            )

        print(f'  Batch {batch_idx}: processed')

    with open(out_dir / 'comparison.json', 'w') as f:
        json.dump(
            {
                'forward_strategy_counts': forward_counts,
                'langevin_strategy_counts': langevin_counts,
                'per_puzzle': comparison,
            },
            f,
            indent=2,
        )

    # Plot comparison
    if comparison:
        _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        fwd_acc = [c['forward_accuracy'] for c in comparison]
        lang_acc = [c['langevin_accuracy'] for c in comparison]
        axes[0].scatter(fwd_acc, lang_acc, alpha=0.5)
        axes[0].plot([0, 1], [0, 1], 'r--')
        axes[0].set_xlabel('Forward Pass Accuracy')
        axes[0].set_ylabel('Langevin Accuracy')
        axes[0].set_title('Forward vs Langevin Accuracy')

        all_strategies = sorted(set(list(forward_counts) + list(langevin_counts)))
        x = range(len(all_strategies))
        width = 0.35
        axes[1].bar(
            [i - width / 2 for i in x], [forward_counts.get(s, 0) for s in all_strategies], width, label='Forward'
        )
        axes[1].bar(
            [i + width / 2 for i in x], [langevin_counts.get(s, 0) for s in all_strategies], width, label='Langevin'
        )
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(all_strategies, rotation=45, ha='right', fontsize=7)
        axes[1].set_ylabel('Count')
        axes[1].set_title('Strategy Distribution: Forward vs Langevin')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(out_dir / 'forward_vs_langevin.png', dpi=150)
        plt.close()

    print(f'Results saved to {out_dir}')


EXPERIMENT_RUNNERS = {
    'trajectory-decomposition': run_trajectory_decomposition,
    'strategy-progression': run_strategy_progression,
    'attention-specialization': run_attention_specialization,
    'causal-ablation': run_causal_ablation,
    'forward-vs-langevin': run_forward_vs_langevin,
}


def main() -> None:
    """Entry point."""
    args = parse_args()
    if not args.checkpoint.exists():
        print(f'Error: checkpoint not found at {args.checkpoint}', file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, device)

    print(f'Loading {args.num_puzzles} test puzzles...')
    loader = load_test_data(args.num_puzzles, args.batch_size)

    print(f'Running experiment: {args.experiment}')
    runner = EXPERIMENT_RUNNERS[args.experiment]
    runner(args, model, loader)

    print('Done.')


if __name__ == '__main__':
    main()
