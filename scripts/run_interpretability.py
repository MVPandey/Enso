#!/usr/bin/env python3
"""
Run mechanistic interpretability experiments on trained ENSO EBM checkpoints.

Usage:
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment trajectory-decomposition
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment strategy-progression
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment attention-specialization
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment causal-ablation
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment forward-vs-langevin
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment lr-sweep
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment difficulty-stratification
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment z-dependence
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment energy-landscape
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment latent-trajectory
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment multi-chain-divergence
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --experiment probability-curves

"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ebm.dataset import SudokuDataset, SudokuTorchDataset
from ebm.interpretability import (
    STRATEGY_DIFFICULTY,
    AttentionAnalyzer,
    EnergyEvaluator,
    HeadAblator,
    MetricsComputer,
    StrategyDetector,
    TrajectoryAnalyzer,
    TrajectoryRecorder,
)
from ebm.interpretability.difficulty import DifficultyBucket, classify_difficulty
from ebm.model.jepa import InferenceConfig, SudokuJEPA
from ebm.utils.config import ArchitectureConfig

EXPERIMENTS = [
    'trajectory-decomposition',
    'strategy-progression',
    'attention-specialization',
    'causal-ablation',
    'forward-vs-langevin',
    'lr-sweep',
    'difficulty-stratification',
    'z-dependence',
    'energy-landscape',
    'latent-trajectory',
    'multi-chain-divergence',
    'probability-curves',
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
    # Phase 3 CLI arguments
    parser.add_argument(
        '--learning-rates',
        type=str,
        default='0.001,0.005,0.01,0.05,0.1,0.5',
        help='Comma-separated learning rates for lr-sweep experiment.',
    )
    parser.add_argument('--n-chains', type=int, default=16, help='Number of chains for multi-chain experiment.')
    parser.add_argument(
        '--n-interpolation-points',
        type=int,
        default=50,
        help='Number of interpolation points for energy-landscape experiment.',
    )
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
    """
    Load test puzzles from the Kaggle dataset.

    Uses the tail of the full dataset to avoid overlap with training data
    (training uses the head via split_dataset).
    """
    ds = SudokuDataset()
    df = ds.load_all()
    test_df = df.tail(num_puzzles).reset_index(drop=True)
    test_ds = SudokuTorchDataset(test_df)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Phase 2 experiments (original)
# ---------------------------------------------------------------------------


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

        # Accumulate attention from all captured steps, averaging per layer
        attn_accum: dict[str, list[torch.Tensor]] = defaultdict(list)
        for step in traj.steps:
            for attn_dict in (step.encoder_attention, step.decoder_attention):
                if attn_dict:
                    for key, tensor in attn_dict.items():
                        attn_accum[key].append(tensor)
        all_attention: dict[str, torch.Tensor] = {k: torch.stack(v).mean(dim=0) for k, v in attn_accum.items()}

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
    attn_accum: dict[str, list[torch.Tensor]] = defaultdict(list)
    for step in traj.steps:
        for attn_dict in (step.encoder_attention, step.decoder_attention):
            if attn_dict:
                for key, tensor in attn_dict.items():
                    attn_accum[key].append(tensor)
    all_attention: dict[str, torch.Tensor] = {k: torch.stack(v).mean(dim=0) for k, v in attn_accum.items()}

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
    """
    Experiment 5: Compare forward pass (oracle) vs Langevin dynamics.

    The forward pass encodes the solution through the target encoder to get
    the ideal latent z, then decodes — an oracle upper bound that reveals
    the decoder's capacity but tells us nothing about *how* the model reasons.

    Langevin dynamics starts from random z and iteratively refines. The key
    finding is not whether Langevin matches forward-pass accuracy, but that
    it recovers most of the performance through an interpretable strategy
    progression: easy cells (naked singles) lock in first, followed by
    increasingly complex strategies as the latent state is optimized.
    """
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

        # Forward pass: encode solution to ideal z, then decode
        with torch.no_grad():
            z_context = model.context_encoder(puzzle)
            z_target = model.target_encoder(solution.permute(0, 3, 1, 2))
            z_ideal = torch.nn.functional.normalize(model.z_encoder(z_target), dim=-1)
            forward_logits = model.decoder(z_context, z_ideal, puzzle, mask)
            forward_board = forward_logits.argmax(dim=-1) + 1  # (B, 9, 9)

        # Langevin dynamics from random z
        traj = recorder.record(puzzle, mask, solution, inference_cfg)

        # Build clue board for strategy classification
        clue_board = puzzle[:, 1:].permute(0, 2, 3, 1).argmax(dim=-1) + 1  # (B, 9, 9)
        clue_board = clue_board * mask.long()

        for b in range(puzzle.shape[0]):
            sol_board = solution[b].argmax(dim=-1) + 1

            events_fwd = detector.classify(clue_board[b], forward_board[b], mask[b])
            forward_correct = int(((forward_board[b] == sol_board) & (mask[b] == 0)).sum().item())

            board_final = traj.final_board[b]
            events_final = detector.classify(clue_board[b], board_final, mask[b])
            langevin_correct = int(((board_final == sol_board) & (mask[b] == 0)).sum().item())

            total_cells = int((mask[b] == 0).sum().item())

            for e in events_fwd:
                label = e.strategy.value if e.strategy else 'unknown'
                forward_counts[label] = forward_counts.get(label, 0) + 1
            for e in events_final:
                label = e.strategy.value if e.strategy else 'unknown'
                langevin_counts[label] = langevin_counts.get(label, 0) + 1

            # Per-strategy breakdown for this puzzle's Langevin result
            langevin_strat_detail: dict[str, int] = {}
            for e in events_final:
                label = e.strategy.value if e.strategy else 'unknown'
                langevin_strat_detail[label] = langevin_strat_detail.get(label, 0) + 1

            comparison.append(
                {
                    'forward_accuracy': forward_correct / max(total_cells, 1),
                    'langevin_accuracy': langevin_correct / max(total_cells, 1),
                    'recovery_ratio': langevin_correct / max(forward_correct, 1),
                    'forward_events': len(events_fwd),
                    'langevin_events': len(events_final),
                    'langevin_strategy_breakdown': langevin_strat_detail,
                }
            )

        print(f'  Batch {batch_idx}: processed')

    # Compute summary statistics
    if comparison:
        mean_fwd = sum(c['forward_accuracy'] for c in comparison) / len(comparison)
        mean_lang = sum(c['langevin_accuracy'] for c in comparison) / len(comparison)
        mean_recovery = sum(c['recovery_ratio'] for c in comparison) / len(comparison)
        print(f'\n  Forward pass (oracle) accuracy: {mean_fwd:.3f}')
        print(f'  Langevin dynamics accuracy:     {mean_lang:.3f}')
        print(f'  Recovery ratio:                 {mean_recovery:.3f}')
        print(f'  Langevin strategy distribution: {langevin_counts}')

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
        axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[0].set_xlabel('Oracle (Ideal z) Accuracy')
        axes[0].set_ylabel('Langevin (Random z) Accuracy')
        axes[0].set_title('Performance Recovery: Langevin vs Oracle')

        all_strategies = sorted(set(list(forward_counts) + list(langevin_counts)))
        x = range(len(all_strategies))
        width = 0.35
        axes[1].bar(
            [i - width / 2 for i in x], [forward_counts.get(s, 0) for s in all_strategies], width, label='Oracle'
        )
        axes[1].bar(
            [i + width / 2 for i in x], [langevin_counts.get(s, 0) for s in all_strategies], width, label='Langevin'
        )
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(all_strategies, rotation=45, ha='right', fontsize=7)
        axes[1].set_ylabel('Count')
        axes[1].set_title('Strategy Distribution: Oracle vs Langevin Reasoning')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(out_dir / 'forward_vs_langevin.png', dpi=150)
        plt.close()

    print(f'Results saved to {out_dir}')


# ---------------------------------------------------------------------------
# Phase 3 experiments — Crystallization Validation
# ---------------------------------------------------------------------------


def run_lr_sweep(args: argparse.Namespace, model: SudokuJEPA, loader: DataLoader) -> None:
    """
    Experiment A: Learning Rate Sweep.

    If crystallization happens at the same *energy level* regardless of lr,
    the landscape has a genuine phase transition. If lock-in always occurs at
    step 1-2, it's a step-size artifact.
    """
    out_dir = args.output_dir / 'lr-sweep'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    lr_values = [float(x) for x in args.learning_rates.split(',')]
    recorder = TrajectoryRecorder(model, record_attention=False)
    metrics_computer = MetricsComputer()

    # Collect first batch
    batch = next(iter(loader))
    puzzle = batch['puzzle'].to(device)
    solution = batch['solution'].to(device)
    mask = batch['mask'].to(device)

    lr_results: list[dict] = []

    for lr in lr_values:
        n_steps = max(50, int(50 * 0.01 / lr))
        noise = 0.005 * math.sqrt(lr / 0.01)
        step_stride = max(1, n_steps // 100)

        cfg = InferenceConfig(n_steps=n_steps, n_chains=1, lr=lr, noise_scale=noise)
        traj = recorder.record(puzzle, mask, solution, cfg, step_stride=step_stride)

        lock_in_steps_per_lr: list[int] = []
        energy_at_lock_in: list[float] = []

        for b in range(puzzle.shape[0]):
            lock_ins = metrics_computer.compute_lock_in(traj, solution, b)
            for ev in lock_ins:
                lock_in_steps_per_lr.append(ev.lock_in_step)
                # Find energy at lock-in step from stored snapshots
                for snap in traj.steps:
                    if snap.step >= ev.lock_in_step:
                        energy_at_lock_in.append(float(snap.energy[b].item()))
                        break

        # Energy trajectory (mean across batch)
        energy_traj = []
        for snap in traj.steps:
            energy_traj.append({'step': snap.step, 'mean_energy': float(snap.energy.mean().item())})

        lr_results.append(
            {
                'lr': lr,
                'n_steps': n_steps,
                'noise_scale': noise,
                'step_stride': step_stride,
                'mean_lock_in_step': float(np.mean(lock_in_steps_per_lr)) if lock_in_steps_per_lr else None,
                'mean_energy_at_lock_in': float(np.mean(energy_at_lock_in)) if energy_at_lock_in else None,
                'n_lock_ins': len(lock_in_steps_per_lr),
                'energy_trajectory': energy_traj,
            }
        )
        print(f'  lr={lr:.4f}: {len(lock_in_steps_per_lr)} lock-ins, mean step={lr_results[-1]["mean_lock_in_step"]}')

    with open(out_dir / 'lr_sweep_results.json', 'w') as f:
        json.dump(lr_results, f, indent=2)

    # Plot (a) energy at lock-in vs lr, (b) lock-in step vs lr, (c) energy trajectories
    _fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    lrs_with_data = [r for r in lr_results if r['mean_energy_at_lock_in'] is not None]
    if lrs_with_data:
        axes[0].plot([r['lr'] for r in lrs_with_data], [r['mean_energy_at_lock_in'] for r in lrs_with_data], 'o-')
        axes[0].set_xlabel('Learning Rate')
        axes[0].set_ylabel('Mean Energy at Lock-in')
        axes[0].set_title('Energy at Lock-in vs LR (constant = genuine)')
        axes[0].set_xscale('log')

        axes[1].plot([r['lr'] for r in lrs_with_data], [r['mean_lock_in_step'] for r in lrs_with_data], 's-')
        axes[1].set_xlabel('Learning Rate')
        axes[1].set_ylabel('Mean Lock-in Step')
        axes[1].set_title('Lock-in Step vs LR')
        axes[1].set_xscale('log')

    for r in lr_results:
        steps = [e['step'] for e in r['energy_trajectory']]
        energies = [e['mean_energy'] for e in r['energy_trajectory']]
        axes[2].plot(steps, energies, label=f'lr={r["lr"]}', alpha=0.7)
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Mean Energy')
    axes[2].set_title('Energy Trajectories by LR')
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / 'lr_sweep.png', dpi=150)
    plt.close()
    print(f'Results saved to {out_dir}')


def run_difficulty_stratification(args: argparse.Namespace, model: SudokuJEPA, loader: DataLoader) -> None:
    """
    Experiment B: Difficulty Stratification.

    See if harder puzzles (fewer givens) show genuinely slower convergence.
    """
    out_dir = args.output_dir / 'difficulty-stratification'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    inference_cfg = InferenceConfig(n_steps=args.langevin_steps, n_chains=1)
    recorder = TrajectoryRecorder(model, record_attention=False)
    metrics_computer = MetricsComputer()

    bucket_lock_ins: dict[str, list[int]] = {b.value: [] for b in DifficultyBucket}
    bucket_energies: dict[str, list[list[float]]] = {b.value: [] for b in DifficultyBucket}
    bucket_accuracies: dict[str, list[float]] = {b.value: [] for b in DifficultyBucket}

    for batch_idx, batch in enumerate(loader):
        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)

        traj = recorder.record(puzzle, mask, solution, inference_cfg)

        for b in range(puzzle.shape[0]):
            n_givens = int(mask[b].sum().item())
            bucket = classify_difficulty(n_givens)

            lock_ins = metrics_computer.compute_lock_in(traj, solution, b)
            for ev in lock_ins:
                bucket_lock_ins[bucket.value].append(ev.lock_in_step)

            # Energy trajectory for this sample
            sample_energy = [float(snap.energy[b].item()) for snap in traj.steps]
            bucket_energies[bucket.value].append(sample_energy)

            # Accuracy
            sol_board = solution[b].argmax(dim=-1) + 1
            final_board = traj.final_board[b]
            non_clue = mask[b] == 0
            total = int(non_clue.sum().item())
            correct = int(((final_board == sol_board) & non_clue).sum().item())
            bucket_accuracies[bucket.value].append(correct / max(total, 1))

        print(f'  Batch {batch_idx}: processed')

    # Save results
    results = {}
    for bk in DifficultyBucket:
        bv = bk.value
        results[bv] = {
            'n_lock_ins': len(bucket_lock_ins[bv]),
            'mean_lock_in_step': float(np.mean(bucket_lock_ins[bv])) if bucket_lock_ins[bv] else None,
            'mean_accuracy': float(np.mean(bucket_accuracies[bv])) if bucket_accuracies[bv] else None,
            'n_puzzles': len(bucket_accuracies[bv]),
        }
    with open(out_dir / 'stratification_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot (a) lock-in distribution per bucket, (b) mean energy per bucket, (c) accuracy
    _fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for bk in DifficultyBucket:
        bv = bk.value
        if bucket_lock_ins[bv]:
            axes[0].hist(bucket_lock_ins[bv], bins=15, alpha=0.5, label=bv, edgecolor='black')
    axes[0].set_xlabel('Lock-in Step')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Lock-in Step Distribution by Difficulty')
    axes[0].legend()

    for bk in DifficultyBucket:
        bv = bk.value
        if bucket_energies[bv]:
            mean_curve = np.mean(bucket_energies[bv], axis=0)
            axes[1].plot(range(len(mean_curve)), mean_curve, label=bv)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Mean Energy')
    axes[1].set_title('Mean Energy Trajectory by Difficulty')
    axes[1].legend()

    buckets_with_data = [bk.value for bk in DifficultyBucket if bucket_accuracies[bk.value]]
    if buckets_with_data:
        means = [float(np.mean(bucket_accuracies[bv])) for bv in buckets_with_data]
        axes[2].bar(range(len(buckets_with_data)), means)
        axes[2].set_xticks(range(len(buckets_with_data)))
        axes[2].set_xticklabels(buckets_with_data)
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Accuracy by Difficulty')

    plt.tight_layout()
    plt.savefig(out_dir / 'difficulty_stratification.png', dpi=150)
    plt.close()
    print(f'Results saved to {out_dir}')


def run_z_dependence(args: argparse.Namespace, model: SudokuJEPA, loader: DataLoader) -> None:
    """
    Experiment C: Z-Dependence Test.

    If z=0 accuracy ~ Langevin accuracy, the decoder ignores z and Langevin
    is cosmetic.
    """
    out_dir = args.output_dir / 'z-dependence'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    inference_cfg = InferenceConfig(n_steps=args.langevin_steps, n_chains=1)
    evaluator = EnergyEvaluator(model)
    recorder = TrajectoryRecorder(model, record_attention=False)

    conditions = ['z=zeros', 'z=randn', 'langevin', 'oracle']
    accuracy_per_cond: dict[str, list[float]] = {c: [] for c in conditions}

    for batch_idx, batch in enumerate(loader):
        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)

        z_context = evaluator.compute_z_context(puzzle)
        batch_size = puzzle.shape[0]

        # (a) z = zeros
        z_zeros = torch.zeros(batch_size, model.arch_cfg.d_latent, device=device)
        profile_zeros = evaluator.evaluate(puzzle, mask, z_zeros, z_context=z_context)
        board_zeros = profile_zeros.logits.argmax(dim=-1) + 1

        # (b) z = randn (no dynamics)
        z_rand = torch.randn(batch_size, model.arch_cfg.d_latent, device=device)
        profile_rand = evaluator.evaluate(puzzle, mask, z_rand, z_context=z_context)
        board_rand = profile_rand.logits.argmax(dim=-1) + 1

        # (c) Langevin 50 steps
        traj = recorder.record(puzzle, mask, solution, inference_cfg)
        board_langevin = traj.final_board

        # (d) oracle z
        z_oracle = evaluator.compute_oracle_z(solution)
        profile_oracle = evaluator.evaluate(puzzle, mask, z_oracle, z_context=z_context)
        board_oracle = profile_oracle.logits.argmax(dim=-1) + 1

        boards = {'z=zeros': board_zeros, 'z=randn': board_rand, 'langevin': board_langevin, 'oracle': board_oracle}

        for b in range(batch_size):
            sol_board = solution[b].argmax(dim=-1) + 1
            non_clue = mask[b] == 0
            total = int(non_clue.sum().item())
            for cond, board in boards.items():
                correct = int(((board[b] == sol_board) & non_clue).sum().item())
                accuracy_per_cond[cond].append(correct / max(total, 1))

        print(f'  Batch {batch_idx}: processed')

    # Save results
    summary = {
        c: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)} for c, v in accuracy_per_cond.items()
    }
    with open(out_dir / 'z_dependence_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot box plot of 4 conditions
    _fig, ax = plt.subplots(figsize=(8, 5))
    data = [accuracy_per_cond[c] for c in conditions]
    bp = ax.boxplot(data, labels=conditions, patch_artist=True)
    colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Accuracy')
    ax.set_title('Z-Dependence: Does the Decoder Actually Use z?')
    plt.tight_layout()
    plt.savefig(out_dir / 'z_dependence.png', dpi=150)
    plt.close()

    for c in conditions:
        print(f'  {c}: accuracy = {summary[c]["mean"]:.3f} +/- {summary[c]["std"]:.3f}')
    print(f'Results saved to {out_dir}')


def run_energy_landscape(args: argparse.Namespace, model: SudokuJEPA, loader: DataLoader) -> None:
    """
    Experiment D: Energy Cross-Section.

    Directly visualize energy landscape geometry along the random-to-oracle
    z path. Monotonic decrease = funnel topology.
    """
    out_dir = args.output_dir / 'energy-landscape'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    evaluator = EnergyEvaluator(model)
    n_points = args.n_interpolation_points

    batch = next(iter(loader))
    puzzle = batch['puzzle'].to(device)
    solution = batch['solution'].to(device)
    mask = batch['mask'].to(device)

    z_context = evaluator.compute_z_context(puzzle)
    z_oracle = evaluator.compute_oracle_z(solution)
    z_start = torch.randn_like(z_oracle)

    profiles = evaluator.interpolate(puzzle, mask, z_start, z_oracle, n_points=n_points, z_context=z_context)

    # Collect data
    alphas = np.linspace(0, 1, n_points)
    mean_energy = [float(p.energy.mean().item()) for p in profiles]
    mean_sc = [float(p.self_consistency.mean().item()) for p in profiles]
    mean_cp = [float(p.constraint_penalty.mean().item()) for p in profiles]

    # Compute accuracy at each alpha
    accuracies = []
    for p in profiles:
        board = p.logits.argmax(dim=-1) + 1
        sol_board = solution.argmax(dim=-1) + 1
        non_clue = mask == 0
        total = non_clue.sum().item()
        correct = ((board == sol_board) & non_clue).sum().item()
        accuracies.append(correct / max(total, 1))

    results = {
        'alphas': alphas.tolist(),
        'mean_energy': mean_energy,
        'mean_self_consistency': mean_sc,
        'mean_constraint_penalty': mean_cp,
        'accuracy': accuracies,
    }
    with open(out_dir / 'landscape_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot (a) energy vs alpha, (b) components vs alpha, (c) accuracy vs alpha
    _fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(alphas, mean_energy, 'b-', linewidth=2)
    axes[0].set_xlabel('alpha (0=random, 1=oracle)')
    axes[0].set_ylabel('Mean Energy')
    axes[0].set_title('Energy Cross-Section: Random -> Oracle')

    axes[1].plot(alphas, mean_sc, 'r-', label='Self-Consistency')
    axes[1].plot(alphas, mean_cp, 'g-', label='Constraint Penalty')
    axes[1].set_xlabel('alpha')
    axes[1].set_ylabel('Component Value')
    axes[1].set_title('Energy Components Along Path')
    axes[1].legend()

    axes[2].plot(alphas, accuracies, 'k-', linewidth=2)
    axes[2].set_xlabel('alpha')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Accuracy Along Random -> Oracle Path')

    plt.tight_layout()
    plt.savefig(out_dir / 'energy_landscape.png', dpi=150)
    plt.close()
    print(f'Results saved to {out_dir}')


def run_latent_trajectory(args: argparse.Namespace, model: SudokuJEPA, loader: DataLoader) -> None:
    """
    Experiment E: Latent Space Convergence.

    Track whether Langevin z converges toward the oracle z* or a different
    minimum, using L2 distance and cosine similarity.
    """
    out_dir = args.output_dir / 'latent-trajectory'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    inference_cfg = InferenceConfig(n_steps=args.langevin_steps, n_chains=1)
    evaluator = EnergyEvaluator(model)
    recorder = TrajectoryRecorder(model, record_attention=False)

    all_l2: list[list[float]] = []
    all_cosine: list[list[float]] = []

    for batch_idx, batch in enumerate(loader):
        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)

        z_oracle = evaluator.compute_oracle_z(solution)
        traj = recorder.record(puzzle, mask, solution, inference_cfg)

        for b in range(puzzle.shape[0]):
            l2_series = []
            cos_series = []
            z_star = z_oracle[b].unsqueeze(0)
            for snap in traj.steps:
                z_t = snap.z[b].unsqueeze(0)
                l2_series.append(float((z_t - z_star).norm().item()))
                cos_series.append(float(F.cosine_similarity(z_t, z_star).item()))
            all_l2.append(l2_series)
            all_cosine.append(cos_series)

        print(f'  Batch {batch_idx}: processed')

    # Save results
    steps_list = [snap.step for snap in traj.steps] if all_l2 else []
    results = {
        'steps': steps_list,
        'mean_l2': np.mean(all_l2, axis=0).tolist() if all_l2 else [],
        'mean_cosine': np.mean(all_cosine, axis=0).tolist() if all_cosine else [],
        'n_samples': len(all_l2),
    }
    with open(out_dir / 'latent_trajectory_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot (a) L2 distance vs step, (b) cosine similarity vs step
    if all_l2:
        _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        mean_l2 = np.mean(all_l2, axis=0)
        std_l2 = np.std(all_l2, axis=0)
        axes[0].plot(steps_list, mean_l2, 'b-')
        axes[0].fill_between(steps_list, mean_l2 - std_l2, mean_l2 + std_l2, alpha=0.2)
        axes[0].set_xlabel('Langevin Step')
        axes[0].set_ylabel('L2 Distance to Oracle z*')
        axes[0].set_title('Convergence: L2 Distance to z*')

        mean_cos = np.mean(all_cosine, axis=0)
        std_cos = np.std(all_cosine, axis=0)
        axes[1].plot(steps_list, mean_cos, 'r-')
        axes[1].fill_between(steps_list, mean_cos - std_cos, mean_cos + std_cos, alpha=0.2)
        axes[1].set_xlabel('Langevin Step')
        axes[1].set_ylabel('Cosine Similarity to Oracle z*')
        axes[1].set_title('Convergence: Cosine Similarity to z*')

        plt.tight_layout()
        plt.savefig(out_dir / 'latent_trajectory.png', dpi=150)
        plt.close()

    print(f'Results saved to {out_dir}')


def run_multi_chain_divergence(args: argparse.Namespace, model: SudokuJEPA, loader: DataLoader) -> None:
    """
    Experiment F: Multi-Chain Basin Analysis.

    Single basin (all chains converge to same z) vs multiple basins
    (different z, same board) vs disagreement.
    """
    out_dir = args.output_dir / 'multi-chain-divergence'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    n_chains = args.n_chains
    inference_cfg = InferenceConfig(n_steps=args.langevin_steps, n_chains=1)
    recorder = TrajectoryRecorder(model, record_attention=False)

    # Cap batch size for memory
    effective_batch = min(args.batch_size, 4)

    all_z_convergence: list[float] = []
    all_board_agreement: list[float] = []
    classifications: list[str] = []

    for batch_idx, batch in enumerate(loader):
        puzzle = batch['puzzle'][:effective_batch].to(device)
        solution = batch['solution'][:effective_batch].to(device)
        mask = batch['mask'][:effective_batch].to(device)

        result = recorder.record_multi_chain(puzzle, mask, solution, inference_cfg, n_chains=n_chains)

        for b in range(puzzle.shape[0]):
            # Pairwise cosine similarity of final z
            z_chains = result.final_z[b]  # (n_chains, d_latent)
            z_norm = F.normalize(z_chains, dim=-1)
            cos_sim = z_norm @ z_norm.T  # (n_chains, n_chains)
            # Mean off-diagonal cosine similarity
            mask_offdiag = ~torch.eye(n_chains, dtype=torch.bool, device=device)
            mean_z_conv = float(cos_sim[mask_offdiag].mean().item())

            # Pairwise board agreement
            boards = result.final_boards[b]  # (n_chains, 9, 9)
            agreements = []
            for i in range(n_chains):
                for j in range(i + 1, n_chains):
                    agree = float((boards[i] == boards[j]).float().mean().item())
                    agreements.append(agree)
            mean_board_agree = float(np.mean(agreements))

            all_z_convergence.append(mean_z_conv)
            all_board_agreement.append(mean_board_agree)

            # Classify basin topology
            z_conv_single_threshold = 0.95
            z_conv_multi_threshold = 0.5
            board_agree_threshold = 0.99
            if mean_z_conv > z_conv_single_threshold:
                classifications.append('single_basin')
            elif mean_z_conv < z_conv_multi_threshold and mean_board_agree > board_agree_threshold:
                classifications.append('multiple_basins')
            else:
                classifications.append('disagreement')

        print(f'  Batch {batch_idx}: processed')

    # Save results
    results_data = {
        'z_convergence': all_z_convergence,
        'board_agreement': all_board_agreement,
        'classifications': classifications,
        'classification_counts': {
            'single_basin': classifications.count('single_basin'),
            'multiple_basins': classifications.count('multiple_basins'),
            'disagreement': classifications.count('disagreement'),
        },
    }
    with open(out_dir / 'multi_chain_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    # Plot (a) histogram of z-convergence, (b) scatter z-conv vs board-agree
    if all_z_convergence:
        _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(all_z_convergence, bins=20, edgecolor='black')
        axes[0].set_xlabel('Mean Pairwise Cosine Similarity (z)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Z-Space Convergence Across Chains')

        colors = {'single_basin': 'green', 'multiple_basins': 'blue', 'disagreement': 'red'}
        for cls in ['single_basin', 'multiple_basins', 'disagreement']:
            idxs = [i for i, c in enumerate(classifications) if c == cls]
            if idxs:
                axes[1].scatter(
                    [all_z_convergence[i] for i in idxs],
                    [all_board_agreement[i] for i in idxs],
                    c=colors[cls],
                    label=cls,
                    alpha=0.6,
                )
        axes[1].set_xlabel('Z-Space Convergence')
        axes[1].set_ylabel('Board Agreement')
        axes[1].set_title('Multi-Chain Basin Analysis')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(out_dir / 'multi_chain_divergence.png', dpi=150)
        plt.close()

    print(f'Results saved to {out_dir}')


def _extract_sample_curves(
    traj: 'Trajectory',  # noqa: F821
    solution: torch.Tensor,
    mask: torch.Tensor,
    clue_board: torch.Tensor,
    b: int,
    detector: StrategyDetector,
    metrics_computer: MetricsComputer,
    strategy_curves: dict[str, list[list[float]]],
    all_curves_with_lock_in: list[tuple[list[float], int]],
) -> None:
    """Extract P(correct) curves for one sample and append to accumulators."""
    sol_board = solution[b].argmax(dim=-1) + 1
    final_board = traj.final_board[b]

    events = detector.classify(clue_board[b], final_board, mask[b])
    cell_strategy: dict[tuple[int, int], str] = {}
    for e in events:
        cell_strategy[(e.row, e.col)] = e.strategy.value if e.strategy else 'unknown'

    lock_ins = metrics_computer.compute_lock_in(traj, solution, b)
    cell_lock_in: dict[tuple[int, int], int] = {}
    for ev in lock_ins:
        cell_lock_in[(ev.row, ev.col)] = ev.lock_in_step

    for r in range(9):
        for c in range(9):
            if mask[b, r, c] > 0:
                continue
            correct_digit = int(sol_board[r, c].item())
            d_idx = correct_digit - 1
            curve = [float(snap.probs[b, r, c, d_idx].item()) for snap in traj.steps]
            strat = cell_strategy.get((r, c), 'unknown')
            strategy_curves.setdefault(strat, []).append(curve)
            if (r, c) in cell_lock_in:
                all_curves_with_lock_in.append((curve, cell_lock_in[(r, c)]))


def run_probability_curves(args: argparse.Namespace, model: SudokuJEPA, loader: DataLoader) -> None:
    """
    Experiment G: Cell Confidence Trajectories.

    Even if argmax doesn't change, confidence curves may correlate with
    strategy difficulty.
    """
    out_dir = args.output_dir / 'probability-curves'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    inference_cfg = InferenceConfig(n_steps=args.langevin_steps, n_chains=1)
    recorder = TrajectoryRecorder(model, record_attention=False)
    detector = StrategyDetector()
    metrics_computer = MetricsComputer()

    strategy_curves: dict[str, list[list[float]]] = {}
    all_curves_with_lock_in: list[tuple[list[float], int]] = []

    for batch_idx, batch in enumerate(loader):
        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)

        traj = recorder.record(puzzle, mask, solution, inference_cfg)

        clue_board = puzzle[:, 1:].permute(0, 2, 3, 1).argmax(dim=-1) + 1
        clue_board = clue_board * mask.long()

        for b in range(puzzle.shape[0]):
            _extract_sample_curves(
                traj,
                solution,
                mask,
                clue_board,
                b,
                detector,
                metrics_computer,
                strategy_curves,
                all_curves_with_lock_in,
            )

        print(f'  Batch {batch_idx}: processed')

    # Save results
    summary: dict[str, dict] = {}
    for strat, curves in strategy_curves.items():
        mean_curve = np.mean(curves, axis=0).tolist()
        summary[strat] = {'mean_curve': mean_curve, 'n_cells': len(curves)}
    with open(out_dir / 'probability_curves.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot (a) mean P(correct) curve per strategy, (b) heatmap sorted by lock-in
    _fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for strat, curves in sorted(strategy_curves.items()):
        if curves:
            mean_curve = np.mean(curves, axis=0)
            steps_range = list(range(len(mean_curve)))
            axes[0].plot(steps_range, mean_curve, label=f'{strat} (n={len(curves)})')
    axes[0].set_xlabel('Langevin Step')
    axes[0].set_ylabel('P(correct digit)')
    axes[0].set_title('Mean Confidence Curve by Strategy')
    axes[0].legend(fontsize=7)

    # Heatmap of all curves sorted by lock-in step
    if all_curves_with_lock_in:
        sorted_curves = sorted(all_curves_with_lock_in, key=lambda x: x[1])
        curve_matrix = np.array([c[0] for c in sorted_curves])
        # Limit to at most max_display curves for visualization
        max_display = 200
        if len(curve_matrix) > max_display:
            indices = np.linspace(0, len(curve_matrix) - 1, max_display, dtype=int)
            curve_matrix = curve_matrix[indices]
        im = axes[1].imshow(curve_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        axes[1].set_xlabel('Langevin Step')
        axes[1].set_ylabel('Cell (sorted by lock-in step)')
        axes[1].set_title('Confidence Heatmap (sorted by lock-in)')
        plt.colorbar(im, ax=axes[1], label='P(correct)')

    plt.tight_layout()
    plt.savefig(out_dir / 'probability_curves.png', dpi=150)
    plt.close()
    print(f'Results saved to {out_dir}')


# ---------------------------------------------------------------------------
# Registry & main
# ---------------------------------------------------------------------------

EXPERIMENT_RUNNERS = {
    'trajectory-decomposition': run_trajectory_decomposition,
    'strategy-progression': run_strategy_progression,
    'attention-specialization': run_attention_specialization,
    'causal-ablation': run_causal_ablation,
    'forward-vs-langevin': run_forward_vs_langevin,
    'lr-sweep': run_lr_sweep,
    'difficulty-stratification': run_difficulty_stratification,
    'z-dependence': run_z_dependence,
    'energy-landscape': run_energy_landscape,
    'latent-trajectory': run_latent_trajectory,
    'multi-chain-divergence': run_multi_chain_divergence,
    'probability-curves': run_probability_curves,
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
