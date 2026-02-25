#!/usr/bin/env python3
"""
Run mechanistic interpretability analysis on trained ENSO EBM checkpoints.

Usage:
    uv run python scripts/run_interpretability.py --checkpoint checkpoints/best.pt --num-puzzles 10

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from ebm.interpretability import StrategyDetector, TrajectoryAnalyzer, TrajectoryRecorder
from ebm.model.jepa import InferenceConfig, SudokuJEPA
from ebm.utils.config import ArchitectureConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='ENSO Mechanistic Interpretability Experiment')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to model checkpoint.')
    parser.add_argument('--num-puzzles', type=int, default=10, help='Number of puzzles to analyze.')
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


def make_random_puzzles(num_puzzles: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random puzzle tensors for smoke-testing (no real Sudoku structure)."""
    puzzle = torch.zeros(num_puzzles, 10, 9, 9, device=device)
    puzzle[:, 0] = 1.0  # all cells empty
    solution = torch.zeros(num_puzzles, 9, 9, 9, device=device)
    solution[:, :, :, 0] = 1.0  # dummy solution
    mask = torch.zeros(num_puzzles, 9, 9, device=device)
    return puzzle, solution, mask


def run_analysis(args: argparse.Namespace) -> None:
    """Run the full interpretability analysis pipeline."""
    device = torch.device(args.device)
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, device)

    inference_cfg = InferenceConfig(n_steps=args.langevin_steps, n_chains=1)

    print(f'Recording trajectories for {args.num_puzzles} puzzles...')
    recorder = TrajectoryRecorder(model, record_attention=True, attention_stride=args.attention_stride)

    puzzle, solution, mask = make_random_puzzles(args.num_puzzles, device)
    trajectory = recorder.record(puzzle, mask, solution, inference_cfg)

    print(f'Recorded {len(trajectory.steps)} steps')

    detector = StrategyDetector()
    analyzer = TrajectoryAnalyzer(strategy_detector=detector)

    all_results = []
    for b in range(args.num_puzzles):
        result = analyzer.analyze_trajectory(trajectory, batch_idx=b)
        all_results.append(result)
        print(f'  Puzzle {b}: {len(result.events)} events, strategies: {result.strategy_counts}')

    # Aggregate statistics
    total_events = sum(len(r.events) for r in all_results)
    total_counts: dict[str, int] = {}
    for r in all_results:
        for strategy, count in r.strategy_counts.items():
            total_counts[strategy] = total_counts.get(strategy, 0) + count

    # Check attention capture
    attn_steps = sum(1 for s in trajectory.steps if s.encoder_attention is not None)
    attn_shapes = {}
    for s in trajectory.steps:
        if s.encoder_attention:
            for key, tensor in s.encoder_attention.items():
                attn_shapes[key] = list(tensor.shape)
            break

    summary = {
        'num_puzzles': args.num_puzzles,
        'langevin_steps': args.langevin_steps,
        'attention_stride': args.attention_stride,
        'total_cell_events': total_events,
        'strategy_counts': total_counts,
        'attention_steps_captured': attn_steps,
        'attention_shapes': attn_shapes,
    }

    print('\n--- Summary ---')
    print(f'Total cell-fill events: {total_events}')
    print(f'Strategy distribution: {total_counts}')
    print(f'Attention maps captured at {attn_steps}/{len(trajectory.steps)} steps')
    if attn_shapes:
        print(f'Attention tensor shapes: {attn_shapes}')

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary saved to {summary_path}')

    # Save per-puzzle events
    events_path = args.output_dir / 'events.json'
    events_data = []
    for b, result in enumerate(all_results):
        for event in result.events:
            events_data.append({
                'puzzle_idx': b,
                'step': event.step,
                'row': event.row,
                'col': event.col,
                'digit': event.digit,
                'strategy': event.strategy.value if event.strategy else 'unknown',
                'confidence': event.confidence,
            })
    with open(events_path, 'w') as f:
        json.dump(events_data, f, indent=2)
    print(f'Events saved to {events_path}')


def main() -> None:
    """Entry point."""
    args = parse_args()
    if not args.checkpoint.exists():
        print(f'Error: checkpoint not found at {args.checkpoint}', file=sys.stderr)
        sys.exit(1)
    run_analysis(args)


if __name__ == '__main__':
    main()
