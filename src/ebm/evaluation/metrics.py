"""Evaluation metrics for Sudoku solutions."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ebm.model.constraints import GROUP_INDICES


@dataclass
class EvalMetrics:
    """Container for evaluation results."""

    cell_accuracy: float
    puzzle_accuracy: float
    constraint_satisfaction: float
    n_puzzles: int


def evaluate(
    preds: list[Tensor],
    solutions: list[Tensor],
    masks: list[Tensor],
) -> EvalMetrics:
    """
    Compute evaluation metrics across batches.

    Args:
        preds: List of (B, 9, 9) predicted digit grids (1-9).
        solutions: List of (B, 9, 9, 9) one-hot solutions.
        masks: List of (B, 9, 9) binary masks.

    Returns:
        EvalMetrics with cell accuracy, puzzle accuracy, and constraint satisfaction.

    """
    total_empty = 0
    correct_empty = 0
    total_puzzles = 0
    solved_puzzles = 0
    total_groups = 0
    satisfied_groups = 0

    for pred, solution, mask in zip(preds, solutions, masks, strict=True):
        target_digits = solution.argmax(dim=-1) + 1
        empty = mask == 0

        total_empty += empty.sum().item()
        correct_empty += (pred[empty] == target_digits[empty]).sum().item()

        fully_correct = (pred == target_digits).all(dim=(1, 2))
        total_puzzles += pred.shape[0]
        solved_puzzles += fully_correct.sum().item()

        sat, total = _constraint_satisfaction(pred)
        satisfied_groups += sat
        total_groups += total

    return EvalMetrics(
        cell_accuracy=correct_empty / total_empty if total_empty > 0 else 1.0,
        puzzle_accuracy=solved_puzzles / max(total_puzzles, 1),
        constraint_satisfaction=satisfied_groups / max(total_groups, 1),
        n_puzzles=total_puzzles,
    )


def _constraint_satisfaction(pred: Tensor) -> tuple[int, int]:
    """
    Count how many constraint groups are satisfied.

    A group is satisfied if all 9 digits 1-9 appear exactly once.

    Args:
        pred: (B, 9, 9) integer grid with digits 1-9.

    Returns:
        Tuple of (satisfied_count, total_count).

    """
    batch_size = pred.shape[0]
    flat = pred.reshape(batch_size, 81)

    group_idx = GROUP_INDICES.to(pred.device)
    group_vals = flat[:, group_idx]

    total = batch_size * 27
    satisfied = 0
    for group_id in range(27):
        group = group_vals[:, group_id]
        sorted_vals, _ = group.sort(dim=-1)
        expected = torch.arange(1, 10, device=pred.device).unsqueeze(0).expand(batch_size, -1)
        satisfied += int((sorted_vals == expected).all(dim=-1).sum().item())

    return satisfied, total
