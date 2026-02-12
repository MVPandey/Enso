"""Differentiable Sudoku constraint penalties for inference guidance."""

import torch
from torch import Tensor

_GROUPS: list[list[int]] = []
_GROUPS.extend([list(range(row * 9, row * 9 + 9)) for row in range(9)])
_GROUPS.extend([list(range(col, 81, 9)) for col in range(9)])
_GROUPS.extend([
    [(box_r * 3 + dr) * 9 + box_c * 3 + dc for dr in range(3) for dc in range(3)]
    for box_r in range(3) for box_c in range(3)
])

GROUP_INDICES = torch.tensor(_GROUPS, dtype=torch.long)


def constraint_penalty(probs: Tensor) -> Tensor:
    """
    Compute differentiable Sudoku constraint violation penalty.

    For each constraint group (9 rows + 9 columns + 9 boxes), each digit
    should appear exactly once — meaning the sum of probabilities for each
    digit across the 9 cells in a group should be 1.0. The penalty is the
    squared deviation from 1.0, summed over all groups and digits.

    Args:
        probs: (B, 9, 9, 9) softmax probabilities over digits for each cell.

    Returns:
        (B,) per-sample constraint penalty.

    """
    batch_size = probs.shape[0]
    flat_probs = probs.reshape(batch_size, 81, 9)

    group_idx = GROUP_INDICES.to(flat_probs.device)
    group_probs = flat_probs[:, group_idx]
    digit_sums = group_probs.sum(dim=2)

    return ((digit_sums - 1.0) ** 2).sum(dim=(1, 2))
