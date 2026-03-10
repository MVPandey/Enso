"""Solve Sudoku puzzles using a trained SudokuJEPA model."""

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ebm.model.jepa import InferenceConfig, SudokuJEPA


def solve_batch(model: SudokuJEPA, puzzle: Tensor, mask: Tensor, cfg: InferenceConfig) -> Tensor:
    """
    Solve a batch of puzzles using iterative inference.

    Args:
        model: Trained SudokuJEPA model.
        puzzle: (B, 10, 9, 9) one-hot encoded puzzles.
        mask: (B, 9, 9) binary mask.
        cfg: Inference parameters.

    Returns:
        (B, 9, 9) integer solution grids with digits 1-9.

    """
    model.eval()
    return model.solve(puzzle, mask, cfg)


@torch.no_grad()
def solve_dataset(
    model: SudokuJEPA,
    loader: DataLoader,
    cfg: InferenceConfig,
    device: torch.device | None = None,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """
    Solve all puzzles in a DataLoader.

    Args:
        model: Trained SudokuJEPA model.
        loader: DataLoader yielding batches with 'puzzle', 'solution', 'mask'.
        cfg: Inference parameters.
        device: Device to run on. Defaults to model's device.

    Returns:
        Tuple of (predictions, solutions, masks) as lists of tensors.

    """
    if not device:
        device = next(model.parameters()).device

    model.eval()
    all_preds = []
    all_solutions = []
    all_masks = []

    for batch in loader:
        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)

        pred = model.solve(puzzle, mask, cfg)
        all_preds.append(pred.cpu())
        all_solutions.append(solution.cpu())
        all_masks.append(mask.cpu())

    return all_preds, all_solutions, all_masks
