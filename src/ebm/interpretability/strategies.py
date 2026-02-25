"""Rule-based Sudoku strategy classifier for cell-fill events."""

from __future__ import annotations

import torch
from torch import Tensor

from ebm.interpretability.types import CellEvent, StrategyLabel


def get_candidates(board: Tensor, mask: Tensor) -> Tensor:
    """
    Compute valid candidates for each cell based on Sudoku rules.

    Args:
        board: (9, 9) integer grid where 0 means unfilled, 1-9 means digit.
        mask: (9, 9) binary mask, 1 = given clue.

    Returns:
        (9, 9, 9) boolean tensor where [r, c, d] is True if digit d+1 is a
        valid candidate for cell (r, c).

    """
    candidates = torch.ones(9, 9, 9, dtype=torch.bool, device=board.device)

    for r in range(9):
        for c in range(9):
            if board[r, c] > 0:
                # Filled cell has no candidates
                candidates[r, c] = False
                continue

            for d in range(9):
                digit = d + 1
                # Check row
                if (board[r] == digit).any():
                    candidates[r, c, d] = False
                    continue
                # Check column
                if (board[:, c] == digit).any():
                    candidates[r, c, d] = False
                    continue
                # Check 3x3 box
                box_r, box_c = 3 * (r // 3), 3 * (c // 3)
                if (board[box_r : box_r + 3, box_c : box_c + 3] == digit).any():
                    candidates[r, c, d] = False

    return candidates


class StrategyDetector:
    """
    Classify cell-fill events using human Sudoku solving strategies.

    Compares consecutive board snapshots and classifies each newly-filled cell
    as a Naked Single, Hidden Single, or Unknown.
    """

    def classify(
        self, board_before: Tensor, board_after: Tensor, mask: Tensor, probs: Tensor | None = None,
    ) -> list[CellEvent]:
        """
        Detect and classify cell changes between two board states.

        Handles both traditional boards (0=unfilled) and Langevin trajectory
        boards where all cells always have a digit (from argmax). Detects
        cells that changed digit or went from empty to filled.

        Args:
            board_before: (9, 9) integer board at step t.
            board_after: (9, 9) integer board at step t+1.
            mask: (9, 9) binary clue mask.
            probs: (9, 9, 9) optional softmax probabilities at step t+1
                for confidence values.

        Returns:
            List of CellEvent for each changed cell.

        """
        events: list[CellEvent] = []

        # Build a "reference board" for candidate computation: treat non-clue
        # cells that are identical in both boards as placed digits, and cells
        # that differ (or were 0) as empty for candidate analysis.
        ref_board = board_before.clone()
        changed_mask = board_before != board_after
        ref_board[changed_mask] = 0

        candidates = get_candidates(ref_board, mask)

        for r in range(9):
            for c in range(9):
                # Skip clue cells
                if mask[r, c] > 0:
                    continue
                old_digit = int(board_before[r, c].item())
                new_digit = int(board_after[r, c].item())
                # Skip unchanged cells or cells that became empty
                if new_digit in (old_digit, 0):
                    continue

                confidence = 0.0
                if probs is not None:
                    confidence = float(probs[r, c, new_digit - 1].item())

                strategy = self._classify_cell(candidates, r, c, new_digit)
                events.append(
                    CellEvent(step=0, row=r, col=c, digit=new_digit, strategy=strategy, confidence=confidence)
                )

        return events

    def _classify_cell(self, candidates: Tensor, row: int, col: int, digit: int) -> StrategyLabel:
        """Determine which strategy explains a cell fill."""
        # Naked Single: only one candidate in this cell
        cell_candidates = candidates[row, col]
        if cell_candidates.sum().item() == 1:
            return StrategyLabel.NAKED_SINGLE

        # Hidden Single: digit can only go in this cell within a group
        d = digit - 1
        # Check row
        if candidates[row, :, d].sum().item() == 1:
            return StrategyLabel.HIDDEN_SINGLE
        # Check column
        if candidates[:, col, d].sum().item() == 1:
            return StrategyLabel.HIDDEN_SINGLE
        # Check box
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        if candidates[box_r : box_r + 3, box_c : box_c + 3, d].sum().item() == 1:
            return StrategyLabel.HIDDEN_SINGLE

        return StrategyLabel.UNKNOWN
