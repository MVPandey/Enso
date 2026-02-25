"""
Rule-based Sudoku strategy classifier for cell-fill events.

Note on Langevin trajectory classification:
    During continuous Langevin dynamics all 81 cells always carry a digit
    (the argmax of the current distribution). The ``classify`` method detects
    cells whose digit *changed* between consecutive steps and builds a
    reference board by treating those changed cells as empty. This means the
    candidate analysis — and therefore the strategy label — depends on the
    specific ordering in which cells change across the trajectory. The same
    final board could receive different strategy labels depending on which
    cells happened to flip first. Strategy classifications during
    intermediate Langevin steps are therefore *approximate*; they reflect the
    minimum elimination technique needed given the trajectory's particular
    sequence of fills, not a canonical solver ordering.
"""

from __future__ import annotations

from itertools import combinations

import torch
from torch import Tensor

from ebm.interpretability.types import CellEvent, StrategyLabel

STRATEGY_DIFFICULTY: dict[StrategyLabel, int] = {
    StrategyLabel.NAKED_SINGLE: 1,
    StrategyLabel.HIDDEN_SINGLE: 2,
    StrategyLabel.POINTING_PAIR: 3,
    StrategyLabel.BOX_LINE_REDUCTION: 3,
    StrategyLabel.NAKED_PAIR: 4,
    StrategyLabel.HIDDEN_PAIR: 4,
    StrategyLabel.NAKED_TRIPLE: 4,
    StrategyLabel.HIDDEN_TRIPLE: 4,
    StrategyLabel.X_WING: 5,
    StrategyLabel.UNKNOWN: 6,
}


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

    Uses a hierarchical approach: for each cell fill, determines the minimum
    elimination technique needed to make the cell solvable by direct placement.
    """

    def classify(
        self,
        board_before: Tensor,
        board_after: Tensor,
        mask: Tensor,
        probs: Tensor | None = None,
    ) -> list[CellEvent]:
        """
        Detect and classify cell changes between two board states.

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

        ref_board = board_before.clone()
        changed_mask = board_before != board_after
        ref_board[changed_mask] = 0

        candidates = get_candidates(ref_board, mask)

        for r in range(9):
            for c in range(9):
                if mask[r, c] > 0:
                    continue
                old_digit = int(board_before[r, c].item())
                new_digit = int(board_after[r, c].item())
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
        """Determine which strategy explains a cell fill using hierarchical elimination."""
        # Level 1-2: check raw candidates
        if self._is_naked_single(candidates, row, col):
            return StrategyLabel.NAKED_SINGLE
        if self._is_hidden_single(candidates, row, col, digit):
            return StrategyLabel.HIDDEN_SINGLE

        # Level 3: try each L3 technique independently
        for eliminate_fn, label in [
            (self._eliminate_pointing_pairs, StrategyLabel.POINTING_PAIR),
            (self._eliminate_box_line, StrategyLabel.BOX_LINE_REDUCTION),
        ]:
            reduced = candidates.clone()
            if eliminate_fn(reduced):
                if self._is_naked_single(reduced, row, col) or self._is_hidden_single(reduced, row, col, digit):
                    return label

        # Level 4: cumulative L3 + L4 techniques
        reduced = candidates.clone()
        self._eliminate_pointing_pairs(reduced)
        self._eliminate_box_line(reduced)
        for eliminate_fn, label in [
            (self._eliminate_naked_pairs, StrategyLabel.NAKED_PAIR),
            (self._eliminate_hidden_pairs, StrategyLabel.HIDDEN_PAIR),
            (self._eliminate_naked_triples, StrategyLabel.NAKED_TRIPLE),
            (self._eliminate_hidden_triples, StrategyLabel.HIDDEN_TRIPLE),
        ]:
            r2 = reduced.clone()
            if eliminate_fn(r2):
                if self._is_naked_single(r2, row, col) or self._is_hidden_single(r2, row, col, digit):
                    return label

        # Level 5: cumulative L3 + L4 + L5
        reduced2 = reduced.clone()
        self._eliminate_naked_pairs(reduced2)
        self._eliminate_hidden_pairs(reduced2)
        if self._eliminate_x_wing(reduced2):
            if self._is_naked_single(reduced2, row, col) or self._is_hidden_single(reduced2, row, col, digit):
                return StrategyLabel.X_WING

        return StrategyLabel.UNKNOWN

    @staticmethod
    def _is_naked_single(candidates: Tensor, row: int, col: int) -> bool:
        """Check if the cell has exactly one candidate."""
        return candidates[row, col].sum().item() == 1

    @staticmethod
    def _is_hidden_single(candidates: Tensor, row: int, col: int, digit: int) -> bool:
        """Check if the digit is unique in any group containing this cell."""
        d = digit - 1
        # Check row
        if candidates[row, :, d].sum().item() == 1 and candidates[row, col, d]:
            return True
        # Check column
        if candidates[:, col, d].sum().item() == 1 and candidates[row, col, d]:
            return True
        # Check box
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        if candidates[box_r : box_r + 3, box_c : box_c + 3, d].sum().item() == 1 and candidates[row, col, d]:
            return True
        return False

    @staticmethod
    def _eliminate_pointing_pairs(candidates: Tensor) -> bool:
        """Eliminate candidates via pointing pairs/triples within boxes."""
        changed = False
        for box_r in range(3):
            for box_c in range(3):
                br, bc = box_r * 3, box_c * 3
                box_cands = candidates[br : br + 3, bc : bc + 3]
                for d in range(9):
                    positions = box_cands[:, :, d].nonzero(as_tuple=False)
                    if len(positions) < 2:
                        continue
                    rows = positions[:, 0]
                    cols = positions[:, 1]
                    # All in same row within the box
                    if (rows == rows[0]).all():
                        global_row = br + rows[0].item()
                        for c in range(9):
                            if c < bc or c >= bc + 3:
                                if candidates[global_row, c, d]:
                                    candidates[global_row, c, d] = False
                                    changed = True
                    # All in same column within the box
                    if (cols == cols[0]).all():
                        global_col = bc + cols[0].item()
                        for r in range(9):
                            if r < br or r >= br + 3:
                                if candidates[r, global_col, d]:
                                    candidates[r, global_col, d] = False
                                    changed = True
        return changed

    @staticmethod
    def _eliminate_box_line(candidates: Tensor) -> bool:
        """Eliminate candidates via box-line reduction."""
        changed = False
        for d in range(9):
            # Row-based: if all candidates for digit d in a row fall in one box
            for r in range(9):
                cols = candidates[r, :, d].nonzero(as_tuple=False).squeeze(-1)
                if len(cols) < 2:
                    continue
                box_cols = cols // 3
                if (box_cols == box_cols[0]).all():
                    box_c = box_cols[0].item() * 3
                    box_r = 3 * (r // 3)
                    for rr in range(box_r, box_r + 3):
                        if rr == r:
                            continue
                        for cc in range(box_c, box_c + 3):
                            if candidates[rr, cc, d]:
                                candidates[rr, cc, d] = False
                                changed = True
            # Column-based: if all candidates for digit d in a column fall in one box
            for c in range(9):
                rows = candidates[:, c, d].nonzero(as_tuple=False).squeeze(-1)
                if len(rows) < 2:
                    continue
                box_rows = rows // 3
                if (box_rows == box_rows[0]).all():
                    box_r = box_rows[0].item() * 3
                    box_c = 3 * (c // 3)
                    for rr in range(box_r, box_r + 3):
                        for cc in range(box_c, box_c + 3):
                            if cc == c:
                                continue
                            if candidates[rr, cc, d]:
                                candidates[rr, cc, d] = False
                                changed = True
        return changed

    @staticmethod
    def _eliminate_naked_pairs(candidates: Tensor) -> bool:
        """Eliminate candidates via naked pairs in constraint groups."""
        changed = False
        groups = _get_groups()
        for group in groups:
            # Find cells with exactly 2 candidates
            pair_cells = []
            for r, c in group:
                cands = candidates[r, c]
                if cands.sum().item() == 2:
                    pair_cells.append((r, c, tuple(cands.nonzero(as_tuple=False).squeeze(-1).tolist())))
            # Find two cells with same 2 candidates
            for i, j in combinations(range(len(pair_cells)), 2):
                if pair_cells[i][2] == pair_cells[j][2]:
                    d1, d2 = pair_cells[i][2]
                    pair_positions = {(pair_cells[i][0], pair_cells[i][1]), (pair_cells[j][0], pair_cells[j][1])}
                    for r, c in group:
                        if (r, c) not in pair_positions:
                            for d in (d1, d2):
                                if candidates[r, c, d]:
                                    candidates[r, c, d] = False
                                    changed = True
        return changed

    @staticmethod
    def _eliminate_hidden_pairs(candidates: Tensor) -> bool:
        """Eliminate candidates via hidden pairs in constraint groups."""
        changed = False
        groups = _get_groups()
        for group in groups:
            # For each pair of digits, check if they only appear in the same 2 cells
            digit_cells: dict[int, list[tuple[int, int]]] = {}
            for r, c in group:
                for d in range(9):
                    if candidates[r, c, d]:
                        digit_cells.setdefault(d, []).append((r, c))
            digits_with_2 = [d for d, cells in digit_cells.items() if len(cells) == 2]
            for d1, d2 in combinations(digits_with_2, 2):
                if set(digit_cells[d1]) == set(digit_cells[d2]):
                    # These 2 digits only in these 2 cells — remove all other candidates
                    for r, c in digit_cells[d1]:
                        for d in range(9):
                            if d not in (d1, d2) and candidates[r, c, d]:
                                candidates[r, c, d] = False
                                changed = True
        return changed

    @staticmethod
    def _eliminate_naked_triples(candidates: Tensor) -> bool:
        """Eliminate candidates via naked triples in constraint groups."""
        changed = False
        groups = _get_groups()
        for group in groups:
            # Find cells with 2 or 3 candidates
            eligible = []
            for r, c in group:
                n = candidates[r, c].sum().item()
                if 2 <= n <= 3:
                    cand_set = frozenset(candidates[r, c].nonzero(as_tuple=False).squeeze(-1).tolist())
                    eligible.append((r, c, cand_set))
            for combo in combinations(range(len(eligible)), 3):
                union = eligible[combo[0]][2] | eligible[combo[1]][2] | eligible[combo[2]][2]
                if len(union) == 3:
                    triple_positions = {(eligible[i][0], eligible[i][1]) for i in combo}
                    for r, c in group:
                        if (r, c) not in triple_positions:
                            for d in union:
                                if candidates[r, c, d]:
                                    candidates[r, c, d] = False
                                    changed = True
        return changed

    @staticmethod
    def _eliminate_hidden_triples(candidates: Tensor) -> bool:
        """Eliminate candidates via hidden triples in constraint groups."""
        changed = False
        groups = _get_groups()
        for group in groups:
            digit_cells: dict[int, set[tuple[int, int]]] = {}
            for r, c in group:
                for d in range(9):
                    if candidates[r, c, d]:
                        digit_cells.setdefault(d, set()).add((r, c))
            digits_with_2_or_3 = [d for d, cells in digit_cells.items() if 2 <= len(cells) <= 3]
            for combo in combinations(digits_with_2_or_3, 3):
                union_cells = digit_cells[combo[0]] | digit_cells[combo[1]] | digit_cells[combo[2]]
                if len(union_cells) == 3:
                    digit_set = set(combo)
                    for r, c in union_cells:
                        for d in range(9):
                            if d not in digit_set and candidates[r, c, d]:
                                candidates[r, c, d] = False
                                changed = True
        return changed

    @staticmethod
    def _eliminate_x_wing(candidates: Tensor) -> bool:
        """Eliminate candidates via X-Wing pattern."""
        changed = False
        for d in range(9):
            # Row-based X-Wing
            row_cols: dict[int, list[int]] = {}
            for r in range(9):
                cols = candidates[r, :, d].nonzero(as_tuple=False).squeeze(-1).tolist()
                if isinstance(cols, int):
                    cols = [cols]
                if len(cols) == 2:
                    row_cols[r] = cols
            for r1, r2 in combinations(row_cols, 2):
                if row_cols[r1] == row_cols[r2]:
                    c1, c2 = row_cols[r1]
                    for r in range(9):
                        if r not in (r1, r2):
                            if candidates[r, c1, d]:
                                candidates[r, c1, d] = False
                                changed = True
                            if candidates[r, c2, d]:
                                candidates[r, c2, d] = False
                                changed = True
            # Column-based X-Wing
            col_rows: dict[int, list[int]] = {}
            for c in range(9):
                rows = candidates[:, c, d].nonzero(as_tuple=False).squeeze(-1).tolist()
                if isinstance(rows, int):
                    rows = [rows]
                if len(rows) == 2:
                    col_rows[c] = rows
            for c1, c2 in combinations(col_rows, 2):
                if col_rows[c1] == col_rows[c2]:
                    r1, r2 = col_rows[c1]
                    for c in range(9):
                        if c not in (c1, c2):
                            if candidates[r1, c, d]:
                                candidates[r1, c, d] = False
                                changed = True
                            if candidates[r2, c, d]:
                                candidates[r2, c, d] = False
                                changed = True
        return changed


def _get_groups() -> list[list[tuple[int, int]]]:
    """Return all 27 constraint groups as lists of (row, col) tuples."""
    groups: list[list[tuple[int, int]]] = []
    # Rows
    for r in range(9):
        groups.append([(r, c) for c in range(9)])
    # Columns
    for c in range(9):
        groups.append([(r, c) for r in range(9)])
    # Boxes
    for box_r in range(3):
        for box_c in range(3):
            groups.append([(box_r * 3 + dr, box_c * 3 + dc) for dr in range(3) for dc in range(3)])
    return groups
