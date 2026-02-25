import torch

from ebm.interpretability.strategies import StrategyDetector, get_candidates
from ebm.interpretability.types import StrategyLabel

# A valid Sudoku solution (0-indexed digits for constructing boards, 1-indexed in grid)
VALID_SOLUTION = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 5, 6, 7, 8, 9, 1, 2, 3],
    [7, 8, 9, 1, 2, 3, 4, 5, 6],
    [2, 3, 1, 5, 6, 4, 8, 9, 7],
    [5, 6, 4, 8, 9, 7, 2, 3, 1],
    [8, 9, 7, 2, 3, 1, 5, 6, 4],
    [3, 1, 2, 6, 4, 5, 9, 7, 8],
    [6, 4, 5, 9, 7, 8, 3, 1, 2],
    [9, 7, 8, 3, 1, 2, 6, 4, 5],
]


def _board_from_list(grid):
    return torch.tensor(grid, dtype=torch.long)


def test_get_candidates_empty_board():
    board = torch.zeros(9, 9, dtype=torch.long)
    mask = torch.zeros(9, 9)
    candidates = get_candidates(board, mask)

    assert candidates.shape == (9, 9, 9)
    # All digits valid for all cells on an empty board
    assert candidates.all()


def test_get_candidates_filled_cell_has_no_candidates():
    board = torch.zeros(9, 9, dtype=torch.long)
    board[0, 0] = 5
    mask = torch.zeros(9, 9)
    candidates = get_candidates(board, mask)

    assert not candidates[0, 0].any()


def test_get_candidates_row_elimination():
    board = torch.zeros(9, 9, dtype=torch.long)
    board[0, 0] = 1
    mask = torch.zeros(9, 9)
    candidates = get_candidates(board, mask)

    # Digit 1 (index 0) should be eliminated from entire row 0
    for c in range(1, 9):
        assert not candidates[0, c, 0].item()


def test_get_candidates_col_elimination():
    board = torch.zeros(9, 9, dtype=torch.long)
    board[0, 0] = 1
    mask = torch.zeros(9, 9)
    candidates = get_candidates(board, mask)

    # Digit 1 (index 0) should be eliminated from entire col 0
    for r in range(1, 9):
        assert not candidates[r, 0, 0].item()


def test_get_candidates_box_elimination():
    board = torch.zeros(9, 9, dtype=torch.long)
    board[0, 0] = 1
    mask = torch.zeros(9, 9)
    candidates = get_candidates(board, mask)

    # Digit 1 should be eliminated from entire top-left box
    for r in range(3):
        for c in range(3):
            if r == 0 and c == 0:
                continue
            assert not candidates[r, c, 0].item()


def test_naked_single_detection():
    """When a cell has exactly one candidate, it's a Naked Single."""
    board = _board_from_list(VALID_SOLUTION)
    # Remove one cell to create a naked single scenario
    # Remove (0,0) which is 1 — all other digits in row/col/box are filled
    board_before = board.clone()
    board_before[0, 0] = 0
    mask = torch.zeros(9, 9)

    board_after = board.clone()

    detector = StrategyDetector()
    events = detector.classify(board_before, board_after, mask)

    assert len(events) == 1
    assert events[0].row == 0
    assert events[0].col == 0
    assert events[0].digit == 1
    assert events[0].strategy == StrategyLabel.NAKED_SINGLE


def test_hidden_single_detection():
    """When a digit can only go in one cell within a group, it's a Hidden Single."""
    # Construct a board where cell (0,0) has multiple candidates but digit 1
    # can only go in (0,0) within row 0 (hidden single in row).
    board_before = torch.zeros(9, 9, dtype=torch.long)
    # Fill row 0 partially — leave (0,0), (0,1), (0,2) empty
    board_before[0, 3] = 4
    board_before[0, 4] = 5
    board_before[0, 5] = 6
    board_before[0, 6] = 7
    board_before[0, 7] = 8
    board_before[0, 8] = 9
    # (0,0), (0,1), (0,2) all have candidates {1, 2, 3} in row 0.
    # Put digit 1 in col 1 and col 2 to eliminate it from (0,1) and (0,2)
    board_before[1, 1] = 1
    board_before[2, 2] = 1
    # Now digit 1 can only go at (0,0) in row 0 — hidden single.
    # But (0,0) still has multiple candidates: {1, 2, 3} minus box eliminations.
    # Box (0-2, 0-2) has 1 at (1,1) — wait, that eliminates 1 from (0,0) too!
    # Fix: place digit 1 outside the box of (0,0).
    board_before[1, 1] = 0  # undo
    board_before[2, 2] = 0  # undo
    # Instead: put 1 in col 1 row 3+ and col 2 row 3+ (outside box 0)
    board_before[3, 1] = 1
    board_before[4, 2] = 1
    # Now digit 1 is eliminated from col 1 and col 2.
    # In row 0: (0,0) can have 1, (0,1) cannot (col 1), (0,2) cannot (col 2).
    # So digit 1 is a hidden single in row 0 at (0,0).
    # But (0,0) has candidates {1, 2, 3} → more than 1 → NOT naked single.

    board_after = board_before.clone()
    board_after[0, 0] = 1
    mask = torch.zeros(9, 9)

    detector = StrategyDetector()
    events = detector.classify(board_before, board_after, mask)

    assert len(events) == 1
    assert events[0].row == 0
    assert events[0].col == 0
    assert events[0].digit == 1
    assert events[0].strategy == StrategyLabel.HIDDEN_SINGLE


def test_unknown_strategy():
    """When neither naked single nor hidden single applies, classify as UNKNOWN."""
    board_before = torch.zeros(9, 9, dtype=torch.long)
    # Only one digit placed — many candidates remain everywhere
    board_before[4, 4] = 5

    board_after = board_before.clone()
    board_after[0, 0] = 1  # digit 1 at (0,0) — many other cells could hold 1
    mask = torch.zeros(9, 9)

    detector = StrategyDetector()
    events = detector.classify(board_before, board_after, mask)

    assert len(events) == 1
    assert events[0].strategy == StrategyLabel.UNKNOWN


def test_skips_clue_cells():
    board_before = torch.zeros(9, 9, dtype=torch.long)
    board_after = board_before.clone()
    board_after[0, 0] = 5
    mask = torch.zeros(9, 9)
    mask[0, 0] = 1  # (0,0) is a clue

    detector = StrategyDetector()
    events = detector.classify(board_before, board_after, mask)

    assert len(events) == 0


def test_detects_digit_change():
    board_before = torch.zeros(9, 9, dtype=torch.long)
    board_before[0, 0] = 5
    board_after = board_before.clone()
    board_after[0, 0] = 3  # changed from 5 to 3
    mask = torch.zeros(9, 9)

    detector = StrategyDetector()
    events = detector.classify(board_before, board_after, mask)

    assert len(events) == 1
    assert events[0].digit == 3


def test_skips_unchanged_cells():
    board_before = torch.zeros(9, 9, dtype=torch.long)
    board_before[0, 0] = 5
    board_after = board_before.clone()  # no change
    mask = torch.zeros(9, 9)

    detector = StrategyDetector()
    events = detector.classify(board_before, board_after, mask)

    assert len(events) == 0


def test_multiple_fills_at_once():
    board_before = _board_from_list(VALID_SOLUTION)
    board_before[0, 0] = 0
    board_before[0, 1] = 0
    board_after = _board_from_list(VALID_SOLUTION)
    mask = torch.zeros(9, 9)

    detector = StrategyDetector()
    events = detector.classify(board_before, board_after, mask)

    assert len(events) == 2
    positions = {(e.row, e.col) for e in events}
    assert (0, 0) in positions
    assert (0, 1) in positions


def test_confidence_from_probs():
    board_before = _board_from_list(VALID_SOLUTION)
    board_before[0, 0] = 0
    board_after = _board_from_list(VALID_SOLUTION)
    mask = torch.zeros(9, 9)

    probs = torch.ones(9, 9, 9) * 0.1
    probs[0, 0, 0] = 0.95  # digit 1 at (0,0) has high confidence

    detector = StrategyDetector()
    events = detector.classify(board_before, board_after, mask, probs=probs)

    assert len(events) == 1
    assert abs(events[0].confidence - 0.95) < 1e-5
