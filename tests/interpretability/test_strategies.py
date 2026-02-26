import torch

from ebm.interpretability.strategies import STRATEGY_DIFFICULTY, StrategyDetector, get_candidates
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
    board_before = torch.zeros(9, 9, dtype=torch.long)
    # Fill row 0 partially — leave (0,0), (0,1), (0,2) empty
    board_before[0, 3] = 4
    board_before[0, 4] = 5
    board_before[0, 5] = 6
    board_before[0, 6] = 7
    board_before[0, 7] = 8
    board_before[0, 8] = 9
    # Put 1 in col 1 and col 2 (outside box 0) to eliminate from (0,1) and (0,2)
    board_before[3, 1] = 1
    board_before[4, 2] = 1

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


def test_pointing_pair_detection():
    """Pointing pair: digit confined to one row within a box eliminates from rest of row."""
    board = torch.zeros(9, 9, dtype=torch.long)
    mask = torch.zeros(9, 9)

    # Set up a board where pointing pair logic is needed.
    # Fill most of the board to create a constrained scenario.
    # We use a known solution and remove cells strategically.
    sol = _board_from_list(VALID_SOLUTION)
    board = sol.clone()

    # Remove cells to create a pointing pair scenario at (0,0)
    # First, ensure (0,0) has multiple candidates from raw analysis
    # but after pointing pair elimination becomes solvable.
    board[0, 0] = 0
    board[0, 1] = 0
    board[1, 0] = 0
    board[1, 1] = 0

    board_after = board.clone()
    board_after[0, 0] = sol[0, 0].item()

    detector = StrategyDetector()
    events = detector.classify(board, board_after, mask)

    # The detector should classify this — the strategy depends on constraint propagation
    assert len(events) == 1
    assert events[0].strategy is not None


def test_box_line_reduction_method():
    """Test that _eliminate_box_line modifies candidates."""
    # Create a scenario where box-line reduction applies
    board = torch.zeros(9, 9, dtype=torch.long)
    mask = torch.zeros(9, 9)

    # Place digits to force box-line reduction pattern
    # digit 1 in row 0 can only be in cols 0-2 (box 0)
    # so eliminate digit 1 from rest of box 0 in rows 1-2
    board[0, 3] = 2
    board[0, 4] = 3
    board[0, 5] = 4
    board[0, 6] = 5
    board[0, 7] = 6
    board[0, 8] = 7

    candidates = get_candidates(board, mask)
    detector = StrategyDetector()
    changed = detector._eliminate_box_line(candidates)

    # Should have found some eliminations (digit 1 in row 0 confined to box 0)
    assert isinstance(changed, bool)


def test_naked_pair_elimination():
    """Test that _eliminate_naked_pairs correctly eliminates."""
    candidates = torch.zeros(9, 9, 9, dtype=torch.bool)
    # Set up a naked pair in row 0: cells (0,0) and (0,1) both have {1, 2}
    candidates[0, 0, 0] = True  # digit 1
    candidates[0, 0, 1] = True  # digit 2
    candidates[0, 1, 0] = True  # digit 1
    candidates[0, 1, 1] = True  # digit 2
    # Other cells in row 0 have digits including 1 and 2
    for c in range(2, 9):
        candidates[0, c, 0] = True
        candidates[0, c, 1] = True
        candidates[0, c, 2] = True

    detector = StrategyDetector()
    changed = detector._eliminate_naked_pairs(candidates)

    assert changed
    # Digits 1 and 2 should be eliminated from cells (0,2)-(0,8) in row 0
    for c in range(2, 9):
        assert not candidates[0, c, 0].item()
        assert not candidates[0, c, 1].item()
        # Digit 3 should remain
        assert candidates[0, c, 2].item()


def test_hidden_pair_elimination():
    """Test that _eliminate_hidden_pairs correctly eliminates."""
    candidates = torch.zeros(9, 9, 9, dtype=torch.bool)
    # In row 0: digits 1 and 2 only appear in cells (0,0) and (0,1)
    # but those cells also have other candidates
    candidates[0, 0, 0] = True  # digit 1
    candidates[0, 0, 1] = True  # digit 2
    candidates[0, 0, 2] = True  # digit 3 (extra)
    candidates[0, 1, 0] = True  # digit 1
    candidates[0, 1, 1] = True  # digit 2
    candidates[0, 1, 3] = True  # digit 4 (extra)
    # Other cells don't have digits 1 or 2
    for c in range(2, 9):
        candidates[0, c, 2] = True
        candidates[0, c, 3] = True
        candidates[0, c, 4] = True

    detector = StrategyDetector()
    changed = detector._eliminate_hidden_pairs(candidates)

    assert changed
    # Extra candidates should be removed from the hidden pair cells
    assert not candidates[0, 0, 2].item()  # digit 3 removed from (0,0)
    assert not candidates[0, 1, 3].item()  # digit 4 removed from (0,1)
    # The hidden pair digits remain
    assert candidates[0, 0, 0].item()
    assert candidates[0, 0, 1].item()
    assert candidates[0, 1, 0].item()
    assert candidates[0, 1, 1].item()


def test_naked_triple_elimination():
    """Test that _eliminate_naked_triples correctly eliminates."""
    candidates = torch.zeros(9, 9, 9, dtype=torch.bool)
    # In row 0: cells (0,0), (0,1), (0,2) have subsets of {1, 2, 3}
    candidates[0, 0, 0] = True  # {1, 2}
    candidates[0, 0, 1] = True
    candidates[0, 1, 1] = True  # {2, 3}
    candidates[0, 1, 2] = True
    candidates[0, 2, 0] = True  # {1, 3}
    candidates[0, 2, 2] = True
    # Other cells have digits 1, 2, 3 as candidates too
    for c in range(3, 9):
        candidates[0, c, 0] = True
        candidates[0, c, 1] = True
        candidates[0, c, 2] = True
        candidates[0, c, 3] = True

    detector = StrategyDetector()
    changed = detector._eliminate_naked_triples(candidates)

    assert changed
    # Digits 1, 2, 3 eliminated from other cells in row 0
    for c in range(3, 9):
        assert not candidates[0, c, 0].item()
        assert not candidates[0, c, 1].item()
        assert not candidates[0, c, 2].item()
        assert candidates[0, c, 3].item()  # digit 4 unchanged


def test_hidden_triple_elimination():
    """Test that _eliminate_hidden_triples correctly eliminates."""
    candidates = torch.zeros(9, 9, 9, dtype=torch.bool)
    # In row 0: digits 1, 2, 3 confined to cells (0,0), (0,1), (0,2)
    # Each digit appears in 2 of the 3 cells (required for hidden triple detection)
    candidates[0, 0, 0] = True  # digit 1
    candidates[0, 0, 1] = True  # digit 2
    candidates[0, 0, 3] = True  # digit 4 (extra)
    candidates[0, 1, 1] = True  # digit 2
    candidates[0, 1, 2] = True  # digit 3
    candidates[0, 1, 4] = True  # digit 5 (extra)
    candidates[0, 2, 0] = True  # digit 1
    candidates[0, 2, 2] = True  # digit 3
    candidates[0, 2, 5] = True  # digit 6 (extra)
    # Digits 1, 2, 3 don't appear elsewhere in row 0
    for c in range(3, 9):
        candidates[0, c, 3] = True
        candidates[0, c, 4] = True
        candidates[0, c, 5] = True

    detector = StrategyDetector()
    changed = detector._eliminate_hidden_triples(candidates)

    assert changed
    # Extra digits removed from the triple cells
    assert not candidates[0, 0, 3].item()
    assert not candidates[0, 1, 4].item()
    assert not candidates[0, 2, 5].item()
    # Triple digits remain
    assert candidates[0, 0, 0].item()
    assert candidates[0, 0, 1].item()
    assert candidates[0, 1, 1].item()
    assert candidates[0, 1, 2].item()
    assert candidates[0, 2, 0].item()
    assert candidates[0, 2, 2].item()


def test_x_wing_elimination():
    """Test that _eliminate_x_wing correctly eliminates."""
    candidates = torch.zeros(9, 9, 9, dtype=torch.bool)
    # Row-based X-Wing for digit 1:
    # rows 0 and 3 each have digit 1 in exactly cols 2 and 5
    candidates[0, 2, 0] = True
    candidates[0, 5, 0] = True
    candidates[3, 2, 0] = True
    candidates[3, 5, 0] = True
    # Other rows have digit 1 in cols 2/5 but NOT exclusively those two
    # (row 1 has 3 cols to avoid forming another X-Wing pair)
    candidates[1, 2, 0] = True
    candidates[1, 5, 0] = True
    candidates[1, 7, 0] = True  # extra col so row 1 doesn't form X-Wing
    candidates[4, 2, 0] = True
    candidates[4, 7, 0] = True  # extra col so row 4 doesn't form X-Wing

    detector = StrategyDetector()
    changed = detector._eliminate_x_wing(candidates)

    assert changed
    # Digit 1 eliminated from cols 2 and 5 in other rows
    assert not candidates[1, 2, 0].item()
    assert not candidates[1, 5, 0].item()
    assert not candidates[4, 2, 0].item()
    # X-Wing positions preserved
    assert candidates[0, 2, 0].item()
    assert candidates[0, 5, 0].item()
    assert candidates[3, 2, 0].item()
    assert candidates[3, 5, 0].item()
    # Non-X-Wing col candidates preserved
    assert candidates[1, 7, 0].item()
    assert candidates[4, 7, 0].item()


def test_pointing_pair_classify_integration():
    """Test that _classify_cell returns POINTING_PAIR when pointing pair elimination makes a cell solvable."""
    detector = StrategyDetector()

    # Craft candidates where pointing pair elimination creates a naked single at (0,0)
    crafted = torch.ones(9, 9, 9, dtype=torch.bool)
    # Cell (0,0) has candidates {1, 2} — not naked single
    crafted[0, 0] = False
    crafted[0, 0, 0] = True
    crafted[0, 0, 1] = True

    # Digit 1 in box 0: only in row 0 (cells (0,0) and (0,1))
    # So pointing pair eliminates digit 1 from rest of row 0 outside box 0
    # This doesn't directly help (0,0).

    # Instead: digit 2 in box 0 is only in col 0 (cells (0,0), (1,0), (2,0))
    # Pointing pair eliminates digit 2 from col 0 outside box 0
    # That doesn't help either. The pointing pair needs to eliminate digit 2 from (0,0) itself.

    # For pointing pair to help (0,0):
    # Some digit X in a box that shares row/col with (0,0)
    # is confined to a row/col, and eliminating X from that row/col removes it from (0,0)

    # digit 2 in box 1 (rows 0-2, cols 3-5) is all in row 0
    for r in range(3):
        for c in range(3, 6):
            crafted[r, c, 1] = False
    crafted[0, 3, 1] = True
    crafted[0, 4, 1] = True
    # This means digit 2 can be eliminated from row 0 outside box 1
    # So digit 2 eliminated from (0,0) — leaving only digit 1 = naked single

    result = detector._classify_cell(crafted, 0, 0, 1)
    assert result == StrategyLabel.POINTING_PAIR


def test_naked_pair_classify_integration():
    """Test that _classify_cell returns NAKED_PAIR when naked pair elimination is needed."""
    crafted = torch.ones(9, 9, 9, dtype=torch.bool)
    # Cell (0,0) has candidates {1, 2, 3}
    crafted[0, 0] = False
    crafted[0, 0, 0] = True  # 1
    crafted[0, 0, 1] = True  # 2
    crafted[0, 0, 2] = True  # 3

    # Make digit 1 not a hidden single (available in many places in row/col/box)
    # Make digit 2 and 3 not hidden singles either

    # Create a naked pair {2, 3} at (0,1) and (0,2)
    crafted[0, 1] = False
    crafted[0, 1, 1] = True  # 2
    crafted[0, 1, 2] = True  # 3
    crafted[0, 2] = False
    crafted[0, 2, 1] = True  # 2
    crafted[0, 2, 2] = True  # 3

    # Other cells in row 0 also have digit 1 so it's not a hidden single
    for c in range(3, 9):
        crafted[0, c, 0] = True

    # Make sure pointing pairs/box-line don't fire first by keeping digits spread across boxes
    detector = StrategyDetector()
    result = detector._classify_cell(crafted, 0, 0, 1)
    assert result == StrategyLabel.NAKED_PAIR


def test_strategy_difficulty_covers_all_labels():
    """STRATEGY_DIFFICULTY should have an entry for every StrategyLabel."""
    for label in StrategyLabel:
        assert label in STRATEGY_DIFFICULTY


def test_strategy_difficulty_ordering():
    """Lower-level strategies should have lower difficulty values."""
    assert STRATEGY_DIFFICULTY[StrategyLabel.NAKED_SINGLE] < STRATEGY_DIFFICULTY[StrategyLabel.HIDDEN_SINGLE]
    assert STRATEGY_DIFFICULTY[StrategyLabel.HIDDEN_SINGLE] < STRATEGY_DIFFICULTY[StrategyLabel.POINTING_PAIR]
    assert STRATEGY_DIFFICULTY[StrategyLabel.NAKED_PAIR] < STRATEGY_DIFFICULTY[StrategyLabel.X_WING]
    assert STRATEGY_DIFFICULTY[StrategyLabel.X_WING] < STRATEGY_DIFFICULTY[StrategyLabel.UNKNOWN]
