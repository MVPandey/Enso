import pandas as pd
import torch

from ebm.dataset.torch_dataset import SudokuTorchDataset


def _make_sample_df(n: int = 5) -> pd.DataFrame:
    """Create a tiny DataFrame with valid 81-char puzzle/solution strings."""
    # A known valid Sudoku pair
    puzzle = '004300209005009001070060043006002087190007400050083000600000105003508690042910300'
    solution = '864371259325849761971265843436192587198657432257483916689734125713528694542916378'
    return pd.DataFrame({'puzzle': [puzzle] * n, 'solution': [solution] * n})


def test_dataset_length():
    df = _make_sample_df(10)
    ds = SudokuTorchDataset(df)
    assert len(ds) == 10


def test_puzzle_shape_and_dtype():
    ds = SudokuTorchDataset(_make_sample_df())
    sample = ds[0]
    assert sample['puzzle'].shape == (10, 9, 9)
    assert sample['puzzle'].dtype == torch.float32


def test_solution_shape_and_dtype():
    ds = SudokuTorchDataset(_make_sample_df())
    sample = ds[0]
    assert sample['solution'].shape == (9, 9, 9)
    assert sample['solution'].dtype == torch.float32


def test_mask_shape_and_dtype():
    ds = SudokuTorchDataset(_make_sample_df())
    sample = ds[0]
    assert sample['mask'].shape == (9, 9)
    assert sample['mask'].dtype == torch.float32


def test_puzzle_onehot_valid():
    """Each cell should have exactly one channel active."""
    ds = SudokuTorchDataset(_make_sample_df())
    puzzle = ds[0]['puzzle']  # (10, 9, 9)
    sums = puzzle.sum(dim=0)  # (9, 9)
    assert torch.allclose(sums, torch.ones(9, 9))


def test_solution_onehot_valid():
    """Each cell should have exactly one digit channel active."""
    ds = SudokuTorchDataset(_make_sample_df())
    solution = ds[0]['solution']  # (9, 9, 9) = (row, col, digit)
    sums = solution.sum(dim=-1)  # (9, 9)
    assert torch.allclose(sums, torch.ones(9, 9))


def test_mask_matches_puzzle():
    """Mask should be 1 where puzzle has a clue (non-zero), 0 where empty."""
    ds = SudokuTorchDataset(_make_sample_df())
    sample = ds[0]
    mask = sample['mask']
    empty_channel = sample['puzzle'][0]  # channel 0 = empty indicator
    # mask = 1 means given clue, empty_channel = 1 means empty — they should be complementary
    assert torch.allclose(mask + empty_channel, torch.ones(9, 9))


def test_empty_cell_encoding():
    """Empty cells (0 in puzzle string) should only have channel 0 active."""
    puzzle_str = '0' * 81
    solution_str = '1' * 81  # dummy solution
    df = pd.DataFrame({'puzzle': [puzzle_str], 'solution': [solution_str]})
    ds = SudokuTorchDataset(df)
    sample = ds[0]
    # All cells empty — channel 0 should be all 1s, channels 1-9 all 0s
    assert sample['puzzle'][0].sum() == 81
    assert sample['puzzle'][1:].sum() == 0
