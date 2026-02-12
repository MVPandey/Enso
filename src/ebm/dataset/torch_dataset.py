"""PyTorch Dataset wrapper for Sudoku puzzles and solutions."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SudokuTorchDataset(Dataset):
    """
    PyTorch Dataset that converts 81-char Sudoku strings to tensors.

    Encodings:
        puzzle: (10, 9, 9) one-hot — channel 0 = empty, channels 1-9 = digits.
        solution: (9, 9, 9) one-hot — channels 0-8 for digits 1-9.
        mask: (9, 9) binary — 1 = given clue, 0 = empty cell.

    Pre-parses all strings into int8 numpy arrays at init time for fast
    __getitem__ access.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize from a DataFrame with 'puzzle' and 'solution' columns.

        Args:
            df: DataFrame containing 81-character 'puzzle' and 'solution' strings.

        """
        self.puzzles = self._parse_strings(df['puzzle'].values)
        self.solutions = self._parse_strings(df['solution'].values)

    @staticmethod
    def _parse_strings(strings: np.ndarray) -> np.ndarray:
        """Convert array of 81-char digit strings to (N, 9, 9) int8 array."""
        count = len(strings)
        result = np.zeros((count, 9, 9), dtype=np.int8)
        for idx, string in enumerate(strings):
            digits = np.frombuffer(string.encode('ascii'), dtype=np.uint8) - ord('0')
            result[idx] = digits.reshape(9, 9)
        return result

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.puzzles)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Return encoded puzzle, solution, and mask tensors.

        Args:
            index: Sample index.

        Returns:
            Dict with keys 'puzzle' (10,9,9), 'solution' (9,9,9), 'mask' (9,9).

        """
        puzzle_grid = self.puzzles[index]
        solution_grid = self.solutions[index]

        puzzle_tensor = np.zeros((10, 9, 9), dtype=np.float32)
        empty_mask = puzzle_grid == 0
        puzzle_tensor[0] = empty_mask
        for digit in range(1, 10):
            puzzle_tensor[digit] = puzzle_grid == digit

        solution_tensor = np.zeros((9, 9, 9), dtype=np.float32)
        for digit in range(1, 10):
            solution_tensor[:, :, digit - 1] = solution_grid == digit

        mask = (~empty_mask).astype(np.float32)

        return {
            'puzzle': torch.from_numpy(puzzle_tensor),
            'solution': torch.from_numpy(solution_tensor),
            'mask': torch.from_numpy(mask),
        }
