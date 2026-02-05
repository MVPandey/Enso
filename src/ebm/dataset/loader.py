"""Dataset loader for the Sudoku Kaggle dataset."""

import random

import kagglehub
import pandas as pd

from ebm.utils.config import config

CSV_NAME = 'sudoku.csv'


class SudokuDataset:
    """
    Provides access to the Sudoku Kaggle dataset.

    Downloads the dataset on first use if not already present locally.

    """

    def __init__(self) -> None:
        """Initialize the dataset."""
        self._data_dir = config.data_dir
        self._csv_path = self._data_dir / CSV_NAME
        self._ensure_downloaded()

    def _ensure_downloaded(self) -> None:
        """Download the dataset from Kaggle if not already present."""
        if not self._csv_path.exists():
            kagglehub.dataset_download('rohanrao/sudoku', output_dir=str(self._data_dir))

    def load_all(self) -> pd.DataFrame:
        """Load and return the full dataset."""
        return pd.read_csv(self._csv_path)

    def load_head(self, k: int = 100) -> pd.DataFrame:
        """Load only the first *k* rows."""
        return pd.read_csv(self._csv_path, nrows=k)

    def load_fraction(self, frac: float = 0.1, seed: int = 42) -> pd.DataFrame:
        """
        Load a random fraction of the dataset without reading the entire file.

        Args:
            frac: Fraction of the dataset to return (0.0, 1.0].
            seed: Random seed for reproducibility.

        """
        rng = random.Random(seed)
        return pd.read_csv(self._csv_path, skiprows=lambda i: i > 0 and rng.random() > frac)
