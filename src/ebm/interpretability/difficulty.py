"""Puzzle difficulty classification for stratified analysis."""

from __future__ import annotations

from enum import Enum

import pandas as pd

EASY_THRESHOLD = 30  # 30+ givens = easy
MEDIUM_THRESHOLD = 25  # 25-29 givens = medium; <25 = hard


class DifficultyBucket(Enum):
    """Difficulty categories based on number of given clues."""

    EASY = 'easy'  # 30+ givens
    MEDIUM = 'medium'  # 25-29 givens
    HARD = 'hard'  # <25 givens


def count_givens(puzzle_str: str) -> int:
    """
    Count the number of given (non-zero) digits in a puzzle string.

    Args:
        puzzle_str: 81-character string where '0' or '.' means empty.

    Returns:
        Number of given clues.

    """
    return sum(1 for ch in puzzle_str if ch not in ('0', '.'))


def classify_difficulty(n_givens: int) -> DifficultyBucket:
    """
    Classify puzzle difficulty from number of givens.

    Args:
        n_givens: Number of given clues.

    Returns:
        DifficultyBucket category.

    """
    if n_givens >= EASY_THRESHOLD:
        return DifficultyBucket.EASY
    if n_givens >= MEDIUM_THRESHOLD:
        return DifficultyBucket.MEDIUM
    return DifficultyBucket.HARD


def stratify_dataframe(df: pd.DataFrame) -> dict[DifficultyBucket, pd.DataFrame]:
    """
    Split a puzzle dataframe into difficulty buckets.

    Expects the dataframe to have a 'puzzle' column with 81-character strings.

    Args:
        df: DataFrame with a 'puzzle' column.

    Returns:
        Dict mapping DifficultyBucket to the corresponding subset DataFrame.

    """
    n_givens = df['puzzle'].apply(count_givens)
    buckets = n_givens.apply(classify_difficulty)

    result: dict[DifficultyBucket, pd.DataFrame] = {}
    for bucket in DifficultyBucket:
        subset = df[buckets == bucket]
        if len(subset) > 0:
            result[bucket] = subset.reset_index(drop=True)

    return result
