import pandas as pd

from ebm.interpretability.difficulty import DifficultyBucket, classify_difficulty, count_givens, stratify_dataframe


def test_count_givens_all_zeros():
    puzzle_str = '0' * 81
    assert count_givens(puzzle_str) == 0


def test_count_givens_all_filled():
    puzzle_str = '1' * 81
    assert count_givens(puzzle_str) == 81


def test_count_givens_mixed():
    puzzle_str = '123456789' + '0' * 72
    assert count_givens(puzzle_str) == 9


def test_count_givens_dots():
    puzzle_str = '12345....' + '0' * 72
    assert count_givens(puzzle_str) == 5


def test_classify_difficulty_easy():
    assert classify_difficulty(30) == DifficultyBucket.EASY
    assert classify_difficulty(40) == DifficultyBucket.EASY
    assert classify_difficulty(81) == DifficultyBucket.EASY


def test_classify_difficulty_medium():
    assert classify_difficulty(25) == DifficultyBucket.MEDIUM
    assert classify_difficulty(29) == DifficultyBucket.MEDIUM
    assert classify_difficulty(27) == DifficultyBucket.MEDIUM


def test_classify_difficulty_hard():
    assert classify_difficulty(24) == DifficultyBucket.HARD
    assert classify_difficulty(17) == DifficultyBucket.HARD
    assert classify_difficulty(0) == DifficultyBucket.HARD


def test_classify_difficulty_boundaries():
    assert classify_difficulty(30) == DifficultyBucket.EASY
    assert classify_difficulty(29) == DifficultyBucket.MEDIUM
    assert classify_difficulty(25) == DifficultyBucket.MEDIUM
    assert classify_difficulty(24) == DifficultyBucket.HARD


def test_stratify_dataframe_all_buckets():
    df = pd.DataFrame(
        {
            'puzzle': [
                '1' * 35 + '0' * 46,  # 35 givens = easy
                '1' * 27 + '0' * 54,  # 27 givens = medium
                '1' * 20 + '0' * 61,  # 20 givens = hard
            ]
        }
    )

    result = stratify_dataframe(df)

    assert DifficultyBucket.EASY in result
    assert DifficultyBucket.MEDIUM in result
    assert DifficultyBucket.HARD in result
    assert len(result[DifficultyBucket.EASY]) == 1
    assert len(result[DifficultyBucket.MEDIUM]) == 1
    assert len(result[DifficultyBucket.HARD]) == 1


def test_stratify_dataframe_single_bucket():
    df = pd.DataFrame(
        {
            'puzzle': [
                '1' * 35 + '0' * 46,
                '1' * 40 + '0' * 41,
            ]
        }
    )

    result = stratify_dataframe(df)

    assert DifficultyBucket.EASY in result
    assert DifficultyBucket.MEDIUM not in result
    assert DifficultyBucket.HARD not in result
    assert len(result[DifficultyBucket.EASY]) == 2


def test_stratify_dataframe_empty():
    df = pd.DataFrame({'puzzle': []})

    result = stratify_dataframe(df)

    assert len(result) == 0


def test_stratify_dataframe_preserves_columns():
    df = pd.DataFrame(
        {
            'puzzle': ['1' * 35 + '0' * 46],
            'solution': ['1' * 81],
            'extra': [42],
        }
    )

    result = stratify_dataframe(df)

    bucket_df = result[DifficultyBucket.EASY]
    assert 'solution' in bucket_df.columns
    assert 'extra' in bucket_df.columns
    assert bucket_df['extra'].iloc[0] == 42


def test_difficulty_bucket_values():
    assert DifficultyBucket.EASY.value == 'easy'
    assert DifficultyBucket.MEDIUM.value == 'medium'
    assert DifficultyBucket.HARD.value == 'hard'
