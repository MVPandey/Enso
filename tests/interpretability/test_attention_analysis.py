import torch

from ebm.interpretability.attention_analysis import AttentionAnalyzer
from ebm.interpretability.types import CellEvent, HeadProfile, StrategyLabel


def _uniform_attention(batch=2, n_heads=4, seq_len=81):
    """Create uniform attention maps."""
    return torch.ones(batch, n_heads, seq_len, seq_len) / seq_len


def _row_biased_attention(batch=2, n_heads=4, seq_len=81):
    """Create attention maps biased toward same-row positions."""
    attn = torch.ones(batch, n_heads, seq_len, seq_len) * 0.001
    for i in range(seq_len):
        row = i // 9
        for j in range(row * 9, row * 9 + 9):
            attn[:, :, i, j] = 1.0
    # Normalize
    attn = attn / attn.sum(dim=-1, keepdim=True)
    return attn


def test_analyzer_init():
    analyzer = AttentionAnalyzer()
    assert analyzer._row_mask.shape == (81, 81)
    assert analyzer._col_mask.shape == (81, 81)
    assert analyzer._box_mask.shape == (81, 81)


def test_row_mask_correctness():
    analyzer = AttentionAnalyzer()
    # Position 0 (row 0, col 0) and position 8 (row 0, col 8) share a row
    assert analyzer._row_mask[0, 8].item()
    # Position 0 (row 0) and position 9 (row 1) do not share a row
    assert not analyzer._row_mask[0, 9].item()


def test_col_mask_correctness():
    analyzer = AttentionAnalyzer()
    # Position 0 (col 0) and position 9 (col 0) share a column
    assert analyzer._col_mask[0, 9].item()
    # Position 0 (col 0) and position 1 (col 1) do not share a column
    assert not analyzer._col_mask[0, 1].item()


def test_box_mask_correctness():
    analyzer = AttentionAnalyzer()
    # Position 0 (row 0, col 0) and position 10 (row 1, col 1) share a box
    assert analyzer._box_mask[0, 10].item()
    # Position 0 (row 0, col 0) and position 27 (row 3, col 0) do not share a box
    assert not analyzer._box_mask[0, 27].item()


def test_compute_head_profiles_uniform():
    """Uniform attention should yield mixed specialization."""
    analyzer = AttentionAnalyzer()
    maps = {'encoder.layer0': _uniform_attention()}
    profiles = analyzer.compute_head_profiles(maps)

    assert len(profiles) == 4
    for p in profiles:
        assert isinstance(p, HeadProfile)
        assert p.layer == 'encoder.layer0'
        # Uniform attention: row/col/box scores all near 1.0
        assert abs(p.row_score - 1.0) < 0.2
        assert abs(p.col_score - 1.0) < 0.2
        assert abs(p.box_score - 1.0) < 0.2
        assert p.specialization == 'mixed'


def test_compute_head_profiles_row_biased():
    """Row-biased attention should yield row specialization."""
    analyzer = AttentionAnalyzer()
    maps = {'encoder.layer0': _row_biased_attention()}
    profiles = analyzer.compute_head_profiles(maps)

    for p in profiles:
        assert p.row_score > p.col_score
        assert p.row_score > p.box_score
        assert p.specialization == 'row'


def test_compute_head_profiles_multiple_layers():
    analyzer = AttentionAnalyzer()
    maps = {
        'encoder.layer0': _uniform_attention(n_heads=4),
        'decoder.layer0': _uniform_attention(n_heads=2),
    }
    profiles = analyzer.compute_head_profiles(maps)

    assert len(profiles) == 6
    encoder_profiles = [p for p in profiles if p.layer == 'encoder.layer0']
    decoder_profiles = [p for p in profiles if p.layer == 'decoder.layer0']
    assert len(encoder_profiles) == 4
    assert len(decoder_profiles) == 2


def test_within_group_score_identity():
    """When attention is uniform, score should be ~1.0."""
    analyzer = AttentionAnalyzer()
    attn = torch.ones(81, 81) / 81
    mask = analyzer._row_mask
    score = analyzer._within_group_score(attn, mask)
    assert abs(score - 1.0) < 0.01


def test_within_group_score_zero_attention():
    """Zero attention should return 1.0 (fallback)."""
    analyzer = AttentionAnalyzer()
    attn = torch.zeros(81, 81)
    score = analyzer._within_group_score(attn, analyzer._row_mask)
    assert score == 1.0


def test_correlate_with_events():
    analyzer = AttentionAnalyzer()
    maps = {'encoder.layer0': _uniform_attention(n_heads=2)}
    profiles = analyzer.compute_head_profiles(maps)

    events = [
        CellEvent(step=1, row=0, col=0, digit=1, strategy=StrategyLabel.NAKED_SINGLE, confidence=0.9),
        CellEvent(step=2, row=1, col=1, digit=2, strategy=StrategyLabel.HIDDEN_SINGLE, confidence=0.8),
    ]

    result = analyzer.correlate_with_events(profiles, events, maps)

    assert 'naked_single' in result
    assert 'hidden_single' in result
    assert len(result['naked_single']) == 2
    assert all(isinstance(p, HeadProfile) for p in result['naked_single'])


def test_correlate_with_empty_events():
    analyzer = AttentionAnalyzer()
    maps = {'encoder.layer0': _uniform_attention(n_heads=2)}
    profiles = analyzer.compute_head_profiles(maps)

    result = analyzer.correlate_with_events(profiles, [], maps)
    assert result == {}
