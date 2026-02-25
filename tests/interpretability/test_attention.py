import torch

from ebm.interpretability.attention import AttentionExtractor
from ebm.model.jepa import SudokuJEPA
from ebm.utils.config import ArchitectureConfig, TrainingConfig

SMALL_ARCH = ArchitectureConfig(
    d_model=32,
    n_layers=1,
    n_heads=4,
    d_ffn=64,
    d_latent=16,
    predictor_hidden=32,
    decoder_layers=1,
    decoder_heads=2,
    decoder_d_cell=16,
)
SMALL_TRAIN = TrainingConfig(langevin_steps=3, n_chains=2)


def _make_batch(b: int = 2):
    puzzle = torch.zeros(b, 10, 9, 9)
    puzzle[:, 0] = 1.0
    solution = torch.zeros(b, 9, 9, 9)
    solution[:, :, :, 0] = 1.0
    mask = torch.zeros(b, 9, 9)
    return puzzle, solution, mask


def test_attach_detach():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    extractor = AttentionExtractor(model)

    extractor.attach()
    assert len(extractor._handles) > 0

    extractor.detach()
    assert len(extractor._handles) == 0


def test_context_manager():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    with AttentionExtractor(model) as ext:
        assert len(ext._handles) > 0
    assert len(ext._handles) == 0


def test_captures_attention_on_forward():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    puzzle, solution, mask = _make_batch()

    with AttentionExtractor(model) as ext:
        model(puzzle, solution, mask)
        maps = ext.get_attention_maps()

    assert len(maps) > 0
    for key, attn in maps.items():
        assert attn.dim() == 4  # (B, n_heads, seq, seq)
        assert attn.shape[0] == 2  # batch size
        assert attn.shape[2] == attn.shape[3] == 81  # 81 cells


def test_encoder_and_decoder_keys():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    puzzle, solution, mask = _make_batch()

    with AttentionExtractor(model) as ext:
        model(puzzle, solution, mask)
        maps = ext.get_attention_maps()

    encoder_keys = [k for k in maps if k.startswith('encoder')]
    decoder_keys = [k for k in maps if k.startswith('decoder')]

    assert len(encoder_keys) >= 1
    assert len(decoder_keys) >= 1


def test_get_clears_buffer():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    puzzle, solution, mask = _make_batch()

    with AttentionExtractor(model) as ext:
        model(puzzle, solution, mask)
        maps1 = ext.get_attention_maps()
        maps2 = ext.get_attention_maps()

    assert len(maps1) > 0
    assert len(maps2) == 0


def test_attention_shapes_per_head():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    puzzle, solution, mask = _make_batch(3)

    with AttentionExtractor(model) as ext:
        model(puzzle, solution, mask)
        maps = ext.get_attention_maps()

    for key, attn in maps.items():
        if key.startswith('encoder'):
            assert attn.shape[1] == SMALL_ARCH.n_heads
        elif key.startswith('decoder'):
            assert attn.shape[1] == SMALL_ARCH.decoder_heads


def test_no_hooks_after_detach():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    extractor = AttentionExtractor(model)

    extractor.attach()
    extractor.detach()

    puzzle, solution, mask = _make_batch()
    model(puzzle, solution, mask)

    maps = extractor.get_attention_maps()
    assert len(maps) == 0
