import torch

from ebm.interpretability.ablation import HeadAblator
from ebm.interpretability.types import AblationResult, HeadProfile
from ebm.model.jepa import InferenceConfig, SudokuJEPA
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
SMALL_INFERENCE = InferenceConfig(n_steps=3, n_chains=1)


def _make_batch(b=1):
    puzzle = torch.zeros(b, 10, 9, 9)
    puzzle[:, 0] = 1.0
    solution = torch.zeros(b, 9, 9, 9)
    solution[:, :, :, 0] = 1.0
    mask = torch.zeros(b, 9, 9)
    return puzzle, solution, mask


def test_apply_weight_ablation():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    ablator = HeadAblator(model)

    # Find an MHA module
    mha_paths = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.MultiheadAttention):
            mha_paths.append(name)

    if mha_paths:
        path = mha_paths[0]
        module = dict(model.named_modules())[path]
        original_weight = module.out_proj.weight.data.clone()
        d_head = module.embed_dim // module.num_heads

        saved = ablator._apply_weight_ablation([(path, 0)])

        # Head 0 columns should be zeroed
        assert (module.out_proj.weight.data[:, :d_head] == 0).all()
        # Other columns should be unchanged
        assert torch.allclose(module.out_proj.weight.data[:, d_head:], original_weight[:, d_head:])

        # Restore should bring back original weights
        ablator._restore_weights(saved)
        assert torch.allclose(module.out_proj.weight.data, original_weight)


def test_weight_ablation_multiple_heads():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    ablator = HeadAblator(model)

    mha_paths = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.MultiheadAttention):
            mha_paths.append(name)

    if mha_paths:
        path = mha_paths[0]
        module = dict(model.named_modules())[path]
        d_head = module.embed_dim // module.num_heads

        saved = ablator._apply_weight_ablation([(path, 0), (path, 1)])

        # Heads 0 and 1 should be zeroed
        assert (module.out_proj.weight.data[:, : 2 * d_head] == 0).all()

        ablator._restore_weights(saved)


def test_head_ablator_init():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    ablator = HeadAblator(model)
    assert ablator._model is model


def test_ablate_and_solve():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    puzzle, solution, mask = _make_batch()
    ablator = HeadAblator(model)

    # Use a module path that exists — find it from the model
    mha_paths = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.MultiheadAttention):
            mha_paths.append(name)

    # Run without ablation specs first to get baseline
    result = ablator.ablate_and_solve(
        puzzle,
        mask,
        solution,
        SMALL_INFERENCE,
        head_specs=[],
    )
    assert isinstance(result, AblationResult)
    assert 0 <= result.overall_accuracy <= 1.0
    assert 0 <= result.baseline_accuracy <= 1.0


def test_ablate_and_solve_with_ablation():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    puzzle, solution, mask = _make_batch()
    ablator = HeadAblator(model)

    # Find an MHA module path
    mha_paths = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.MultiheadAttention):
            mha_paths.append(name)

    if mha_paths:
        # Ablate head 0 of the first MHA module
        result = ablator.ablate_and_solve(
            puzzle,
            mask,
            solution,
            SMALL_INFERENCE,
            head_specs=[(mha_paths[0], 0)],
        )
        assert isinstance(result, AblationResult)
        assert result.ablated_heads == [(mha_paths[0], 0)]


def test_run_ablation_sweep():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    model.eval()
    puzzle, solution, mask = _make_batch()
    ablator = HeadAblator(model)

    # Find MHA modules
    mha_paths = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.MultiheadAttention):
            mha_paths.append(name)

    if mha_paths:
        profiles = [
            HeadProfile(
                layer=mha_paths[0], head_idx=0, row_score=1.5, col_score=1.0, box_score=1.0, specialization='row'
            ),
        ]
        results = ablator.run_ablation_sweep(
            puzzle,
            mask,
            solution,
            SMALL_INFERENCE,
            profiles,
        )
        assert len(results) == 1
        assert isinstance(results[0], AblationResult)


def test_find_module_returns_none_for_bad_path():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    ablator = HeadAblator(model)
    result = ablator._find_module('nonexistent.path.module')
    assert result is None


def test_compute_accuracy():
    model = SudokuJEPA(SMALL_ARCH, SMALL_TRAIN)
    ablator = HeadAblator(model)

    # Perfect board
    puzzle = torch.zeros(1, 10, 9, 9)
    puzzle[:, 0] = 1.0  # empty channel active (no clues)
    solution = torch.zeros(1, 9, 9, 9)
    solution[:, :, :, 0] = 1.0  # all digit 1
    board = torch.ones(1, 9, 9, dtype=torch.long)  # all 1s
    mask = torch.zeros(1, 9, 9)

    acc, _strat_acc = ablator._compute_accuracy(board, puzzle, solution, mask)
    assert acc == 1.0
