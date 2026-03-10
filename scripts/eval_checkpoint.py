"""Evaluate a checkpoint: forward-pass accuracy + Langevin dynamics on a small sample."""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ebm.dataset.loader import SudokuDataset
from ebm.dataset.splits import split_dataset
from ebm.dataset.torch_dataset import SudokuTorchDataset
from ebm.evaluation.metrics import evaluate
from ebm.model.jepa import InferenceConfig, SudokuJEPA
from ebm.training.checkpoint import CheckpointManager
from ebm.utils.config import ArchitectureConfig, TrainingConfig

CHECKPOINT = Path('checkpoints/checkpoint_epoch019_acc0.9926.pt')
N_SAMPLES = 20_000  # gives ~1K test puzzles
LANGEVIN_PUZZLES = 100
LANGEVIN_STEPS = 50
LANGEVIN_CHAINS = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load data
ds = SudokuDataset()
df = ds.load_head(N_SAMPLES)
_, _, test_df = split_dataset(df, val_size=max(1, int(len(df) * 0.05)), test_size=max(1, int(len(df) * 0.05)))
test_ds = SudokuTorchDataset(test_df)
print(f'Test puzzles: {len(test_ds)}')

# Load model
arch_cfg = ArchitectureConfig()
train_cfg = TrainingConfig()
model = SudokuJEPA(arch_cfg, train_cfg)
CheckpointManager.load(CHECKPOINT, model)
model.to(device)
model.eval()

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Parameters: {param_count:,}')

# --- Forward pass evaluation ---
print('\n=== Forward Pass Evaluation ===')
loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
all_preds, all_solutions, all_masks = [], [], []

t0 = time.time()
with torch.no_grad():
    for batch in loader:
        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)

        out = model(puzzle, solution, mask)
        pred = out.decode_logits.argmax(dim=-1) + 1
        all_preds.append(pred.cpu())
        all_solutions.append(solution.cpu())
        all_masks.append(mask.cpu())

fwd_time = time.time() - t0
fwd_metrics = evaluate(all_preds, all_solutions, all_masks)
print(f'Puzzles:                 {fwd_metrics.n_puzzles}')
print(f'Cell accuracy:           {fwd_metrics.cell_accuracy:.4f} ({fwd_metrics.cell_accuracy * 100:.1f}%)')
print(f'Puzzle accuracy:         {fwd_metrics.puzzle_accuracy:.4f} ({fwd_metrics.puzzle_accuracy * 100:.1f}%)')
print(
    f'Constraint satisfaction: {fwd_metrics.constraint_satisfaction:.4f} ({fwd_metrics.constraint_satisfaction * 100:.1f}%)'
)
print(f'Time: {fwd_time:.1f}s ({fwd_metrics.n_puzzles / fwd_time:.0f} puzzles/sec)')

# --- Langevin dynamics evaluation ---
print(
    f'\n=== Langevin Dynamics Evaluation ({LANGEVIN_PUZZLES} puzzles, {LANGEVIN_STEPS} steps, {LANGEVIN_CHAINS} chains) ==='
)
langevin_ds = SudokuTorchDataset(test_df.head(LANGEVIN_PUZZLES))
langevin_loader = DataLoader(langevin_ds, batch_size=min(LANGEVIN_PUZZLES, 50), shuffle=False, num_workers=0)
inference_cfg = InferenceConfig(n_steps=LANGEVIN_STEPS, n_chains=LANGEVIN_CHAINS, lr=0.01, noise_scale=0.005)

lang_preds, lang_solutions, lang_masks = [], [], []
t0 = time.time()
for batch in langevin_loader:
    puzzle = batch['puzzle'].to(device)
    solution = batch['solution'].to(device)
    mask = batch['mask'].to(device)
    pred = model.solve(puzzle, mask, inference_cfg)
    lang_preds.append(pred.cpu())
    lang_solutions.append(solution.cpu())
    lang_masks.append(mask.cpu())

lang_time = time.time() - t0
lang_metrics = evaluate(lang_preds, lang_solutions, lang_masks)
print(f'Puzzles:                 {lang_metrics.n_puzzles}')
print(f'Cell accuracy:           {lang_metrics.cell_accuracy:.4f} ({lang_metrics.cell_accuracy * 100:.1f}%)')
print(f'Puzzle accuracy:         {lang_metrics.puzzle_accuracy:.4f} ({lang_metrics.puzzle_accuracy * 100:.1f}%)')
print(
    f'Constraint satisfaction: {lang_metrics.constraint_satisfaction:.4f} ({lang_metrics.constraint_satisfaction * 100:.1f}%)'
)
print(f'Time: {lang_time:.1f}s ({lang_metrics.n_puzzles / lang_time:.1f} puzzles/sec)')

# --- Print example puzzles ---
print('\n=== Example Puzzles (Langevin) ===')
first_preds = lang_preds[0]
first_solutions = lang_solutions[0]
first_masks = lang_masks[0]

for i in range(min(5, first_preds.shape[0])):
    pred = first_preds[i]
    sol = first_solutions[i]
    mask = first_masks[i]
    correct = (pred == sol).all().item()
    empty_cells = (mask == 0).sum().item()
    empty_correct = ((pred == sol) | (mask == 1)).sum().item() - (mask == 1).sum().item()

    print(f'\nPuzzle {i + 1} — {"SOLVED" if correct else "FAILED"} ({empty_correct}/{empty_cells} empty cells correct)')
    print('  Puzzle:     ', end='')
    for r in range(9):
        if r > 0:
            print('              ', end='')
        for c in range(9):
            if mask[r, c] == 1:
                print(f' {sol[r, c].item()}', end='')
            else:
                print(' .', end='')
            if c in (2, 5):
                print(' |', end='')
        print()
        if r in (2, 5):
            print('              ------+-------+------')

    print('  Predicted:  ', end='')
    for r in range(9):
        if r > 0:
            print('              ', end='')
        for c in range(9):
            digit = pred[r, c].item()
            is_correct = (pred[r, c] == sol[r, c]).item()
            if mask[r, c] == 1:
                print(f' {digit}', end='')
            elif is_correct:
                print(f' {digit}', end='')
            else:
                print(f' \033[91m{digit}\033[0m', end='')
            if c in (2, 5):
                print(' |', end='')
        print()
        if r in (2, 5):
            print('              ------+-------+------')

# --- Summary ---
print('\n=== Summary ===')
print(f'Model: {param_count:,} parameters (epoch 19)')
print(f'Forward pass:      {fwd_metrics.puzzle_accuracy * 100:.1f}% puzzle accuracy on {fwd_metrics.n_puzzles} puzzles')
print(
    f'Langevin dynamics: {lang_metrics.puzzle_accuracy * 100:.1f}% puzzle accuracy on {lang_metrics.n_puzzles} puzzles'
)
delta = lang_metrics.puzzle_accuracy - fwd_metrics.puzzle_accuracy
print(f'Langevin delta:    {"+" if delta >= 0 else ""}{delta * 100:.1f}%')
