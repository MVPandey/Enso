"""Smoke test: train 100 steps on a small subset, verify loss decreases."""

import logging
import time

import torch
from torch.utils.data import DataLoader

from ebm.dataset.loader import SudokuDataset
from ebm.dataset.torch_dataset import SudokuTorchDataset
from ebm.model.jepa import SudokuJEPA
from ebm.training.losses import compute_loss
from ebm.training.scheduler import create_lr_scheduler, get_ema_momentum
from ebm.utils.config import ArchitectureConfig, TrainingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

BATCH_SIZE = 2048
N_STEPS = 100
N_SAMPLES = 10_000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Using device: %s', device)
if device.type == 'cuda':
    logger.info('GPU: %s', torch.cuda.get_device_name(0))

# Load a small subset
logger.info('Loading %d samples...', N_SAMPLES)
ds = SudokuDataset()
df = ds.load_head(N_SAMPLES)
torch_ds = SudokuTorchDataset(df)
loader = DataLoader(torch_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Build model with full-size architecture
arch_cfg = ArchitectureConfig()
train_cfg = TrainingConfig(batch_size=BATCH_SIZE)
model = SudokuJEPA(arch_cfg, train_cfg).to(device)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info('Trainable parameters: %s', f'{param_count:,}')

optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
total_steps = N_STEPS
scheduler = create_lr_scheduler(optimizer, train_cfg, total_steps)

# Track losses
losses = []
step = 0
start = time.time()

model.train()
while step < N_STEPS:
    for batch in loader:
        if step >= N_STEPS:
            break

        puzzle = batch['puzzle'].to(device)
        solution = batch['solution'].to(device)
        mask = batch['mask'].to(device)

        out = model(puzzle, solution, mask)
        loss_out = compute_loss(out, solution, mask, train_cfg)

        optimizer.zero_grad()
        loss_out.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        momentum = get_ema_momentum(step, total_steps, train_cfg)
        model.update_target_encoder(momentum)

        losses.append(loss_out.total.item())

        if step % 10 == 0:
            logger.info(
                'Step %3d | total=%.4f | energy=%.4f | vicreg=%.4f | decode=%.4f | lr=%.2e',
                step, loss_out.total.item(), loss_out.energy.item(),
                loss_out.vicreg.item(), loss_out.decode.item(),
                scheduler.get_last_lr()[0],
            )

        step += 1

elapsed = time.time() - start
logger.info('Completed %d steps in %.1fs (%.1f steps/sec)', N_STEPS, elapsed, N_STEPS / elapsed)

# Verify loss decreased
first_10 = sum(losses[:10]) / 10
last_10 = sum(losses[-10:]) / 10
logger.info('Mean loss first 10 steps: %.4f', first_10)
logger.info('Mean loss last 10 steps:  %.4f', last_10)

if last_10 < first_10:
    logger.info('PASS: Loss decreased (%.4f -> %.4f)', first_10, last_10)
else:
    logger.warning('FAIL: Loss did not decrease (%.4f -> %.4f)', first_10, last_10)

# Check GPU memory usage
if device.type == 'cuda':
    allocated = torch.cuda.max_memory_allocated() / 1e9
    logger.info('Peak GPU memory: %.2f GB / 34.2 GB', allocated)
