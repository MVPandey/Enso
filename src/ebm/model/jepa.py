"""SudokuJEPA — top-level orchestrator wiring all model components."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ebm.model.constraints import constraint_penalty
from ebm.model.decoder import SudokuDecoder
from ebm.model.encoder import SudokuEncoder
from ebm.model.energy import energy_fn
from ebm.model.predictor import LatentPredictor
from ebm.utils.config import ArchitectureConfig, TrainingConfig


@dataclass
class JEPAOutput:
    """Container for SudokuJEPA forward pass outputs."""

    energy: Tensor
    z_context: Tensor
    z_pred: Tensor
    z_target: Tensor
    decode_logits: Tensor


@dataclass
class InferenceConfig:
    """Parameters for Langevin dynamics inference."""

    n_steps: int = 50
    n_chains: int = 8
    lr: float = 0.01
    noise_scale: float = 0.005

    @classmethod
    def from_training_config(cls, cfg: TrainingConfig) -> 'InferenceConfig':
        """Create from a TrainingConfig, using its Langevin defaults."""
        return cls(
            n_steps=cfg.langevin_steps,
            n_chains=cfg.n_chains,
            lr=cfg.langevin_lr,
            noise_scale=cfg.langevin_noise_scale,
        )


class SudokuJEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture for Sudoku.

    Wires together context encoder, target encoder (EMA), latent predictor,
    decoder, and energy function. Provides forward() for training and
    solve() for inference via Langevin dynamics.
    """

    def __init__(
        self,
        arch_cfg: ArchitectureConfig | None = None,
        train_cfg: TrainingConfig | None = None,
    ) -> None:
        """
        Initialize all sub-modules.

        Args:
            arch_cfg: Architecture hyperparameters.
            train_cfg: Training hyperparameters (used for inference defaults).

        """
        super().__init__()
        if not arch_cfg:
            arch_cfg = ArchitectureConfig()
        if not train_cfg:
            train_cfg = TrainingConfig()

        self.arch_cfg = arch_cfg
        self.train_cfg = train_cfg

        self.context_encoder = SudokuEncoder(input_channels=10, cfg=arch_cfg)
        self.target_encoder = SudokuEncoder(input_channels=9, cfg=arch_cfg)
        self.z_encoder = nn.Linear(arch_cfg.d_model, arch_cfg.d_latent)
        self.predictor = LatentPredictor(cfg=arch_cfg)
        self.decoder = SudokuDecoder(cfg=arch_cfg)

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self._init_target_encoder()

    def _init_target_encoder(self) -> None:
        """Copy context encoder weights to target encoder (matching layers only)."""
        target_state = self.target_encoder.state_dict()
        context_state = self.context_encoder.state_dict()
        for key in target_state:
            if key in context_state and target_state[key].shape == context_state[key].shape:
                target_state[key].copy_(context_state[key])

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """
        EMA update: target = momentum * target + (1 - momentum) * context.

        Only updates parameters with matching shapes (skips input_proj which
        differs between 10-channel context and 9-channel target).

        Args:
            momentum: EMA momentum in [0, 1].

        """
        for t_param, c_param in zip(self.target_encoder.parameters(), self.context_encoder.parameters(), strict=False):
            if t_param.shape == c_param.shape:
                t_param.lerp_(c_param, 1.0 - momentum)

    def forward(self, puzzle: Tensor, solution: Tensor, mask: Tensor) -> JEPAOutput:
        """
        Training forward pass.

        Args:
            puzzle: (B, 10, 9, 9) one-hot encoded puzzle.
            solution: (B, 9, 9, 9) one-hot encoded solution.
            mask: (B, 9, 9) binary mask, 1 = given clue.

        Returns:
            JEPAOutput with energy, z_pred, z_target, and decode_logits.

        """
        z_context = self.context_encoder(puzzle)

        with torch.no_grad():
            z_target = self.target_encoder(solution.permute(0, 3, 1, 2))

        z_target_latent = F.normalize(self.z_encoder(z_target), dim=-1)
        z = z_target_latent + self.train_cfg.z_noise_scale * torch.randn_like(z_target_latent)

        z_pred = self.predictor(z_context, z)
        energy = energy_fn(z_pred, z_target)
        decode_logits = self.decoder(z_context, z, puzzle, mask)

        return JEPAOutput(
            energy=energy,
            z_context=z_context,
            z_pred=z_pred,
            z_target=z_target,
            decode_logits=decode_logits,
        )

    def solve(
        self,
        puzzle: Tensor,
        mask: Tensor,
        inference_cfg: InferenceConfig | None = None,
    ) -> Tensor:
        """
        Solve a puzzle via Langevin dynamics in latent space.

        Args:
            puzzle: (B, 10, 9, 9) one-hot encoded puzzle.
            mask: (B, 9, 9) binary mask.
            inference_cfg: Langevin dynamics parameters. Defaults from train_cfg.

        Returns:
            (B, 9, 9) integer solution grid with digits 1-9.

        """
        if not inference_cfg:
            inference_cfg = InferenceConfig.from_training_config(self.train_cfg)

        batch_size = puzzle.shape[0]
        device = puzzle.device
        n_chains = inference_cfg.n_chains

        with torch.no_grad():
            z_context = self.context_encoder(puzzle)

        z_context_exp = z_context.repeat_interleave(n_chains, dim=0)
        puzzle_exp = puzzle.repeat_interleave(n_chains, dim=0)
        mask_exp = mask.repeat_interleave(n_chains, dim=0)

        z = torch.randn(batch_size * n_chains, self.arch_cfg.d_latent, device=device)

        # autograd.grad requires z to have gradients even inside no_grad contexts
        total_energy = torch.zeros(batch_size * n_chains, device=device)
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            for step in range(inference_cfg.n_steps):
                z_pred = self.predictor(z_context_exp, z)

                logits = self.decoder(z_context_exp, z, puzzle_exp, mask_exp)
                probs = torch.softmax(logits, dim=-1)

                # Self-consistency: decode → re-encode → compare with z_pred
                z_target_est = self.target_encoder(probs.permute(0, 3, 1, 2))
                self_consistency = ((z_pred - z_target_est) ** 2).sum(dim=-1)

                c_penalty = constraint_penalty(probs)

                temp = 1.0 - step / max(inference_cfg.n_steps, 1)
                total_energy = self_consistency + c_penalty * (1.0 + 2.0 * (1.0 - temp))

                grad_z = torch.autograd.grad(total_energy.sum(), z)[0]
                noise = inference_cfg.noise_scale * temp * torch.randn_like(z)
                z = (z - inference_cfg.lr * grad_z + noise).detach().requires_grad_(True)

        with torch.no_grad():
            final_logits = self.decoder(z_context_exp, z, puzzle_exp, mask_exp)

        final_logits = final_logits.reshape(batch_size, n_chains, 9, 9, 9)
        chain_energy = total_energy.detach().reshape(batch_size, n_chains)
        best_chain = chain_energy.argmin(dim=1)
        best_logits = final_logits[torch.arange(batch_size, device=device), best_chain]

        return best_logits.argmax(dim=-1) + 1
