"""Instrumented Langevin solve loop that records full trajectory state."""

from __future__ import annotations

import torch
from torch import Tensor

from ebm.interpretability.attention import AttentionExtractor
from ebm.interpretability.types import StepSnapshot, Trajectory
from ebm.model.constraints import constraint_penalty
from ebm.model.jepa import InferenceConfig, SudokuJEPA


class TrajectoryRecorder:
    """
    Record complete Langevin dynamics trajectories for analysis.

    Re-implements the solve loop from ``SudokuJEPA.solve()`` as a single-chain
    trajectory that captures z, logits, probs, energy components, gradient
    norms, and optionally attention maps at each step.
    """

    def __init__(self, model: SudokuJEPA, record_attention: bool = True, attention_stride: int = 5) -> None:
        """Initialize with model and attention capture settings."""
        self._model = model
        self._record_attention = record_attention
        self._attention_stride = attention_stride

    def record(
        self,
        puzzle: Tensor,
        mask: Tensor,
        solution: Tensor,
        inference_cfg: InferenceConfig | None = None,
    ) -> Trajectory:
        """
        Run instrumented Langevin dynamics and record the full trajectory.

        Operates on a **single chain** (no multi-chain ensemble) to provide
        a clean, traceable trajectory for interpretability analysis.

        Args:
            puzzle: (B, 10, 9, 9) one-hot encoded puzzle.
            mask: (B, 9, 9) binary clue mask.
            solution: (B, 9, 9, 9) one-hot ground truth.
            inference_cfg: Langevin dynamics parameters.

        Returns:
            Trajectory with complete per-step state snapshots.

        """
        if inference_cfg is None:
            inference_cfg = InferenceConfig.from_training_config(self._model.train_cfg)

        model = self._model
        device = puzzle.device
        batch_size = puzzle.shape[0]

        extractor = AttentionExtractor(model) if self._record_attention else None

        with torch.no_grad():
            z_context = model.context_encoder(puzzle)

        z = torch.randn(batch_size, model.arch_cfg.d_latent, device=device)
        steps: list[StepSnapshot] = []

        with torch.enable_grad():
            z = z.detach().requires_grad_(True)

            for step_idx in range(inference_cfg.n_steps):
                capture_attention = (
                    self._record_attention and extractor is not None and step_idx % self._attention_stride == 0
                )

                if capture_attention:
                    extractor.attach()

                z_pred = model.predictor(z_context, z)
                logits = model.decoder(z_context, z, puzzle, mask)
                probs = torch.softmax(logits, dim=-1)

                z_target_est = model.target_encoder(probs.permute(0, 3, 1, 2))
                self_consistency = ((z_pred - z_target_est) ** 2).sum(dim=-1)

                c_penalty = constraint_penalty(probs)

                temp = 1.0 - step_idx / max(inference_cfg.n_steps, 1)
                total_energy = self_consistency + c_penalty * (1.0 + 2.0 * (1.0 - temp))

                grad_z = torch.autograd.grad(total_energy.sum(), z)[0]
                grad_norm = grad_z.norm(dim=-1)

                encoder_attn = None
                decoder_attn = None
                if capture_attention:
                    all_maps = extractor.get_attention_maps()
                    encoder_attn = {k: v for k, v in all_maps.items() if k.startswith('encoder')}
                    decoder_attn = {k: v for k, v in all_maps.items() if k.startswith('decoder')}
                    extractor.detach()

                board = logits.detach().argmax(dim=-1) + 1  # (B, 9, 9)

                steps.append(
                    StepSnapshot(
                        step=step_idx,
                        z=z.detach().clone(),
                        logits=logits.detach().clone(),
                        probs=probs.detach().clone(),
                        board=board,
                        energy=total_energy.detach().clone(),
                        self_consistency=self_consistency.detach().clone(),
                        constraint_penalty=c_penalty.detach().clone(),
                        grad_norm=grad_norm.detach().clone(),
                        encoder_attention=encoder_attn,
                        decoder_attention=decoder_attn,
                    )
                )

                noise = inference_cfg.noise_scale * temp * torch.randn_like(z)
                z = (z - inference_cfg.lr * grad_z + noise).detach().requires_grad_(True)

        # Final board from last step
        final_board = steps[-1].board

        return Trajectory(
            puzzle=puzzle.detach().clone(),
            solution=solution.detach().clone(),
            mask=mask.detach().clone(),
            steps=steps,
            final_board=final_board,
        )
