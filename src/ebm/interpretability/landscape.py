"""Energy landscape probing utilities for mechanistic interpretability."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ebm.interpretability.types import EnergyProfile
from ebm.model.constraints import constraint_penalty
from ebm.model.jepa import SudokuJEPA


class EnergyEvaluator:
    """
    Evaluate energy at arbitrary latent states without gradient updates.

    Provides utilities for probing the energy landscape geometry:
    point evaluation, linear interpolation, and oracle z computation.
    """

    def __init__(self, model: SudokuJEPA) -> None:
        """Initialize with a trained SudokuJEPA model."""
        self._model = model

    def evaluate(
        self,
        puzzle: Tensor,
        mask: Tensor,
        z: Tensor,
        z_context: Tensor | None = None,
        temp: float = 0.5,
    ) -> EnergyProfile:
        """
        Compute energy at an arbitrary z without gradient updates.

        Replicates the recorder's energy computation exactly.

        Args:
            puzzle: (B, 10, 9, 9) one-hot encoded puzzle.
            mask: (B, 9, 9) binary clue mask.
            z: (B, d_latent) latent state to evaluate.
            z_context: (B, d_model) precomputed context. Computed if None.
            temp: Temperature for constraint penalty weighting.

        Returns:
            EnergyProfile with energy components and decoded outputs.

        """
        model = self._model

        with torch.no_grad():
            if z_context is None:
                z_context = model.context_encoder(puzzle)

            z_pred = model.predictor(z_context, z)
            logits = model.decoder(z_context, z, puzzle, mask)
            probs = torch.softmax(logits, dim=-1)

            z_target_est = model.target_encoder(probs.permute(0, 3, 1, 2))
            self_consistency = ((z_pred - z_target_est) ** 2).sum(dim=-1)

            c_penalty = constraint_penalty(probs)

            energy = self_consistency + c_penalty * (1.0 + 2.0 * (1.0 - temp))

        return EnergyProfile(
            z=z.detach().clone(),
            energy=energy,
            self_consistency=self_consistency,
            constraint_penalty=c_penalty,
            logits=logits,
            probs=probs,
        )

    def interpolate(
        self,
        puzzle: Tensor,
        mask: Tensor,
        z_start: Tensor,
        z_end: Tensor,
        n_points: int = 50,
        z_context: Tensor | None = None,
        temp: float = 0.5,
    ) -> list[EnergyProfile]:
        """
        Evaluate energy along a linear path from z_start to z_end.

        Args:
            puzzle: (B, 10, 9, 9) one-hot encoded puzzle.
            mask: (B, 9, 9) binary clue mask.
            z_start: (B, d_latent) starting latent state.
            z_end: (B, d_latent) ending latent state.
            n_points: Number of evaluation points along the path.
            z_context: (B, d_model) precomputed context. Computed if None.
            temp: Temperature for constraint penalty weighting.

        Returns:
            List of EnergyProfile, one per interpolation point.

        """
        model = self._model

        with torch.no_grad():
            if z_context is None:
                z_context = model.context_encoder(puzzle)

        profiles: list[EnergyProfile] = []
        for alpha in torch.linspace(0.0, 1.0, n_points):
            z = (1.0 - alpha) * z_start + alpha * z_end
            profile = self.evaluate(puzzle, mask, z, z_context=z_context, temp=temp)
            profiles.append(profile)

        return profiles

    def compute_oracle_z(self, solution: Tensor) -> Tensor:
        """
        Compute the oracle latent z from a known solution.

        Matches the oracle computation in the forward-vs-langevin experiment:
        ``z_target = target_encoder(solution)`` then
        ``z = normalize(z_encoder(z_target))``.

        Args:
            solution: (B, 9, 9, 9) one-hot encoded solution.

        Returns:
            (B, d_latent) normalized oracle latent vector.

        """
        model = self._model
        with torch.no_grad():
            z_target = model.target_encoder(solution.permute(0, 3, 1, 2))
            return F.normalize(model.z_encoder(z_target), dim=-1)

    def compute_z_context(self, puzzle: Tensor) -> Tensor:
        """
        Compute context encoding for reuse across multiple evaluate calls.

        Args:
            puzzle: (B, 10, 9, 9) one-hot encoded puzzle.

        Returns:
            (B, d_model) context encoding.

        """
        with torch.no_grad():
            return self._model.context_encoder(puzzle)
