"""Weight-based head ablation for causal validation experiments."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor, nn

from ebm.interpretability.strategies import StrategyDetector
from ebm.interpretability.types import AblationResult, HeadProfile
from ebm.model.constraints import constraint_penalty
from ebm.model.jepa import InferenceConfig, SudokuJEPA


class HeadAblator:
    """Weight-based ablation of specific attention heads during Langevin dynamics."""

    def __init__(self, model: SudokuJEPA) -> None:
        """Initialize with the model to ablate."""
        self._model = model
        self._detector = StrategyDetector()

    def ablate_and_solve(
        self,
        puzzle: Tensor,
        mask: Tensor,
        solution: Tensor,
        inference_cfg: InferenceConfig,
        head_specs: list[tuple[str, int]],
    ) -> AblationResult:
        """
        Run Langevin dynamics with specified heads ablated.

        Args:
            puzzle: (B, 10, 9, 9) one-hot encoded puzzle.
            mask: (B, 9, 9) binary clue mask.
            solution: (B, 9, 9, 9) one-hot ground truth.
            inference_cfg: Langevin dynamics parameters.
            head_specs: List of (module_path, head_index) to ablate.

        Returns:
            AblationResult with accuracy metrics.

        """
        # First compute baseline (no ablation)
        baseline_board = self._run_langevin(puzzle, mask, inference_cfg, ablation_specs=[])
        baseline_acc, baseline_strat_acc = self._compute_accuracy(
            baseline_board,
            puzzle,
            solution,
            mask,
        )

        # Run with ablation
        ablated_board = self._run_langevin(puzzle, mask, inference_cfg, ablation_specs=head_specs)
        ablated_acc, ablated_strat_acc = self._compute_accuracy(
            ablated_board,
            puzzle,
            solution,
            mask,
        )

        return AblationResult(
            ablated_heads=head_specs,
            overall_accuracy=ablated_acc,
            strategy_accuracy=ablated_strat_acc,
            baseline_accuracy=baseline_acc,
            baseline_strategy_accuracy=baseline_strat_acc,
        )

    def run_ablation_sweep(
        self,
        puzzle: Tensor,
        mask: Tensor,
        solution: Tensor,
        inference_cfg: InferenceConfig,
        profiles: list[HeadProfile],
    ) -> list[AblationResult]:
        """
        Ablate each head individually and measure impact.

        Args:
            puzzle: (B, 10, 9, 9) one-hot encoded puzzle.
            mask: (B, 9, 9) binary clue mask.
            solution: (B, 9, 9, 9) one-hot ground truth.
            inference_cfg: Langevin dynamics parameters.
            profiles: Head profiles to sweep over.

        Returns:
            List of AblationResult sorted by accuracy degradation (most impactful first).

        """
        results: list[AblationResult] = []
        for profile in profiles:
            result = self.ablate_and_solve(
                puzzle,
                mask,
                solution,
                inference_cfg,
                head_specs=[(profile.layer, profile.head_idx)],
            )
            results.append(result)

        results.sort(key=lambda r: r.baseline_accuracy - r.overall_accuracy, reverse=True)
        return results

    def _run_langevin(
        self,
        puzzle: Tensor,
        mask: Tensor,
        inference_cfg: InferenceConfig,
        ablation_specs: list[tuple[str, int]],
    ) -> Tensor:
        """Run Langevin dynamics with optional head ablation via weight zeroing."""
        model = self._model
        device = puzzle.device
        batch_size = puzzle.shape[0]

        saved_weights = self._apply_weight_ablation(ablation_specs) if ablation_specs else {}

        try:
            with torch.no_grad():
                z_context = model.context_encoder(puzzle)

            z = torch.randn(batch_size, model.arch_cfg.d_latent, device=device)

            with torch.enable_grad():
                z = z.detach().requires_grad_(True)

                for step_idx in range(inference_cfg.n_steps):
                    z_pred = model.predictor(z_context, z)
                    logits = model.decoder(z_context, z, puzzle, mask)
                    probs = torch.softmax(logits, dim=-1)

                    z_target_est = model.target_encoder(probs.permute(0, 3, 1, 2))
                    self_consistency = ((z_pred - z_target_est) ** 2).sum(dim=-1)
                    c_penalty = constraint_penalty(probs)

                    temp = 1.0 - step_idx / max(inference_cfg.n_steps, 1)
                    total_energy = self_consistency + c_penalty * (1.0 + 2.0 * (1.0 - temp))

                    grad_z = torch.autograd.grad(total_energy.sum(), z)[0]
                    noise = inference_cfg.noise_scale * temp * torch.randn_like(z)
                    z = (z - inference_cfg.lr * grad_z + noise).detach().requires_grad_(True)

            final_logits = model.decoder(z_context, z.detach(), puzzle, mask)
            return final_logits.argmax(dim=-1) + 1  # (B, 9, 9)

        finally:
            self._restore_weights(saved_weights)

    def _apply_weight_ablation(
        self,
        head_specs: list[tuple[str, int]],
    ) -> dict[str, Tensor]:
        """
        Zero out_proj weight columns for specified heads.

        PyTorch's MHA concatenates per-head outputs as [h0|h1|...|hN] and
        projects through out_proj. Zeroing columns [h*d_head:(h+1)*d_head]
        of out_proj.weight eliminates that head's contribution cleanly,
        before the linear mixing.

        Args:
            head_specs: List of (module_path, head_index) to ablate.

        Returns:
            Dict mapping module_path to original weight tensors for restoration.

        """
        saved: dict[str, Tensor] = {}

        module_heads: dict[str, list[int]] = {}
        for module_path, head_idx in head_specs:
            module_heads.setdefault(module_path, []).append(head_idx)

        for module_path, head_indices in module_heads.items():
            module = self._find_module(module_path)
            if module is None or not isinstance(module, nn.MultiheadAttention):
                continue

            d_head = module.embed_dim // module.num_heads
            for h in head_indices:
                if not isinstance(h, int) or h < 0 or h >= module.num_heads:
                    msg = (
                        f"Invalid head index {h} for module '{module_path}'; "
                        f'expected 0 <= head_idx < {module.num_heads}.'
                    )
                    raise ValueError(msg)
            saved[module_path] = module.out_proj.weight.data.clone()
            with torch.no_grad():
                for h in head_indices:
                    start = h * d_head
                    end = start + d_head
                    module.out_proj.weight.data[:, start:end] = 0.0

        return saved

    def _restore_weights(self, saved: dict[str, Tensor]) -> None:
        """Restore original out_proj weights after ablation."""
        for module_path, weight_data in saved.items():
            module = self._find_module(module_path)
            if module is not None:
                # Restore in-place to avoid breaking optimizer/state_dict references.
                with torch.no_grad():
                    module.out_proj.weight.copy_(weight_data)

    # Mapping from generic prefixes (as produced by AttentionExtractor)
    # to the concrete model attribute names.
    _PREFIX_ALIASES: ClassVar[dict[str, str]] = {
        'encoder': 'context_encoder',
        'context': 'context_encoder',
        'decoder': 'decoder',
        'target': 'target_encoder',
    }

    def _find_module(self, path: str) -> nn.Module | None:
        """Find a submodule by dot-separated path."""
        # Try looking in common parent modules
        for parent_name in ['context_encoder', 'decoder', 'target_encoder']:
            parent = getattr(self._model, parent_name, None)
            if parent is None:
                continue

            # Normalize generic encoder prefix to the concrete module name.
            # This allows keys like 'encoder.layers.0.self_attn' (from
            # AttentionExtractor) to resolve to
            # self._model.context_encoder.layers[0].self_attn.
            normalized_path = path
            first_segment = path.split('.', 1)[0]
            alias_target = self._PREFIX_ALIASES.get(first_segment)
            if alias_target == parent_name and first_segment != parent_name:
                normalized_path = f'{parent_name}.{path.split(".", 1)[1]}' if '.' in path else parent_name

            prefix = f'{parent_name}.'
            if normalized_path.startswith(prefix):
                subpath = normalized_path[len(prefix) :]
            else:
                continue

            try:
                module = parent
                for part in subpath.split('.'):
                    module = getattr(module, part, None) if not part.isdigit() else module[int(part)]
                    if module is None:
                        break
                if module is not None:
                    return module
            except (AttributeError, IndexError, TypeError):
                continue
        return None

    def _compute_accuracy(
        self,
        board: Tensor,
        puzzle: Tensor,
        solution: Tensor,
        mask: Tensor,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute overall and per-strategy accuracy.

        Args:
            board: (B, 9, 9) predicted digit grid.
            puzzle: (B, 10, 9, 9) one-hot encoded puzzle (used for clue board).
            solution: (B, 9, 9, 9) one-hot ground truth.
            mask: (B, 9, 9) binary clue mask.

        Returns:
            Tuple of (overall_accuracy, per_strategy_accuracy).

        """
        batch_size = board.shape[0]
        sol_board = solution.argmax(dim=-1) + 1  # (B, 9, 9)

        # Build clue board from puzzle: channels 1-9 are digits
        clue_board = puzzle[:, 1:].permute(0, 2, 3, 1).argmax(dim=-1) + 1  # (B, 9, 9)
        clue_board = clue_board * mask.long()  # zero out non-clue cells

        total_correct = 0
        total_cells = 0

        for b in range(batch_size):
            non_clue = mask[b] == 0
            total_cells += int(non_clue.sum().item())
            total_correct += int(((board[b] == sol_board[b]) & non_clue).sum().item())

        overall_acc = total_correct / max(total_cells, 1)

        # Per-strategy accuracy: classify from clue board to solution
        strategy_correct: dict[str, int] = {}
        strategy_total: dict[str, int] = {}

        for b in range(batch_size):
            events = self._detector.classify(
                clue_board[b],
                sol_board[b],
                mask[b],
            )
            for event in events:
                label = event.strategy.value if event.strategy else 'unknown'
                strategy_total[label] = strategy_total.get(label, 0) + 1
                if board[b, event.row, event.col] == sol_board[b, event.row, event.col]:
                    strategy_correct[label] = strategy_correct.get(label, 0) + 1

        strategy_acc = {
            label: strategy_correct.get(label, 0) / max(count, 1) for label, count in strategy_total.items()
        }

        return overall_acc, strategy_acc
