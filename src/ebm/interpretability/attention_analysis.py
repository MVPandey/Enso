"""Attention head specialization scoring and strategy-attention correlation."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor

from ebm.interpretability.types import CellEvent, HeadProfile


class AttentionAnalyzer:
    """Analyze 81x81 attention matrices to detect structural specialization."""

    def __init__(self) -> None:
        """Precompute group masks from the 9x9 Sudoku grid structure."""
        self._row_mask = torch.zeros(81, 81, dtype=torch.bool)
        self._col_mask = torch.zeros(81, 81, dtype=torch.bool)
        self._box_mask = torch.zeros(81, 81, dtype=torch.bool)

        for i in range(81):
            ri, ci = divmod(i, 9)
            for j in range(81):
                rj, cj = divmod(j, 9)
                if ri == rj:
                    self._row_mask[i, j] = True
                if ci == cj:
                    self._col_mask[i, j] = True
                if ri // 3 == rj // 3 and ci // 3 == cj // 3:
                    self._box_mask[i, j] = True

    def compute_head_profiles(self, attention_maps: dict[str, Tensor]) -> list[HeadProfile]:
        """
        Compute structural specialization scores for all attention heads.

        Args:
            attention_maps: {layer_key: (B, n_heads, 81, 81)} from AttentionExtractor.

        Returns:
            List of HeadProfile with specialization labels.

        """
        profiles: list[HeadProfile] = []
        for layer_key, attn in attention_maps.items():
            # Average over batch
            attn_mean = attn.float().mean(dim=0)  # (n_heads, 81, 81)
            n_heads = attn_mean.shape[0]
            for h in range(n_heads):
                head_attn = attn_mean[h]  # (81, 81)
                row_score = self._within_group_score(head_attn, self._row_mask.to(head_attn.device))
                col_score = self._within_group_score(head_attn, self._col_mask.to(head_attn.device))
                box_score = self._within_group_score(head_attn, self._box_mask.to(head_attn.device))

                scores = {'row': row_score, 'column': col_score, 'box': box_score}
                max_label = max(scores, key=scores.get)
                # Only label as specialized if the max score is substantially above 1.0
                specialization = max_label if scores[max_label] > 1.2 else 'mixed'

                profiles.append(
                    HeadProfile(
                        layer=layer_key,
                        head_idx=h,
                        row_score=row_score,
                        col_score=col_score,
                        box_score=box_score,
                        specialization=specialization,
                    )
                )
        return profiles

    @staticmethod
    def _within_group_score(attention: Tensor, group_mask: Tensor) -> float:
        """
        Compute ratio of within-group attention to overall attention.

        Args:
            attention: (81, 81) single-head attention matrix.
            group_mask: (81, 81) boolean mask where True = same group.

        Returns:
            Ratio of mean within-group attention to overall mean attention.

        """
        overall_mean = attention.mean().item()
        if overall_mean < 1e-10:
            return 1.0
        within_mean = attention[group_mask].mean().item()
        return within_mean / overall_mean

    def correlate_with_events(
        self,
        profiles: list[HeadProfile],
        events: list[CellEvent],
        attention_maps: dict[str, Tensor],
    ) -> dict[str, list[HeadProfile]]:
        """
        Find which heads are most active at changed cell positions per strategy.

        Args:
            profiles: Head profiles from compute_head_profiles.
            events: Classified cell events.
            attention_maps: {layer_key: (B, n_heads, 81, 81)}.

        Returns:
            Mapping from strategy label to ranked list of relevant head profiles.

        """
        strategy_events: dict[str, list[CellEvent]] = defaultdict(list)
        for event in events:
            label = event.strategy.value if event.strategy else 'unknown'
            strategy_events[label].append(event)

        result: dict[str, list[HeadProfile]] = {}
        for strategy_label, strat_events in strategy_events.items():
            # Collect positions for this strategy
            positions = [event.row * 9 + event.col for event in strat_events]
            if not positions:
                continue

            # Score each head by its attention to these positions
            head_scores: list[tuple[HeadProfile, float]] = []
            for profile in profiles:
                if profile.layer not in attention_maps:
                    continue
                attn = attention_maps[profile.layer].float().mean(dim=0)  # (n_heads, 81, 81)
                head_attn = attn[profile.head_idx]  # (81, 81)
                # Average attention received at the changed positions
                score = sum(head_attn[:, pos].mean().item() for pos in positions) / len(positions)
                head_scores.append((profile, score))

            head_scores.sort(key=lambda x: x[1], reverse=True)
            result[strategy_label] = [hs[0] for hs in head_scores]

        return result
