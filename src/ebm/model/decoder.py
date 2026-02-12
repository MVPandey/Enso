"""Decoder that maps latent representations back to Sudoku cell logits."""

import torch
from torch import Tensor, nn

from ebm.model.encoder import SudokuPositionalEncoding
from ebm.utils.config import ArchitectureConfig


class SudokuDecoder(nn.Module):
    """
    Decode (z, z_context) back to per-cell digit logits.

    Concatenates z and z_context, projects to per-cell features, adds
    Sudoku-aware positional encoding, refines through lightweight Transformer
    layers, and outputs (B, 9, 9, 9) logits. Given clues are hard-enforced
    via the puzzle mask.
    """

    def __init__(self, cfg: ArchitectureConfig | None = None) -> None:
        """
        Initialize decoder.

        Args:
            cfg: Architecture config. Uses defaults if None.

        """
        super().__init__()
        if not cfg:
            cfg = ArchitectureConfig()

        input_dim = cfg.d_model + cfg.d_latent
        self.cell_proj = nn.Linear(input_dim, 81 * cfg.decoder_d_cell)
        self.d_cell = cfg.decoder_d_cell
        self.pos_enc = SudokuPositionalEncoding(cfg.decoder_d_cell)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.decoder_d_cell,
            nhead=cfg.decoder_heads,
            dim_feedforward=cfg.decoder_d_cell * 4,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=cfg.decoder_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(cfg.decoder_d_cell, 9)

    def forward(self, z_context: Tensor, z: Tensor, puzzle: Tensor, mask: Tensor) -> Tensor:
        """
        Decode latent variables to cell logits with clue enforcement.

        Args:
            z_context: (B, d_model) from context encoder.
            z: (B, d_latent) latent variable.
            puzzle: (B, 10, 9, 9) one-hot encoded puzzle.
            mask: (B, 9, 9) binary mask, 1 = given clue.

        Returns:
            (B, 9, 9, 9) logits over digits 1-9 for each cell.

        """
        batch_size = z_context.shape[0]
        x = torch.cat([z_context, z], dim=-1)
        x = self.cell_proj(x)
        x = x.reshape(batch_size, 81, self.d_cell)
        x = self.pos_enc(x)
        x = self.transformer(x)
        logits = self.head(x)
        logits = logits.reshape(batch_size, 9, 9, 9)

        clue_logits = puzzle[:, 1:].permute(0, 2, 3, 1)
        clue_mask = mask.unsqueeze(-1)
        logits = logits * (1 - clue_mask) + clue_logits * clue_mask * 1e6

        return logits
