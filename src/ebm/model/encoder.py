"""Sudoku-aware Transformer encoder for puzzle and solution grids."""

import torch
from torch import Tensor, nn

from ebm.utils.config import ArchitectureConfig

# Box indices 0-8 map to the 3x3 sub-grids, left-to-right, top-to-bottom.
BOX_INDICES = torch.tensor(
    [(r // 3) * 3 + c // 3 for r in range(9) for c in range(9)],
    dtype=torch.long,
)


class SudokuPositionalEncoding(nn.Module):
    """
    Learned positional encoding that encodes Sudoku structure.

    Each cell gets the sum of three learned embeddings: row (0-8),
    column (0-8), and box (0-8). This explicitly encodes the constraint
    groups that define Sudoku.
    """

    def __init__(self, d_model: int) -> None:
        """
        Initialize row, column, and box embedding tables.

        Args:
            d_model: Embedding dimension.

        """
        super().__init__()
        self.row_embed = nn.Embedding(9, d_model)
        self.col_embed = nn.Embedding(9, d_model)
        self.box_embed = nn.Embedding(9, d_model)

        rows = torch.arange(9).unsqueeze(1).expand(9, 9).reshape(81)
        cols = torch.arange(9).unsqueeze(0).expand(9, 9).reshape(81)
        self.register_buffer('row_ids', rows)
        self.register_buffer('col_ids', cols)
        self.register_buffer('box_ids', BOX_INDICES.clone())

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to token embeddings.

        Args:
            x: (B, 81, d_model) token embeddings.

        Returns:
            (B, 81, d_model) with positional information added.

        """
        pos = self.row_embed(self.row_ids) + self.col_embed(self.col_ids) + self.box_embed(self.box_ids)
        return x + pos


class SudokuEncoder(nn.Module):
    """
    Transformer encoder for Sudoku grids.

    Takes a (B, C, 9, 9) grid, reshapes to 81 tokens, projects to d_model,
    adds Sudoku-aware positional encoding, and processes through pre-norm
    Transformer layers. Returns a (B, d_model) representation via mean pooling.
    """

    def __init__(self, input_channels: int, cfg: ArchitectureConfig | None = None) -> None:
        """
        Initialize encoder.

        Args:
            input_channels: Number of input channels (10 for puzzle, 9 for solution).
            cfg: Architecture config. Uses defaults if None.

        """
        super().__init__()
        if not cfg:
            cfg = ArchitectureConfig()

        self.d_model = cfg.d_model
        self.input_proj = nn.Linear(input_channels, cfg.d_model)
        self.pos_enc = SudokuPositionalEncoding(cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ffn,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode a Sudoku grid to a fixed-size representation.

        Args:
            x: (B, C, 9, 9) input grid.

        Returns:
            (B, d_model) pooled representation.

        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, x.shape[1], 81).permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x.mean(dim=1)
