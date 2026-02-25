"""Hook-based attention weight extraction from Transformer layers."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

from ebm.model.jepa import SudokuJEPA


class AttentionExtractor:
    """
    Extract attention weights from MultiheadAttention modules via hooks.

    Registers forward pre-hooks to inject ``need_weights=True`` and
    ``average_attn_weights=False``, and forward hooks to capture the returned
    attention weight tensors. Supports context manager usage for clean
    attach/detach lifecycle.
    """

    def __init__(self, model: SudokuJEPA) -> None:
        """Initialize with the model to extract attention from."""
        self._model = model
        self._handles: list[nn.utils.hooks.RemovableHook] = []
        self._buffer: dict[str, Tensor] = {}

    def attach(self) -> None:
        """Register hooks on all MultiheadAttention sub-modules."""
        self._discover_and_hook(self._model.context_encoder, 'encoder')
        self._discover_and_hook(self._model.decoder, 'decoder')

    def detach(self) -> None:
        """Remove all registered hooks and clear the buffer."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._buffer.clear()

    def get_attention_maps(self) -> dict[str, Tensor]:
        """Return captured attention maps and clear the buffer."""
        maps = dict(self._buffer)
        self._buffer.clear()
        return maps

    def __enter__(self) -> AttentionExtractor:
        """Attach hooks on context manager entry."""
        self.attach()
        return self

    def __exit__(self, *_: object) -> None:
        """Detach hooks on context manager exit."""
        self.detach()

    def _discover_and_hook(self, module: nn.Module, prefix: str) -> None:
        """Find MHA sub-modules and register pre-hook + post-hook pairs."""
        for name, submodule in module.named_modules():
            if isinstance(submodule, nn.MultiheadAttention):
                key = f'{prefix}.{name}'
                self._handles.append(submodule.register_forward_pre_hook(self._make_pre_hook(), with_kwargs=True))
                self._handles.append(submodule.register_forward_hook(self._make_post_hook(key)))

    @staticmethod
    def _make_pre_hook() -> Callable:
        """Create a pre-hook that forces attention weight output."""

        def hook(_module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
            kwargs = dict(kwargs)
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False
            return args, kwargs

        return hook

    def _make_post_hook(self, key: str) -> Callable:
        """Create a post-hook that captures attention weights."""

        def hook(_module: nn.Module, _input: tuple, output: tuple) -> None:
            # MHA returns (attn_output, attn_weights) when need_weights=True
            _min_output_len = 2
            if isinstance(output, tuple) and len(output) >= _min_output_len and output[1] is not None:
                self._buffer[key] = output[1].detach()

        return hook
