from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor


def wan_latents_to_token_summary(latents: Tensor, patch_size: tuple[int, int] = (2, 2)) -> Tensor:
    """Convert Wan latent frames to TokenTrim spatial-token summaries.

    For the default 60x104 latent grid, a 2x2 spatial patch gives 30x52=1560
    spatial tokens per frame, matching RollingForcing's cache layout.
    """

    if latents.ndim != 5:
        raise ValueError("Wan latents must have shape [batch, frames, channels, height, width]")

    batch, frames, channels, height, width = latents.shape
    patch_h, patch_w = patch_size
    if height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(f"latent grid {(height, width)} is not divisible by patch size {patch_size}")

    patched = latents.reshape(
        batch,
        frames,
        channels,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    patched = patched.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    patched = patched.reshape(batch, frames, (height // patch_h) * (width // patch_w), channels * patch_h * patch_w)
    return patched.mean(dim=1)


def suppress_rolling_forcing_cache_tokens(
    kv_cache: list[dict],
    token_indices: Tensor | Iterable[int],
    frame_seq_length: int = 1560,
    block_length: int | None = None,
    sink_blocks: int = 1,
    value: float = 0.0,
    scale: float | None = None,
) -> None:
    """Suppress selected spatial tokens in RollingForcing's working KV cache.

    The cache has fixed storage and positional offsets, so this zeroes selected
    K/V entries rather than physically removing entries.
    """

    if not kv_cache:
        return

    if isinstance(token_indices, Tensor):
        indices = token_indices.detach().flatten().to(dtype=torch.long)
    else:
        indices = torch.tensor(list(token_indices), dtype=torch.long)

    if indices.numel() == 0:
        return
    if block_length is None:
        block_length = 3 * frame_seq_length
    if torch.any(indices < 0) or torch.any(indices >= frame_seq_length):
        raise ValueError("token_indices must refer to transformer spatial tokens in one frame")

    sink_tokens = sink_blocks * block_length

    for layer_cache in kv_cache:
        local_end = int(layer_cache["local_end_index"].item())
        if local_end <= sink_tokens:
            continue

        device_indices = indices.to(device=layer_cache["k"].device)
        frame_starts = torch.arange(
            sink_tokens,
            local_end,
            frame_seq_length,
            device=layer_cache["k"].device,
            dtype=torch.long,
        )
        positions = (frame_starts[:, None] + device_indices[None, :]).flatten()
        positions = positions[positions < local_end]
        if positions.numel() == 0:
            continue

        if scale is None:
            layer_cache["k"][:, positions] = value
            layer_cache["v"][:, positions] = value
        else:
            layer_cache["k"][:, positions] *= scale
            layer_cache["v"][:, positions] *= scale
