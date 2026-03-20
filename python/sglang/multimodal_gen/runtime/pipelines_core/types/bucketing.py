from dataclasses import dataclass

import torch


@dataclass
class BucketMeta:
    """Metadata for a bucketed diffusion request.

    v1 primarily targets DiT-style HW bucketing, but the same container is also
    used by utilities that operate on packed-token layouts (e.g. "BSD").
    """

    enabled: bool
    layout: str  # e.g. "BCTHW", "BCHW", "BSD"

    # Original logical dims (execution should be cropped back to these)
    orig_t: int
    orig_h: int
    orig_w: int

    # Bucketed execution dims
    bucket_h: int
    bucket_w: int

    # Optional: canonical bool [B, S] mask.
    attention_mask: torch.Tensor | None = None

    # Optional: authoritative token count for the bucketed execution shape.
    num_tokens: int | None = None

    # Spatial padding ratio: 1 - (H*W)/(Hb*Wb)
    padding_ratio: float = 0.0

    # Optional, but useful for graph cache keying / metrics
    bucket_id: int = -1
