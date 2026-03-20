from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.pipelines_core.types.bucketing import BucketMeta
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def pad_like_video_latents_hw(
    latents: torch.Tensor,
    bucket_h: int,
    bucket_w: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pads latents on H/W only (v1).

    Supports 5D BCTHW latents.
    """
    if latents.ndim != 5:
        raise ValueError(f"Expected 5D latents, got shape={tuple(latents.shape)}")

    H, W = int(latents.shape[-2]), int(latents.shape[-1])

    if bucket_h < H or bucket_w < W:
        raise ValueError(
            f"Bucket ({bucket_h},{bucket_w}) smaller than request ({H},{W})"
        )

    pad_h = int(bucket_h) - H
    pad_w = int(bucket_w) - W

    if pad_h == 0 and pad_w == 0:
        return latents, (0, 0)

    pad_args = (0, pad_w, 0, pad_h, 0, 0)
    padded = F.pad(latents, pad_args, mode="constant", value=0).contiguous()
    return padded, (pad_h, pad_w)


def _default_hw_buckets() -> List[Tuple[int, int]]:
    # v1 default: square-only buckets in latent H/W.
    # For a typical VAE scale factor of 8, these correspond to 512/768/1024px images.
    return [(64, 64), (96, 96), (128, 128)]


def _get_hw_buckets_from_args(server_args: ServerArgs) -> List[Tuple[int, int]]:
    """Get HW bucket table from server args.

    Buckets are specified in latent H/W units (i.e. the spatial resolution of the
    latent tensor after VAE encoding / before DiT/UNet).
    """
    buckets = getattr(server_args, "diffusion_hw_buckets", None)
    if not buckets:
        return _default_hw_buckets()

    # Handle comma-separated string input (e.g., "64x64,96x96")
    if isinstance(buckets, str):
        buckets = [s.strip() for s in buckets.split(",") if s.strip()]

    # Accept either ["64x64", ...] or [(64,64), ...]
    parsed: List[Tuple[int, int]] = []
    for b in buckets:
        if isinstance(b, str):
            parts = b.lower().split("x")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    f"Invalid diffusion_hw_buckets entry '{b}'. Expected 'HxW' like '64x64'."
                )
            try:
                h = int(parts[0])
                w = int(parts[1])
            except ValueError as e:
                raise ValueError(
                    f"Invalid diffusion_hw_buckets entry '{b}'. Expected 'HxW' like '64x64'."
                ) from e
        else:
            try:
                h, w = b
            except Exception as e:
                raise ValueError(
                    f"Invalid diffusion_hw_buckets entry {b!r}. Expected (H, W) or 'HxW' like '64x64'."
                ) from e
            try:
                h = int(h)
                w = int(w)
            except ValueError as e:
                raise ValueError(
                    f"Invalid diffusion_hw_buckets entry {b!r}. Expected integer (H, W)."
                ) from e

        if h <= 0 or w <= 0:
            raise ValueError(
                f"Invalid diffusion_hw_buckets entry {b!r}. Height/width must be positive."
            )

        parsed.append((h, w))

    return parsed


def _select_hw_bucket(
    H: int,
    W: int,
    buckets: List[Tuple[int, int]],
    max_waste: float,
) -> Optional[Tuple[int, int, float, int]]:
    """
    Returns (Hb, Wb, padding_ratio, bucket_id) using the smallest-area feasible bucket.
    padding_ratio = 1 - (H*W)/(Hb*Wb)
    """
    best = None  # (area, Hb, Wb, ratio, idx)
    for idx, (Hb, Wb) in enumerate(buckets):
        if Hb < H or Wb < W:
            continue
        ratio = 1.0 - (H * W) / float(Hb * Wb)
        if ratio > max_waste:
            continue
        area = Hb * Wb
        cand = (area, Hb, Wb, ratio, idx)
        if best is None or cand < best:
            best = cand

    if best is None:
        return None
    _, Hb, Wb, ratio, idx = best
    return Hb, Wb, ratio, idx


def apply_shape_bucketing(
    latents: torch.Tensor,
    server_args: ServerArgs,
    orig_thw: tuple[int, int, int] | None = None,
) -> Tuple[torch.Tensor, BucketMeta]:
    enabled = bool(
        getattr(
            server_args,
            "enable_diffusion_hw_bucketing",
            getattr(server_args, "enable_shape_bucketing", False),
        )
    )
    max_waste = float(getattr(server_args, "diffusion_hw_bucketing_max_waste", 0.20))

    if isinstance(orig_thw, (list, tuple)) and len(orig_thw) == 3:
        orig_thw = (int(orig_thw[0]), int(orig_thw[1]), int(orig_thw[2]))
    else:
        orig_thw = None

    if latents.ndim != 5:
        if enabled:
            raise ValueError(
                f"v1 diffusion HW bucketing expects 5D BCTHW latents, got shape={tuple(latents.shape)}"
            )
        return latents, BucketMeta(
            enabled=False,
            layout="BCTHW",
            orig_t=1,
            orig_h=0,
            orig_w=0,
            bucket_h=0,
            bucket_w=0,
            attention_mask=None,
            padding_ratio=0.0,
            bucket_id=-1,
        )

    T_lat, H_lat, W_lat = (
        int(latents.shape[2]),
        int(latents.shape[3]),
        int(latents.shape[4]),
    )
    if orig_thw is None:
        orig_thw = (T_lat, H_lat, W_lat)
    T, H, W = orig_thw

    if not enabled:
        return latents, BucketMeta(
            enabled=False,
            layout="BCTHW",
            orig_t=T,
            orig_h=H,
            orig_w=W,
            bucket_h=H,
            bucket_w=W,
            attention_mask=None,
            padding_ratio=0.0,
            bucket_id=-1,
        )

    buckets = _get_hw_buckets_from_args(server_args)
    sel = _select_hw_bucket(H, W, buckets=buckets, max_waste=max_waste)
    if sel is None:
        return latents, BucketMeta(
            enabled=False,
            layout="BCTHW",
            orig_t=T,
            orig_h=H,
            orig_w=W,
            bucket_h=H,
            bucket_w=W,
            attention_mask=None,
            padding_ratio=0.0,
            bucket_id=-1,
        )

    Hb, Wb, ratio, bucket_id = sel
    padded, _ = pad_like_video_latents_hw(latents, Hb, Wb)

    meta = BucketMeta(
        enabled=True,
        layout="BCTHW",
        orig_t=T,
        orig_h=H,
        orig_w=W,
        bucket_h=Hb,
        bucket_w=Wb,
        attention_mask=None,
        padding_ratio=ratio,
        bucket_id=bucket_id,
    )
    return padded, meta


def remove_shape_bucketing(
    latents: torch.Tensor,
    meta: BucketMeta,
) -> torch.Tensor:
    if not meta.enabled:
        return latents

    if meta.layout == "BCTHW":
        if latents.ndim == 5:
            return latents[:, :, :, : meta.orig_h, : meta.orig_w]
        return latents

    return latents


def get_orig_thw(meta: BucketMeta) -> tuple[int, int, int]:
    """Return the original (T, H, W) dimensions from BucketMeta."""
    return (int(meta.orig_t), int(meta.orig_h), int(meta.orig_w))
