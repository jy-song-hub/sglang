import torch

from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def build_key_padding_mask_bcthw(
    B: int,
    exec_thw: tuple[int, int, int],
    orig_thw: tuple[int, int, int],
    patch_thw: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """
    Builds a canonical boolean key-padding mask for a BCTHW layout that has been
    patched and flattened into a token sequence.

    This logic is centralized here to be easily testable and reusable. It assumes
    the patch embedding process uses floor-division semantics, common in Conv3D layers.

    Args:
        B: Batch size.
        exec_thw: The bucketed (padded) execution dimensions (T, H, W).
        orig_thw: The original, logical dimensions (T, H, W).
        patch_thw: The patch size dimensions (pt, ph, pw).
        device: The torch device to create the mask on.

    Returns:
        A tuple containing:
        - A boolean tensor of shape [B, S] where True indicates a valid token
          and False indicates a padding token.
        - An integer S, the total number of tokens after padding.
    """
    T_b, H_b, W_b = exec_thw
    T_o, H_o, W_o = orig_thw
    pt, ph, pw = patch_thw

    # Calculate token dimensions based on floor division, mimicking Conv3D
    Tt_b, Ht_b, Wt_b = T_b // pt, H_b // ph, W_b // pw
    Tt_o, Ht_o, Wt_o = T_o // pt, H_o // ph, W_o // pw

    # Build a 3D grid representing the token layout
    mask_3d = torch.zeros((B, Tt_b, Ht_b, Wt_b), dtype=torch.bool, device=device)

    # Mark the subgrid corresponding to the original dimensions as valid (True)
    mask_3d[:, :Tt_o, :Ht_o, :Wt_o] = True

    # Flatten to the final [B, S] sequence length, assuming T-H-W flatten order
    mask_flat = mask_3d.reshape(B, -1)
    num_tokens = mask_flat.shape[1]

    return mask_flat, num_tokens


def should_use_masked_bucketing(
    attn_backend: AttentionBackendEnum,
    sp_world_size: int,
    is_dit: bool,
) -> bool:
    """
    Centralized whitelist logic to determine if masked bucketing is allowed.
    V1: DiT-only, single-GPU, SDPA-only.
    """
    return (
        attn_backend == AttentionBackendEnum.TORCH_SDPA
        and sp_world_size == 1
        and is_dit
    )
