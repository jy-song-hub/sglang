# python_tests/test_sdpa_masked_bucketing_cpu.py
import types

import pytest
import torch

import sglang.multimodal_gen.runtime.layers.attention.backends.sdpa as sdpa_mod

# Import the backend under test (this file is lightweight)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl


def build_valid_mask_floor(
    *,
    B: int,
    exec_thw: tuple[int, int, int],  # (T_b, H_b, W_b)
    orig_thw: tuple[int, int, int],  # (T_o, H_o, W_o)
    patch_thw: tuple[int, int, int],  # (pt, ph, pw)
    device: torch.device,
) -> torch.Tensor:
    """
    Minimal replica of your denoising._build_key_padding_mask() logic:
    - Conv3D patch embedding uses FLOOR division (//)
    - Token order is T-major then H then W => flatten(T,H,W)
    - Returns bool [B,S] where True=VALID token, False=PAD token
    """
    T_b, H_b, W_b = exec_thw
    T_o, H_o, W_o = orig_thw
    pt, ph, pw = patch_thw

    assert (
        T_b % pt == 0 and H_b % ph == 0 and W_b % pw == 0
    ), "exec dims must be patch-aligned"
    assert (
        T_o % pt == 0 and H_o % ph == 0 and W_o % pw == 0
    ), "orig dims must be patch-aligned for this unit test"

    Tt_b, Ht_b, Wt_b = T_b // pt, H_b // ph, W_b // pw
    Tt_o, Ht_o, Wt_o = T_o // pt, H_o // ph, W_o // pw

    mask_3d = torch.zeros((B, Tt_b, Ht_b, Wt_b), dtype=torch.bool, device=device)
    mask_3d[:, :Tt_o, :Ht_o, :Wt_o] = True
    mask = mask_3d.reshape(B, -1)  # [B, S]
    return mask


def make_qkv(B: int, S: int, H: int, Dh: int, device: torch.device):
    """
    SDPAImpl.forward expects query/key/value shaped like [B, S, H, Dh]
    because it does transpose(1,2) to [B, H, S, Dh].
    """
    q = torch.randn(B, S, H, Dh, device=device, dtype=torch.float32)
    k = torch.randn(B, S, H, Dh, device=device, dtype=torch.float32)
    v = torch.randn(B, S, H, Dh, device=device, dtype=torch.float32)
    return q, k, v


def test_sdpa_impl_calls_torch_scaled_dot_product_attention_cpu(monkeypatch):
    device = torch.device("cpu")
    B, S, H_heads, Dh = 1, 16, 2, 32

    sdpa = SDPAImpl(
        num_heads=H_heads,
        head_size=Dh,
        causal=False,
        softmax_scale=1.0,
        dropout_p=0.0,
    )

    q, k, v = make_qkv(B, S, H_heads, Dh, device=device)

    counter = {"n": 0}
    orig = sdpa_mod.F.scaled_dot_product_attention

    def wrapped(*args, **kwargs):
        counter["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(sdpa_mod.F, "scaled_dot_product_attention", wrapped)

    out = sdpa.forward(q, k, v, attn_metadata=None)
    assert out.shape == (B, S, H_heads, Dh)
    assert counter["n"] == 1


def test_sdpa_impl_consumes_attention_mask_and_builds_attn_mask_cpu():
    device = torch.device("cpu")
    B = 2
    # Exec (bucketed) dims, Orig dims => padding exists
    exec_thw = (8, 16, 16)
    orig_thw = (8, 8, 16)  # H padded from 8 -> 16
    patch_thw = (1, 8, 8)  # example patch size (pt,ph,pw)

    valid_mask = build_valid_mask_floor(
        B=B, exec_thw=exec_thw, orig_thw=orig_thw, patch_thw=patch_thw, device=device
    )
    assert valid_mask.dtype == torch.bool
    assert valid_mask.ndim == 2
    assert valid_mask.shape[0] == B

    # S = (T//pt)*(H//ph)*(W//pw)
    T_b, H_b, W_b = exec_thw
    pt, ph, pw = patch_thw
    S = (T_b // pt) * (H_b // ph) * (W_b // pw)
    assert valid_mask.shape[1] == S

    # Build SDPA impl
    H_heads = 4
    Dh = 32
    sdpa = SDPAImpl(
        num_heads=H_heads,
        head_size=Dh,
        causal=False,
        softmax_scale=1.0,
        dropout_p=0.0,
    )

    q, k, v = make_qkv(B, S, H_heads, Dh, device=device)

    # Provide attn_metadata with attention_mask (True=VALID)
    attn_metadata = types.SimpleNamespace(attention_mask=valid_mask)

    # Forward should succeed and return correct shape
    out = sdpa.forward(q, k, v, attn_metadata=attn_metadata)
    assert out.shape == (B, S, H_heads, Dh)


def test_sdpa_impl_mask_shape_validation_cpu():
    device = torch.device("cpu")
    B, S, H, Dh = 2, 32, 2, 16
    sdpa = SDPAImpl(
        num_heads=H, head_size=Dh, causal=False, softmax_scale=1.0, dropout_p=0.0
    )
    q, k, v = make_qkv(B, S, H, Dh, device=device)

    # Wrong S in mask => should fail (your code checks against key.shape[-2] after transpose)
    bad_mask = torch.ones((B, S + 1), dtype=torch.bool, device=device)
    attn_metadata = types.SimpleNamespace(attention_mask=bad_mask)

    with pytest.raises(ValueError):
        _ = sdpa.forward(q, k, v, attn_metadata=attn_metadata)


def test_sdpa_masked_matches_truncated_baseline_simple_cpu():
    """
    Robust CPU-only masking check:

    If the mask polarity and wiring are correct, then changing K/V at padded token
    positions must NOT affect outputs at valid query positions (because those
    padded keys are masked out).
    """
    device = torch.device("cpu")
    B = 1
    exec_thw = (8, 16, 16)
    orig_thw = (8, 8, 16)  # padding exists
    patch_thw = (1, 8, 8)

    valid_mask = build_valid_mask_floor(
        B=B, exec_thw=exec_thw, orig_thw=orig_thw, patch_thw=patch_thw, device=device
    )
    S = valid_mask.shape[1]
    valid_idx = valid_mask[0].nonzero(as_tuple=False).squeeze(-1)
    padded_idx = (~valid_mask[0]).nonzero(as_tuple=False).squeeze(-1)
    assert valid_idx.numel() > 0
    assert padded_idx.numel() > 0

    H_heads = 2
    Dh = 16
    sdpa = SDPAImpl(
        num_heads=H_heads,
        head_size=Dh,
        causal=False,
        softmax_scale=1.0,
        dropout_p=0.0,
    )

    torch.manual_seed(0)
    q, k, v = make_qkv(B, S, H_heads, Dh, device=device)

    attn_metadata = types.SimpleNamespace(attention_mask=valid_mask)
    out_1 = sdpa.forward(q, k, v, attn_metadata=attn_metadata)  # [B,S,H,Dh]

    # Perturb ONLY the padded token K/V drastically. Valid outputs must stay stable.
    k2 = k.clone()
    v2 = v.clone()
    k2[:, padded_idx] = torch.randn_like(k2[:, padded_idx]) * 1000.0
    v2[:, padded_idx] = torch.randn_like(v2[:, padded_idx]) * 1000.0

    out_2 = sdpa.forward(q, k2, v2, attn_metadata=attn_metadata)

    torch.testing.assert_close(
        out_1[:, valid_idx],
        out_2[:, valid_idx],
        rtol=1e-4,
        atol=1e-4,
    )


def test_sdpa_mask_adversarial_pad_constant_leakage_cpu():
    """Adversarial mask-leakage check.

    Set padded-token K/V values to a huge constant and verify:
    - With mask: outputs at valid token positions do not change.
    - Without mask: outputs at valid positions do change (sanity that test is meaningful).
    """
    device = torch.device("cpu")
    B = 1
    exec_thw = (8, 16, 16)
    orig_thw = (8, 8, 16)  # padding exists
    patch_thw = (1, 8, 8)

    valid_mask = build_valid_mask_floor(
        B=B, exec_thw=exec_thw, orig_thw=orig_thw, patch_thw=patch_thw, device=device
    )
    S = valid_mask.shape[1]
    valid_idx = valid_mask[0].nonzero(as_tuple=False).squeeze(-1)
    padded_idx = (~valid_mask[0]).nonzero(as_tuple=False).squeeze(-1)
    assert valid_idx.numel() > 0
    assert padded_idx.numel() > 0

    H_heads = 2
    Dh = 16
    sdpa = SDPAImpl(
        num_heads=H_heads,
        head_size=Dh,
        causal=False,
        softmax_scale=1.0,
        dropout_p=0.0,
    )

    torch.manual_seed(0)
    q, k, v = make_qkv(B, S, H_heads, Dh, device=device)

    huge = 1.0e3
    k_huge = k.clone()
    v_huge = v.clone()
    k_huge[:, padded_idx] = huge
    v_huge[:, padded_idx] = huge

    # Masked: should be invariant at valid positions
    attn_metadata = types.SimpleNamespace(attention_mask=valid_mask)
    out_masked_base = sdpa.forward(q, k, v, attn_metadata=attn_metadata)
    out_masked_huge = sdpa.forward(q, k_huge, v_huge, attn_metadata=attn_metadata)
    torch.testing.assert_close(
        out_masked_base[:, valid_idx],
        out_masked_huge[:, valid_idx],
        rtol=1e-4,
        atol=1e-4,
    )

    # Unmasked: should change at valid positions (otherwise test is not adversarial)
    out_nomask_base = sdpa.forward(q, k, v, attn_metadata=None)
    out_nomask_huge = sdpa.forward(q, k_huge, v_huge, attn_metadata=None)
    max_diff = (
        (out_nomask_base[:, valid_idx] - out_nomask_huge[:, valid_idx])
        .abs()
        .max()
        .item()
    )
    assert max_diff > 1e-2
