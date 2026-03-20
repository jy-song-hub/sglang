from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    _infer_orig_thw_from_latents,
)

# Import the new, lightweight utility functions under test
from sglang.multimodal_gen.runtime.pipelines_core.utils.bucketing_masks import (
    build_key_padding_mask_bcthw,
    should_use_masked_bucketing,
)
from sglang.multimodal_gen.runtime.pipelines_core.utils.shape_bucketing import (
    apply_shape_bucketing,
)

# Import the correct Enum from the correct location
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

# === Test A: Pure function test for mask generation ===


@pytest.mark.parametrize(
    "orig_h, orig_w, bucket_h, bucket_w",
    [
        (32, 32, 32, 32),  # Scenario 1: No padding
        (24, 32, 32, 32),  # Scenario 2: H is padded
        (32, 24, 32, 32),  # Scenario 3: W is padded
        (16, 24, 32, 32),  # Scenario 4: Both H and W are padded
    ],
)
def test_build_key_padding_mask_bcthw_logic(orig_h, orig_w, bucket_h, bucket_w):
    """
    Verifies the pure mask generation utility `build_key_padding_mask_bcthw`
    correctly generates a mask using floor-division semantics.
    This test is now stable, fast, and has no heavy dependencies.
    """
    device = "cpu"
    B, T = 2, 8
    patch_thw = (1, 8, 8)
    pt, ph, pw = patch_thw

    exec_thw = (T, bucket_h, bucket_w)
    orig_thw = (T, orig_h, orig_w)

    # --- Method under test ---
    mask, num_tokens = build_key_padding_mask_bcthw(
        B, exec_thw, orig_thw, patch_thw, device
    )

    # --- Verification ---
    assert isinstance(num_tokens, int)
    assert mask.dtype == torch.bool
    assert mask.device.type == device

    # Total tokens for the bucketed (padded) shape
    total_tokens = (T // pt) * (bucket_h // ph) * (bucket_w // pw)
    assert mask.shape == (B, total_tokens)
    assert num_tokens == total_tokens

    # Number of valid tokens for the original shape
    valid_tokens = (T // pt) * (orig_h // ph) * (orig_w // pw)
    assert mask.sum().item() == B * valid_tokens

    # Reconstruct the expected mask manually to confirm the logic is identical
    Tt_b, Ht_b, Wt_b = T // pt, bucket_h // ph, bucket_w // pw
    Tt_o, Ht_o, Wt_o = T // pt, orig_h // ph, orig_w // pw

    mask_3d = torch.zeros((B, Tt_b, Ht_b, Wt_b), dtype=torch.bool, device=device)
    mask_3d[:, :Tt_o, :Ht_o, :Wt_o] = True
    expected_mask_flat = mask_3d.reshape(B, -1)

    torch.testing.assert_close(mask, expected_mask_flat)


def test_build_key_padding_mask_bcthw_exact_match_all_true_and_num_tokens_correct():
    device = "cpu"
    B = 3
    patch_thw = (1, 8, 8)
    exec_thw = orig_thw = (8, 32, 48)

    mask, num_tokens = build_key_padding_mask_bcthw(
        B, exec_thw, orig_thw, patch_thw, device
    )

    expected_tokens = (8 // 1) * (32 // 8) * (48 // 8)
    assert num_tokens == expected_tokens
    assert mask.shape == (B, expected_tokens)
    assert mask.dtype == torch.bool
    assert mask.all().item() is True


def test_mask_token_order_matches_visual_patch_embed_thw():
    from sglang.multimodal_gen.runtime.layers.visual_embedding import PatchEmbed

    device = torch.device("cpu")
    B, C = 1, 1
    patch_thw = (1, 2, 2)
    pt, ph, pw = patch_thw

    exec_thw = (2, 4, 6)
    orig_thw = (2, 4, 4)

    T_b, H_b, W_b = exec_thw
    Tt_b, Ht_b, Wt_b = T_b // pt, H_b // ph, W_b // pw

    x = torch.zeros((B, C, T_b, H_b, W_b), device=device, dtype=torch.float32)
    token_grid_ids = torch.arange(
        Tt_b * Ht_b * Wt_b, device=device, dtype=torch.float32
    ).reshape(Tt_b, Ht_b, Wt_b)
    x[0, 0, ::pt, ::ph, ::pw] = token_grid_ids

    embed = PatchEmbed(
        patch_size=patch_thw,
        in_chans=C,
        embed_dim=1,
        flatten=False,
        bias=False,
    ).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        embed.proj.weight.zero_()
        embed.proj.weight[0, 0, 0, 0, 0] = 1.0

    tokens_grid = embed(x)
    tokens = tokens_grid.flatten(2).transpose(1, 2)
    token_ids = tokens[0, :, 0].to(dtype=torch.int64)

    expected_flat = torch.tensor(
        [
            ((t * Ht_b + h) * Wt_b + w)
            for t in range(Tt_b)
            for h in range(Ht_b)
            for w in range(Wt_b)
        ],
        device=device,
        dtype=torch.int64,
    )
    torch.testing.assert_close(token_ids, expected_flat)

    mask, num_tokens = build_key_padding_mask_bcthw(
        B, exec_thw, orig_thw, patch_thw, device
    )
    assert mask.shape == (B, num_tokens)
    assert num_tokens == token_ids.shape[0]

    T_o, H_o, W_o = orig_thw
    Tt_o, Ht_o, Wt_o = T_o // pt, H_o // ph, W_o // pw
    expected_valid = [
        ((t * Ht_b + h) * Wt_b + w)
        for t in range(Tt_o)
        for h in range(Ht_o)
        for w in range(Wt_o)
    ]

    valid_idx = mask[0].nonzero(as_tuple=False).flatten().tolist()
    assert valid_idx == expected_valid
    torch.testing.assert_close(
        token_ids[mask[0]],
        torch.tensor(expected_valid, device=device, dtype=torch.int64),
    )


def test_sdpa_consumes_bucket_mask_with_correct_sequence_length():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl

    device = torch.device("cpu")
    B, H, D = 2, 2, 8

    patch_thw = (1, 2, 2)
    exec_thw = (2, 4, 6)
    orig_thw = (2, 4, 4)

    mask, num_tokens = build_key_padding_mask_bcthw(
        B, exec_thw, orig_thw, patch_thw, device
    )
    attn_metadata = SimpleNamespace(key_padding_mask=mask)

    q = torch.randn((B, num_tokens, H, D), device=device)
    k = torch.randn((B, num_tokens, H, D), device=device)
    v = torch.randn((B, num_tokens, H, D), device=device)

    impl = SDPAImpl(num_heads=H, head_size=D, causal=False, softmax_scale=1.0)
    out = impl.forward(q, k, v, attn_metadata)
    assert out.shape == q.shape


def test_sdpa_raises_on_mismatched_mask_length():
    if not __debug__:
        pytest.skip("SDPA mask shape checks run only when __debug__ is enabled")

    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl

    device = torch.device("cpu")
    B, H, D = 2, 2, 8

    patch_thw = (1, 2, 2)
    exec_thw = (2, 4, 6)
    orig_thw = (2, 4, 4)

    mask, num_tokens = build_key_padding_mask_bcthw(
        B, exec_thw, orig_thw, patch_thw, device
    )
    bad_mask = mask[:, :-1]
    attn_metadata = SimpleNamespace(key_padding_mask=bad_mask)

    q = torch.randn((B, num_tokens, H, D), device=device)
    k = torch.randn((B, num_tokens, H, D), device=device)
    v = torch.randn((B, num_tokens, H, D), device=device)

    impl = SDPAImpl(num_heads=H, head_size=D, causal=False, softmax_scale=1.0)
    with pytest.raises(ValueError, match="Mask S != key S"):
        impl.forward(q, k, v, attn_metadata)


def test_mask_token_order_matches_qwen_image_pack_latents_full_order():
    from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import _pack_latents

    device = torch.device("cpu")
    B, C = 1, 1

    exec_thw = (1, 6, 8)
    orig_thw = (1, 4, 6)
    patch_thw = (1, 2, 2)
    _, ph, pw = patch_thw

    _, H_b, W_b = exec_thw
    Ht_b, Wt_b = H_b // ph, W_b // pw

    latents = torch.zeros((B, C, H_b, W_b), device=device, dtype=torch.float32)
    latents[0, 0, ::ph, ::pw] = torch.arange(
        Ht_b * Wt_b, device=device, dtype=torch.float32
    ).reshape(Ht_b, Wt_b)

    packed = _pack_latents(
        latents,
        batch_size=B,
        num_channels_latents=C,
        height=H_b,
        width=W_b,
    )

    token_ids = packed[0, :, 0].to(dtype=torch.int64)
    torch.testing.assert_close(
        token_ids,
        torch.arange(token_ids.shape[0], device=device, dtype=torch.int64),
    )

    mask, _ = build_key_padding_mask_bcthw(B, exec_thw, orig_thw, patch_thw, device)

    _, H_o, W_o = orig_thw
    Ht_o, Wt_o = H_o // ph, W_o // pw
    expected_valid = [(h * Wt_b + w) for h in range(Ht_o) for w in range(Wt_o)]

    valid_idx = mask[0].nonzero(as_tuple=False).flatten().tolist()
    assert valid_idx == expected_valid
    torch.testing.assert_close(
        token_ids[mask[0]],
        torch.tensor(expected_valid, device=device, dtype=torch.int64),
    )


def test_qwen_text_plus_image_concat_mask_content():
    qwen_mod = pytest.importorskip(
        "sglang.multimodal_gen.configs.pipeline_configs.qwen_image"
    )
    QwenImagePipelineConfig = qwen_mod.QwenImagePipelineConfig

    B, txt_len, img_len = 2, 5, 12

    base_img_mask = torch.tensor(
        [[True] * 8 + [False] * 4, [True] * 8 + [False] * 4],
        dtype=torch.bool,
    )
    base_txt_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]],
        dtype=torch.long,
    )

    stage = MagicMock(spec=DenoisingStage)
    stage._maybe_attach_bucket_mask = DenoisingStage._maybe_attach_bucket_mask.__get__(
        stage
    )
    stage._try_attach_attention_mask = (
        DenoisingStage._try_attach_attention_mask.__get__(stage)
    )
    stage._should_use_bucketing_with_masks = MagicMock(return_value=True)
    stage.server_args = SimpleNamespace(pipeline_config=QwenImagePipelineConfig())

    batch = SimpleNamespace(
        is_cfg_negative=False,
        prompt_embeds=[torch.zeros((B, txt_len, 8), dtype=torch.float32)],
        prompt_embeds_mask=[base_txt_mask],
        latents=torch.zeros((B, 4, 1, 32, 32), device="cpu"),
        exec_latent_shape=(1, 32, 32),
        bucket_meta=SimpleNamespace(
            enabled=True,
            padding_ratio=0.25,
            attention_mask=base_img_mask,
            num_tokens=img_len,
        ),
    )

    attn_md = stage._maybe_attach_bucket_mask(batch, None)

    expected = torch.cat([base_txt_mask.bool(), base_img_mask], dim=1)

    assert attn_md is batch.sdpa_attn_metadata_storage
    assert attn_md.key_padding_mask.shape == (B, txt_len + img_len)
    assert attn_md.key_padding_mask.dtype == torch.bool
    assert attn_md.key_padding_mask.device == batch.latents.device
    torch.testing.assert_close(attn_md.key_padding_mask, expected)

    assert hasattr(attn_md, "attention_mask")
    assert attn_md.attention_mask.shape == (B, txt_len + img_len)
    torch.testing.assert_close(attn_md.attention_mask, expected)
    assert attn_md.key_padding_mask.sum().item() == expected.sum().item()


# === Test B: Orchestration test for the decision-making helper ===


@pytest.mark.parametrize(
    "attn_backend, sp_world_size, is_dit, expected",
    [
        # V1 happy path: SDPA, single GPU, DiT
        (AttentionBackendEnum.TORCH_SDPA, 1, True, True),
        # Fallback cases
        (AttentionBackendEnum.VMOBA_ATTN, 1, True, False),  # Incorrect backend
        (AttentionBackendEnum.TORCH_SDPA, 2, True, False),  # SP > 1 is not supported
        (AttentionBackendEnum.TORCH_SDPA, 1, False, False),  # Not a DiT pipeline
    ],
)
def test_should_use_masked_bucketing_logic(
    attn_backend, sp_world_size, is_dit, expected
):
    """
    Verifies the pure orchestration utility `should_use_masked_bucketing`
    correctly returns True or False based on the system state.
    This test requires no mocking or patching.
    """
    # --- Method under test ---
    result = should_use_masked_bucketing(
        attn_backend=attn_backend,
        sp_world_size=sp_world_size,
        is_dit=is_dit,
    )

    # --- Verification ---
    assert result == expected


# === Test C: Orchestration test for the in-loop mask attachment helper ===

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# We need to import DenoisingStage to attach our method to a mock instance of it.
# This is safe because we will not be initializing it.
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
    _infer_orig_thw_from_latents,
)


def test_get_patch_size_thw_int_respects_patch_size_t_and_defaults_to_one():
    stage = MagicMock(spec=DenoisingStage)
    stage._get_patch_size_thw = DenoisingStage._get_patch_size_thw.__get__(stage)

    stage.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(patch_size=8, patch_size_t=2)
        )
    )
    assert stage._get_patch_size_thw() == (2, 8, 8)

    stage.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(dit_config=SimpleNamespace(patch_size=8))
    )
    assert stage._get_patch_size_thw() == (1, 8, 8)

    stage.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(patch_size=8, patch_size_t=None)
        )
    )
    assert stage._get_patch_size_thw() == (1, 8, 8)


def test_build_attn_metadata_vmoba_not_none(monkeypatch):
    import sglang.multimodal_gen.runtime.pipelines_core.stages.denoising as denoising_mod

    class DummyBuilder:
        def build(self, **kwargs):
            return kwargs

    stage = MagicMock(spec=DenoisingStage)
    stage._build_attn_metadata = DenoisingStage._build_attn_metadata.__get__(stage)
    stage._exec_thw = DenoisingStage._exec_thw.__get__(stage)
    stage._get_patch_size_thw = MagicMock(return_value=(1, 8, 8))

    stage.attn_backend = SimpleNamespace(
        get_enum=MagicMock(return_value=AttentionBackendEnum.VMOBA_ATTN),
        get_builder_cls=MagicMock(return_value=DummyBuilder),
    )

    monkeypatch.setattr(
        denoising_mod, "get_local_torch_device", lambda: torch.device("cpu")
    )

    batch = SimpleNamespace(raw_latent_shape=(1, 4, 8, 32, 32), exec_latent_shape=None)
    server_args = SimpleNamespace(
        attention_backend_config=SimpleNamespace(
            moba_config={
                "temporal_chunk_size": 2,
                "temporal_topk": 2,
                "spatial_chunk_size": [4, 13],
                "spatial_topk": 6,
                "st_chunk_size": [4, 4, 13],
                "st_topk": 18,
                "moba_select_mode": "topk",
                "moba_threshold": 0.25,
                "moba_threshold_type": "query_head",
            }
        )
    )

    md = stage._build_attn_metadata(0, batch, server_args)

    assert md is not None
    assert md["raw_latent_shape"] == (8, 32, 32)
    assert md["patch_size"] == (1, 8, 8)


class TestMaybeAttachBucketMask:
    """
    Unit tests for the `_maybe_attach_bucket_mask` helper method.
    This uses a mock object to simulate the DenoisingStage, allowing us to
    test the orchestration logic in isolation.
    """

    def setup_method(self):
        # Create a mock stage instance. We don't need to call __init__.
        self.stage = MagicMock(spec=DenoisingStage)

        # Bind the real helper method to our mock instance.
        # This allows us to call it as if it were part of a real DenoisingStage.
        self.stage._maybe_attach_bucket_mask = (
            DenoisingStage._maybe_attach_bucket_mask.__get__(self.stage)
        )
        self.stage._try_attach_attention_mask = (
            DenoisingStage._try_attach_attention_mask.__get__(self.stage)
        )

        # Mock out the dependencies that the helper calls.
        self.stage._should_use_bucketing_with_masks = MagicMock(return_value=True)
        self.stage._get_patch_size_thw = MagicMock(return_value=(1, 8, 8))

    def test_attach_mask_happy_path(self):
        """
        Verifies the mask is correctly attached when all conditions are met.
        """
        # --- Setup ---
        batch = SimpleNamespace()
        batch.bucket_meta = SimpleNamespace(
            enabled=True,
            padding_ratio=0.25,
            attention_mask=torch.ones(2, 100, dtype=torch.bool),  # A dummy mask
        )
        batch.exec_latent_shape = (8, 32, 32)  # Dummy shape for debug validation

        attn_metadata = None  # Start with None, as SDPA might provide.

        # Provide latents for device validation inside _try_attach_attention_mask
        batch.latents = torch.zeros((2, 4, 8, 32, 32), device="cpu")

        # Provide authoritative token count for __debug__ assertion
        batch.bucket_meta.num_tokens = batch.bucket_meta.attention_mask.shape[1]

        # --- Method under test ---
        result_metadata = self.stage._maybe_attach_bucket_mask(batch, attn_metadata)

        # --- Verification ---
        assert result_metadata is not None
        assert hasattr(result_metadata, "key_padding_mask")
        assert result_metadata.key_padding_mask is batch.bucket_meta.attention_mask
        assert result_metadata.key_padding_mask.dtype == torch.bool
        assert result_metadata.key_padding_mask.ndim == 2
        assert result_metadata.key_padding_mask.shape == (2, 100)
        assert result_metadata.key_padding_mask.device == batch.latents.device

    @pytest.mark.parametrize(
        "bucket_meta_config, should_use_masks, reason",
        [
            (None, True, "Bucketing is not enabled for the batch"),
            (SimpleNamespace(enabled=False), True, "Bucketing is disabled in meta"),
            (
                SimpleNamespace(enabled=True, padding_ratio=0.0),
                True,
                "No padding exists",
            ),
            (
                SimpleNamespace(enabled=True, padding_ratio=0.25, attention_mask=None),
                True,
                "Mask is missing",
            ),
            (
                SimpleNamespace(
                    enabled=True,
                    padding_ratio=0.25,
                    attention_mask=torch.ones(2, 100, dtype=torch.bool),
                ),
                False,
                "Config forbids masks",
            ),
        ],
    )
    def test_attach_mask_fallback_cases(
        self, bucket_meta_config, should_use_masks, reason
    ):
        """
        Verifies the mask is NOT attached if any condition is not met.
        """
        # --- Setup ---
        batch = SimpleNamespace()
        batch.bucket_meta = bucket_meta_config
        self.stage._should_use_bucketing_with_masks.return_value = should_use_masks

        attn_metadata = None

        # --- Method under test ---
        result_metadata = self.stage._maybe_attach_bucket_mask(batch, attn_metadata)

        # --- Verification ---
        assert result_metadata is None, f"Test failed for reason: {reason}"


# === Test D: Functional test for mask polarity and correctness ===


class TestDenoisingStageFunctionalMasking:
    """
    Performs a functional test on a mocked DenoisingStage to verify that
    the generated mask has the correct polarity and correctly identifies
    padding tokens when used in a simulated SDPA backend.
    """

    def setup_method(self):
        self.stage = MagicMock(spec=DenoisingStage)

        # Bind real methods to the mock instance
        self.stage._prepare_bucketing_for_step = (
            DenoisingStage._prepare_bucketing_for_step.__get__(self.stage)
        )
        self.stage._maybe_attach_bucket_mask = (
            DenoisingStage._maybe_attach_bucket_mask.__get__(self.stage)
        )
        self.stage._try_attach_attention_mask = (
            DenoisingStage._try_attach_attention_mask.__get__(self.stage)
        )
        self.stage._should_use_bucketing_with_masks = MagicMock(return_value=True)
        self.stage._is_dit_pipeline = MagicMock(return_value=True)
        self.stage._get_patch_size_thw = MagicMock(return_value=(1, 8, 8))
        self.stage._exec_thw = DenoisingStage._exec_thw.__get__(self.stage)

        # Mock attn backend selection used by _prepare_bucketing_for_step
        self.stage.attn_backend = SimpleNamespace(
            get_enum=MagicMock(return_value=AttentionBackendEnum.TORCH_SDPA)
        )

        # Mock server_args and pipeline_config
        # Use a max_waste large enough to allow the 32x32 bucket for our chosen (H,W).
        self.stage.server_args = SimpleNamespace(
            enable_diffusion_hw_bucketing=True,
            diffusion_hw_bucketing_max_waste=1.0,
            diffusion_hw_buckets="32x32",
            pipeline_config=SimpleNamespace(
                dit_config=SimpleNamespace(patch_size=(1, 8, 8))
            ),
        )

    def test_mask_functional_correctness(self):
        """
        Tests the end-to-end flow:
        1. Create unpadded latents.
        2. Apply bucketing to pad them and generate a mask.
        3. Attach the mask to metadata.
        4. Simulate the SDPA backend's use of the mask to verify correctness.
        """
        # --- Setup: Create unpadded latents ---
        B, C, T, H, W = 2, 4, 8, 24, 16
        unpadded_latents = torch.ones((B, C, T, H, W))
        batch = SimpleNamespace(
            raw_latent_shape=unpadded_latents.shape,
            latents=unpadded_latents,
            bucket_meta=None,
            exec_latent_shape=None,
            reqs=[],
        )

        # --- Step 1: Apply bucketing and generate mask ---
        # _prepare_bucketing_for_step mutates `batch` and returns the (possibly) padded latents.
        # Patch distributed query to avoid requiring parallel_state initialization.
        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.get_sp_world_size",
            return_value=1,
        ):
            padded_latents = self.stage._prepare_bucketing_for_step(
                unpadded_latents, batch, self.stage.server_args
            )
        batch_out = batch

        # Verification after bucketing
        assert batch_out.bucket_meta is not None
        assert batch_out.bucket_meta.enabled
        assert hasattr(batch_out.bucket_meta, "attention_mask")
        assert padded_latents.shape == (B, C, T, 32, 32)

        # --- Step 2: Attach the mask ---
        attn_metadata = self.stage._maybe_attach_bucket_mask(batch_out, None)

        # Verification after attachment
        assert attn_metadata is not None
        assert hasattr(attn_metadata, "key_padding_mask")
        mask = attn_metadata.key_padding_mask
        assert mask.dtype == torch.bool

        # --- Step 3: Simulate SDPA backend usage ---
        # The SDPA backend receives a mask where True=VALID.

        # Create a dummy token sequence (e.g., all ones)
        pt, ph, pw = self.stage._get_patch_size_thw()
        dummy_tokens = torch.ones(B, (T // pt) * (32 // ph) * (32 // pw))

        # Apply the mask. The result should have zeros where padding was.
        masked_tokens = dummy_tokens * mask.float()

        # --- Final Verification ---
        # The number of non-zero elements should equal the number of valid tokens.
        valid_tokens_per_req = (T // pt) * (H // ph) * (W // pw)
        assert torch.count_nonzero(masked_tokens).item() == B * valid_tokens_per_req

        # The sum of the masked tokens should also equal this number.
        assert masked_tokens.sum().item() == B * valid_tokens_per_req


def test_prepare_bucketing_vmoba_fast_path_resets_req_metrics():
    stage = MagicMock(spec=DenoisingStage)
    stage._prepare_bucketing_for_step = (
        DenoisingStage._prepare_bucketing_for_step.__get__(stage)
    )
    stage._is_dit_pipeline = MagicMock(return_value=True)

    stage.attn_backend = SimpleNamespace(
        get_enum=MagicMock(return_value=AttentionBackendEnum.VMOBA_ATTN)
    )

    server_args = SimpleNamespace(
        enable_diffusion_hw_bucketing=True,
        diffusion_hw_bucketing_max_waste=1.0,
        diffusion_hw_buckets="32x32",
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(patch_size=(1, 8, 8))
        ),
    )

    latents = torch.zeros((1, 4, 1, 16, 16), dtype=torch.float32)

    req = SimpleNamespace(
        bucket_id=123,
        padding_ratio=0.9,
        bucket_hit=True,
        bucket_applied=True,
        bucket_padded=True,
        bucket_exact_match=True,
    )
    batch = SimpleNamespace(
        raw_latent_shape=latents.shape,
        latents=latents,
        bucket_meta=SimpleNamespace(enabled=True),
        exec_latent_shape=(1, 32, 32),
        reqs=[req],
    )

    out = stage._prepare_bucketing_for_step(latents, batch, server_args)

    assert out is latents
    assert batch.bucket_meta is None
    assert batch.exec_latent_shape is None

    assert req.bucket_id == -1
    assert req.padding_ratio == 0.0
    assert req.bucket_applied is False
    assert req.bucket_padded is False
    assert req.bucket_exact_match is False
    assert req.bucket_hit is False


def test_bucket_hit_semantics_exact_match_vs_padded():
    stage = MagicMock(spec=DenoisingStage)
    stage._prepare_bucketing_for_step = (
        DenoisingStage._prepare_bucketing_for_step.__get__(stage)
    )
    stage._maybe_attach_bucket_mask = DenoisingStage._maybe_attach_bucket_mask.__get__(
        stage
    )
    stage._try_attach_attention_mask = (
        DenoisingStage._try_attach_attention_mask.__get__(stage)
    )
    stage._should_use_bucketing_with_masks = MagicMock(return_value=True)
    stage._is_dit_pipeline = MagicMock(return_value=True)
    stage._get_patch_size_thw = MagicMock(return_value=(1, 8, 8))
    stage._exec_thw = DenoisingStage._exec_thw.__get__(stage)

    stage.attn_backend = SimpleNamespace(
        get_enum=MagicMock(return_value=AttentionBackendEnum.TORCH_SDPA)
    )

    server_args = SimpleNamespace(
        enable_diffusion_hw_bucketing=True,
        diffusion_hw_bucketing_max_waste=1.0,
        diffusion_hw_buckets="32x32",
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(patch_size=(1, 8, 8))
        ),
    )

    def _run_case(H: int, W: int):
        B, C, T = 1, 4, 1
        latents = torch.zeros((B, C, T, H, W))
        req = SimpleNamespace(
            bucket_id=-1,
            padding_ratio=0.0,
            bucket_hit=False,
            bucket_applied=False,
            bucket_padded=False,
            bucket_exact_match=False,
        )
        batch = SimpleNamespace(
            raw_latent_shape=latents.shape,
            latents=latents,
            bucket_meta=None,
            exec_latent_shape=None,
            reqs=[req],
        )

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.get_sp_world_size",
            return_value=1,
        ):
            stage._prepare_bucketing_for_step(latents, batch, server_args)
        return req, batch

    req_exact, batch_exact = _run_case(32, 32)
    assert batch_exact.bucket_meta is not None and batch_exact.bucket_meta.enabled
    assert req_exact.bucket_applied is True
    assert req_exact.bucket_exact_match is True
    assert req_exact.bucket_padded is False
    assert req_exact.bucket_hit is False
    assert req_exact.padding_ratio == 0.0

    req_pad, batch_pad = _run_case(24, 16)
    assert batch_pad.bucket_meta is not None and batch_pad.bucket_meta.enabled
    assert req_pad.bucket_applied is True
    assert req_pad.bucket_exact_match is False
    assert req_pad.bucket_padded is True
    assert req_pad.bucket_hit is True
    assert req_pad.padding_ratio > 0.0


def test_prepare_bucketing_reverts_when_masked_bucketing_not_allowed():
    stage = MagicMock(spec=DenoisingStage)
    stage._prepare_bucketing_for_step = (
        DenoisingStage._prepare_bucketing_for_step.__get__(stage)
    )
    stage._should_use_bucketing_with_masks = (
        DenoisingStage._should_use_bucketing_with_masks.__get__(stage)
    )
    stage._is_dit_pipeline = MagicMock(return_value=True)
    stage._get_patch_size_thw = MagicMock(return_value=(1, 8, 8))
    stage._exec_thw = DenoisingStage._exec_thw.__get__(stage)

    stage.attn_backend = SimpleNamespace(
        get_enum=MagicMock(return_value=AttentionBackendEnum.FA)
    )

    server_args = SimpleNamespace(
        enable_diffusion_hw_bucketing=True,
        diffusion_hw_bucketing_max_waste=1.0,
        diffusion_hw_buckets="32x32",
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(patch_size=(1, 8, 8))
        ),
    )

    B, C, T, H, W = 1, 4, 1, 16, 16
    latents = torch.randn((B, C, T, H, W))

    req = SimpleNamespace(
        bucket_id=-1,
        padding_ratio=0.0,
        bucket_hit=False,
        bucket_applied=False,
        bucket_padded=False,
        bucket_exact_match=False,
    )
    batch = SimpleNamespace(
        raw_latent_shape=latents.shape,
        latents=latents,
        bucket_meta=None,
        exec_latent_shape=None,
        reqs=[req],
    )

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.get_sp_world_size",
        return_value=1,
    ):
        out = stage._prepare_bucketing_for_step(latents, batch, server_args)

    assert out.shape == latents.shape
    torch.testing.assert_close(out, latents)
    assert batch.bucket_meta is None
    assert batch.exec_latent_shape is None

    assert req.bucket_id == -1
    assert req.bucket_applied is False
    assert req.bucket_hit is False
    assert req.bucket_padded is False
    assert req.bucket_exact_match is False


def _rel_l2(z_a: torch.Tensor, z_b: torch.Tensor, eps: float = 1e-8) -> float:
    return (z_a.sub(z_b).norm() / (z_b.norm() + eps)).item()


def _pad_hw_bcthw(latents: torch.Tensor, bucket_h: int, bucket_w: int) -> torch.Tensor:
    pad_h = bucket_h - latents.shape[3]
    pad_w = bucket_w - latents.shape[4]
    if pad_h < 0 or pad_w < 0:
        raise ValueError("bucket must be >= latents H/W")
    if pad_h == 0 and pad_w == 0:
        return latents
    return F.pad(latents, (0, pad_w, 0, pad_h, 0, 0), mode="constant", value=0)


def _unpad_hw_bcthw(latents: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
    return latents[:, :, :, :orig_h, :orig_w]


def test_build_key_padding_mask_bcthw_floor_division_when_orig_not_divisible_by_patch():
    device = "cpu"
    B, T = 1, 8
    patch_thw = (1, 8, 8)
    pt, ph, pw = patch_thw

    exec_thw = (T, 32, 32)
    orig_thw = (T, 30, 30)

    mask, num_tokens = build_key_padding_mask_bcthw(
        B, exec_thw, orig_thw, patch_thw, device
    )

    expected_total = (T // pt) * (32 // ph) * (32 // pw)
    expected_valid = (T // pt) * (30 // ph) * (30 // pw)

    assert num_tokens == expected_total
    assert mask.shape == (B, expected_total)
    assert mask.sum().item() == expected_valid


def test_infer_orig_thw_falls_back_to_latents_shape_when_unpacked_shape_mismatches_bc():
    latents = torch.zeros((2, 4, 1, 16, 32), dtype=torch.float32)

    unpacked_shape_b_mismatch = torch.Size([1, 4, 1, 64, 64])
    assert _infer_orig_thw_from_latents(latents, unpacked_shape_b_mismatch) == (
        1,
        16,
        32,
    )

    unpacked_shape_c_mismatch = (2, 8, 1, 64, 64)
    assert _infer_orig_thw_from_latents(latents, unpacked_shape_c_mismatch) == (
        1,
        16,
        32,
    )


def test_apply_shape_bucketing_invalid_diffusion_hw_buckets_string_has_clean_error():
    latents = torch.zeros((1, 4, 1, 32, 32), dtype=torch.float32)
    server_args = SimpleNamespace(
        enable_diffusion_hw_bucketing=True,
        enable_shape_bucketing=False,
        diffusion_hw_bucketing_max_waste=0.2,
        diffusion_hw_buckets="64x,96x96",
    )

    with pytest.raises(
        ValueError,
        match=r"Invalid diffusion_hw_buckets entry '64x'\. Expected 'HxW' like '64x64'\.",
    ):
        apply_shape_bucketing(latents, server_args, orig_thw=(1, 32, 32))


def test_apply_shape_bucketing_invalid_diffusion_hw_buckets_tuple_has_clean_error():
    latents = torch.zeros((1, 4, 1, 32, 32), dtype=torch.float32)
    server_args = SimpleNamespace(
        enable_diffusion_hw_bucketing=True,
        enable_shape_bucketing=False,
        diffusion_hw_bucketing_max_waste=0.2,
        diffusion_hw_buckets=[(64,), (96, 96)],
    )

    with pytest.raises(
        ValueError,
        match=r"Invalid diffusion_hw_buckets entry \(64,\)\. Expected \(H, W\) or 'HxW' like '64x64'\.",
    ):
        apply_shape_bucketing(latents, server_args, orig_thw=(1, 32, 32))


def test_sdpa_mask_dtype_validation_allows_bool_and_rejects_non_bool():
    impl = SDPAImpl(num_heads=2, head_size=8, causal=False, softmax_scale=1.0)

    B, L, S, H, D = 2, 3, 4, 2, 8
    q = torch.randn((B, L, H, D), dtype=torch.float32)
    k = torch.randn((B, S, H, D), dtype=torch.float32)
    v = torch.randn((B, S, H, D), dtype=torch.float32)

    ok_mask = torch.ones((B, S), dtype=torch.bool)
    out = impl.forward(q, k, v, attn_metadata=SimpleNamespace(attention_mask=ok_mask))
    assert out.shape == (B, L, H, D)

    bad_mask = torch.ones((B, S), dtype=torch.int64)
    with pytest.raises(ValueError, match=r"Expected bool mask"):
        impl.forward(q, k, v, attn_metadata=SimpleNamespace(attention_mask=bad_mask))


def _run_toy_accuracy_harness(
    *,
    z0: torch.Tensor,
    enable_masked_bucketing: bool,
    bucket_hw: tuple[int, int],
    num_steps: int,
) -> torch.Tensor:
    from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl

    stage = MagicMock(spec=DenoisingStage)
    stage._maybe_attach_bucket_mask = DenoisingStage._maybe_attach_bucket_mask.__get__(
        stage
    )
    stage._try_attach_attention_mask = (
        DenoisingStage._try_attach_attention_mask.__get__(stage)
    )
    stage._should_use_bucketing_with_masks = MagicMock(return_value=True)

    B, C, T, H, W = z0.shape
    bucket_h, bucket_w = bucket_hw

    if enable_masked_bucketing:
        latents = _pad_hw_bcthw(z0, bucket_h, bucket_w)
        exec_thw = (T, bucket_h, bucket_w)
        orig_thw = (T, H, W)
        patch_thw = (1, 1, 1)

        mask, num_tokens = build_key_padding_mask_bcthw(
            B=B,
            exec_thw=exec_thw,
            orig_thw=orig_thw,
            patch_thw=patch_thw,
            device=latents.device,
        )

        padding_ratio = 1.0 - (H * W) / float(bucket_h * bucket_w)
        bucket_meta = SimpleNamespace(
            enabled=True,
            padding_ratio=padding_ratio,
            attention_mask=mask,
            num_tokens=num_tokens,
        )
        batch = SimpleNamespace(
            raw_latent_shape=z0.shape,
            latents=latents,
            bucket_meta=bucket_meta,
            exec_latent_shape=exec_thw,
            reqs=[],
        )
    else:
        latents = z0.clone()
        batch = SimpleNamespace(
            raw_latent_shape=z0.shape,
            latents=latents,
            bucket_meta=None,
            exec_latent_shape=None,
            reqs=[],
        )

    sdpa = SDPAImpl(
        num_heads=1,
        head_size=8,
        causal=False,
        softmax_scale=1.0,
        dropout_p=0.0,
    )

    proj = torch.linspace(
        0.1, 1.0, steps=8, device=latents.device, dtype=latents.dtype
    ).view(1, 1, 1, 8)

    for _ in range(num_steps):
        batch.latents = latents
        attn_metadata = stage._maybe_attach_bucket_mask(batch, None)

        if enable_masked_bucketing:
            bm = batch.bucket_meta
            assert bm is not None
            assert bm.enabled
            assert bm.padding_ratio > 0.0
            assert bm.attention_mask is not None
            expected_tokens = bm.num_tokens
            assert bm.attention_mask.shape == (B, expected_tokens)
            assert attn_metadata is not None
            assert hasattr(attn_metadata, "key_padding_mask")
            assert attn_metadata.key_padding_mask is bm.attention_mask

        tokens = latents.reshape(B, C, -1).transpose(1, 2).contiguous()  # [B,S,C]
        x0 = tokens[:, :, 0].unsqueeze(-1)  # [B,S,1]
        q = x0.unsqueeze(2) * proj
        k = x0.unsqueeze(2) * (proj * 0.9 + 0.1)
        v = x0.unsqueeze(2) * (proj * 1.1)

        out = sdpa.forward(q, k, v, attn_metadata=attn_metadata)  # [B,S,H,Dh]
        y = out.mean(dim=(2, 3)).reshape(B, 1, *latents.shape[2:])
        latents = latents + 0.01 * y

    if enable_masked_bucketing:
        latents = _unpad_hw_bcthw(latents, H, W)

    return latents


@pytest.mark.parametrize(
    "hw",
    [
        (31, 31),
        (30, 30),
        (31, 32),
        (32, 31),
        (23, 23),
    ],
)
def test_masked_bucketing_latent_drift_band_cpu(monkeypatch, hw):
    import sglang.multimodal_gen.runtime.layers.attention.backends.sdpa as sdpa_mod

    counter = {"n": 0}
    orig = sdpa_mod.F.scaled_dot_product_attention

    def wrapped(*args, **kwargs):
        counter["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(sdpa_mod.F, "scaled_dot_product_attention", wrapped)

    H, W = hw
    torch.manual_seed(0)
    z0 = torch.randn((1, 1, 2, H, W), device=torch.device("cpu"), dtype=torch.float32)

    z_off_1 = _run_toy_accuracy_harness(
        z0=z0,
        enable_masked_bucketing=False,
        bucket_hw=(32, 32),
        num_steps=2,
    )
    z_off_2 = _run_toy_accuracy_harness(
        z0=z0,
        enable_masked_bucketing=False,
        bucket_hw=(32, 32),
        num_steps=2,
    )
    z_on = _run_toy_accuracy_harness(
        z0=z0,
        enable_masked_bucketing=True,
        bucket_hw=(32, 32),
        num_steps=2,
    )

    assert counter["n"] == 2 * 3

    baseline_band = _rel_l2(z_off_2, z_off_1)
    treatment_rel = _rel_l2(z_on, z_off_1)
    tol = max(1e-6, baseline_band * 2.0 + 1e-6)

    assert (
        treatment_rel <= tol
    ), f"shape={hw} treatment_rel={treatment_rel:.3e} baseline_band={baseline_band:.3e} tol={tol:.3e}"
