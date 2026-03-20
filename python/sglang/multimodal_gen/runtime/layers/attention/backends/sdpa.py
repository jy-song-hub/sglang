# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (  # FlashAttentionMetadata,
    AttentionBackend,
    AttentionImpl,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SDPABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.TORCH_SDPA

    @staticmethod
    def get_impl_cls() -> type["SDPAImpl"]:
        return SDPAImpl

    # @staticmethod
    # def get_metadata_cls() -> Type["AttentionMetadata"]:
    #     return FlashAttentionMetadata


class SDPAImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Optional[Any] = None,
    ) -> torch.Tensor:
        # transpose to bs, heads, seq_len, head_dim
        query = query.transpose(1, 2)  # [B, Hq, L, Dh]
        key = key.transpose(1, 2)  # [B, Hk, S, Dh]
        value = value.transpose(1, 2)  # [B, Hv, S, Dv]

        attn_mask = None
        valid_mask = None

        # Support both attn_metadata=None and field fallbacks
        if attn_metadata is not None:
            valid_mask = getattr(attn_metadata, "attention_mask", None)
            if valid_mask is None:
                valid_mask = getattr(attn_metadata, "key_padding_mask", None)

        if valid_mask is not None:
            if __debug__:
                if valid_mask.dtype != torch.bool:
                    raise ValueError(f"Expected bool mask, got {valid_mask.dtype}")
                if valid_mask.ndim != 2:
                    raise ValueError(f"Expected [B,S], got {tuple(valid_mask.shape)}")
                if valid_mask.shape[0] != query.shape[0]:
                    raise ValueError("Mask B != query B")
                if valid_mask.shape[1] != key.shape[-2]:
                    raise ValueError("Mask S != key S")
                if valid_mask.device != query.device:
                    raise ValueError(
                        f"Mask device {valid_mask.device} != query device {query.device}"
                    )

            # PyTorch SDPA boolean attn_mask uses True=participate, False=masked.
            # Broadcast to [B, 1, 1, S] to apply key-padding mask across heads and query length.
            attn_mask = valid_mask[:, None, None, :]

        attn_kwargs = {
            "attn_mask": attn_mask,
            "dropout_p": self.dropout,
            "is_causal": self.causal,
            "scale": self.softmax_scale,
        }
        if query.shape[1] != key.shape[1]:
            attn_kwargs["enable_gqa"] = True

        output = F.scaled_dot_product_attention(query, key, value, **attn_kwargs)
        output = output.transpose(1, 2)
        return output
