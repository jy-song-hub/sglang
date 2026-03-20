import types

import numpy as np
import pytest

from sglang.multimodal_gen.benchmarks.datasets import _resolve_request_size


def test_resolve_request_size_fixed_uses_width_height():
    args = types.SimpleNamespace(
        size_set_list=[(1, 2)],
        size_policy="fixed",
        width=512,
        height=768,
        square=False,
    )
    w, h = _resolve_request_size(args, idx=123, rng=np.random.default_rng(0))
    assert (w, h) == (512, 768)


def test_resolve_request_size_round_robin_picks_by_index():
    args = types.SimpleNamespace(
        size_set_list=[(10, 11), (20, 21), (30, 31)],
        size_policy="round_robin",
        width=None,
        height=None,
        square=False,
    )
    assert _resolve_request_size(args, idx=0, rng=None) == (10, 11)
    assert _resolve_request_size(args, idx=1, rng=None) == (20, 21)
    assert _resolve_request_size(args, idx=2, rng=None) == (30, 31)
    assert _resolve_request_size(args, idx=3, rng=None) == (10, 11)


def test_resolve_request_size_random_requires_rng():
    args = types.SimpleNamespace(
        size_set_list=[(10, 11), (20, 21)],
        size_policy="random",
        size_weights=None,
        width=None,
        height=None,
        square=False,
    )
    with pytest.raises(ValueError, match=r"rng is None"):
        _resolve_request_size(args, idx=0, rng=None)


def test_resolve_request_size_random_weighted_is_deterministic_for_degenerate_weights():
    # Degenerate weights (always pick index 1) avoids flakiness.
    args = types.SimpleNamespace(
        size_set_list=[(10, 11), (20, 21)],
        size_policy="random",
        size_weights=np.array([0.0, 1.0], dtype=np.float64),
        width=None,
        height=None,
        square=False,
    )
    rng = np.random.default_rng(0)
    for idx in range(5):
        assert _resolve_request_size(args, idx=idx, rng=rng) == (20, 21)


def test_resolve_request_size_square_forces_square_after_selection():
    # From a rectangular entry, square flag should force side=width (if provided).
    args = types.SimpleNamespace(
        size_set_list=[(10, 99)],
        size_policy="round_robin",
        width=None,
        height=None,
        square=True,
    )
    assert _resolve_request_size(args, idx=0, rng=None) == (10, 10)
