from types import SimpleNamespace

import pytest

from torchspec import train_entry


def _args(**overrides):
    values = {
        "attention_backend": "usp",
        "sp_size": 1,
        "sp_ulysses_size": 1,
        "sp_ring_size": 1,
        "training_num_gpus_per_node": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_draft_config(monkeypatch, *, heads=32, kv_heads=8):
    monkeypatch.setattr(
        train_entry,
        "_get_draft_model_config",
        lambda _args: SimpleNamespace(
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
        ),
    )


def test_derive_usp_topology_from_sp_size(monkeypatch):
    _patch_draft_config(monkeypatch, heads=32, kv_heads=8)
    args = _args(sp_size=4)

    train_entry._derive_usp_topology(args)

    assert args.sp_size == 4
    assert args.sp_ulysses_size == 4
    assert args.sp_ring_size == 1


def test_derive_usp_topology_uses_ring_when_kv_heads_do_not_divide(monkeypatch):
    _patch_draft_config(monkeypatch, heads=32, kv_heads=8)
    args = _args(sp_size=16)

    train_entry._derive_usp_topology(args)

    assert args.sp_size == 16
    assert args.sp_ulysses_size == 8
    assert args.sp_ring_size == 2


def test_derive_usp_topology_accepts_legacy_ulysses_ring_fields(monkeypatch):
    def fail_get_config(_args):
        raise AssertionError("_get_draft_model_config should not be called")

    monkeypatch.setattr(train_entry, "_get_draft_model_config", fail_get_config)
    args = _args(sp_ulysses_size=2, sp_ring_size=2)

    train_entry._derive_usp_topology(args)

    assert args.sp_size == 4
    assert args.sp_ulysses_size == 2
    assert args.sp_ring_size == 2


def test_derive_usp_topology_rejects_conflicting_explicit_topology(monkeypatch):
    _patch_draft_config(monkeypatch, heads=32, kv_heads=8)
    args = _args(sp_size=8, sp_ulysses_size=2, sp_ring_size=2)

    with pytest.raises(ValueError, match="sp_size must match"):
        train_entry._derive_usp_topology(args)


def test_derive_usp_topology_skips_non_usp_without_loading_config(monkeypatch):
    def fail_get_config(_args):
        raise AssertionError("_get_draft_model_config should not be called")

    monkeypatch.setattr(train_entry, "_get_draft_model_config", fail_get_config)
    args = _args(attention_backend="sdpa")

    train_entry._derive_usp_topology(args)

    assert args.sp_size == 1


def test_derive_usp_topology_requires_head_counts(monkeypatch):
    monkeypatch.setattr(
        train_entry,
        "_get_draft_model_config",
        lambda _args: SimpleNamespace(),
    )
    args = _args(sp_size=2)

    with pytest.raises(ValueError, match="attention head counts"):
        train_entry._derive_usp_topology(args)
