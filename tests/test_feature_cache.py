import torch

from torchspec.cache import CacheManifest, FeatureCache, FeatureHandle


class _MockTargetOutput:
    def __init__(self, tensors):
        self._tensors = tensors

    def to_tensor_dict(self):
        return dict(self._tensors)


class _MockMooncakeStore:
    def __init__(self):
        self.existing = set()
        self.get_calls = 0

    def exists(self, key: str) -> bool:
        return key in self.existing

    def get(self, key, shapes, dtypes, device):
        self.get_calls += 1
        tensors = {
            name: torch.zeros(shape, dtype=dtypes.get(name, torch.float32), device=device)
            for name, shape in shapes.items()
        }
        return _MockTargetOutput(tensors)


class _MockRemoteClient:
    def __init__(self, handle: FeatureHandle):
        self.handle = handle
        self.calls = 0

    def request_features(self, **kwargs):
        self.calls += 1
        return self.handle


def test_feature_cache_hit_uses_manifest(tmp_path):
    manifest = CacheManifest(str(tmp_path / "manifest.sqlite3"))
    handle = FeatureHandle(
        sample_key="sample-1",
        mooncake_key="feature:sample-1",
        tensor_shapes={"hidden_states": (2, 4), "input_ids": (2,)},
        tensor_dtypes={"hidden_states": "float32", "input_ids": "int64"},
        feature_schema_version="eagle3.v1",
        created_at=1.0,
    )
    manifest.upsert(handle)

    mooncake_store = _MockMooncakeStore()
    mooncake_store.existing.add("feature:sample-1_hs")
    remote_client = _MockRemoteClient(handle)
    cache = FeatureCache(manifest, remote_client, mooncake_store)

    result = cache.resolve_and_load(
        {"sample_key": "sample-1", "input_ids": [1, 2], "packed_loss_mask": "mask"},
        device=torch.device("cpu"),
    )

    assert remote_client.calls == 0
    assert cache.metrics["hits"] == 1
    assert result["hidden_states"].shape == (2, 4)


def test_feature_cache_miss_fetches_remote(tmp_path):
    manifest = CacheManifest(str(tmp_path / "manifest.sqlite3"))
    handle = FeatureHandle(
        sample_key="sample-2",
        mooncake_key="feature:sample-2",
        tensor_shapes={"hidden_states": (2, 4), "input_ids": (2,)},
        tensor_dtypes={"hidden_states": "float32", "input_ids": "int64"},
        feature_schema_version="eagle3.v1",
        created_at=2.0,
    )

    mooncake_store = _MockMooncakeStore()
    mooncake_store.existing.add("feature:sample-2_hs")
    remote_client = _MockRemoteClient(handle)
    cache = FeatureCache(manifest, remote_client, mooncake_store)

    resolved = cache.resolve_handle({"sample_key": "sample-2", "input_ids": [1], "packed_loss_mask": "mask"})

    assert remote_client.calls == 1
    assert resolved == handle
    assert manifest.get("sample-2") == handle
    assert cache.metrics["misses"] == 1


def test_feature_cache_regenerates_stale_entry(tmp_path):
    manifest = CacheManifest(str(tmp_path / "manifest.sqlite3"))
    stale = FeatureHandle(
        sample_key="sample-3",
        mooncake_key="feature:sample-3-old",
        tensor_shapes={"hidden_states": (2, 4), "input_ids": (2,)},
        tensor_dtypes={"hidden_states": "float32", "input_ids": "int64"},
        feature_schema_version="eagle3.v1",
        created_at=3.0,
    )
    fresh = FeatureHandle(
        sample_key="sample-3",
        mooncake_key="feature:sample-3-new",
        tensor_shapes={"hidden_states": (2, 4), "input_ids": (2,)},
        tensor_dtypes={"hidden_states": "float32", "input_ids": "int64"},
        feature_schema_version="eagle3.v1",
        created_at=4.0,
    )
    manifest.upsert(stale)

    mooncake_store = _MockMooncakeStore()
    mooncake_store.existing.add("feature:sample-3-new_hs")
    remote_client = _MockRemoteClient(fresh)
    cache = FeatureCache(manifest, remote_client, mooncake_store)

    resolved = cache.resolve_handle({"sample_key": "sample-3", "input_ids": [1], "packed_loss_mask": "mask"})

    assert remote_client.calls == 1
    assert resolved.mooncake_key == "feature:sample-3-new"
    assert cache.metrics["stale_regenerations"] == 1
