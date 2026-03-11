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
        self.config = type(
            "Config",
            (),
            {
                "master_server_address": "10.0.0.1:50051",
                "metadata_server": "http://10.0.0.1:8090/metadata",
                "protocol": "tcp",
                "device_name": "",
                "enable_gpu_direct": False,
            },
        )()

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
        self.last_kwargs = None

    def request_features(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
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
    assert remote_client.handle == handle


def test_feature_cache_passes_mooncake_target(tmp_path):
    manifest = CacheManifest(str(tmp_path / "manifest.sqlite3"))
    handle = FeatureHandle(
        sample_key="sample-target",
        mooncake_key="feature:sample-target",
        tensor_shapes={"hidden_states": (2, 4), "input_ids": (2,)},
        tensor_dtypes={"hidden_states": "float32", "input_ids": "int64"},
        feature_schema_version="eagle3.v1",
        created_at=2.0,
    )

    mooncake_store = _MockMooncakeStore()
    remote_client = _MockRemoteClient(handle)
    cache = FeatureCache(manifest, remote_client, mooncake_store)

    cache.resolve_handle(
        {"sample_key": "sample-target", "input_ids": [1, 2], "packed_loss_mask": "11"}
    )

    assert remote_client.calls == 1
    assert remote_client.last_kwargs["mooncake_target"] == {
        "master_server_address": "10.0.0.1:50051",
        "metadata_server": "http://10.0.0.1:8090/metadata",
        "protocol": "tcp",
        "device_name": "",
        "enable_gpu_direct": False,
    }


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


def test_feature_cache_stitches_cached_prefix_and_suffix(tmp_path):
    manifest = CacheManifest(str(tmp_path / "manifest.sqlite3"))
    prefix_handle = FeatureHandle(
        sample_key="prefix",
        mooncake_key="feature:prefix",
        tensor_shapes={
            "hidden_states": (2, 4),
            "input_ids": (2,),
            "last_hidden_states": (2, 2),
        },
        tensor_dtypes={
            "hidden_states": "float32",
            "input_ids": "int64",
            "last_hidden_states": "float32",
        },
        feature_schema_version="eagle3.v1",
        created_at=1.0,
    )
    manifest.upsert(prefix_handle)
    composite_handle = FeatureHandle(
        sample_key="sample-4",
        mooncake_key="feature:sample-4",
        tensor_shapes={
            "hidden_states": (1, 4),
            "input_ids": (3,),
            "last_hidden_states": (1, 2),
        },
        tensor_dtypes={
            "hidden_states": "float32",
            "input_ids": "int64",
            "last_hidden_states": "float32",
        },
        feature_schema_version="eagle3.v1",
        created_at=2.0,
        prefix_sample_key="prefix",
        cached_tokens=2,
    )

    mooncake_store = _MockMooncakeStore()
    mooncake_store.existing.update({"feature:prefix_hs", "feature:sample-4_hs"})
    remote_client = _MockRemoteClient(composite_handle)
    cache = FeatureCache(manifest, remote_client, mooncake_store)

    result = cache.resolve_and_load(
        {"sample_key": "sample-4", "input_ids": [1, 2, 3], "packed_loss_mask": "111"},
        device=torch.device("cpu"),
    )

    assert remote_client.calls == 1
    assert result["hidden_states"].shape == (3, 4)
    assert result["last_hidden_states"].shape == (3, 2)
    assert result["input_ids"].tolist() == [1, 2, 3]


def test_feature_cache_fetches_missing_prefix_on_demand(tmp_path):
    manifest = CacheManifest(str(tmp_path / "manifest.sqlite3"))
    prefix_key = FeatureCache.build_sample_key_from_values(
        input_ids=[1, 2], packed_loss_mask="11", multimodal_inputs=None
    )
    prefix_handle = FeatureHandle(
        sample_key=prefix_key,
        mooncake_key="feature:prefix",
        tensor_shapes={
            "hidden_states": (2, 4),
            "input_ids": (2,),
            "last_hidden_states": (2, 2),
        },
        tensor_dtypes={
            "hidden_states": "float32",
            "input_ids": "int64",
            "last_hidden_states": "float32",
        },
        feature_schema_version="eagle3.v1",
        created_at=1.0,
    )
    composite_handle = FeatureHandle(
        sample_key="sample-5",
        mooncake_key="feature:sample-5",
        tensor_shapes={
            "hidden_states": (1, 4),
            "input_ids": (3,),
            "last_hidden_states": (1, 2),
        },
        tensor_dtypes={
            "hidden_states": "float32",
            "input_ids": "int64",
            "last_hidden_states": "float32",
        },
        feature_schema_version="eagle3.v1",
        created_at=2.0,
        prefix_sample_key=prefix_key,
        cached_tokens=2,
    )
    manifest.upsert(composite_handle)

    mooncake_store = _MockMooncakeStore()
    mooncake_store.existing.update({"feature:prefix_hs", "feature:sample-5_hs"})

    class _RemoteClient:
        def __init__(self):
            self.calls = []

        def request_features(self, **kwargs):
            self.calls.append(kwargs["sample_key"])
            if kwargs["sample_key"] == prefix_key:
                return prefix_handle
            return composite_handle

    remote_client = _RemoteClient()
    cache = FeatureCache(manifest, remote_client, mooncake_store)

    result = cache.resolve_and_load(
        {"sample_key": "sample-5", "input_ids": [1, 2, 3], "packed_loss_mask": "111"},
        device=torch.device("cpu"),
    )

    assert remote_client.calls == ["sample-5", prefix_key]
    assert result["hidden_states"].shape == (3, 4)
    assert manifest.get(prefix_key) == prefix_handle


def test_slice_prefix_sample_ignores_parent_data_id():
    cache = FeatureCache(manifest=object(), remote_client=None, mooncake_store=None)

    prefix_sample = cache._slice_prefix_sample(
        {
            "data_id": "dataset:123",
            "input_ids": [1, 2, 3],
            "packed_loss_mask": "111",
            "multimodal_inputs": None,
        },
        2,
    )

    expected_key = FeatureCache.build_sample_key_from_values(
        input_ids=[1, 2],
        packed_loss_mask="11",
        multimodal_inputs=None,
    )

    assert prefix_sample["sample_key"] == expected_key
    assert "data_id" not in prefix_sample
