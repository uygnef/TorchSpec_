from torchspec.cache.cache_manifest import CacheManifest, FeatureHandle


def test_cache_manifest_upsert_and_get(tmp_path):
    manifest = CacheManifest(str(tmp_path / "feature_cache.sqlite3"))
    handle = FeatureHandle(
        sample_key="sample-1",
        mooncake_key="feature:sample-1",
        tensor_shapes={"hidden_states": (2, 4), "input_ids": (2,)},
        tensor_dtypes={"hidden_states": "bfloat16", "input_ids": "int64"},
        feature_schema_version="eagle3.v1",
        created_at=100.0,
    )

    manifest.upsert(handle)
    loaded = manifest.get("sample-1")

    assert loaded == handle


def test_cache_manifest_persists_across_instances(tmp_path):
    db_path = tmp_path / "feature_cache.sqlite3"
    handle = FeatureHandle(
        sample_key="sample-2",
        mooncake_key="feature:sample-2",
        tensor_shapes={"hidden_states": (3, 4)},
        tensor_dtypes={"hidden_states": "bfloat16"},
        feature_schema_version="eagle3.v1",
        created_at=200.0,
    )

    CacheManifest(str(db_path)).upsert(handle)
    loaded = CacheManifest(str(db_path)).get("sample-2")

    assert loaded == handle


def test_cache_manifest_delete(tmp_path):
    manifest = CacheManifest(str(tmp_path / "feature_cache.sqlite3"))
    handle = FeatureHandle(
        sample_key="sample-3",
        mooncake_key="feature:sample-3",
        tensor_shapes={"hidden_states": (1, 4)},
        tensor_dtypes={"hidden_states": "bfloat16"},
        feature_schema_version="eagle3.v1",
        created_at=300.0,
    )

    manifest.upsert(handle)
    manifest.delete("sample-3")

    assert manifest.get("sample-3") is None


def test_cache_manifest_persists_composite_fields(tmp_path):
    manifest = CacheManifest(str(tmp_path / "feature_cache.sqlite3"))
    handle = FeatureHandle(
        sample_key="sample-4",
        mooncake_key="feature:sample-4",
        tensor_shapes={"hidden_states": (2, 4)},
        tensor_dtypes={"hidden_states": "bfloat16"},
        feature_schema_version="eagle3.v1",
        created_at=400.0,
        prefix_sample_key="sample-prefix",
        cached_tokens=3,
    )

    manifest.upsert(handle)
    loaded = manifest.get("sample-4")

    assert loaded == handle
