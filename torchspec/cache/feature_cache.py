import hashlib
import json
from collections.abc import Mapping
from typing import Any

from torchspec.cache.cache_manifest import CacheManifest, FeatureHandle


class FeatureCache:
    def __init__(
        self,
        manifest: CacheManifest,
        remote_client,
        mooncake_store,
        *,
        feature_schema_version: str = "eagle3.v1",
        validate_missing_as_stale: bool = True,
    ):
        self.manifest = manifest
        self.remote_client = remote_client
        self.mooncake_store = mooncake_store
        self.feature_schema_version = feature_schema_version
        self.validate_missing_as_stale = validate_missing_as_stale
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "stale_regenerations": 0,
        }

    def build_sample_key(self, sample: Mapping[str, Any]) -> str:
        explicit_key = sample.get("sample_key")
        if explicit_key:
            return str(explicit_key)
        data_id = sample.get("data_id")
        if data_id:
            return str(data_id)
        payload = {
            "input_ids": sample.get("input_ids"),
            "packed_loss_mask": sample.get("packed_loss_mask"),
            "multimodal_inputs": sample.get("multimodal_inputs"),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def resolve_handle(self, sample: Mapping[str, Any]) -> FeatureHandle:
        sample_key = self.build_sample_key(sample)
        cached = self.manifest.get(sample_key)
        if cached is not None and self._handle_exists(cached):
            self.metrics["hits"] += 1
            return cached

        if cached is not None and self.validate_missing_as_stale:
            self.manifest.delete(sample_key)
            self.metrics["stale_regenerations"] += 1

        self.metrics["misses"] += 1
        handle = self.remote_client.request_features(
            sample_key=sample_key,
            input_ids=list(sample["input_ids"]),
            packed_loss_mask=sample["packed_loss_mask"],
            multimodal_inputs=sample.get("multimodal_inputs"),
            feature_schema_version=sample.get(
                "feature_schema_version", self.feature_schema_version
            ),
        )
        self.manifest.upsert(handle)
        return handle

    def resolve_and_load(self, sample: Mapping[str, Any], *, device) -> dict[str, Any]:
        handle = self.resolve_handle(sample)
        tensors = self.mooncake_store.get(
            key=handle.mooncake_key,
            shapes=handle.tensor_shapes,
            dtypes=self._torch_dtypes(handle.tensor_dtypes),
            device=device,
        )
        result = tensors.to_tensor_dict()
        if "packed_loss_mask" in sample:
            result["packed_loss_mask"] = sample["packed_loss_mask"]
        return result

    def _handle_exists(self, handle: FeatureHandle) -> bool:
        return self.mooncake_store.exists(f"{handle.mooncake_key}_hs")

    @staticmethod
    def _torch_dtypes(dtypes: dict[str, str]) -> dict[str, Any]:
        import torch

        resolved = {}
        for key, value in dtypes.items():
            resolved[key] = getattr(torch, value.replace("torch.", ""))
        return resolved
