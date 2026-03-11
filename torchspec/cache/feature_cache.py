import hashlib
import json
from collections.abc import Mapping
from typing import Any

from torchspec.cache.cache_manifest import CacheManifest, FeatureHandle
from torchspec.inference.client.errors import RemoteSGLangError


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
        return self.build_sample_key_from_values(
            input_ids=sample.get("input_ids"),
            packed_loss_mask=sample.get("packed_loss_mask"),
            multimodal_inputs=sample.get("multimodal_inputs"),
            data_id=sample.get("data_id"),
        )

    @staticmethod
    def build_sample_key_from_values(
        *,
        input_ids: Any,
        packed_loss_mask: Any,
        multimodal_inputs: Any,
        data_id: Any = None,
    ) -> str:
        if data_id:
            return str(data_id)
        payload = {
            "input_ids": input_ids,
            "packed_loss_mask": packed_loss_mask,
            "multimodal_inputs": multimodal_inputs,
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
            mooncake_target=self._build_mooncake_target(),
        )
        self.manifest.upsert(handle)
        return handle

    def resolve_and_load(self, sample: Mapping[str, Any], *, device) -> dict[str, Any]:
        handle = self.resolve_handle(sample)
        result = self._load_handle(handle, sample=sample, device=device)
        if "packed_loss_mask" in sample:
            result["packed_loss_mask"] = sample["packed_loss_mask"]
        return result

    def _handle_exists(self, handle: FeatureHandle) -> bool:
        if not self.mooncake_store.exists(f"{handle.mooncake_key}_hs"):
            return False
        if handle.prefix_sample_key:
            prefix = self.manifest.get(handle.prefix_sample_key, touch=False)
            if prefix is None:
                return False
            return self._handle_exists(prefix)
        return True

    def _load_handle(
        self,
        handle: FeatureHandle,
        *,
        sample: Mapping[str, Any],
        device,
    ) -> dict[str, Any]:
        tensors = self.mooncake_store.get(
            key=handle.mooncake_key,
            shapes=handle.tensor_shapes,
            dtypes=self._torch_dtypes(handle.tensor_dtypes),
            device=device,
        )
        result = tensors.to_tensor_dict()
        if not handle.prefix_sample_key:
            return result

        prefix_sample = self._slice_prefix_sample(sample, handle.cached_tokens)
        prefix_handle = self.manifest.get(handle.prefix_sample_key)
        if prefix_handle is None:
            # The remote radix cache can hit prefixes that this local cache has not
            # seen before, so recursively fetch the prefix features on demand.
            prefix_handle = self.resolve_handle(prefix_sample)
        if prefix_handle is None:
            raise RemoteSGLangError(
                "Remote SGLang returned cached prompt tokens, but the local feature cache "
                f"could not resolve the required prefix sample {handle.prefix_sample_key} "
                f"(cached_tokens={handle.cached_tokens})."
            )

        prefix_result = self._load_handle(prefix_handle, sample=prefix_sample, device=device)
        merged = dict(result)
        for key in ("hidden_states", "last_hidden_states", "target"):
            prefix_tensor = prefix_result.get(key)
            suffix_tensor = result.get(key)
            if prefix_tensor is None:
                continue
            if suffix_tensor is None:
                merged[key] = prefix_tensor
                continue
            merged[key] = self._concat_feature_tensors(prefix_tensor, suffix_tensor)

        merged["input_ids"] = self._make_input_ids_tensor(sample["input_ids"], device=device)
        merged["input_ids_cpu"] = self._make_input_ids_tensor(sample["input_ids"], device="cpu")
        return merged

    @staticmethod
    def _concat_feature_tensors(prefix_tensor, suffix_tensor):
        import torch

        if prefix_tensor.dim() == 1:
            return torch.cat([prefix_tensor, suffix_tensor], dim=0)
        return torch.cat([prefix_tensor, suffix_tensor], dim=0)

    def _slice_prefix_sample(self, sample: Mapping[str, Any], prefix_len: int) -> dict[str, Any]:
        prefix_sample = dict(sample)
        prefix_sample["input_ids"] = list(sample["input_ids"][:prefix_len])
        packed_loss_mask = sample.get("packed_loss_mask")
        if isinstance(packed_loss_mask, str):
            prefix_sample["packed_loss_mask"] = packed_loss_mask[:prefix_len]
        # Prefix cache entries must be keyed by their actual prefix payload, not by
        # the parent sample's data_id/sample_key, otherwise cached-prefix stitching
        # can recurse back into the same manifest entry.
        prefix_sample.pop("data_id", None)
        prefix_sample["sample_key"] = self.build_sample_key_from_values(
            input_ids=prefix_sample["input_ids"],
            packed_loss_mask=prefix_sample.get("packed_loss_mask"),
            multimodal_inputs=prefix_sample.get("multimodal_inputs"),
        )
        return prefix_sample

    @staticmethod
    def _make_input_ids_tensor(input_ids, *, device):
        import torch

        return torch.tensor(list(input_ids), dtype=torch.int64, device=device)

    def _build_mooncake_target(self) -> dict[str, Any] | None:
        config = getattr(self.mooncake_store, "config", None)
        if config is None:
            return None

        target = {
            "master_server_address": getattr(config, "master_server_address", None),
            "metadata_server": getattr(config, "metadata_server", None),
            "protocol": getattr(config, "protocol", None),
            "device_name": getattr(config, "device_name", None),
            "enable_gpu_direct": getattr(config, "enable_gpu_direct", None),
        }
        return {key: value for key, value in target.items() if value is not None}

    @staticmethod
    def _torch_dtypes(dtypes: dict[str, str]) -> dict[str, Any]:
        import torch

        resolved = {}
        for key, value in dtypes.items():
            resolved[key] = getattr(torch, value.replace("torch.", ""))
        return resolved
