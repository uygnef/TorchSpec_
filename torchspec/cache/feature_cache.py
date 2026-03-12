import hashlib
import json
import os
import logging
import uuid
from collections.abc import Mapping
from typing import Any

from torchspec.cache.cache_manifest import CacheManifest, FeatureHandle

logger = logging.getLogger(__name__)


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
        self.chunk_size = max(
            int(os.environ.get("TORCHSPEC_FEATURE_CACHE_CHUNK_SIZE", "64")),
            1,
        )

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
        feature_schema_version = sample.get(
            "feature_schema_version", self.feature_schema_version
        )
        return self.remote_client.request_features(
            sample_key=sample_key,
            input_ids=list(sample["input_ids"]),
            packed_loss_mask=sample.get("packed_loss_mask", ""),
            multimodal_inputs=sample.get("multimodal_inputs"),
            feature_schema_version=feature_schema_version,
            mooncake_target=self._build_mooncake_target(),
            tensor_mode="full",
        )

    def resolve_and_load(self, sample: Mapping[str, Any], *, device) -> dict[str, Any]:
        result = self._load_from_local_chunks(sample, device=device)
        if result is not None:
            self.metrics["hits"] += 1
        else:
            self.metrics["misses"] += 1
            full_handle = self.resolve_handle(sample)
            logger.warning(
                "FeatureCache remote full_handle sample_key=%s request_len=%d mooncake_key=%s shapes=%s",
                full_handle.sample_key,
                len(sample.get("input_ids") or []),
                full_handle.mooncake_key,
                full_handle.tensor_shapes,
            )
            tensors = self.mooncake_store.get(
                key=full_handle.mooncake_key,
                shapes=full_handle.tensor_shapes,
                dtypes=self._torch_dtypes(full_handle.tensor_dtypes),
                device=device,
            )
            result = tensors.to_tensor_dict()
            self._store_missing_chunks(sample, result, feature_schema_version=full_handle.feature_schema_version)
            self._cleanup_full_fetch(full_handle)
        if "packed_loss_mask" in sample:
            result["packed_loss_mask"] = sample["packed_loss_mask"]
        result["input_ids"] = self._make_input_ids_tensor(sample["input_ids"], device=device)
        result["input_ids_cpu"] = self._make_input_ids_tensor(sample["input_ids"], device="cpu")
        return result

    def _load_from_local_chunks(self, sample: Mapping[str, Any], *, device) -> dict[str, Any] | None:
        chunk_specs = self._build_chunk_specs(sample)
        if not chunk_specs:
            return {
                "hidden_states": self._empty_hidden_states(device=device),
                "last_hidden_states": self._empty_last_hidden_states(device=device),
                "input_ids": self._make_input_ids_tensor([], device=device),
                "input_ids_cpu": self._make_input_ids_tensor([], device="cpu"),
            }

        hidden_states_parts = []
        last_hidden_states_parts = []
        has_last_hidden_states = True
        for spec in chunk_specs:
            handle = self.manifest.get(spec["chunk_key"], touch=False)
            if handle is None:
                return None
            if not self._handle_exists(handle):
                if self.validate_missing_as_stale:
                    self.manifest.delete(spec["chunk_key"])
                    self.metrics["stale_regenerations"] += 1
                return None
            tensors = self.mooncake_store.get(
                key=handle.mooncake_key,
                shapes=handle.tensor_shapes,
                dtypes=self._torch_dtypes(handle.tensor_dtypes),
                device=device,
            ).to_tensor_dict()
            hidden_states_parts.append(tensors["hidden_states"])
            chunk_last_hidden = tensors.get("last_hidden_states")
            if chunk_last_hidden is None:
                has_last_hidden_states = False
            elif has_last_hidden_states:
                last_hidden_states_parts.append(chunk_last_hidden)

        result = {
            "hidden_states": self._concat_feature_tensors(hidden_states_parts),
            "input_ids": self._make_input_ids_tensor(sample["input_ids"], device=device),
            "input_ids_cpu": self._make_input_ids_tensor(sample["input_ids"], device="cpu"),
        }
        if has_last_hidden_states and last_hidden_states_parts:
            result["last_hidden_states"] = self._concat_feature_tensors(last_hidden_states_parts)
        return result

    def _store_missing_chunks(
        self,
        sample: Mapping[str, Any],
        tensors: Mapping[str, Any],
        *,
        feature_schema_version: str,
    ) -> None:
        chunk_specs = self._build_chunk_specs(sample)
        for spec in chunk_specs:
            existing = self.manifest.get(spec["chunk_key"], touch=False)
            if existing is not None and self._handle_exists(existing):
                continue
            if existing is not None and self.validate_missing_as_stale:
                self.manifest.delete(spec["chunk_key"])
                self.metrics["stale_regenerations"] += 1

            chunk_hidden_states = tensors["hidden_states"][spec["start"] : spec["end"]].contiguous()
            chunk_input_ids = self._make_input_ids_tensor(
                sample["input_ids"][spec["start"] : spec["end"]],
                device=chunk_hidden_states.device,
            )
            chunk_last_hidden_states = None
            if tensors.get("last_hidden_states") is not None:
                chunk_last_hidden_states = tensors["last_hidden_states"][
                    spec["start"] : spec["end"]
                ].contiguous()

            mooncake_key = self._make_chunk_mooncake_key(spec["chunk_key"])
            shapes = self.mooncake_store.put(
                key=mooncake_key,
                hidden_states=chunk_hidden_states,
                input_ids=chunk_input_ids,
                last_hidden_states=chunk_last_hidden_states,
            )
            handle = FeatureHandle(
                sample_key=spec["chunk_key"],
                mooncake_key=mooncake_key,
                tensor_shapes=shapes,
                tensor_dtypes={
                    "hidden_states": str(chunk_hidden_states.dtype).replace("torch.", ""),
                    "input_ids": "int64",
                    "last_hidden_states": (
                        str(chunk_last_hidden_states.dtype).replace("torch.", "")
                        if chunk_last_hidden_states is not None
                        else str(chunk_hidden_states.dtype).replace("torch.", "")
                    ),
                },
                feature_schema_version=feature_schema_version,
                created_at=self._now(),
            )
            self.manifest.upsert(handle)

    def _build_chunk_specs(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        input_ids = list(sample.get("input_ids") or [])
        packed_loss_mask = sample.get("packed_loss_mask")
        multimodal_inputs = sample.get("multimodal_inputs")
        specs = []
        for start in range(0, len(input_ids), self.chunk_size):
            end = min(start + self.chunk_size, len(input_ids))
            prefix_ids = input_ids[:end]
            prefix_loss_mask = None
            if isinstance(packed_loss_mask, str):
                prefix_loss_mask = packed_loss_mask[:end]
            prefix_key = self.build_sample_key_from_values(
                input_ids=prefix_ids,
                packed_loss_mask=prefix_loss_mask,
                multimodal_inputs=multimodal_inputs,
            )
            specs.append(
                {
                    "start": start,
                    "end": end,
                    "chunk_key": f"{prefix_key}:chunk:{start}:{end}",
                }
            )
        return specs

    def _handle_exists(self, handle: FeatureHandle) -> bool:
        def _numel(shape) -> int:
            total = 1
            for dim in shape:
                total *= int(dim)
            return total

        if _numel(handle.tensor_shapes.get("input_ids", ())) > 0 and not self.mooncake_store.exists(
            f"{handle.mooncake_key}_ids"
        ):
            return False
        if _numel(handle.tensor_shapes.get("hidden_states", ())) > 0 and not self.mooncake_store.exists(
            f"{handle.mooncake_key}_hs"
        ):
            return False
        if _numel(handle.tensor_shapes.get("last_hidden_states", ())) > 0 and not self.mooncake_store.exists(
            f"{handle.mooncake_key}_lhs"
        ):
            return False
        if _numel(handle.tensor_shapes.get("target", ())) > 0 and not self.mooncake_store.exists(
            f"{handle.mooncake_key}_tgt"
        ):
            return False
        return True

    def _cleanup_full_fetch(self, handle: FeatureHandle) -> None:
        try:
            self.mooncake_store.remove_eagle3_tensors(
                key=handle.mooncake_key,
                has_last_hidden_states=("last_hidden_states" in handle.tensor_shapes),
                has_target=("target" in handle.tensor_shapes),
            )
        except Exception:
            pass

    def _make_chunk_mooncake_key(self, chunk_key: str) -> str:
        digest = hashlib.sha1(chunk_key.encode("utf-8")).hexdigest()[:16]
        return f"chunk_{digest}_{uuid.uuid4().hex[:8]}"

    def _empty_hidden_states(self, *, device):
        import torch

        width = self.remote_client.hidden_size * self.remote_client.num_aux_hidden_layers
        dtype = getattr(torch, self.remote_client.torch_dtype)
        return torch.empty((0, width), dtype=dtype, device=device)

    def _empty_last_hidden_states(self, *, device):
        import torch

        dtype = getattr(torch, self.remote_client.torch_dtype)
        return torch.empty((0, self.remote_client.hidden_size), dtype=dtype, device=device)

    @staticmethod
    def _concat_feature_tensors(parts):
        import torch

        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=0)

    @staticmethod
    def _make_input_ids_tensor(input_ids, *, device):
        import torch

        return torch.tensor(list(input_ids), dtype=torch.int64, device=device)

    def _build_mooncake_target(self) -> dict[str, Any] | None:
        config = getattr(self.mooncake_store, "config", None)
        if config is None:
            return None

        target = {
            "local_hostname": getattr(config, "local_hostname", None),
            "master_server_address": getattr(config, "master_server_address", None),
            "metadata_server": getattr(config, "metadata_server", None),
            "global_segment_size": getattr(config, "global_segment_size", None),
            "local_buffer_size": getattr(config, "local_buffer_size", None),
            "host_buffer_size": getattr(config, "host_buffer_size", None),
            "protocol": getattr(config, "protocol", None),
            "device_name": getattr(config, "device_name", None),
            "enable_gpu_direct": getattr(config, "enable_gpu_direct", None),
            "async_put_pool_size": getattr(config, "async_put_pool_size", None),
            "kv_lease_ttl_s": getattr(config, "kv_lease_ttl_s", None),
        }
        return {key: value for key, value in target.items() if value is not None}

    @staticmethod
    def _torch_dtypes(dtypes: dict[str, str]) -> dict[str, Any]:
        import torch

        resolved = {}
        for key, value in dtypes.items():
            resolved[key] = getattr(torch, value.replace("torch.", ""))
        return resolved

    @staticmethod
    def _now() -> float:
        import time

        return time.time()
