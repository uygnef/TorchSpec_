import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from torchspec.cache.cache_manifest import FeatureHandle
from torchspec.inference.client.errors import RemoteSGLangError

logger = logging.getLogger(__name__)


class RemoteSGLangClient:
    def __init__(
        self,
        endpoint: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        *,
        hidden_size: int,
        num_aux_hidden_layers: int,
        torch_dtype: str = "bfloat16",
    ):
        if not endpoint:
            raise ValueError("Remote SGLang endpoint must be configured")
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.hidden_size = hidden_size
        self.num_aux_hidden_layers = num_aux_hidden_layers
        self.torch_dtype = self._normalize_torch_dtype(torch_dtype)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request_features(
        self,
        sample_key: str,
        input_ids: list[int],
        packed_loss_mask: str,
        multimodal_inputs: Optional[dict[str, Any]],
        feature_schema_version: str,
        mooncake_target: Optional[dict[str, Any]] = None,
        *,
        tensor_mode: str = "full",
    ) -> FeatureHandle:
        del tensor_mode
        payload = {
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": 0},
            "return_hidden_states": True,
            "spec_training_data_id": sample_key,
            "packed_loss_mask": packed_loss_mask,
            "spec_training_tensor_mode": "full",
        }
        if mooncake_target:
            payload["mooncake_target"] = mooncake_target
        if multimodal_inputs:
            payload.update(multimodal_inputs)
        request = urllib.request.Request(
            url=f"{self.endpoint}/generate_for_spec_training",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._opener.open(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return self._parse_response(
                    body=body,
                    sample_key=sample_key,
                    input_ids=input_ids,
                    feature_schema_version=feature_schema_version,
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8") if hasattr(exc, "read") else ""
                raise RemoteSGLangError(
                    f"Remote SGLang HTTP {exc.code}: {body or exc.reason}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(0.25 * (attempt + 1), 1.0))

        raise RemoteSGLangError(f"Remote SGLang request failed: {last_error}") from last_error

    def _parse_response(
        self,
        *,
        body: dict[str, Any] | list[dict[str, Any]],
        sample_key: str,
        input_ids: list[int],
        feature_schema_version: str,
    ) -> FeatureHandle:
        if isinstance(body, list):
            if not body:
                raise RemoteSGLangError("Malformed feature handle response: empty response list")
            body = body[0]
        if not isinstance(body, dict):
            raise RemoteSGLangError(
                f"Malformed feature handle response: expected object, got {type(body).__name__}"
            )
        status = body.get("status", "ok")
        if status != "ok":
            raise RemoteSGLangError(
                f"{body.get('error_code', 'UNKNOWN_ERROR')}: {body.get('message', 'request failed')}"
            )
        try:
            meta_info = body["meta_info"]
            store_keys = meta_info["spec_training_mooncake_store_keys"]
            if not store_keys:
                raise RemoteSGLangError("Spec training response missing mooncake store keys")
            tensor_shapes = self._parse_tensor_shapes(meta_info.get("spec_training_tensor_shapes"))
            logger.warning(
                "Remote spec-training response sample_key=%s request_len=%d raw_shapes=%s store_keys=%s",
                sample_key,
                len(input_ids),
                tensor_shapes,
                store_keys,
            )
            if tensor_shapes is None:
                seq_len = len(input_ids)
                tensor_shapes = {
                    "hidden_states": (
                        seq_len,
                        self.hidden_size * self.num_aux_hidden_layers,
                    ),
                    "input_ids": (seq_len,),
                    "last_hidden_states": (seq_len, self.hidden_size),
                }

            return FeatureHandle(
                sample_key=sample_key,
                mooncake_key=store_keys[0],
                tensor_shapes=tensor_shapes,
                tensor_dtypes={
                    "hidden_states": self.torch_dtype,
                    "input_ids": "int64",
                    "last_hidden_states": self.torch_dtype,
                },
                feature_schema_version=feature_schema_version,
                created_at=time.time(),
                expires_at=None,
                prefix_sample_key=None,
                cached_tokens=0,
            )
        except KeyError as exc:
            raise RemoteSGLangError(f"Malformed feature handle response: missing {exc.args[0]}") from exc

    @staticmethod
    def _resolve_full_seq_len(
        tensor_shapes: dict[str, tuple[int, ...]] | None,
        input_ids: list[int],
    ) -> int:
        del tensor_shapes
        return len(input_ids)

    @staticmethod
    def _normalize_full_shapes(
        tensor_shapes: dict[str, tuple[int, ...]],
        seq_len: int,
    ) -> dict[str, tuple[int, ...]]:
        normalized = dict(tensor_shapes)
        normalized["input_ids"] = (seq_len,)
        if "hidden_states" in normalized and len(normalized["hidden_states"]) >= 2:
            normalized["hidden_states"] = (seq_len, *normalized["hidden_states"][1:])
        if "last_hidden_states" in normalized and len(normalized["last_hidden_states"]) >= 2:
            normalized["last_hidden_states"] = (
                seq_len,
                *normalized["last_hidden_states"][1:],
            )
        return normalized

    @staticmethod
    def _normalize_torch_dtype(value: Any) -> str:
        if isinstance(value, str):
            return value.replace("torch.", "")
        dtype_name = getattr(value, "__str__", None)
        if callable(dtype_name):
            return str(value).replace("torch.", "")
        return "bfloat16"

    @staticmethod
    def _parse_tensor_shapes(payload: Any) -> dict[str, tuple[int, ...]] | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise RemoteSGLangError(
                "Malformed feature handle response: spec_training_tensor_shapes must be an object"
            )
        parsed: dict[str, tuple[int, ...]] = {}
        for name, shape in payload.items():
            if not isinstance(name, str) or not isinstance(shape, list):
                raise RemoteSGLangError(
                    "Malformed feature handle response: invalid spec_training_tensor_shapes entry"
                )
            parsed[name] = tuple(int(dim) for dim in shape)
        return parsed
