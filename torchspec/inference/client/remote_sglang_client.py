import json
import hashlib
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from torchspec.cache.cache_manifest import FeatureHandle


class RemoteSGLangError(RuntimeError):
    pass


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

    def request_features(
        self,
        sample_key: str,
        input_ids: list[int],
        packed_loss_mask: str,
        multimodal_inputs: Optional[dict[str, Any]],
        feature_schema_version: str,
    ) -> FeatureHandle:
        payload = {
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": 0},
            "return_hidden_states": True,
            "spec_training_data_id": sample_key,
            "packed_loss_mask": packed_loss_mask,
        }
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
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return self._parse_response(
                    body=body,
                    sample_key=sample_key,
                    input_ids=input_ids,
                    packed_loss_mask=packed_loss_mask,
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
        packed_loss_mask: str,
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
            cached_tokens = int(meta_info.get("cached_tokens", 0) or 0)
            total_seq_len = int(meta_info.get("prompt_tokens", len(input_ids)))
            suffix_seq_len = max(total_seq_len - cached_tokens, 0)
            prefix_sample_key = None
            if cached_tokens > 0:
                prefix_sample_key = self._build_sample_key(
                    input_ids=input_ids[:cached_tokens],
                    packed_loss_mask=packed_loss_mask[:cached_tokens],
                    multimodal_inputs=None,
                )
            return FeatureHandle(
                sample_key=sample_key,
                mooncake_key=store_keys[0],
                tensor_shapes={
                    "hidden_states": (suffix_seq_len, self.hidden_size * self.num_aux_hidden_layers),
                    "input_ids": (total_seq_len,),
                    "last_hidden_states": (suffix_seq_len, self.hidden_size),
                },
                tensor_dtypes={
                    "hidden_states": self.torch_dtype,
                    "input_ids": "int64",
                    "last_hidden_states": self.torch_dtype,
                },
                feature_schema_version=feature_schema_version,
                created_at=time.time(),
                expires_at=None,
                prefix_sample_key=prefix_sample_key,
                cached_tokens=cached_tokens,
            )
        except KeyError as exc:
            raise RemoteSGLangError(f"Malformed feature handle response: missing {exc.args[0]}") from exc

    @staticmethod
    def _normalize_torch_dtype(value: Any) -> str:
        if isinstance(value, str):
            return value.replace("torch.", "")
        dtype_name = getattr(value, "__str__", None)
        if callable(dtype_name):
            return str(value).replace("torch.", "")
        return "bfloat16"

    @staticmethod
    def _build_sample_key(
        *,
        input_ids: list[int],
        packed_loss_mask: str,
        multimodal_inputs: Optional[dict[str, Any]],
    ) -> str:
        payload = {
            "input_ids": input_ids,
            "packed_loss_mask": packed_loss_mask,
            "multimodal_inputs": multimodal_inputs,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
