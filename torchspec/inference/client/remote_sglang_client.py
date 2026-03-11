import json
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from torchspec.cache.cache_manifest import FeatureHandle


class RemoteSGLangError(RuntimeError):
    pass


class RemoteSGLangClient:
    def __init__(self, endpoint: str, timeout_seconds: float = 30.0, max_retries: int = 2):
        if not endpoint:
            raise ValueError("Remote SGLang endpoint must be configured")
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def request_features(
        self,
        sample_key: str,
        input_ids: list[int],
        packed_loss_mask: str,
        multimodal_inputs: Optional[dict[str, Any]],
        feature_schema_version: str,
    ) -> FeatureHandle:
        payload = {
            "sample_key": sample_key,
            "input_ids": input_ids,
            "packed_loss_mask": packed_loss_mask,
            "multimodal_inputs": multimodal_inputs,
            "feature_schema_version": feature_schema_version,
        }
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
                return self._parse_response(body)
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

    @staticmethod
    def _parse_response(body: dict[str, Any]) -> FeatureHandle:
        status = body.get("status", "ok")
        if status != "ok":
            raise RemoteSGLangError(
                f"{body.get('error_code', 'UNKNOWN_ERROR')}: {body.get('message', 'request failed')}"
            )
        try:
            return FeatureHandle(
                sample_key=body["sample_key"],
                mooncake_key=body["mooncake_key"],
                tensor_shapes={
                    name: tuple(shape) for name, shape in body.get("tensor_shapes", {}).items()
                },
                tensor_dtypes=body.get("tensor_dtypes", {}),
                feature_schema_version=body["feature_schema_version"],
                created_at=float(body["created_at"]),
                expires_at=body.get("expires_at"),
            )
        except KeyError as exc:
            raise RemoteSGLangError(f"Malformed feature handle response: missing {exc.args[0]}") from exc
