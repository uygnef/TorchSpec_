import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from torchspec.inference.client.remote_sglang_client import (
    RemoteSGLangClient,
    RemoteSGLangError,
)


class _MockHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_request_features_success():
    client = RemoteSGLangClient("http://127.0.0.1:8000", timeout_seconds=1.0, max_retries=0)
    payload = {
        "status": "ok",
        "sample_key": "dataset:1",
        "mooncake_key": "feature:dataset:1",
        "tensor_shapes": {"hidden_states": [2, 4], "input_ids": [2]},
        "tensor_dtypes": {"hidden_states": "bfloat16", "input_ids": "int64"},
        "feature_schema_version": "eagle3.v1",
        "created_at": 123.0,
    }

    with patch("urllib.request.urlopen", return_value=_MockHTTPResponse(payload)) as mock_urlopen:
        handle = client.request_features(
            sample_key="dataset:1",
            input_ids=[1, 2],
            packed_loss_mask="mask",
            multimodal_inputs=None,
            feature_schema_version="eagle3.v1",
        )

    assert mock_urlopen.call_count == 1
    assert handle.sample_key == "dataset:1"
    assert handle.tensor_shapes["hidden_states"] == (2, 4)


def test_request_features_service_error():
    client = RemoteSGLangClient("http://127.0.0.1:8000", timeout_seconds=1.0, max_retries=0)
    payload = {
        "status": "error",
        "error_code": "MOONCAKE_WRITE_FAILED",
        "message": "batch_put_from failed",
    }

    with patch("urllib.request.urlopen", return_value=_MockHTTPResponse(payload)):
        with pytest.raises(RemoteSGLangError, match="MOONCAKE_WRITE_FAILED"):
            client.request_features(
                sample_key="dataset:1",
                input_ids=[1, 2],
                packed_loss_mask="mask",
                multimodal_inputs=None,
                feature_schema_version="eagle3.v1",
            )


def test_request_features_http_error():
    client = RemoteSGLangClient("http://127.0.0.1:8000", timeout_seconds=1.0, max_retries=0)
    error = urllib.error.HTTPError(
        url="http://127.0.0.1:8000/generate_for_spec_training",
        code=500,
        msg="internal error",
        hdrs=None,
        fp=io.BytesIO(b'{"message":"server exploded"}'),
    )

    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(RemoteSGLangError, match="HTTP 500"):
            client.request_features(
                sample_key="dataset:1",
                input_ids=[1, 2],
                packed_loss_mask="mask",
                multimodal_inputs=None,
                feature_schema_version="eagle3.v1",
            )
