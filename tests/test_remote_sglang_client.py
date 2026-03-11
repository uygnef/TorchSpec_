import io
import json
import urllib.error
from unittest.mock import patch

import pytest
import torch

from torchspec.inference.client.remote_sglang_client import (
    RemoteSGLangClient,
    RemoteSGLangError,
)


class _MockHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_request_features_success():
    client = RemoteSGLangClient(
        "http://127.0.0.1:8000",
        timeout_seconds=1.0,
        max_retries=0,
        hidden_size=4,
        num_aux_hidden_layers=2,
    )
    payload = [
        {
            "meta_info": {
                "spec_training_data_id": "dataset:1",
                "packed_loss_mask": "mask",
                "spec_training_mooncake_store_keys": ["feature:dataset:1"],
                "prompt_tokens": 2,
            }
        }
    ]

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
    assert handle.mooncake_key == "feature:dataset:1"
    assert handle.tensor_shapes["hidden_states"] == (2, 8)


def test_request_features_service_error():
    client = RemoteSGLangClient(
        "http://127.0.0.1:8000",
        timeout_seconds=1.0,
        max_retries=0,
        hidden_size=4,
        num_aux_hidden_layers=2,
    )
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


def test_request_features_service_error_from_list_payload():
    client = RemoteSGLangClient(
        "http://127.0.0.1:8000",
        timeout_seconds=1.0,
        max_retries=0,
        hidden_size=4,
        num_aux_hidden_layers=2,
    )
    payload = [
        {
            "status": "error",
            "error_code": "MOONCAKE_WRITE_FAILED",
            "message": "batch_put_from failed",
        }
    ]

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
    client = RemoteSGLangClient(
        "http://127.0.0.1:8000",
        timeout_seconds=1.0,
        max_retries=0,
        hidden_size=4,
        num_aux_hidden_layers=2,
    )
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


def test_request_features_missing_store_keys():
    client = RemoteSGLangClient(
        "http://127.0.0.1:8000",
        timeout_seconds=1.0,
        max_retries=0,
        hidden_size=4,
        num_aux_hidden_layers=2,
    )
    payload = {"meta_info": {"prompt_tokens": 2, "spec_training_mooncake_store_keys": []}}

    with patch("urllib.request.urlopen", return_value=_MockHTTPResponse(payload)):
        with pytest.raises(RemoteSGLangError, match="missing mooncake store keys"):
            client.request_features(
                sample_key="dataset:1",
                input_ids=[1, 2],
                packed_loss_mask="mask",
                multimodal_inputs=None,
                feature_schema_version="eagle3.v1",
            )


def test_normalize_dtype_object_to_string():
    client = RemoteSGLangClient(
        "http://127.0.0.1:8000",
        timeout_seconds=1.0,
        max_retries=0,
        hidden_size=4,
        num_aux_hidden_layers=2,
        torch_dtype=torch.bfloat16,
    )

    assert client.torch_dtype == "bfloat16"
