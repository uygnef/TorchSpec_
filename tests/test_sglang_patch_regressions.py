from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = PROJECT_ROOT / "patches" / "sglang" / "v0.5.8.post1"


def test_sglang_patch_handles_missing_enable_gpu_direct():
    for patch_name in ("sglang.patch", "sglang_decode.patch"):
        patch_text = (PATCH_DIR / patch_name).read_text()
        assert 'if hasattr(config, "enable_gpu_direct")' in patch_text
        assert 'getattr(config, "enable_gpu_direct", False)' in patch_text


def test_sglang_patch_routes_mooncake_target_per_request():
    for patch_name in ("sglang.patch", "sglang_decode.patch"):
        patch_text = (PATCH_DIR / patch_name).read_text()
        assert "mooncake_target" in patch_text
        assert "def get_eagle_mooncake_store" in patch_text


def test_sglang_patch_reports_actual_tensor_shapes():
    for patch_name in ("sglang.patch", "sglang_decode.patch"):
        patch_text = (PATCH_DIR / patch_name).read_text()
        assert "spec_training_tensor_shapes" in patch_text
        assert "req.spec_training_tensor_shapes = {" in patch_text
        assert 'meta_info["spec_training_tensor_shapes"]' in patch_text


def test_sglang_patch_adds_tensor_shapes_to_all_batch_outputs():
    required = "spec_training_tensor_shapes: Optional[List[Optional[Dict[str, List[int]]]]] = None"
    for patch_name in ("sglang.patch", "sglang_decode.patch"):
        patch_text = (PATCH_DIR / patch_name).read_text()
        assert patch_text.count(required) >= 2
        assert "class BatchTokenIDOutput(" in patch_text
        assert "class BatchStrOutput(" in patch_text
