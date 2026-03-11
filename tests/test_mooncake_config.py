from types import SimpleNamespace

from torchspec.config.mooncake_config import MooncakeConfig


def test_from_flat_args_builds_metadata_url_from_host_only_value():
    args = SimpleNamespace(
        mooncake_master_server_address="127.0.0.1:50051",
        mooncake_metadata_server="127.0.0.1",
        mooncake_metadata_port=50052,
        mooncake_local_hostname="10.0.0.1",
    )

    config = MooncakeConfig.from_flat_args(args)

    assert config.metadata_server == "http://127.0.0.1:50052/metadata"


def test_from_flat_args_preserves_full_metadata_url():
    args = SimpleNamespace(
        mooncake_master_server_address="127.0.0.1:50051",
        mooncake_metadata_server="http://10.0.0.2:50052/metadata",
        mooncake_metadata_port=50052,
        mooncake_local_hostname="10.0.0.1",
    )

    config = MooncakeConfig.from_flat_args(args)

    assert config.metadata_server == "http://10.0.0.2:50052/metadata"
