from argparse import Namespace
from unittest.mock import MagicMock, patch


def _make_args(**overrides):
    defaults = dict(
        inference_mode="remote_sglang",
        training_num_nodes=1,
        training_num_gpus_per_node=2,
        per_dp_rank_batch_size=2,
        max_concurrent_batches=1,
        dp_size=2,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_setup_async_training_with_engines_remote_mode_skips_inference_manager():
    args = _make_args()
    train_group = MagicMock()
    controller = MagicMock()
    controller.get_train_queues.remote.return_value = ["q0", "q1"]
    controller.get_eval_queues.remote.return_value = ["eq0", "eq1"]
    mooncake_config = object()

    with patch("torchspec.controller.setup.ray.get", side_effect=lambda x: x):
        from torchspec.controller.setup import setup_async_training_with_engines

        result_controller, inference_manager = setup_async_training_with_engines(
            args=args,
            train_group=train_group,
            mooncake_config=mooncake_config,
            inference_engines=[],
            controller=controller,
        )

    assert result_controller is controller
    assert inference_manager is None
    train_group.set_train_queues.assert_called_once_with(
        ["q0", "q1"], mooncake_config, per_dp_rank_batch_size=2
    )
    train_group.set_eval_queues.assert_called_once_with(
        ["eq0", "eq1"], mooncake_config, per_dp_rank_batch_size=1
    )


def test_expected_gpu_count_ignores_inference_in_remote_mode():
    from torchspec.ray.placement_group import _get_expected_gpu_count

    args = _make_args(
        training_num_nodes=2,
        training_num_gpus_per_node=4,
        inference_num_gpus=8,
    )

    assert _get_expected_gpu_count(args) == 8
