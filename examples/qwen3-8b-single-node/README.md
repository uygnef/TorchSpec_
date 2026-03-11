# SGLang Single-Node

Demo of disaggregated training with SGLang async inference engine.

## Prerequisites

- 4+ GPUs (2 inference + 2 training by default)
- Model access to `Qwen/Qwen3-8B`
- SGLang installed (included in the `torchspec` conda environment)

## Config

Uses [`configs/sglang_qwen3_8b.yaml`](../../configs/sglang_qwen3_8b.yaml):
- **Backend:** SGLang engine with async inference
- **Training:** 2 GPUs with FSDP, flex_attention
- **Inference:** 2 GPUs in duplicate mode (each engine has full model copy)

## How to run

```bash
./examples/qwen3-8b-single-node/run.sh
```

With a custom config:

```bash
./examples/qwen3-8b-single-node/run.sh configs/sglang_qwen3_8b.yaml
```

Override settings:

```bash
./examples/qwen3-8b-single-node/run.sh configs/sglang_qwen3_8b.yaml training.num_train_steps=10
```

## What to expect

Training launches with SGLang serving the target model for inference. Loss should decrease steadily. Logs are printed to stdout.

## Common customizations

```bash
# Use all 8 GPUs (4 inference + 4 training)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./examples/qwen3-8b-single-node/run.sh \
    training.training_num_gpus_per_node=4 \
    inference.inference_num_gpus=4
```

## Remote-SGLang Smoke

If you want to test the new remote-SGLang feature path first, use:

Service machine:

```bash
bash examples/qwen3-8b-single-node/launch_remote_sglang_server.sh
```

In Luban job environments, the server port defaults to `LUBAN_AVAILABLE_PORT_1` when present.
The launcher also enables `--enable-return-hidden-states`, which is required by TorchSpec remote feature extraction.

Training machine:

```bash
REMOTE_SGLANG_ENDPOINT=http://<sglang_host>:30000 \
bash examples/qwen3-8b-single-node/run_remote_smoke.sh
```

If you want to skip the config-only precheck:

```bash
CHECK_CONFIG=false \
REMOTE_SGLANG_ENDPOINT=http://<sglang_host>:30000 \
bash examples/qwen3-8b-single-node/run_remote_smoke.sh
```

In job environments that expose `DISTRIBUTED_MASTER_HOSTS`, the script will use that hostname for Mooncake by default. If you want to force DNS resolution to a concrete IP, add `RESOLVE_MASTER_IP=true`.
By default the smoke script does not pin Mooncake ports; it lets TorchSpec auto-pick free ports unless you explicitly pass `MOONCAKE_MASTER_ADDRESS` or `MOONCAKE_METADATA_PORT`.
