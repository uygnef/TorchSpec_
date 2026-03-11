# DeepSeek-V3.1 Remote-SGLang Training

Production-style launcher for training an Eagle3 draft model for DeepSeek-V3.1 with the decoupled remote-SGLang architecture.

## Prerequisites

- A reachable remote `sglang` service exposing `/generate_for_spec_training`
- Mooncake metadata/master services reachable from the training job
- Training node(s) with GPUs for TorchSpec
- Model access to `/nfs/ofs-llm-ssd/models/opensource/DeepSeek-V3.1`

## Config

Uses [`configs/sglang_deepseek_v31_3node.yaml`](../../configs/sglang_deepseek_v31_3node.yaml).

Default assumptions:

- `dataset.chat_template=deepseek-v3`
- `inference.mode=remote_sglang`
- `feature_cache.enabled=true`
- `model.draft_model_config` is left unset, so TorchSpec auto-generates a 1-layer Eagle3 config from the target model

## How to run

### 1. Start the remote SGLang service

On the inference/service machine:

```bash
bash examples/deepseek-v31-3node-h100/launch_remote_sglang_server.sh
```

### 2. Start the Ray cluster

On the head node:

```bash
NODE_ROLE=head bash examples/deepseek-v31-3node-h100/setup_ray_cluster.sh
```

On each worker node:

```bash
HEAD_IP=<node0_ip> NODE_ROLE=worker bash examples/deepseek-v31-3node-h100/setup_ray_cluster.sh
```

### 3. Launch training

On the head node:

```bash
REMOTE_SGLANG_ENDPOINT=http://<sglang_host>:30000 \
bash examples/deepseek-v31-3node-h100/run.sh
```

### 4. Launch through the cluster job wrapper

If you run in the same multi-node job environment as `debug` branch `run_job.sh`, use:

```bash
bash examples/deepseek-v31-3node-h100/run_job.sh
```

### 5. One-command smoke on the training machine

On the head node:

```bash
NODE_ROLE=head \
REMOTE_SGLANG_ENDPOINT=http://<sglang_host>:30000 \
bash examples/deepseek-v31-3node-h100/run_remote_smoke.sh
```

## Common customizations

```bash
# Override model path and dataset config
MODEL_PATH=/nfs/ofs-llm-ssd/models/opensource/DeepSeek-V3.1 \
bash examples/deepseek-v31-3node-h100/run.sh \
  dataset.train_data_path=/path/to/train.jsonl

# Use a different DeepSeek chat template
CHAT_TEMPLATE=deepseek-v32 bash examples/deepseek-v31-3node-h100/run.sh

# Point to a remote service and Mooncake control plane
REMOTE_SGLANG_ENDPOINT=http://10.0.0.8:30000 \
MOONCAKE_MASTER_ADDRESS=10.0.0.8:50051 \
MOONCAKE_METADATA_SERVER=10.0.0.8 \
MOONCAKE_METADATA_PORT=50052 \
bash examples/deepseek-v31-3node-h100/run.sh

# Override config file
CONFIG_FILE=configs/custom_deepseek.yaml bash examples/deepseek-v31-3node-h100/run.sh
```
