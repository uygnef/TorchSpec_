# DeepSeek-V3.1 3-Node Training

Production-style 3-node setup for training an Eagle3 draft model for DeepSeek-V3.1.

## Prerequisites

- 3 nodes with 24 GPUs total:
  - Node 0 (head): 8 GPUs for training
  - Node 1-2 (workers): 8 GPUs each for inference (TP=16)
- Model access to `/nfs/ofs-llm-ssd/models/opensource/DeepSeek-V3.1`
- RDMA network if you want Mooncake RDMA mode

## Config

Uses [`configs/sglang_deepseek_v31_3node.yaml`](../../configs/sglang_deepseek_v31_3node.yaml).

Default assumptions:

- `dataset.chat_template=deepseek-v3`
- `model.draft_model_config` is left unset, so TorchSpec auto-generates a 1-layer Eagle3 config from the target model

## How to run

### 1. Start the Ray cluster

On the head node:

```bash
NODE_ROLE=head bash examples/deepseek-v31-3node-h100/setup_ray_cluster.sh
```

On each worker node:

```bash
HEAD_IP=<node0_ip> NODE_ROLE=worker bash examples/deepseek-v31-3node-h100/setup_ray_cluster.sh
```

### 2. Launch training

On the head node:

```bash
bash examples/deepseek-v31-3node-h100/run.sh
```

### 3. Launch through the cluster job wrapper

If you run in the same multi-node job environment as `debug` branch `run_job.sh`, use:

```bash
bash examples/deepseek-v31-3node-h100/run_job.sh
```

## Common customizations

```bash
# Override model path and dataset config
MODEL_PATH=/nfs/ofs-llm-ssd/models/opensource/DeepSeek-V3.1 \
bash examples/deepseek-v31-3node-h100/run.sh \
  dataset.train_data_path=/path/to/train.jsonl

# Use a different DeepSeek chat template
CHAT_TEMPLATE=deepseek-v32 bash examples/deepseek-v31-3node-h100/run.sh

# Override config file
CONFIG_FILE=configs/custom_deepseek.yaml bash examples/deepseek-v31-3node-h100/run.sh
```
