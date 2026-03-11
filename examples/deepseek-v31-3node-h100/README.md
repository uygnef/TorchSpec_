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

The script prints a `Reachable endpoint` line. Use that value for `REMOTE_SGLANG_ENDPOINT`.
In Luban job environments, the server port defaults to `LUBAN_AVAILABLE_PORT_1` when present.
The distributed rendezvous address is auto-selected and printed as `Dist init addr`; you can override it with `DIST_INIT_ADDR` or `DIST_PORT`.

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

Skip the config-only precheck if you want the script to go straight into training:

```bash
NODE_ROLE=head \
CHECK_CONFIG=false \
REMOTE_SGLANG_ENDPOINT=http://<sglang_host>:30000 \
bash examples/deepseek-v31-3node-h100/run_remote_smoke.sh
```

In job environments that expose `DISTRIBUTED_MASTER_HOSTS`, the scripts will use that hostname for Mooncake by default. If you want to force DNS resolution to a concrete IP, add `RESOLVE_MASTER_IP=true`.
By default the smoke/train scripts do not pin Mooncake ports; they let TorchSpec auto-pick free ports unless you explicitly pass `MOONCAKE_MASTER_ADDRESS` or `MOONCAKE_METADATA_PORT`.

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
