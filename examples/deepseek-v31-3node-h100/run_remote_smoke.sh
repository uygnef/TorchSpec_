#!/bin/bash
# Run a minimal TorchSpec remote-SGLang smoke on the training machine.
#
# This script:
# 1. activates the torchspec environment
# 2. starts/joins the local Ray cluster
# 3. prints the resolved config
# 4. runs a 1-step training smoke
#
# Usage:
#   NODE_ROLE=head REMOTE_SGLANG_ENDPOINT=http://<sglang_host>:30000 \
#   bash examples/deepseek-v31-3node-h100/run_remote_smoke.sh
#
# Environment variables:
#   NODE_ROLE                 - head | worker
#   HEAD_IP                   - required for worker nodes
#   REMOTE_SGLANG_ENDPOINT    - remote SGLang endpoint
#   TRAIN_DATA_PATH           - dataset path
#   TRAIN_GPUS                - training GPUs per node
#   TRAIN_NODES               - number of training nodes
#   OUTPUT_DIR                - output directory for smoke
#   CACHE_DIR                 - cache directory for smoke
#   MOONCAKE_MASTER_ADDRESS   - Mooncake master address
#   MOONCAKE_METADATA_SERVER  - metadata host
#   MOONCAKE_METADATA_PORT    - metadata port

set -euo pipefail
set -x

export PATH="/nfs/ofs-fengyu/env/conda/envs/torchspec/bin:/nfs/ofs-fengyu/env/conda/condabin:/nfs/ofs-fengyu/env/conda/bin/:$PATH"
export MAMBA_EXE="/nfs/ofs-fengyu/env/conda/bin/micromamba"
export MAMBA_ROOT_PREFIX="/nfs/ofs-fengyu/env/conda"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate torchspec

WORKING_DIR="${WORKING_DIR:-/nfs/ofs-llab-volume/users/fengyu/TorchSpec}"
cd "$WORKING_DIR"

NODE_ROLE="${NODE_ROLE:-head}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$WORKING_DIR/examples/data/sample_conversations.jsonl}"
REMOTE_SGLANG_ENDPOINT="${REMOTE_SGLANG_ENDPOINT:?REMOTE_SGLANG_ENDPOINT must be set}"
TRAIN_GPUS="${TRAIN_GPUS:-8}"
TRAIN_NODES="${TRAIN_NODES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs/ofs-llab-volume/users/fengyu/o/deepseek_remote_smoke}"
CACHE_DIR="${CACHE_DIR:-/nfs/ofs-llab-volume/users/fengyu/c/deepseek_remote_smoke}"
LOCAL_IP="$(python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
finally:
    s.close()
PY
)"
CHECK_CONFIG="${CHECK_CONFIG:-true}"
MOONCAKE_MASTER_ADDRESS="${MOONCAKE_MASTER_ADDRESS:-$LOCAL_IP:50051}"
MOONCAKE_METADATA_SERVER="${MOONCAKE_METADATA_SERVER:-$LOCAL_IP}"
MOONCAKE_METADATA_PORT="${MOONCAKE_METADATA_PORT:-50052}"

LOG_DIR="$WORKING_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/deepseek_v31_remote_smoke_${NODE_ROLE}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

if [ "$NODE_ROLE" = "head" ]; then
  NODE_ROLE=head bash examples/deepseek-v31-3node-h100/setup_ray_cluster.sh
elif [ "$NODE_ROLE" = "worker" ]; then
  HEAD_IP="${HEAD_IP:?HEAD_IP must be set when NODE_ROLE=worker}"
  HEAD_IP="$HEAD_IP" NODE_ROLE=worker bash examples/deepseek-v31-3node-h100/setup_ray_cluster.sh
  echo "Worker joined cluster. Nothing else to do in smoke mode."
  exit 0
else
  echo "NODE_ROLE must be head or worker"
  exit 1
fi

if [ "$CHECK_CONFIG" = "true" ]; then
  python -m torchspec.train_entry \
    --print-config-only \
    --config configs/sglang_deepseek_v31_3node.yaml \
    dataset.train_data_path="$TRAIN_DATA_PATH" \
    training.num_train_steps=1 \
    output_dir="$OUTPUT_DIR/config_only" \
    cache_dir="$CACHE_DIR/config_only" \
    inference.remote_sglang.endpoint="$REMOTE_SGLANG_ENDPOINT" \
    mooncake.master_server_address="$MOONCAKE_MASTER_ADDRESS" \
    mooncake.metadata_server="$MOONCAKE_METADATA_SERVER" \
    mooncake.metadata_port="$MOONCAKE_METADATA_PORT"
else
  echo "Skipping config-only check because CHECK_CONFIG=$CHECK_CONFIG"
fi

REMOTE_SGLANG_ENDPOINT="$REMOTE_SGLANG_ENDPOINT" \
MOONCAKE_MASTER_ADDRESS="$MOONCAKE_MASTER_ADDRESS" \
MOONCAKE_METADATA_SERVER="$MOONCAKE_METADATA_SERVER" \
MOONCAKE_METADATA_PORT="$MOONCAKE_METADATA_PORT" \
bash examples/deepseek-v31-3node-h100/run.sh \
  dataset.train_data_path="$TRAIN_DATA_PATH" \
  training.num_train_steps=1 \
  output_dir="$OUTPUT_DIR/train" \
  cache_dir="$CACHE_DIR/train"
