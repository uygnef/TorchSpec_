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

resolve_host_with_python() {
  python - "$1" <<'PY'
import socket
import sys

host = sys.argv[1]
print(socket.gethostbyname(host))
PY
}

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
RESOLVE_MASTER_IP="${RESOLVE_MASTER_IP:-false}"
CHECK_CONFIG="${CHECK_CONFIG:-true}"
if [ -n "${MOONCAKE_METADATA_SERVER:-}" ]; then
  MASTER_HOST_CANDIDATE="$MOONCAKE_METADATA_SERVER"
elif [ -n "${DISTRIBUTED_MASTER_HOSTS:-}" ]; then
  MASTER_HOST_CANDIDATE="$DISTRIBUTED_MASTER_HOSTS"
elif [ -n "${HEAD_IP:-}" ]; then
  MASTER_HOST_CANDIDATE="$HEAD_IP"
else
  MASTER_HOST_CANDIDATE=""
fi
if [ -n "$MASTER_HOST_CANDIDATE" ] && [ "$RESOLVE_MASTER_IP" = "true" ]; then
  MASTER_HOST_CANDIDATE="$(resolve_host_with_python "$MASTER_HOST_CANDIDATE")"
fi

LOG_DIR="$WORKING_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/deepseek_v31_remote_smoke_${NODE_ROLE}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

TRAIN_ENTRY_ARGS=(
  --config configs/sglang_deepseek_v31_3node.yaml
  "dataset.train_data_path=$TRAIN_DATA_PATH"
  "training.num_train_steps=1"
  "inference.remote_sglang.endpoint=$REMOTE_SGLANG_ENDPOINT"
)
if [ -n "${MOONCAKE_MASTER_ADDRESS:-}" ]; then
  TRAIN_ENTRY_ARGS+=("mooncake.master_server_address=$MOONCAKE_MASTER_ADDRESS")
fi
if [ -n "${MOONCAKE_METADATA_SERVER:-}" ]; then
  TRAIN_ENTRY_ARGS+=("mooncake.metadata_server=$MOONCAKE_METADATA_SERVER")
elif [ -n "$MASTER_HOST_CANDIDATE" ]; then
  TRAIN_ENTRY_ARGS+=("mooncake.metadata_server=$MASTER_HOST_CANDIDATE")
fi
if [ -n "${MOONCAKE_METADATA_PORT:-}" ]; then
  TRAIN_ENTRY_ARGS+=("mooncake.metadata_port=$MOONCAKE_METADATA_PORT")
fi

echo "Mooncake metadata host source: ${DISTRIBUTED_MASTER_HOSTS:-${HEAD_IP:-auto}}"
echo "Mooncake metadata target: ${MOONCAKE_METADATA_SERVER:-${MASTER_HOST_CANDIDATE:-auto}}:${MOONCAKE_METADATA_PORT:-auto}"
echo "Mooncake master target: ${MOONCAKE_MASTER_ADDRESS:-auto}"

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
    output_dir="$OUTPUT_DIR/config_only" \
    cache_dir="$CACHE_DIR/config_only" \
    "${TRAIN_ENTRY_ARGS[@]}"
else
  echo "Skipping config-only check because CHECK_CONFIG=$CHECK_CONFIG"
fi

RUN_ENV=()
RUN_ENV+=("REMOTE_SGLANG_ENDPOINT=$REMOTE_SGLANG_ENDPOINT")
if [ -n "${MOONCAKE_MASTER_ADDRESS:-}" ]; then
  RUN_ENV+=("MOONCAKE_MASTER_ADDRESS=$MOONCAKE_MASTER_ADDRESS")
fi
if [ -n "${MOONCAKE_METADATA_SERVER:-}" ]; then
  RUN_ENV+=("MOONCAKE_METADATA_SERVER=$MOONCAKE_METADATA_SERVER")
elif [ -n "$MASTER_HOST_CANDIDATE" ]; then
  RUN_ENV+=("MOONCAKE_METADATA_SERVER=$MASTER_HOST_CANDIDATE")
fi
if [ -n "${MOONCAKE_METADATA_PORT:-}" ]; then
  RUN_ENV+=("MOONCAKE_METADATA_PORT=$MOONCAKE_METADATA_PORT")
fi

env "${RUN_ENV[@]}" bash examples/deepseek-v31-3node-h100/run.sh \
  dataset.train_data_path="$TRAIN_DATA_PATH" \
  training.num_train_steps=1 \
  output_dir="$OUTPUT_DIR/train" \
  cache_dir="$CACHE_DIR/train"
