#!/bin/bash
# Run a minimal remote-SGLang smoke for Qwen3-8B on the training machine.

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

REMOTE_SGLANG_ENDPOINT="${REMOTE_SGLANG_ENDPOINT:?REMOTE_SGLANG_ENDPOINT must be set}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$WORKING_DIR/examples/data/sample_conversations.jsonl}"
MODEL_PATH="${MODEL_PATH:-/nfs/ofs-llm-ssd/models/opensource/Qwen3-8B}"
TRAIN_GPUS="${TRAIN_GPUS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs/ofs-llab-volume/users/fengyu/o/qwen_remote_smoke}"
CACHE_DIR="${CACHE_DIR:-/nfs/ofs-llab-volume/users/fengyu/c/qwen_remote_smoke}"
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
MASTER_HOST_CANDIDATE="${MOONCAKE_METADATA_SERVER:-${DISTRIBUTED_MASTER_HOSTS:-${HEAD_IP:-$LOCAL_IP}}}"
if [ "$RESOLVE_MASTER_IP" = "true" ]; then
  MASTER_HOST_CANDIDATE="$(resolve_host_with_python "$MASTER_HOST_CANDIDATE")"
fi
MOONCAKE_MASTER_ADDRESS="${MOONCAKE_MASTER_ADDRESS:-$MASTER_HOST_CANDIDATE:50051}"
MOONCAKE_METADATA_SERVER="${MOONCAKE_METADATA_SERVER:-$MASTER_HOST_CANDIDATE}"
MOONCAKE_METADATA_PORT="${MOONCAKE_METADATA_PORT:-50052}"

LOG_DIR="$WORKING_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/qwen3_remote_smoke_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"
echo "Mooncake metadata host source: ${DISTRIBUTED_MASTER_HOSTS:-${HEAD_IP:-local_ip}}"
echo "Mooncake metadata target: $MOONCAKE_METADATA_SERVER:$MOONCAKE_METADATA_PORT"
echo "Mooncake master target: $MOONCAKE_MASTER_ADDRESS"

if [ "$CHECK_CONFIG" = "true" ]; then
  python -m torchspec.train_entry \
    --print-config-only \
    --config configs/remote_sglang_qwen3_8b.yaml \
    model.target_model_path="$MODEL_PATH" \
    dataset.train_data_path="$TRAIN_DATA_PATH" \
    training.training_num_gpus_per_node="$TRAIN_GPUS" \
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

python -m torchspec.train_entry \
  --config configs/remote_sglang_qwen3_8b.yaml \
  model.target_model_path="$MODEL_PATH" \
  dataset.train_data_path="$TRAIN_DATA_PATH" \
  training.training_num_gpus_per_node="$TRAIN_GPUS" \
  training.num_train_steps=1 \
  output_dir="$OUTPUT_DIR/train" \
  cache_dir="$CACHE_DIR/train" \
  inference.remote_sglang.endpoint="$REMOTE_SGLANG_ENDPOINT" \
  mooncake.master_server_address="$MOONCAKE_MASTER_ADDRESS" \
  mooncake.metadata_server="$MOONCAKE_METADATA_SERVER" \
  mooncake.metadata_port="$MOONCAKE_METADATA_PORT"
