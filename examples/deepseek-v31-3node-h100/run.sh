#!/bin/bash
# Launch torchspec.train_entry for DeepSeek-V3.1 remote-SGLang training
#
# Run this on the head node AFTER the Ray cluster is fully ready.
#
# Architecture:
#   - TorchSpec runs training on the allocated training node(s)
#   - target-model feature extraction is served by a remote SGLang endpoint
#   - Mooncake is used as the feature transport/cache backend
#
# Usage:
#   bash examples/deepseek-v31-3node-h100/run.sh [extra_overrides...]
#
# Environment variables:
#   TRAIN_GPUS                - GPUs for training on each training node (default: 8)
#   TRAIN_NODES               - Number of training nodes (default: 1)
#   CONFIG_FILE               - Override config file path
#   MODEL_PATH                - Override target model path
#   CHAT_TEMPLATE             - Override dataset.chat_template (default: deepseek-v3)
#   REMOTE_SGLANG_ENDPOINT    - Remote SGLang endpoint (default: http://127.0.0.1:30000)
#   FEATURE_CACHE_ENABLED     - Enable feature cache (default: true)
#   FEATURE_CACHE_INDEX       - Override feature cache index path
#   MOONCAKE_MASTER_ADDRESS   - Mooncake master address (default: 127.0.0.1:50051)
#   MOONCAKE_METADATA_SERVER  - Mooncake metadata server host (default: 127.0.0.1)
#   MOONCAKE_METADATA_PORT    - Mooncake metadata server port (default: 50052)

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCHSPEC_LOG_LEVEL=INFO

resolve_host_with_python() {
  python3 - "$1" <<'PY'
import socket
import sys

host = sys.argv[1]
print(socket.gethostbyname(host))
PY
}

TRAIN_GPUS="${TRAIN_GPUS:-8}"
TRAIN_NODES="${TRAIN_NODES:-1}"
MODEL_PATH="${MODEL_PATH:-/nfs/ofs-llm-ssd/models/opensource/DeepSeek-V3.1}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-deepseek-v3}"
REMOTE_SGLANG_ENDPOINT="${REMOTE_SGLANG_ENDPOINT:-http://127.0.0.1:30000}"
FEATURE_CACHE_ENABLED="${FEATURE_CACHE_ENABLED:-true}"
FEATURE_CACHE_INDEX="${FEATURE_CACHE_INDEX:-$ROOT_DIR/cache/remote_deepseek_v31_feature_cache.sqlite3}"
LOCAL_IP="$(python3 - <<'PY'
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
MASTER_HOST_CANDIDATE="${MOONCAKE_METADATA_SERVER:-${DISTRIBUTED_MASTER_HOSTS:-${HEAD_IP:-$LOCAL_IP}}}"
if [ "$RESOLVE_MASTER_IP" = "true" ]; then
  MASTER_HOST_CANDIDATE="$(resolve_host_with_python "$MASTER_HOST_CANDIDATE")"
fi
MOONCAKE_MASTER_ADDRESS="${MOONCAKE_MASTER_ADDRESS:-$MASTER_HOST_CANDIDATE:50051}"
MOONCAKE_METADATA_SERVER="${MOONCAKE_METADATA_SERVER:-$MASTER_HOST_CANDIDATE}"
MOONCAKE_METADATA_PORT="${MOONCAKE_METADATA_PORT:-50052}"

CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/configs/sglang_deepseek_v31_3node.yaml}"

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/deepseek_v31_remote_train_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

echo "=============================================="
echo "DeepSeek-V3.1 Remote-SGLang Training"
echo "=============================================="
echo "Config:                $CONFIG_FILE"
echo "  Model path:      $MODEL_PATH"
echo "  Chat template:   $CHAT_TEMPLATE"
echo "  Training nodes:  $TRAIN_NODES"
echo "  Training GPUs:   $TRAIN_GPUS per node"
echo "  Remote endpoint: $REMOTE_SGLANG_ENDPOINT"
echo "  Feature cache:   $FEATURE_CACHE_ENABLED ($FEATURE_CACHE_INDEX)"
echo "  Mooncake master: $MOONCAKE_MASTER_ADDRESS"
echo "  Metadata server: $MOONCAKE_METADATA_SERVER:$MOONCAKE_METADATA_PORT"
echo "  Metadata source: ${DISTRIBUTED_MASTER_HOSTS:-${HEAD_IP:-local_ip}}"
echo "  draft config:    auto-generated from target model unless overridden"
echo "=============================================="

if ! ray status &>/dev/null; then
  echo "ERROR: Cannot connect to Ray cluster (is Ray running on this node?)"
  echo "Start the cluster first:"
  echo "  NODE_ROLE=head   bash examples/deepseek-v31-3node-h100/setup_ray_cluster.sh"
  echo "  HEAD_IP=<node0_ip> NODE_ROLE=worker bash examples/deepseek-v31-3node-h100/setup_ray_cluster.sh"
  exit 1
fi

echo "=== Launching training ==="
python3 -m torchspec.train_entry \
  --config "$CONFIG_FILE" \
  model.target_model_path="$MODEL_PATH" \
  dataset.chat_template="$CHAT_TEMPLATE" \
  training.training_num_nodes="$TRAIN_NODES" \
  training.training_num_gpus_per_node="$TRAIN_GPUS" \
  inference.mode="remote_sglang" \
  inference.remote_sglang.endpoint="$REMOTE_SGLANG_ENDPOINT" \
  feature_cache.enabled="$FEATURE_CACHE_ENABLED" \
  feature_cache.index_path="$FEATURE_CACHE_INDEX" \
  mooncake.master_server_address="$MOONCAKE_MASTER_ADDRESS" \
  mooncake.metadata_server="$MOONCAKE_METADATA_SERVER" \
  mooncake.metadata_port="$MOONCAKE_METADATA_PORT" \
  dataset.last_turn_loss_only=true \
  "$@"

echo "=============================================="
echo "Training completed!"
echo "=============================================="
