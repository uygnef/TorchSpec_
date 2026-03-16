#!/bin/bash
# Run a minimal remote-SGLang smoke for Qwen3-8B on the training machine.

set -euo pipefail
set -x

export PATH="/nfs/ofs-fengyu/env/conda/envs/torchspec/bin:/nfs/ofs-fengyu/env/conda/condabin:/nfs/ofs-fengyu/env/conda/bin/:$PATH"
export MAMBA_EXE="/nfs/ofs-fengyu/env/conda/bin/micromamba"
export MAMBA_ROOT_PREFIX="/nfs/ofs-fengyu/env/conda"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate torchspec

export TORCHSPEC_REPO_ROOT="${TORCHSPEC_REPO_ROOT:-/nfs/ofs-llab-volume/users/fengyu/TorchSpec}"
export SGLANG_PYTHON_DIR="${SGLANG_PYTHON_DIR:-/nfs/ofs-llab-volume/users/fengyu/torchspec/_sglang/python}"
export PYTHONPATH="$TORCHSPEC_REPO_ROOT:$SGLANG_PYTHON_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TORCHSPEC_FEATURE_CACHE_BUSY_TIMEOUT_MS="${TORCHSPEC_FEATURE_CACHE_BUSY_TIMEOUT_MS:-60000}"
export TORCHSPEC_FEATURE_CACHE_WRITE_RETRY_TIMEOUT_S="${TORCHSPEC_FEATURE_CACHE_WRITE_RETRY_TIMEOUT_S:-120}"
export TORCHSPEC_FEATURE_CACHE_TOUCH_INTERVAL_S="${TORCHSPEC_FEATURE_CACHE_TOUCH_INTERVAL_S:-300}"
export TORCHSPEC_FEATURE_CACHE_CHUNK_SIZE="${TORCHSPEC_FEATURE_CACHE_CHUNK_SIZE:-64}"

WORKING_DIR="${WORKING_DIR:-/nfs/ofs-llab-volume/users/fengyu/TorchSpec}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-configs/remote_sglang_qwen3_8b.yaml}"
cd "$WORKING_DIR"

export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-300}"
export RAY_agent_register_timeout_ms="${RAY_agent_register_timeout_ms:-300000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
REMOTE_READY_TIMEOUT_SECONDS="${REMOTE_READY_TIMEOUT_SECONDS:-600}"

resolve_host_with_python() {
  python - "$1" <<'PY'
import socket
import sys

host = sys.argv[1]
print(socket.gethostbyname(host))
PY
}

wait_for_remote_sglang() {
  python - "$1" "$2" <<'PY'
import json
import sys
import time
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
endpoint = sys.argv[1].rstrip("/")
timeout_seconds = int(sys.argv[2])
deadline = time.time() + timeout_seconds
url = endpoint + "/model_info"
last_error = None

while time.time() < deadline:
    try:
        with opener.open(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        print(f"Remote SGLang ready: {url} -> {payload.get('model_path', 'unknown')}")
        sys.exit(0)
    except Exception as exc:
        last_error = exc
        print(f"Waiting for Remote SGLang at {url}: {exc}", flush=True)
        time.sleep(5)

raise SystemExit(
    f"Timed out waiting for Remote SGLang readiness at {url}: {last_error}"
)
PY
}

REMOTE_SGLANG_ENDPOINT="${REMOTE_SGLANG_ENDPOINT:?REMOTE_SGLANG_ENDPOINT must be set}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$WORKING_DIR/examples/data/sample_conversations.jsonl}"
MODEL_PATH="${MODEL_PATH:-/nfs/ofs-llm-ssd/models/opensource/Qwen3-8B}"
TRAIN_GPUS="${TRAIN_GPUS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs/ofs-llab-volume/users/fengyu/o/qwen_remote_smoke}"
CACHE_DIR="${CACHE_DIR:-/nfs/ofs-llab-volume/users/fengyu/c/qwen_remote_smoke}"
FEATURE_CACHE_INDEX="${FEATURE_CACHE_INDEX:-$CACHE_DIR/train/remote_sglang_feature_cache.sqlite3}"
MOONCAKE_NATIVE_ROOT_FS_DIR="${MOONCAKE_NATIVE_ROOT_FS_DIR:-$CACHE_DIR/train/mooncake_native_rootfs}"
MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-}
MOONCAKE_LOCAL_BUFFER_SIZE=${MOONCAKE_LOCAL_BUFFER_SIZE:-}
MOONCAKE_HOST_BUFFER_SIZE=${MOONCAKE_HOST_BUFFER_SIZE:-}
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
LOG_FILE="$LOG_DIR/qwen3_remote_smoke_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

echo "Clearing smoke cache directories and feature cache manifest"
rm -rf "$CACHE_DIR/train" "$CACHE_DIR/config_only" "$MOONCAKE_NATIVE_ROOT_FS_DIR"
rm -f "$FEATURE_CACHE_INDEX"
rm -f "$WORKING_DIR/cache/remote_sglang_feature_cache.sqlite3"
mkdir -p "$(dirname "$FEATURE_CACHE_INDEX")"

TRAIN_ENTRY_ARGS=(
  --config "$TRAIN_CONFIG_PATH"
  "model.target_model_path=$MODEL_PATH"
  "dataset.train_data_path=$TRAIN_DATA_PATH"
  "training.training_num_gpus_per_node=$TRAIN_GPUS"
  "training.num_train_steps=100"
  "inference.remote_sglang.endpoint=$REMOTE_SGLANG_ENDPOINT"
  "feature_cache.index_path=$FEATURE_CACHE_INDEX"
  "mooncake.native_disk_eviction_enabled=true"
  "mooncake.native_root_fs_dir=$MOONCAKE_NATIVE_ROOT_FS_DIR"
)
if [ -n "$MOONCAKE_GLOBAL_SEGMENT_SIZE" ]; then
  TRAIN_ENTRY_ARGS+=("mooncake.global_segment_size=$MOONCAKE_GLOBAL_SEGMENT_SIZE")
fi
if [ -n "$MOONCAKE_LOCAL_BUFFER_SIZE" ]; then
  TRAIN_ENTRY_ARGS+=("mooncake.local_buffer_size=$MOONCAKE_LOCAL_BUFFER_SIZE")
fi
if [ -n "$MOONCAKE_HOST_BUFFER_SIZE" ]; then
  TRAIN_ENTRY_ARGS+=("mooncake.host_buffer_size=$MOONCAKE_HOST_BUFFER_SIZE")
fi
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
echo "Training CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Remote SGLang ready timeout: ${REMOTE_READY_TIMEOUT_SECONDS}s"

if [ "$CHECK_CONFIG" = "true" ]; then
  python -m torchspec.train_entry \
    --print-config-only \
    output_dir="$OUTPUT_DIR/config_only" \
    cache_dir="$CACHE_DIR/config_only" \
    "${TRAIN_ENTRY_ARGS[@]}"
else
  echo "Skipping config-only check because CHECK_CONFIG=$CHECK_CONFIG"
fi

wait_for_remote_sglang "$REMOTE_SGLANG_ENDPOINT" "$REMOTE_READY_TIMEOUT_SECONDS"

python -m torchspec.train_entry \
  output_dir="$OUTPUT_DIR/train" \
  cache_dir="$CACHE_DIR/train" \
  "${TRAIN_ENTRY_ARGS[@]}"
