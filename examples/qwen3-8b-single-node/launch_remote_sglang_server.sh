#!/bin/bash
# Launch a remote SGLang service for Qwen3-8B feature extraction.
#
# Run this on the inference/service machine.

set -euo pipefail
set -x

export PATH="/nfs/ofs-fengyu/env/conda/envs/torchspec/bin:/nfs/ofs-fengyu/env/conda/condabin:/nfs/ofs-fengyu/env/conda/bin/:$PATH"
export MAMBA_EXE="/nfs/ofs-fengyu/env/conda/bin/micromamba"
export MAMBA_ROOT_PREFIX="/nfs/ofs-fengyu/env/conda"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate torchspec

export SGLANG_DISABLE_CUDNN_CHECK=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/qwen3_remote_sglang_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

MODEL_PATH="${MODEL_PATH:-/nfs/ofs-llm-ssd/models/opensource/Qwen3-8B}"
TP_SIZE="${TP_SIZE:-8}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
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
ADVERTISE_HOST="${ADVERTISE_HOST:-$LOCAL_IP}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
SGLANG_PYTHON_DIR="${SGLANG_PYTHON_DIR:-/nfs/ofs-llab-volume/users/fengyu/torchspec/_sglang/python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=============================================="
echo "Qwen3-8B Remote SGLang Server"
echo "=============================================="
echo "Model path:          $MODEL_PATH"
echo "Bind address:        $HOST:$PORT"
echo "Reachable endpoint:  http://$ADVERTISE_HOST:$PORT"
echo "TP size:             $TP_SIZE"
echo "Mem fraction:        $MEM_FRACTION_STATIC"
echo "SGLang python dir:   $SGLANG_PYTHON_DIR"
echo "Important flags:     radix cache ON, overlap schedule OFF"
echo "=============================================="

cd "$SGLANG_PYTHON_DIR"

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tp-size "$TP_SIZE" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --trust-remote-code \
  --disable-overlap-schedule \
  ${EXTRA_ARGS}
