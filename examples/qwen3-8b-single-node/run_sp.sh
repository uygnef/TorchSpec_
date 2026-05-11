#!/bin/bash
# Train Qwen3-8B with SGLang async inference and USP sequence parallel training.
#
# GPU allocation (default: 4 GPUs total):
#   - 2 GPUs for inference (duplicate mode: each engine has full model copy)
#   - 2 GPUs for training with USP sequence parallel
#
# Usage:
#   ./examples/qwen3-8b-single-node/run_sp.sh [CONFIG_FILE] [EXTRA_ARGS...]
#
# Examples:
#   ./examples/qwen3-8b-single-node/run_sp.sh
#   SP_SIZE=2 TRAIN_GPUS=2 ./examples/qwen3-8b-single-node/run_sp.sh
#   ./examples/qwen3-8b-single-node/run_sp.sh configs/sglang_qwen3_8b.yaml training.num_train_steps=10

set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}"
export TORCHSPEC_LOG_LEVEL="${TORCHSPEC_LOG_LEVEL:-INFO}"

CONFIG_FILE="${1:-$ROOT_DIR/configs/sglang_qwen3_8b.yaml}"
if [[ -f "$CONFIG_FILE" ]]; then
    shift 1 || true
elif [[ -f "$ROOT_DIR/$CONFIG_FILE" ]]; then
    CONFIG_FILE="$ROOT_DIR/$CONFIG_FILE"
    shift 1 || true
else
    CONFIG_FILE="$ROOT_DIR/configs/sglang_qwen3_8b.yaml"
fi

IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPU_ARRAY[@]}

TRAIN_GPUS="${TRAIN_GPUS:-2}"
INFERENCE_GPUS="${INFERENCE_GPUS:-2}"
INFERENCE_GPUS_PER_ENGINE="${INFERENCE_GPUS_PER_ENGINE:-2}"
SP_SIZE="${SP_SIZE:-2}"

LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")

echo "=============================================="
echo "Train with async inference + USP sequence parallel"
echo "=============================================="
echo "Config: $CONFIG_FILE (nested format)"
echo "Total GPUs: $TOTAL_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  - Training GPUs: $TRAIN_GPUS"
echo "  - Inference GPUs: $INFERENCE_GPUS"
echo "  - Inference GPUs per engine: $INFERENCE_GPUS_PER_ENGINE"
echo "  - USP SP size: $SP_SIZE"
echo "Local IP: $LOCAL_IP"
echo "Extra args: $*"
echo "=============================================="

python3 -m torchspec.train_entry \
    --config "$CONFIG_FILE" \
    training.training_num_gpus_per_node="$TRAIN_GPUS" \
    inference.inference_num_gpus="$INFERENCE_GPUS" \
    inference.inference_num_gpus_per_engine="$INFERENCE_GPUS_PER_ENGINE" \
    inference.inference_num_gpus_per_node="$TOTAL_GPUS" \
    inference.sglang.tp_size="$INFERENCE_GPUS_PER_ENGINE" \
    training.attention_backend=usp \
    training.sp_size="$SP_SIZE" \
    training.micro_batch_size=1 \
    "$@"

echo "=============================================="
echo "Training completed!"
echo "=============================================="
