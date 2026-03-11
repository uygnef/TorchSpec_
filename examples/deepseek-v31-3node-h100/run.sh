#!/bin/bash
# Launch torchspec.train_entry for DeepSeek-V3.1 Eagle3 3-node training
#
# Run this on the head node AFTER the Ray cluster is fully ready.
# See examples/deepseek-v31-3node-h100/setup_ray_cluster.sh to set up the cluster first.
#
# Node layout:
#   Head node (this node): 8 GPUs (GPU 0-7) - FSDP training
#   Worker nodes x2:       8 GPUs each      - SglEngine inference (TP=16)
#
# Usage:
#   bash examples/deepseek-v31-3node-h100/run.sh [extra_overrides...]
#
# Environment variables:
#   TRAIN_GPUS          - GPUs for training (default: 8)
#   INFERENCE_GPUS      - Total inference GPUs across all worker nodes (default: 16)
#   INFERENCE_NODES     - Number of inference worker nodes (default: 2)
#   CONFIG_FILE         - Override config file path
#   MODEL_PATH          - Override target model path
#   CHAT_TEMPLATE       - Override dataset.chat_template (default: deepseek-v3)

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCHSPEC_LOG_LEVEL=INFO

TRAIN_GPUS="${TRAIN_GPUS:-8}"
INFERENCE_GPUS="${INFERENCE_GPUS:-16}"
INFERENCE_NODES="${INFERENCE_NODES:-2}"
MODEL_PATH="${MODEL_PATH:-/nfs/ofs-llm-ssd/models/opensource/DeepSeek-V3.1}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-deepseek-v3}"

CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/configs/sglang_deepseek_v31_3node.yaml}"

LOG_DIR="$ROOT_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/deepseek_v31_3node_train_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

echo "=============================================="
echo "DeepSeek-V3.1 BF16 3-Node Training"
echo "=============================================="
echo "Config:                $CONFIG_FILE"
echo "  Model path:      $MODEL_PATH"
echo "  Chat template:   $CHAT_TEMPLATE"
echo "  Training GPUs:   $TRAIN_GPUS (this node x $TRAIN_GPUS GPUs)"
echo "  Inference GPUs:  $INFERENCE_GPUS ($INFERENCE_NODES nodes x 8 GPUs, TP=$INFERENCE_GPUS)"
echo "  dist_init_addr:  (auto-negotiated via Ray)"
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
  training.training_num_gpus_per_node="$TRAIN_GPUS" \
  inference.inference_engine_type="sgl" \
  inference.inference_num_gpus="$INFERENCE_GPUS" \
  inference.inference_num_gpus_per_engine="$INFERENCE_GPUS" \
  inference.inference_num_gpus_per_node=8 \
  inference.sglang.tp_size="$INFERENCE_GPUS" \
  inference.sglang.nnodes="$INFERENCE_NODES" \
  dataset.last_turn_loss_only=true \
  "$@"

echo "=============================================="
echo "Training completed!"
echo "=============================================="
