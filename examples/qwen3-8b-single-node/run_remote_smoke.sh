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

REMOTE_SGLANG_ENDPOINT="${REMOTE_SGLANG_ENDPOINT:?REMOTE_SGLANG_ENDPOINT must be set}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$WORKING_DIR/examples/data/sample_conversations.jsonl}"
MODEL_PATH="${MODEL_PATH:-/nfs/ofs-llm-ssd/models/opensource/Qwen3-8B}"
TRAIN_GPUS="${TRAIN_GPUS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs/ofs-llab-volume/users/fengyu/o/qwen_remote_smoke}"
CACHE_DIR="${CACHE_DIR:-/nfs/ofs-llab-volume/users/fengyu/c/qwen_remote_smoke}"
MOONCAKE_MASTER_ADDRESS="${MOONCAKE_MASTER_ADDRESS:-127.0.0.1:50051}"
MOONCAKE_METADATA_SERVER="${MOONCAKE_METADATA_SERVER:-127.0.0.1}"
MOONCAKE_METADATA_PORT="${MOONCAKE_METADATA_PORT:-50052}"

LOG_DIR="$WORKING_DIR/running_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/qwen3_remote_smoke_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

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
