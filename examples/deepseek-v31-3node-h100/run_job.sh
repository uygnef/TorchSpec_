#!/bin/bash
set -xeo pipefail

echo "DISTRIBUTED_NODE_COUNT: $DISTRIBUTED_NODE_COUNT"
echo "DISTRIBUTED_NODE_RANK: $DISTRIBUTED_NODE_RANK"
echo "DISTRIBUTED_MASTER_HOSTS: $DISTRIBUTED_MASTER_HOSTS"
echo "PET_MASTER_PORT: $PET_MASTER_PORT"
echo "NP: $NP"
set -euo pipefail
set -x

export PATH="/nfs/ofs-fengyu/env/conda/envs/torchspec/bin:/nfs/ofs-fengyu/env/conda/condabin:/nfs/ofs-fengyu/env/conda/bin/:$PATH"
export MAMBA_EXE="/nfs/ofs-fengyu/env/conda/bin/micromamba"
export MAMBA_ROOT_PREFIX="/nfs/ofs-fengyu/env/conda"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate torchspec
export HF_HOME="${HF_HOME:-/nfs/ofs-llab-volume/users/fengyu/hf_cache}"
export SGLANG_DG_CACHE_DIR="${SGLANG_DG_CACHE_DIR:-/nfs/ofs-fengyu/cache/deep_gemm}"
mkdir -p "$SGLANG_DG_CACHE_DIR"
echo "SGLANG_DG_CACHE_DIR: $SGLANG_DG_CACHE_DIR"

WORKING_DIR="${WORKING_DIR:-/nfs/ofs-llab-volume/users/fengyu/TorchSpec}"
cd "$WORKING_DIR"
RAY_TMP_ROOT="${RAY_TMP_ROOT:-/tmp}"
DATE_TAG="$(date +%m%d_%H%M)"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-$RAY_TMP_ROOT/ray_${DATE_TAG}_$(id -u)}"
export RAY_TMPDIR="$RAY_TEMP_DIR"
export RAY_TEMP_DIR="$RAY_TMPDIR"
mkdir -p "$RAY_TEMP_DIR"

LOCAL_IP="$(ip -4 addr show scope global | awk '/inet /{print $2}' | cut -d/ -f1 | head -n1)"
LOG_SUFFIX="$(printf "%s" "$LOCAL_IP" | tr '.' '-')"
RAY_LOG_ROOT="$WORKING_DIR/tmp/ray_${DATE_TAG}_${LOG_SUFFIX}"
RAY_LOG_DEST="$RAY_LOG_ROOT/node_${DISTRIBUTED_NODE_RANK}"
mkdir -p "$RAY_LOG_DEST"
echo "Ray logs will be synced to: $RAY_LOG_DEST/"

sync_ray_logs() {
    if [ -d "$RAY_TEMP_DIR" ]; then
        cp -a "$RAY_TEMP_DIR" "$RAY_LOG_DEST/" 2>/dev/null || true
    fi
}

sync_ray_logs_loop() {
    while true; do
        sync_ray_logs
        sleep 30
    done
}

sync_ray_logs_loop >/dev/null 2>&1 &
RAY_LOG_SYNC_PID=$!
trap "kill $RAY_LOG_SYNC_PID 2>/dev/null || true; sync_ray_logs" EXIT

export RAY_PORT=$PET_MASTER_PORT
export RAY_ADDRESS="$DISTRIBUTED_MASTER_HOSTS:$PET_MASTER_PORT"
export TORCHSPEC_MASTER_PORT="${TORCHSPEC_MASTER_PORT:-31419}"
echo TORCHSPEC_MASTER_PORT:$TORCHSPEC_MASTER_PORT
export RAY_DISABLE_DASHBOARD=1
export RAY_DISABLE_METRICS=1
ray stop --force

if [ "${DISTRIBUTED_NODE_RANK}" -eq 0 ]; then
    ray stop --force
    NODE_ROLE=head bash examples/deepseek-v31-3node-h100/setup_ray_cluster.sh
    TARGET_NODES=$DISTRIBUTED_NODE_COUNT
    while true; do
        CURRENT_NODES=$(ray list nodes | grep "ALIVE" | wc -l)
        echo "当前节点数: $CURRENT_NODES, 目标节点数: $TARGET_NODES"
        if [ "$CURRENT_NODES" -ge "$TARGET_NODES" ]; then
            echo "已达到目标节点数，开始提交任务..."
            break
        fi
        echo "等待更多节点加入..."
        sleep 10
    done
    echo "执行任务提交..."
    bash examples/deepseek-v31-3node-h100/run.sh
else
    RAY_ADDRESS="${DISTRIBUTED_MASTER_HOSTS}:${PET_MASTER_PORT}"
    HEAD_IP=$DISTRIBUTED_MASTER_HOSTS NODE_ROLE=worker bash examples/deepseek-v31-3node-h100/setup_ray_cluster.sh
    echo "ray start worker join master ${RAY_ADDRESS}"
    sleep infinity
fi
