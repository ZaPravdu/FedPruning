#!/usr/bin/env bash
# Repeat an experiment N times with incrementing seed.
#
# Usage:
#   ./run_feddst_repeat.sh [N] MODEL DATASET CLIENT_NUM WORKER_NUM ROUND EPOCH DENSITY LR [extra args...]
#
#   N          repeat count (default: 3)
#   MODEL~LR   same positional args as run_feddst_distributed_pytorch.sh
#   extra args appended verbatim (e.g. --top_p_aggregate --p 0.98)
#
# Each run gets --seed 0, 1, 2, … N-1.

set -e

REPEAT=3
if [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -ge 1 ] 2>/dev/null; then
    REPEAT=$1
    shift
fi

MODEL=$1
DATASET=$2
CLIENT_NUM=$3
WORKER_NUM=$4
ROUND=$5
EPOCH=$6
DENSITY=$7
LR=$8
shift 8

PROCESS_NUM=$((WORKER_NUM + 1))
echo "Processes: $PROCESS_NUM"

hostname > mpi_host_file

for i in $(seq 1 "$REPEAT"); do
    SEED=$((i - 1))
    echo ""
    echo "============================================"
    echo "  Run $i / $REPEAT    (seed=$SEED)"
    echo "============================================"
    mpirun -np "$PROCESS_NUM" -hostfile ./mpi_host_file python3 ./main_feddst.py \
        --gpu_mapping_file "gpu_mapping.yaml" \
        --gpu_mapping_key "mapping_default" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --client_num_in_total "$CLIENT_NUM" \
        --client_num_per_round "$WORKER_NUM" \
        --comm_round "$ROUND" \
        --epochs "$EPOCH" \
        --lr "$LR" \
        --target_density "$DENSITY" \
        "$@" \
        --seed "$SEED"
done

echo ""
echo "All $REPEAT runs finished."
