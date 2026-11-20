#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
IFS=',' read -r -a gpu_ids <<< "${GPUS:-0}"
pids=()
trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done' INT TERM EXIT
for shard in "${!gpu_ids[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu_ids[$shard]}" "$python_bin" -B "$root/run.py" "$@" --shard "$shard" --shards "${#gpu_ids[@]}" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
trap - INT TERM EXIT
exit "$status"
