#!/usr/bin/env bash
set -euo pipefail

ROOT="${UPGRADE_ROOT:-/root/autodl-tmp/uav_upgrade_20260825}"
SRC="$ROOT/source"
RESULTS="$ROOT/results"
PY="$ROOT/.conda-env/bin/python"
SHARDS="$RESULTS/03_simulation/raw_episodes/shards"
LOGS="$RESULTS/06_logs/simulation_shards"
mkdir -p "$SHARDS" "$LOGS"
rm -f "$SHARDS"/*.jsonl

run_shard() {
  policy="$1"
  seed="$2"
  tag="${policy}_seed${seed}"
  export PYTHONPATH="$SRC"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
  "$PY" "$SRC/run_simulation.py" \
    --models-root "$RESULTS/02_training" \
    --output "$SHARDS/$tag.jsonl" \
    --episodes 100 --seed-start 31000 \
    --training-seeds "$seed" --policies "$policy" --trajectory-seeds 3 \
    > "$LOGS/$tag.log" 2>&1
  printf 'SHARD_RESULT=PASS policy=%s seed=%s rows=%s\n' "$policy" "$seed" "$(wc -l < "$SHARDS/$tag.jsonl")"
}
export -f run_shard
export ROOT SRC RESULTS PY SHARDS LOGS

jobs="$ROOT/simulation_jobs.txt"
: > "$jobs"
for policy in gcbf_local gcbf_privileged mappo_local mappo_privileged; do
  for seed in 1101 1102 1103 1104 1105; do
    printf '%s %s\n' "$policy" "$seed" >> "$jobs"
  done
done
printf 'distributed_cbf -1\n' >> "$jobs"

xargs -P 8 -n 2 bash -c 'run_shard "$0" "$1"' < "$jobs"
final="$RESULTS/03_simulation/raw_episodes/episode_results.jsonl"
find "$SHARDS" -maxdepth 1 -name '*.jsonl' -print0 | sort -z | xargs -0 cat > "$final"
mkdir -p "$RESULTS/03_simulation/raw_episodes/trajectories"
if [[ -d "$SHARDS/trajectories" ]]; then
  cp -a "$SHARDS/trajectories/." "$RESULTS/03_simulation/raw_episodes/trajectories/"
fi
printf 'SIMULATION_SHARDED_RESULT=PASS rows=%s output=%s\n' "$(wc -l < "$final")" "$final"
