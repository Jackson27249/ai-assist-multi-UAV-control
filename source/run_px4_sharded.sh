#!/usr/bin/env bash
set -euo pipefail

ROOT="${UPGRADE_ROOT:-/root/autodl-tmp/uav_upgrade_20260825}"
SRC="$ROOT/source"
RESULTS="$ROOT/results"
PY="$ROOT/.conda-env/bin/python"
PX4_ROOT="/root/autodl-tmp/high_fidelity_review/PX4-Autopilot"
SHARDS="$RESULTS/04_px4/shards"
LOGS="$RESULTS/06_logs/px4_shards"
mkdir -p "$SHARDS" "$LOGS"
export PYTHONPATH="$SRC"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

for slot in 0 1 2 3; do
  : > "$ROOT/px4_jobs_$slot.txt"
done
index=0
for policy in gcbf_local mappo_local; do
  for seed in 1101 1102 1103 1104 1105; do
    for agents in 3 5; do
      slot=$((index % 4))
      printf '%s %s %s\n' "$policy" "$seed" "$agents" >> "$ROOT/px4_jobs_$slot.txt"
      index=$((index + 1))
    done
  done
done

worker() {
  slot="$1"
  offset=$((slot * 20))
  while read -r policy seed agents; do
    tag="${policy}_seed${seed}_n${agents}"
    "$PY" "$SRC/px4_sih_experiment.py" \
      --px4-binary "$PX4_ROOT/build/px4_sitl_default/bin/px4" \
      --px4-romfs "$PX4_ROOT/build/px4_sitl_default/etc" \
      --models-root "$RESULTS/02_training" \
      --output "$SHARDS/$tag" --policies "$policy" --training-seeds "$seed" \
      --agent-counts "$agents" --episodes-per-model 20 --seed-start 41000 --duration 10 \
      --instance-offset "$offset" > "$LOGS/$tag.log" 2>&1
    printf 'PX4_SHARD_RESULT=PASS policy=%s seed=%s agents=%s\n' "$policy" "$seed" "$agents"
  done < "$ROOT/px4_jobs_$slot.txt"
}

worker 0 &
p0=$!
worker 1 &
p1=$!
worker 2 &
p2=$!
worker 3 &
p3=$!
wait "$p0" "$p1" "$p2" "$p3"

: > "$RESULTS/04_px4/episode_summary.jsonl"
for shard in "$SHARDS"/*; do
  test -d "$shard" || continue
  cat "$shard/episode_summary.jsonl" >> "$RESULTS/04_px4/episode_summary.jsonl"
  cp -a "$shard/telemetry/." "$RESULTS/04_px4/telemetry/" 2>/dev/null || true
  cp -a "$shard/ulg/." "$RESULTS/04_px4/ulg/" 2>/dev/null || true
done
printf 'PX4_SHARDED_RESULT=PASS episodes=%s telemetry=%s ulg=%s\n' \
  "$(wc -l < "$RESULTS/04_px4/episode_summary.jsonl")" \
  "$(find "$RESULTS/04_px4/telemetry" -name '*.csv' | wc -l)" \
  "$(find "$RESULTS/04_px4/ulg" -name '*.ulg' | wc -l)"

