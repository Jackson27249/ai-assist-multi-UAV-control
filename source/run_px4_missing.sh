#!/usr/bin/env bash
set -euo pipefail

ROOT="${UPGRADE_ROOT:-/root/autodl-tmp/uav_upgrade_20260825}"
SRC="$ROOT/source"
RESULTS="$ROOT/results"
PY="$ROOT/.conda-env/bin/python"
PX4_ROOT="/root/autodl-tmp/high_fidelity_review/PX4-Autopilot"
export PYTHONPATH="$SRC"
export PX4_WORK_ROOT="$ROOT/px4_work"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$PX4_WORK_ROOT"

run_one() {
  policy="$1"; seed="$2"; agents="$3"; offset="$4"
  tag="${policy}_seed${seed}_n${agents}"
  target="$RESULTS/04_px4/shards/$tag/episode_summary.jsonl"
  if [[ -f "$target" ]] && [[ "$(wc -l < "$target")" -eq 20 ]]; then
    return 0
  fi
  rm -rf "$RESULTS/04_px4/shards/$tag"
  "$PY" "$SRC/px4_sih_experiment.py" \
    --px4-binary "$PX4_ROOT/build/px4_sitl_default/bin/px4" \
    --px4-romfs "$PX4_ROOT/build/px4_sitl_default/etc" \
    --models-root "$RESULTS/02_training" --output "$RESULTS/04_px4/shards/$tag" \
    --policies "$policy" --training-seeds "$seed" --agent-counts "$agents" \
    --episodes-per-model 20 --seed-start 41000 --duration 3 --instance-offset "$offset"
}
export -f run_one
export ROOT SRC RESULTS PY PX4_ROOT PX4_WORK_ROOT

: > "$ROOT/px4_missing_jobs.txt"
slot=0
for policy in gcbf_local mappo_local; do
  for seed in 1101 1102 1103 1104 1105; do
    for agents in 3 5; do
      printf '%s %s %s 0\n' "$policy" "$seed" "$agents" >> "$ROOT/px4_missing_jobs.txt"
      slot=$((slot + 1))
    done
  done
done

awk '{print}' "$ROOT/px4_missing_jobs.txt" | xargs -P 1 -n 4 bash -c 'run_one "$0" "$1" "$2" "$3"'
printf 'PX4_MISSING_RESULT=PASS episodes=%s\n' "$(find "$RESULTS/04_px4/shards" -name episode_summary.jsonl -exec cat {} \; | wc -l)"
