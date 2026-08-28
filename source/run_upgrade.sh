#!/usr/bin/env bash
set -euo pipefail

ROOT="${UPGRADE_ROOT:-/root/autodl-tmp/uav_upgrade_20260825}"
SRC="$ROOT/source"
RESULTS="$ROOT/results"
CONDA="/root/miniconda3/bin/conda"
ENV="$ROOT/.conda-env"
PY="$ENV/bin/python"
PX4_ROOT="/root/autodl-tmp/high_fidelity_review/PX4-Autopilot"
MODE="${1:-all}"

mkdir -p "$RESULTS"/{00_environment,01_tests,02_training/{gcbf_local,gcbf_privileged,mappo_local,mappo_privileged},03_simulation/{raw_episodes,aggregates,statistics,figures},04_px4/{ulg,telemetry,per_episode_figures,summary_figures},05_paper_tables,06_logs}
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"

run_setup() {
  if [[ ! -x "$PY" ]]; then
    "$CONDA" create -y -p "$ENV" python=3.10 pip
  fi
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r "$SRC/requirements-upgrade.txt"
  "$PY" - <<'PY' | tee "$RESULTS/00_environment/dependency_smoke.log"
import json
import jax
import flax
import jraph
import optax
import mavsdk
import pyulog
payload = {
    "jax": jax.__version__,
    "flax": flax.__version__,
    "jraph": getattr(jraph, "__version__", "installed"),
    "optax": optax.__version__,
    "jax_backend": jax.default_backend(),
    "devices": [str(device) for device in jax.devices()],
}
print(json.dumps(payload, indent=2))
assert payload["jax_backend"] == "gpu", payload
print("DEPENDENCY_SMOKE_RESULT=PASS")
PY
  /usr/bin/nvidia-smi -q > "$RESULTS/00_environment/nvidia-smi.txt"
  "$PY" -m pip freeze > "$RESULTS/00_environment/pip-freeze.txt"
  uname -a > "$RESULTS/00_environment/uname.txt"
  git -C "$PX4_ROOT" rev-parse HEAD > "$RESULTS/00_environment/px4-commit.txt"
}

run_tests() {
  set +e
  /root/miniconda3/bin/python -m pytest -q "$ROOT/original_workspace/test_city_uav.py" 2>&1 | tee "$RESULTS/01_tests/baseline.log"
  baseline_rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$baseline_rc" > "$RESULTS/01_tests/baseline.exit"
  "$PY" -m pytest -q "$SRC/test_upgrade.py" 2>&1 | tee "$RESULTS/01_tests/modified.log"
  printf '0\n' > "$RESULTS/01_tests/modified.exit"
  "$PY" "$SRC/train_models.py" --algorithm gcbf --visibility local --seed 1101 \
    --updates 1 --transitions-per-update 256 --epochs 1 --output "$RESULTS/01_tests/training_smoke" \
    2>&1 | tee "$RESULTS/01_tests/training_smoke.log"
}

run_training() {
  for policy in gcbf_local gcbf_privileged mappo_local mappo_privileged; do
    algorithm="${policy%%_*}"
    visibility="${policy#*_}"
    for seed in 1101 1102 1103 1104 1105; do
      output="$RESULTS/02_training/$policy/seed_$seed"
      mkdir -p "$output"
      "$PY" "$SRC/train_models.py" --algorithm "$algorithm" --visibility "$visibility" --seed "$seed" \
        --updates 12 --transitions-per-update 768 --epochs 4 --output "$output" \
        2>&1 | tee "$RESULTS/06_logs/train_${policy}_${seed}.log"
    done
  done
}

run_simulation() {
  bash "$SRC/run_simulation_sharded.sh" 2>&1 | tee "$RESULTS/06_logs/simulation.log"
  "$PY" "$SRC/analyze_results.py" \
    --input "$RESULTS/03_simulation/raw_episodes/episode_results.jsonl" \
    --aggregates "$RESULTS/03_simulation/aggregates" \
    --statistics "$RESULTS/03_simulation/statistics" \
    --figures "$RESULTS/03_simulation/figures" \
    --paper-tables "$RESULTS/05_paper_tables" \
    2>&1 | tee "$RESULTS/06_logs/analysis.log"
}

run_px4() {
  bash "$SRC/run_px4_sharded.sh" 2>&1 | tee "$RESULTS/06_logs/px4.log"
  "$PY" "$SRC/plot_px4_results.py" --px4-root "$RESULTS/04_px4" \
    2>&1 | tee "$RESULTS/06_logs/px4_plot.log"
}

run_manifest() {
  "$PY" "$SRC/write_manifest.py" --root "$ROOT" --results "$RESULTS"
}

case "$MODE" in
  setup) run_setup ;;
  test) run_tests ;;
  train) run_training ;;
  simulate) run_simulation ;;
  px4) run_px4 ;;
  manifest) run_manifest ;;
  all)
    run_setup
    run_tests
    run_training
    run_simulation
    run_px4
    run_manifest
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
