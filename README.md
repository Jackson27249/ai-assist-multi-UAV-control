# Auditable AI-assisted discovery for multi-UAV control

This repository contains the code and structured evidence supporting the manuscript **Auditable AI-assisted discovery of a transferable safety mechanism for multi-UAV control**.

## Evidence map

| Path | Contents |
|---|---|
| `source/` | Continuous environment, policy models, training, simulation, analysis and PX4 SIH scripts |
| `02_training/` | Five-seed checkpoints and training histories for four learned-policy families |
| `03_simulation/raw_episodes/` | Episode-level records for the 44,000 simulation runs |
| `03_simulation/statistics/` | Prespecified paired tests, bootstrap intervals and sensitivity analyses |
| `04_px4/telemetry/` | Telemetry CSV files for 400 PX4 SIH interface runs |
| `04_px4/episode_summary.jsonl` | Episode-level PX4 outcomes and interface diagnostics |
| `05_paper_tables/` | Main numerical results used in the manuscript |
| `paper/` | LaTeX source, bibliography, figures and compiled manuscript |
| `experiment_manifest.json` | Experiment inventory and provenance metadata |
| `PUBLIC_MANIFEST.sha256` | SHA-256 checksums for every file in the public snapshot |

## Recompute the reported aggregates

Create an environment from `source/requirements-upgrade.txt`, then run:

```bash
python source/analyze_results.py \
  --input 03_simulation/raw_episodes/episode_results.jsonl \
  --aggregates recomputed/aggregates \
  --statistics recomputed/statistics \
  --figures recomputed/figures \
  --paper-tables recomputed/paper_tables

python source/plot_px4_results.py --px4-root 04_px4
python -m pytest -q source/test_upgrade.py
```

The original execution environment and literal verification outputs are recorded in `00_environment/`, `source/PROVENANCE.md`, and `VERIFICATION.txt`.

## Scope of the public snapshot

The public snapshot includes the inputs needed to recompute the reported tables, statistics and figures. It excludes 1,600 low-level PX4 ULog binaries, duplicated sharded copies, per-episode renderings and Python caches (approximately 6.17 GB). The retained PX4 episode summaries and 400 telemetry CSV files support the interface-level results reported in the paper. `MANIFEST.sha256` describes the complete local acquisition bundle; `PUBLIC_MANIFEST.sha256` describes this GitHub release.

The PX4 SIH experiment validates command delivery, telemetry capture and multi-instance timing over a 3-s Offboard window. It is not obstacle-avoidance or deployment-safety evidence.

## AI-system disclosure

The automated loop used OpenAI Codex GPT-5.6-sol, version 5.6 (`gpt-5.6-sol`, accessed 26 August 2026), for repository inspection, candidate implementation, test and experiment orchestration, aggregation, plotting, discrepancy checks and initial manuscript drafting. Human researchers fixed the research question, terminal labels, primary contrasts, evidence boundaries and final interpretation.

## Licence and citation

Code is released under the MIT License. See `CITATION.cff` for the preferred citation. The data snapshot is provided for verification and research reuse with provenance retained in the included manifests.
