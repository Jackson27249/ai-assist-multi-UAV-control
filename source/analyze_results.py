from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests


def hierarchical_interval(group: pd.DataFrame, metric: str, rng: np.random.Generator, draws: int = 2000) -> tuple[float, float]:
    matrix = group.pivot_table(index="training_seed", columns="seed", values=metric, aggfunc="mean").to_numpy(dtype=float)
    if matrix.size == 0:
        return float("nan"), float("nan")
    train_count, evaluation_count = matrix.shape
    train_index = rng.integers(0, train_count, size=(draws, train_count))
    sampled_training = matrix[train_index]
    evaluation_index = rng.integers(0, evaluation_count, size=(draws, train_count, evaluation_count))
    sampled = np.take_along_axis(sampled_training, evaluation_index, axis=2)
    values = np.nanmean(sampled, axis=(1, 2))
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def aggregate(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260825)
    keys = ["policy", "num_agents", "scenario"]
    for key, group in data.groupby(keys, sort=True):
        for metric in ["team_success", "collision"]:
            low, high = hierarchical_interval(group, metric, rng)
            rows.append(
                {
                    **dict(zip(keys, key)),
                    "metric": metric,
                    "n": len(group),
                    "estimate": float(group[metric].mean()),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def paired_primary_tests(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for num_agents in [3, 5]:
        subset = data[(data.scenario == "dynamic") & (data.num_agents == num_agents)]
        left = subset[subset.policy == "gcbf_local"]
        right = subset[subset.policy == "mappo_local"]
        merged = left.merge(right, on=["training_seed", "seed"], suffixes=("_gcbf", "_mappo"))
        g_only = int(((merged.collision_gcbf == 1) & (merged.collision_mappo == 0)).sum())
        m_only = int(((merged.collision_gcbf == 0) & (merged.collision_mappo == 1)).sum())
        discordant = g_only + m_only
        if discordant:
            from scipy.stats import binomtest

            p_value = float(binomtest(min(g_only, m_only), discordant, 0.5, alternative="two-sided").pvalue)
        else:
            p_value = 1.0
        records.append(
            {
                "num_agents": num_agents,
                "gcbf_collision_mappo_safe": g_only,
                "gcbf_safe_mappo_collision": m_only,
                "discordant": discordant,
                "p_raw": p_value,
            }
        )
    result = pd.DataFrame(records)
    result["p_holm"] = multipletests(result.p_raw, method="holm")[1]
    return result


def mixed_effect_primary(data: pd.DataFrame) -> dict:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    subset = data[
        data.policy.isin(["gcbf_local", "mappo_local"])
        & data.scenario.eq("dynamic")
        & data.num_agents.isin([3, 5])
    ].copy()
    subset["collision"] = subset.collision.astype(int)
    model = BinomialBayesMixedGLM.from_formula(
        "collision ~ C(policy) * C(num_agents)",
        {"training_seed": "0 + C(training_seed)", "evaluation_seed": "0 + C(seed)"},
        subset,
    )
    fit = model.fit_vb()
    return {
        "method": "BinomialBayesMixedGLM variational Bayes",
        "n": len(subset),
        "fixed_effect_names": model.exog_names,
        "fixed_effect_mean": [float(value) for value in fit.fe_mean],
        "fixed_effect_sd": [float(value) for value in fit.fe_sd],
    }


def make_figures(summary: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    core = summary[summary.scenario.isin(["dynamic", "dense"])].copy()
    for metric, label in [("collision", "Collision rate"), ("team_success", "Team success rate")]:
        frame = core[core.metric == metric]
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        sns.lineplot(data=frame, x="num_agents", y="estimate", hue="policy", style="scenario", markers=True, dashes=False, ax=ax)
        ax.set(xlabel="Number of UAVs", ylabel=label, ylim=(-0.03, 1.03))
        fig.tight_layout()
        fig.savefig(figures / f"simulation_{metric}.png", dpi=240)
        fig.savefig(figures / f"simulation_{metric}.pdf")
        plt.close(fig)

    local_priv = summary[(summary.metric == "collision") & summary.policy.isin(["gcbf_local", "gcbf_privileged", "mappo_local", "mappo_privileged"])]
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    sns.barplot(data=local_priv, x="scenario", y="estimate", hue="policy", errorbar=None, ax=ax)
    ax.set(xlabel="Scenario", ylabel="Collision rate", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(figures / "information_symmetry_ablation.png", dpi=240)
    fig.savefig(figures / "information_symmetry_ablation.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--aggregates", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--paper-tables", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = pd.DataFrame(rows)
    args.aggregates.mkdir(parents=True, exist_ok=True)
    args.statistics.mkdir(parents=True, exist_ok=True)
    args.paper_tables.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.aggregates / "episode_results.csv", index=False)
    summary = aggregate(data)
    summary.to_csv(args.aggregates / "hierarchical_summary.csv", index=False)
    primary = paired_primary_tests(data)
    primary.to_csv(args.statistics / "primary_paired_tests.csv", index=False)
    make_figures(summary, args.figures)
    pivot = summary.pivot_table(index=["policy", "num_agents", "scenario"], columns="metric", values=["estimate", "ci_low", "ci_high"])
    pivot.to_csv(args.paper_tables / "main_results_table.csv")
    try:
        mixed_effect = mixed_effect_primary(data)
    except Exception as exc:
        mixed_effect = {"error": f"{type(exc).__name__}: {exc}"}
    report = {
        "rows": len(data),
        "policies": sorted(data.policy.unique().tolist()),
        "training_seeds": sorted(int(x) for x in data.training_seed.unique()),
        "evaluation_seeds": [int(data.seed.min()), int(data.seed.max())],
        "primary_tests": primary.to_dict(orient="records"),
        "mixed_effect_logistic": mixed_effect,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }
    (args.statistics / "statistical_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"ANALYSIS_RESULT=PASS rows={len(data)}")


if __name__ == "__main__":
    main()
