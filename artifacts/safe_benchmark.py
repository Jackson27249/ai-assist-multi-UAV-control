#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from city_uav_env import CityUAVEnv
from evaluate import MAPPOPolicy, one_hot


UPSTREAM = {
    "gcbfplus": "fb449907bdbf981aa10f0edfecca02663ddc8037",
    "dgppo": "51b3b11c42760cd62f502f9d60cf6d302f413973",
}


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))


def progress(env: CityUAVEnv, positions: np.ndarray) -> np.ndarray:
    before = np.linalg.norm(env.goals - env.pos, axis=1)
    after = np.linalg.norm(env.goals - positions, axis=1)
    return before - after


class GoalPolicy:
    name = "nominal_goal"

    def reset(self, num_agents: int) -> None:
        self.last_interventions = 0

    def act(self, env: CityUAVEnv, obs: np.ndarray) -> np.ndarray:
        selected = np.zeros(env.num_agents, dtype=np.int64)
        for agent in range(env.num_agents):
            scores = []
            for action in range(len(env.ACTIONS)):
                trial = selected.copy()
                trial[agent] = action
                candidate = env._predict_positions(trial)[agent]
                delta = np.linalg.norm(env.goals[agent] - env.pos[agent]) - np.linalg.norm(
                    env.goals[agent] - candidate
                )
                scores.append(8.0 * delta - 0.01 * action)
            selected[agent] = int(np.argmax(scores))
        self.last_interventions = 0
        return selected


class GraphAvoidancePolicy(GoalPolicy):
    name = "graph_avoidance_style"

    def act(self, env: CityUAVEnv, obs: np.ndarray) -> np.ndarray:
        nominal = super().act(env, obs)
        selected = nominal.copy()
        for agent in np.argsort(np.linalg.norm(env.goals - env.pos, axis=1)):
            best_score, best_action = -np.inf, 0
            for action in range(len(env.ACTIONS)):
                trial = selected.copy()
                trial[agent] = action
                positions = env._predict_positions(trial)
                clear = env.minimum_clearance(positions)
                delta = progress(env, positions)[agent]
                score = 8.0 * delta + 0.6 * min(clear, 3.0) - 25.0 * max(0.0, -clear)
                if score > best_score:
                    best_score, best_action = score, action
            selected[agent] = best_action
        self.last_interventions = int(np.count_nonzero(selected != nominal))
        return selected


class LagrangianPolicy(GoalPolicy):
    name = "lagrangian_style"

    def __init__(self, initial_lambda: float = 0.5, lr: float = 0.35):
        self.initial_lambda = initial_lambda
        self.lr = lr

    def reset(self, num_agents: int) -> None:
        super().reset(num_agents)
        self.lagrange = self.initial_lambda

    def act(self, env: CityUAVEnv, obs: np.ndarray) -> np.ndarray:
        nominal = super().act(env, obs)
        selected = nominal.copy()
        for agent in range(env.num_agents):
            best_score, best_action = -np.inf, int(nominal[agent])
            for action in range(len(env.ACTIONS)):
                trial = selected.copy()
                trial[agent] = action
                positions = env._predict_positions(trial)
                cost = max(0.0, env.safety_cost(positions))
                delta = progress(env, positions)[agent]
                score = 8.0 * delta - self.lagrange * cost - 0.04 * (action != nominal[agent])
                if score > best_score:
                    best_score, best_action = score, action
            selected[agent] = best_action
        predicted_cost = max(0.0, env.safety_cost(env._predict_positions(selected)))
        self.lagrange = float(np.clip(self.lagrange + self.lr * predicted_cost, 0.0, 50.0))
        self.last_interventions = int(np.count_nonzero(selected != nominal))
        return selected


class GCBFPlusStylePolicy(GoalPolicy):
    name = "gcbfplus_style_cbf"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def act(self, env: CityUAVEnv, obs: np.ndarray) -> np.ndarray:
        nominal = super().act(env, obs)
        selected = nominal.copy()
        current_cost = env.safety_cost(env.pos)
        for agent in np.argsort(np.linalg.norm(env.goals - env.pos, axis=1)):
            feasible: list[tuple[float, int]] = []
            fallback: list[tuple[float, float, int]] = []
            for action in range(len(env.ACTIONS)):
                trial = selected.copy()
                trial[agent] = action
                positions = env._predict_positions(trial)
                next_cost = env.safety_cost(positions)
                residual = (next_cost - current_cost) / env.dt + self.alpha * current_cost
                delta = progress(env, positions)[agent]
                clearance = env.minimum_clearance(positions)
                if residual <= 0.0 and clearance > env.collision_distance:
                    feasible.append((8.0 * delta + 0.15 * clearance, action))
                fallback.append((-next_cost, 8.0 * delta, action))
            selected[agent] = max(feasible)[1] if feasible else max(fallback)[2]
        self.last_interventions = int(np.count_nonzero(selected != nominal))
        return selected


class DGPPOStylePolicy(GoalPolicy):
    name = "dgppo_style_dgcbf"

    def __init__(
        self,
        alpha: float = 1.0,
        cbf_eps: float = 0.01,
        cbf_weight: float = 12.0,
        progress_weight: float = 8.0,
        deviation_weight: float = 0.03,
        clearance_weight: float = 0.12,
        name: str | None = None,
    ):
        self.alpha = alpha
        self.cbf_eps = cbf_eps
        self.cbf_weight = cbf_weight
        self.progress_weight = progress_weight
        self.deviation_weight = deviation_weight
        self.clearance_weight = clearance_weight
        if name is not None:
            self.name = name

    def act(self, env: CityUAVEnv, obs: np.ndarray) -> np.ndarray:
        nominal = super().act(env, obs)
        selected = nominal.copy()
        current_cost = env.safety_cost(env.pos)
        for agent in np.argsort(np.linalg.norm(env.goals - env.pos, axis=1)):
            best_score, best_action = -np.inf, int(nominal[agent])
            for action in range(len(env.ACTIONS)):
                trial = selected.copy()
                trial[agent] = action
                positions = env._predict_positions(trial)
                next_cost = env.safety_cost(positions)
                cbf_residual = (next_cost - current_cost) / env.dt + self.alpha * current_cost
                cbf_advantage = max(cbf_residual + self.cbf_eps, 0.0)
                delta = progress(env, positions)[agent]
                policy_deviation = float(action != nominal[agent])
                score = self.progress_weight * delta - self.cbf_weight * cbf_advantage
                score -= self.deviation_weight * policy_deviation
                score += self.clearance_weight * min(env.minimum_clearance(positions), 3.0)
                if score > best_score:
                    best_score, best_action = score, action
            selected[agent] = best_action
        self.last_interventions = int(np.count_nonzero(selected != nominal))
        return selected


@dataclass(frozen=True)
class Scenario:
    wind: float
    dropout: float


SCENARIOS = {
    "static": Scenario(0.10, 0.05),
    "dynamic": Scenario(0.25, 0.08),
    "dense": Scenario(0.30, 0.10),
    "wind": Scenario(0.65, 0.15),
    "dropout": Scenario(0.40, 0.45),
}


def run_episode(policy, *, seed: int, num_agents: int, scenario: str, safe_distance: float,
                episode_length: int) -> dict:
    pressure = SCENARIOS[scenario]
    env = CityUAVEnv(
        num_agents=num_agents,
        seed=seed,
        episode_length=episode_length,
        reward_variant="baseline",
        wind_scale=pressure.wind,
        comm_dropout=pressure.dropout,
        enforce_shield=False,
        scenario=scenario,
        safe_distance=safe_distance,
    )
    policy.reset(num_agents)
    obs = env.reset()
    interventions = 0
    done = False
    while not done:
        actions = policy.act(env, obs)
        interventions += getattr(policy, "last_interventions", 0)
        obs, _, dones, infos = env.step(one_hot(actions))
        done = bool(np.all(dones))
    row = dict(infos[0]["episode_metrics"])
    row["mechanism_intervention_rate"] = interventions / max(1, env.step_count * num_agents)
    row["policy"] = policy.name
    row["seed"] = seed
    row["safe_distance"] = safe_distance
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["policy"], row["num_agents"], row["scenario"], row["safe_distance"])
        groups[key].append(row)
    numeric = [
        "episode_return", "steps", "mean_path_length", "total_energy", "mean_safety_cost",
        "max_safety_cost", "near_miss_rate", "mechanism_intervention_rate", "comm_drop_rate",
        "min_pair_distance", "min_dynamic_distance", "min_static_clearance", "min_separation",
        "mean_goal_distance",
    ]
    output = []
    for key, group in sorted(groups.items()):
        policy, num_agents, scenario, safe_distance = key
        item = {
            "policy": policy,
            "num_agents": num_agents,
            "scenario": scenario,
            "safe_distance": safe_distance,
            "episodes": len(group),
            "success_rate": float(np.mean([r["team_success"] for r in group])),
            "collision_rate": float(np.mean([r["collision"] for r in group])),
            "safety_rate": float(np.mean([not r["collision"] for r in group])),
            "collision_types": dict(Counter(r["collision_type"] for r in group)),
        }
        for metric in numeric:
            values = [float(r[metric]) for r in group]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_ci95"] = ci95(values)
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row if key != "collision_types"})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def residual_penalty_sensitivity_variants() -> list[DGPPOStylePolicy]:
    """Return one-at-a-time +/-50% perturbations around the reference RP score."""
    reference = {
        "progress_weight": 8.0,
        "cbf_weight": 12.0,
        "cbf_eps": 0.01,
        "deviation_weight": 0.03,
        "clearance_weight": 0.12,
    }
    variants = [DGPPOStylePolicy(name="residual_penalty", **reference)]
    for parameter, value in reference.items():
        for scale in (0.5, 1.5):
            settings = reference.copy()
            settings[parameter] = value * scale
            tag = "0p5" if scale == 0.5 else "1p5"
            variants.append(DGPPOStylePolicy(name=f"rp_{parameter}_{tag}", **settings))
    return variants


def build_policies(policy_names: list[str], model_dir: str | None, rp_sensitivity: bool) -> list:
    if rp_sensitivity and "residual_penalty" not in policy_names:
        raise ValueError("--rp-sensitivity requires residual_penalty in --policies")

    factories = {
        "nominal_only": GoalPolicy,
        "graph_avoidance": GraphAvoidancePolicy,
        "lagrangian": LagrangianPolicy,
        "residual_filter": GCBFPlusStylePolicy,
        "residual_penalty": DGPPOStylePolicy,
    }
    policies = []
    for name in policy_names:
        if name == "mappo":
            if not model_dir:
                raise ValueError("--model-dir is required when mappo is selected")
            probe = CityUAVEnv(num_agents=3)
            policies.append(MAPPOPolicy(Path(model_dir), probe._obs_dim, "mappo_baseline", 3))
        elif name == "residual_penalty" and rp_sensitivity:
            policies.extend(residual_penalty_sensitivity_variants())
        else:
            policy = factories[name]()
            policy.name = {
                "nominal_only": "nominal_only",
                "graph_avoidance": "graph_avoidance",
                "lagrangian": "lagrangian",
                "residual_filter": "residual_filter",
                "residual_penalty": "residual_penalty",
            }[name]
            policies.append(policy)
    return policies


def plot_results(output: Path, summary: list[dict], safe_distance: float) -> None:
    styles = {
        "mappo_baseline": ("o", "#4c78a8"),
        "nominal_only": ("X", "#7f7f7f"),
        "graph_avoidance": ("s", "#f58518"),
        "lagrangian": ("^", "#54a24b"),
        "residual_filter": ("D", "#e45756"),
        "residual_penalty": ("P", "#72b7b2"),
    }
    base = [r for r in summary if r["num_agents"] == 3 and r["safe_distance"] == safe_distance]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for policy in sorted({r["policy"] for r in base}):
        rows = [r for r in base if r["policy"] == policy]
        x = np.mean([r["collision_rate"] for r in rows])
        y = np.mean([r["success_rate"] for r in rows])
        marker, color = styles.get(policy, ("o", "black"))
        ax.scatter(x, y, s=85, marker=marker, color=color, label=policy)
    ax.set(xlabel="Collision rate (lower is better)", ylabel="Success rate (higher is better)",
           xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), title="Safety-performance Pareto summary (N=3)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "pareto_success_collision.png", dpi=180)
    plt.close(fig)

    scale = [r for r in summary if r["scenario"] == "dynamic" and r["safe_distance"] == safe_distance]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True, sharey=True)
    for policy in sorted({r["policy"] for r in scale}):
        rows = sorted((r for r in scale if r["policy"] == policy), key=lambda r: r["num_agents"])
        marker, color = styles.get(policy, ("o", "black"))
        x = [r["num_agents"] for r in rows]
        axes[0].plot(x, [r["success_rate"] for r in rows], marker=marker, color=color, label=policy)
        axes[1].plot(x, [r["safety_rate"] for r in rows], marker=marker, color=color, label=policy)
    axes[0].set_title("Zero-shot task success")
    axes[1].set_title("Zero-shot safety")
    for ax in axes:
        ax.set(xlabel="Number of UAVs", ylim=(-0.03, 1.03))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Rate")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(output / "scalability_zero_shot.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--episode-length", type=int, default=128)
    parser.add_argument("--agent-counts", type=int, nargs="+", default=[3, 5, 7, 10])
    parser.add_argument("--scenarios", nargs="+", default=["static", "dynamic", "dense", "wind", "dropout"])
    parser.add_argument("--safe-distances", type=float, nargs="+", default=[1.2])
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["mappo", "nominal_only", "graph_avoidance", "lagrangian", "residual_filter", "residual_penalty"],
        default=["mappo", "graph_avoidance", "lagrangian", "residual_filter", "residual_penalty"],
    )
    parser.add_argument("--rp-sensitivity", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    policies = build_policies(args.policies, args.model_dir, args.rp_sensitivity)

    rows = []
    for safe_distance in args.safe_distances:
        for num_agents in args.agent_counts:
            for scenario in args.scenarios:
                if scenario not in SCENARIOS:
                    raise ValueError(f"未知场景: {scenario}")
                for policy in policies:
                    for offset in range(args.episodes):
                        row = run_episode(
                            policy,
                            seed=args.seed_start + offset,
                            num_agents=num_agents,
                            scenario=scenario,
                            safe_distance=safe_distance,
                            episode_length=args.episode_length,
                        )
                        rows.append(row)
                print(f"CELL_RESULT=PASS N={num_agents} scenario={scenario} d_safe={safe_distance}", flush=True)

    summary = aggregate(rows)
    failure_counts = Counter(
        (r["policy"], r["scenario"], r["collision_type"]) for r in rows if r["collision"]
    )
    manifest = {
        "implementation_level": "same-environment mechanism adaptation; not upstream model reproduction",
        "upstream_commits": UPSTREAM,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "protocol": vars(args),
    }
    (output / "episode_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "aggregate_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "failure_analysis.json").write_text(
        json.dumps([
            {"policy": k[0], "scenario": k[1], "collision_type": k[2], "count": v}
            for k, v in sorted(failure_counts.items())
        ], indent=2), encoding="utf-8"
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_csv(output / "aggregate_results.csv", summary)
    plot_results(output, summary, args.safe_distances[0])
    print(f"BENCHMARK_RESULT=PASS output={output} episodes={len(rows)}")


if __name__ == "__main__":
    main()
