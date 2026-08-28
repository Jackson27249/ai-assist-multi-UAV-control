from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from continuous_city_env import ContinuousCityUAVEnv
from models import LoadedPolicy, load_policy


POLICY_SPECS = {
    "gcbf_local": (True, True, False),
    "gcbf_privileged": (True, True, True),
    "mappo_local": (False, False, False),
    "mappo_privileged": (False, False, True),
}


def run_episode(policy: LoadedPolicy, *, privileged: bool, num_agents: int, scenario: str, seed: int, trajectory: bool) -> dict:
    env = ContinuousCityUAVEnv(num_agents=num_agents, seed=seed, scenario=scenario)
    obs = env.observe(privileged=privileged)
    done = False
    interventions = infeasible = 0
    inference_times: list[float] = []
    key = jax.random.PRNGKey(seed)
    while not done:
        key, subkey = jax.random.split(key)
        start = time.perf_counter()
        actions, changed, failed = policy.act(obs, subkey)
        inference_times.append(time.perf_counter() - start)
        interventions += changed
        infeasible += failed
        obs, _, done, metrics = env.step(actions, privileged_observation=privileged)
    metrics.update(
        {
            "policy": policy.name,
            "interventions": interventions,
            "qp_infeasible": infeasible,
            "mean_inference_ms": 1000.0 * float(np.mean(inference_times)),
            "p95_inference_ms": 1000.0 * float(np.percentile(inference_times, 95)),
        }
    )
    if trajectory:
        metrics["trajectory"] = np.asarray(env.trajectory).tolist()
        metrics["goals"] = env.goals.tolist()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=31000)
    parser.add_argument("--agent-counts", type=int, nargs="+", default=[3, 5, 7, 10])
    parser.add_argument("--scenarios", nargs="+", default=["static", "dynamic", "dense", "wind", "dropout"])
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[1101, 1102, 1103, 1104, 1105])
    parser.add_argument("--policies", nargs="+", default=list(POLICY_SPECS) + ["distributed_cbf"])
    parser.add_argument("--trajectory-seeds", type=int, default=3)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trajectory_output = args.output.parent / "trajectories"
    trajectory_output.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for policy_name in args.policies:
            seed_values = args.training_seeds if policy_name != "distributed_cbf" else [-1]
            for training_seed in seed_values:
                if policy_name == "distributed_cbf":
                    policy = LoadedPolicy(name=policy_name, model=None, params=None, use_filter=True)
                    privileged = False
                else:
                    graph_actor, use_filter, privileged = POLICY_SPECS[policy_name]
                    checkpoint = args.models_root / policy_name / f"seed_{training_seed}" / "best.msgpack"
                    policy = load_policy(checkpoint, policy_name, graph_actor=graph_actor, use_filter=use_filter)
                for num_agents in args.agent_counts:
                    for scenario in args.scenarios:
                        for offset in range(args.episodes):
                            evaluation_seed = args.seed_start + offset
                            keep_trajectory = offset < args.trajectory_seeds
                            row = run_episode(
                                policy,
                                privileged=privileged,
                                num_agents=num_agents,
                                scenario=scenario,
                                seed=evaluation_seed,
                                trajectory=keep_trajectory,
                            )
                            row["training_seed"] = training_seed
                            if keep_trajectory:
                                trajectory = row.pop("trajectory")
                                payload = {
                                    "policy": policy_name,
                                    "training_seed": training_seed,
                                    "evaluation_seed": evaluation_seed,
                                    "num_agents": num_agents,
                                    "scenario": scenario,
                                    "goals": row.pop("goals"),
                                    "trajectory": trajectory,
                                }
                                path = trajectory_output / f"{policy_name}_train{training_seed}_n{num_agents}_{scenario}_seed{evaluation_seed}.json"
                                path.write_text(json.dumps(payload), encoding="utf-8")
                                row["trajectory_file"] = str(path.name)
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                            count += 1
                            if count % 100 == 0:
                                handle.flush()
                                print(f"SIM_PROGRESS={count}", flush=True)

        # OOD is evaluated separately to keep the confirmatory family unchanged.
        for policy_name in [name for name in args.policies if name != "distributed_cbf"]:
            graph_actor, use_filter, privileged = POLICY_SPECS[policy_name]
            for training_seed in args.training_seeds:
                checkpoint = args.models_root / policy_name / f"seed_{training_seed}" / "best.msgpack"
                policy = load_policy(checkpoint, policy_name, graph_actor=graph_actor, use_filter=use_filter)
                for offset in range(args.episodes):
                    row = run_episode(
                        policy,
                        privileged=privileged,
                        num_agents=12,
                        scenario="ood_dense",
                        seed=args.seed_start + offset,
                        trajectory=offset < args.trajectory_seeds,
                    )
                    row["training_seed"] = training_seed
                    if "trajectory" in row:
                        payload = {
                            "policy": policy_name,
                            "training_seed": training_seed,
                            "evaluation_seed": args.seed_start + offset,
                            "num_agents": 12,
                            "scenario": "ood_dense",
                            "goals": row.pop("goals"),
                            "trajectory": row.pop("trajectory"),
                        }
                        path = trajectory_output / f"{policy_name}_train{training_seed}_n12_ood_dense_seed{args.seed_start + offset}.json"
                        path.write_text(json.dumps(payload), encoding="utf-8")
                        row["trajectory_file"] = str(path.name)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    count += 1
    print(f"SIMULATION_RESULT=PASS rows={count} output={args.output}")


if __name__ == "__main__":
    main()

