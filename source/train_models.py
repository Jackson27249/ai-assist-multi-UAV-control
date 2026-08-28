from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from continuous_city_env import ContinuousCityUAVEnv
from models import (
    GraphActorCritic,
    barrier_targets,
    cbf_filter,
    gaussian_log_prob,
    observation_batch,
    policy_apply,
    save_checkpoint,
)


def tree_l2(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    return float(np.sqrt(sum(float(jnp.sum(x * x)) for x in leaves)))


def make_loss(model: GraphActorCritic, barrier_coef: float):
    def loss_fn(params, batch):
        mean, log_std, value, barrier = model.apply(
            {"params": params},
            nodes=batch["nodes"],
            mask=batch["mask"],
            adjacency=batch["adjacency"],
            global_state=batch["global_state"],
        )
        new_log_prob = gaussian_log_prob(batch["pre_tanh"], mean, log_std)
        ratio = jnp.exp(new_log_prob - batch["old_log_prob"])
        advantage = (batch["advantage"] - batch["advantage"].mean()) / (batch["advantage"].std() + 1e-6)
        clipped = jnp.clip(ratio, 0.8, 1.2) * advantage
        policy_loss = -jnp.mean(jnp.minimum(ratio * advantage, clipped))
        value_loss = 0.5 * jnp.mean((value - batch["returns"]) ** 2)
        entropy = jnp.mean(jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1))
        barrier_error = (barrier - batch["barrier_target"]) ** 2 * batch["barrier_mask"]
        barrier_loss = barrier_error.sum() / jnp.maximum(batch["barrier_mask"].sum(), 1.0)
        total = policy_loss + 0.5 * value_loss - 0.005 * entropy + barrier_coef * barrier_loss
        return total, {
            "loss": total,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "barrier_loss": barrier_loss,
        }

    return loss_fn


def collect_batch(
    model: GraphActorCritic,
    params,
    key: jax.Array,
    *,
    algorithm: str,
    privileged: bool,
    seed: int,
    update: int,
    target_transitions: int,
) -> tuple[dict[str, np.ndarray], jax.Array, dict[str, float]]:
    stores: dict[str, list[np.ndarray]] = {k: [] for k in [
        "nodes", "mask", "adjacency", "global_state", "pre_tanh", "old_log_prob",
        "returns", "advantage", "barrier_target", "barrier_mask"
    ]}
    episodes = 0
    collisions = 0
    successes = 0
    interventions = 0
    infeasible = 0
    transitions = 0
    scenario_names = ["static", "dynamic", "dense", "wind", "dropout"]
    while transitions < target_transitions:
        episode_seed = seed * 100000 + update * 1000 + episodes
        num_agents = [3, 5, 7, 10][episodes % 4]
        scenario = scenario_names[episodes % len(scenario_names)]
        env = ContinuousCityUAVEnv(num_agents=num_agents, seed=episode_seed, scenario=scenario, episode_length=96)
        obs = env.observe(privileged=privileged)
        done = False
        while not done and transitions < target_transitions:
            batch_np = observation_batch(obs)
            batch_jax = {k: jnp.asarray(v) for k, v in batch_np.items()}
            key, action_key = jax.random.split(key)
            action, pre_tanh, log_prob, value, _ = policy_apply(model, params, batch_jax, action_key, False)
            action_np = np.asarray(action)
            if algorithm == "gcbf":
                executed, count, failed = cbf_filter(action_np, obs["nodes"], obs["mask"])
                interventions += count
                infeasible += failed
            else:
                executed = action_np
            next_obs, reward, done, _ = env.step(executed, privileged_observation=privileged)
            if done:
                next_value = np.zeros(num_agents, dtype=np.float32)
            else:
                next_batch = {k: jnp.asarray(v) for k, v in observation_batch(next_obs).items()}
                next_value = np.asarray(policy_apply(model, params, next_batch, action_key, True)[3])
            value_np = np.asarray(value)
            td_target = reward + 0.99 * next_value * float(not done)
            advantage = td_target - value_np
            target, active = barrier_targets(obs["nodes"], obs["mask"], env.safe_distance)
            for name in ["nodes", "mask", "adjacency", "global_state"]:
                stores[name].append(batch_np[name])
            stores["pre_tanh"].append(np.asarray(pre_tanh))
            stores["old_log_prob"].append(np.asarray(log_prob))
            stores["returns"].append(td_target.astype(np.float32))
            stores["advantage"].append(advantage.astype(np.float32))
            stores["barrier_target"].append(target)
            stores["barrier_mask"].append(active)
            transitions += num_agents
            obs = next_obs
        metrics = env.metrics()
        episodes += 1
        collisions += int(metrics["collision"])
        successes += int(metrics["team_success"])
    output = {name: np.concatenate(values, axis=0) for name, values in stores.items()}
    stats = {
        "episodes": float(episodes),
        "collision_rate": collisions / max(1, episodes),
        "success_rate": successes / max(1, episodes),
        "interventions": float(interventions),
        "infeasible": float(infeasible),
    }
    return output, key, stats


def validate(model, params, *, algorithm: str, privileged: bool, seed: int) -> tuple[float, dict[str, float]]:
    successes = collisions = total = 0
    key = jax.random.PRNGKey(seed + 991)
    for offset in range(10):
        env = ContinuousCityUAVEnv(num_agents=3 + 2 * (offset % 2), seed=21000 + offset, scenario="dynamic", episode_length=96)
        obs = env.observe(privileged=privileged)
        done = False
        while not done:
            key, subkey = jax.random.split(key)
            batch = {k: jnp.asarray(v) for k, v in observation_batch(obs).items()}
            actions = np.asarray(policy_apply(model, params, batch, subkey, True)[0])
            if algorithm == "gcbf":
                actions = cbf_filter(actions, obs["nodes"], obs["mask"])[0]
            obs, _, done, metrics = env.step(actions, privileged_observation=privileged)
        total += 1
        successes += int(metrics["team_success"])
        collisions += int(metrics["collision"])
    rates = {"success_rate": successes / total, "collision_rate": collisions / total}
    return rates["success_rate"] - 2.0 * rates["collision_rate"], rates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["gcbf", "mappo"], required=True)
    parser.add_argument("--visibility", choices=["local", "privileged"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--transitions-per-update", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    graph_actor = args.algorithm == "gcbf"
    privileged = args.visibility == "privileged"
    model = GraphActorCritic(graph_actor=graph_actor)
    dummy_env = ContinuousCityUAVEnv(num_agents=3, seed=args.seed)
    dummy = {k: jnp.asarray(v) for k, v in observation_batch(dummy_env.observe(privileged=privileged)).items()}
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = model.init(init_key, **dummy)["params"]
    schedule = optax.clip_by_global_norm(1.0)
    optimizer = optax.chain(schedule, optax.adam(3e-4))
    opt_state = optimizer.init(params)
    loss_fn = make_loss(model, barrier_coef=0.8 if graph_actor else 0.0)

    @jax.jit
    def update_step(current_params, current_opt_state, batch):
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_params, batch)
        updates, new_opt_state = optimizer.update(grads, current_opt_state, current_params)
        return optax.apply_updates(current_params, updates), new_opt_state, info, grads

    history: list[dict[str, float]] = []
    best_score = -float("inf")
    start = time.time()
    for update in range(args.updates):
        batch_np, key, rollout = collect_batch(
            model,
            params,
            key,
            algorithm=args.algorithm,
            privileged=privileged,
            seed=args.seed,
            update=update,
            target_transitions=args.transitions_per_update,
        )
        order = np.random.default_rng(args.seed + update).permutation(len(batch_np["returns"]))
        info = None
        grads = None
        for _ in range(args.epochs):
            for start_index in range(0, len(order), 256):
                index = order[start_index : start_index + 256]
                batch = {name: jnp.asarray(value[index]) for name, value in batch_np.items()}
                params, opt_state, info, grads = update_step(params, opt_state, batch)
        score, validation = validate(model, params, algorithm=args.algorithm, privileged=privileged, seed=args.seed)
        row = {
            "update": update + 1,
            **{k: float(v) for k, v in rollout.items()},
            **{f"validation_{k}": float(v) for k, v in validation.items()},
            **{k: float(v) for k, v in info.items()},
            "grad_l2": tree_l2(grads),
            "elapsed_seconds": time.time() - start,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        metadata = {
            "algorithm": args.algorithm,
            "visibility": args.visibility,
            "seed": args.seed,
            "update": update + 1,
            "validation": validation,
            "graph_actor": graph_actor,
            "continuous_action": True,
        }
        save_checkpoint(args.output / "final.msgpack", params, metadata)
        if score > best_score:
            best_score = score
            save_checkpoint(args.output / "best.msgpack", params, metadata | {"selection_score": score})

    with (args.output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (args.output / "training_manifest.json").write_text(
        json.dumps(
            {
                "algorithm": args.algorithm,
                "visibility": args.visibility,
                "seed": args.seed,
                "updates": args.updates,
                "transitions_per_update": args.transitions_per_update,
                "epochs": args.epochs,
                "best_score": best_score,
                "jax_backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"TRAINING_RESULT=PASS output={args.output}")


if __name__ == "__main__":
    main()

