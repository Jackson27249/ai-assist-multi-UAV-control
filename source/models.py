from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from continuous_city_env import ContinuousCityUAVEnv


MAX_GLOBAL_DIM = 12 * 9


class GraphActorCritic(nn.Module):
    graph_actor: bool
    hidden_dim: int = 96

    @nn.compact
    def __call__(
        self,
        nodes: jax.Array,
        mask: jax.Array,
        adjacency: jax.Array,
        global_state: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        mask_e = mask[..., None]
        if self.graph_actor:
            hidden = nn.relu(nn.Dense(self.hidden_dim, name="node_embed")(nodes)) * mask_e
            degree = jnp.maximum(adjacency.sum(axis=-1, keepdims=True), 1.0)
            message = jnp.einsum("bij,bjh->bih", adjacency, hidden) / degree
            hidden = nn.relu(nn.Dense(self.hidden_dim, name="message")(jnp.concatenate([hidden, message], axis=-1))) * mask_e
            pooled = hidden[:, 0] + (hidden * mask_e).sum(axis=1) / jnp.maximum(mask_e.sum(axis=1), 1.0)
        else:
            flattened = (nodes * mask_e).reshape((nodes.shape[0], -1))
            pooled = nn.relu(nn.Dense(self.hidden_dim, name="flat_embed")(flattened))
            pooled = nn.relu(nn.Dense(self.hidden_dim, name="flat_hidden")(pooled))
        actor_hidden = nn.relu(nn.Dense(self.hidden_dim, name="actor_hidden")(pooled))
        residual = nn.Dense(3, name="actor_mean")(actor_hidden)
        goal = nodes[:, 0, 6:9] * 20.0
        nominal = goal / jnp.maximum(jnp.linalg.norm(goal, axis=-1, keepdims=True), 1e-6)
        mean = nn.tanh(nominal + 0.35 * residual)
        log_std = self.param("log_std", nn.initializers.constant(-1.5), (3,))
        log_std = jnp.broadcast_to(log_std, mean.shape)
        critic_hidden = nn.relu(nn.Dense(self.hidden_dim, name="critic_hidden")(global_state))
        value = nn.Dense(1, name="critic_value")(critic_hidden).squeeze(-1)
        barrier = nn.Dense(1, name="barrier_head")(hidden if self.graph_actor else nodes).squeeze(-1)
        return mean, log_std, value, barrier


def pad_global(global_state: np.ndarray, count: int) -> np.ndarray:
    padded = np.zeros(MAX_GLOBAL_DIM, dtype=np.float32)
    padded[: min(MAX_GLOBAL_DIM, len(global_state))] = global_state[:MAX_GLOBAL_DIM]
    return np.repeat(padded[None, :], count, axis=0)


def observation_batch(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count = obs["nodes"].shape[0]
    return {
        "nodes": obs["nodes"].astype(np.float32),
        "mask": obs["mask"].astype(np.float32),
        "adjacency": obs["adjacency"].astype(np.float32),
        "global_state": pad_global(obs["global_state"], count),
    }


def gaussian_log_prob(pre_tanh: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    variance = jnp.exp(2.0 * log_std)
    normal = -0.5 * (((pre_tanh - mean) ** 2) / variance + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    action = jnp.tanh(pre_tanh)
    correction = jnp.log(jnp.maximum(1.0 - action**2, 1e-6))
    return (normal - correction).sum(axis=-1)


@partial(jax.jit, static_argnames=("model", "deterministic"))
def policy_apply(
    model: GraphActorCritic,
    params: Any,
    batch: dict[str, jax.Array],
    key: jax.Array,
    deterministic: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    mean, log_std, value, barrier = model.apply({"params": params}, **batch)
    noise = jnp.zeros_like(mean) if deterministic else jax.random.normal(key, mean.shape)
    pre_tanh = mean + jnp.exp(log_std) * noise
    action = jnp.tanh(pre_tanh)
    log_prob = gaussian_log_prob(pre_tanh, mean, log_std)
    return action, pre_tanh, log_prob, value, barrier


def barrier_targets(nodes: np.ndarray, mask: np.ndarray, safe_distance: float) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(nodes[..., :3] * 10.0, axis=-1)
    node_type = np.argmax(nodes[..., 9:13], axis=-1)
    active = (mask > 0.0) & (node_type != 0)
    target = np.clip((distance - safe_distance) / safe_distance, -1.0, 2.0).astype(np.float32)
    return target, active.astype(np.float32)


def cbf_filter(actions: np.ndarray, nodes: np.ndarray, mask: np.ndarray, safe_distance: float = 1.2) -> tuple[np.ndarray, int, int]:
    """Project desired velocities onto local pairwise CBF half-spaces."""
    desired = np.asarray(actions, dtype=np.float32) * 2.0
    projected = desired.copy()
    infeasible = 0
    for i in range(len(projected)):
        constraints: list[tuple[np.ndarray, float]] = []
        for k in range(1, nodes.shape[1]):
            if mask[i, k] <= 0.0:
                continue
            node_type = int(np.argmax(nodes[i, k, 9:13]))
            if node_type == 0:
                continue
            rel = nodes[i, k, :3] * 10.0
            rel_vel = nodes[i, k, 3:6] * 2.0
            distance = float(np.linalg.norm(rel))
            if distance > 2.5 * safe_distance or distance < 1e-5:
                continue
            robust = safe_distance + 0.08 + 0.12 * min(1.0, float(nodes[i, k, 14]))
            h = distance * distance - robust * robust
            # rel dot (v_other - u_i) + alpha*h/2 >= 0 -> rel dot u_i <= rhs
            rhs = float(np.dot(rel, rel_vel) + 0.65 * h)
            constraints.append((rel, rhs))
        for _ in range(4):
            changed = False
            for normal, rhs in constraints:
                violation = float(np.dot(normal, projected[i]) - rhs)
                if violation > 0.0:
                    projected[i] -= (violation / max(float(np.dot(normal, normal)), 1e-6)) * normal
                    changed = True
            projected[i] = np.clip(projected[i], -2.0, 2.0)
            if not changed:
                break
        if constraints and any(float(np.dot(normal, projected[i]) - rhs) > 1e-3 for normal, rhs in constraints):
            projected[i] = 0.0
            infeasible += 1
    normalized = np.clip(projected / 2.0, -1.0, 1.0)
    interventions = int(np.count_nonzero(np.linalg.norm(normalized - actions, axis=1) > 1e-4))
    return normalized.astype(np.float32), interventions, infeasible


@dataclass
class LoadedPolicy:
    name: str
    model: GraphActorCritic | None
    params: Any | None
    use_filter: bool

    def act(self, obs: dict[str, np.ndarray], key: jax.Array) -> tuple[np.ndarray, int, int]:
        batch = observation_batch(obs)
        if self.model is None:
            goal = batch["nodes"][:, 0, 6:9] * 20.0
            norm = np.linalg.norm(goal, axis=1, keepdims=True)
            actions = np.clip(goal / np.maximum(norm, 1e-6), -1.0, 1.0).astype(np.float32)
        else:
            device_batch = {k: jnp.asarray(v) for k, v in batch.items()}
            actions = np.asarray(policy_apply(self.model, self.params, device_batch, key, True)[0])
        if self.use_filter:
            return cbf_filter(actions, obs["nodes"], obs["mask"])
        return actions, 0, 0


def save_checkpoint(path: Path, params: Any, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(params))
    path.with_suffix(".json").write_text(__import__("json").dumps(metadata, indent=2), encoding="utf-8")


def load_policy(path: Path, name: str, graph_actor: bool, use_filter: bool) -> LoadedPolicy:
    model = GraphActorCritic(graph_actor=graph_actor)
    dummy = {
        "nodes": jnp.zeros((1, ContinuousCityUAVEnv.MAX_NODES, ContinuousCityUAVEnv.NODE_DIM)),
        "mask": jnp.ones((1, ContinuousCityUAVEnv.MAX_NODES)),
        "adjacency": jnp.eye(ContinuousCityUAVEnv.MAX_NODES)[None, ...],
        "global_state": jnp.zeros((1, MAX_GLOBAL_DIM)),
    }
    params = model.init(jax.random.PRNGKey(0), **dummy)["params"]
    params = serialization.from_bytes(params, path.read_bytes())
    return LoadedPolicy(name=name, model=model, params=params, use_filter=use_filter)
