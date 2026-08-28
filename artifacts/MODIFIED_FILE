from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Building:
    center: tuple[float, float, float]
    half_extent: tuple[float, float, float]


SCENARIOS = {
    "static": (0.10, 0.05),
    "dynamic": (0.25, 0.08),
    "dense": (0.30, 0.10),
    "wind": (0.65, 0.15),
    "dropout": (0.40, 0.45),
    "ood_dense": (0.45, 0.25),
}


class ContinuousCityUAVEnv:
    """Continuous multi-UAV environment with simultaneous decentralized actions."""

    NODE_DIM = 15
    MAX_NODES = 21
    CONTROL_DIM = 3

    def __init__(
        self,
        num_agents: int = 3,
        seed: int = 1,
        scenario: str = "dynamic",
        episode_length: int = 128,
        safe_distance: float = 1.2,
        sensor_radius: float = 6.0,
    ) -> None:
        if num_agents < 2 or num_agents > 12:
            raise ValueError("num_agents must be in [2, 12]")
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        self.num_agents = num_agents
        self.seed_value = int(seed)
        self.scenario = scenario
        self.episode_length = int(episode_length)
        self.safe_distance = float(safe_distance)
        self.sensor_radius = float(sensor_radius)
        self.dt = 0.25
        self.max_speed = 2.0
        self.max_accel = 2.0
        self.collision_distance = 0.55
        self.goal_radius = 0.65
        self.world_low = np.asarray([-10.0, -8.0, 0.8], dtype=np.float32)
        self.world_high = np.asarray([10.0, 8.0, 8.0], dtype=np.float32)
        self.wind_scale, self.comm_dropout = SCENARIOS[scenario]
        self.rng = np.random.default_rng(self.seed_value)
        self.buildings = self._buildings(scenario)
        self.reset()

    @staticmethod
    def _buildings(scenario: str) -> list[Building]:
        base = [
            Building((-3.5, -1.7, 2.4), (1.0, 1.4, 1.8)),
            Building((3.3, 1.6, 2.8), (1.2, 1.3, 2.2)),
            Building((-1.0, 4.5, 3.0), (1.3, 0.9, 2.4)),
            Building((1.2, -4.6, 2.7), (1.2, 0.9, 2.1)),
        ]
        if scenario in {"dense", "ood_dense"}:
            base.extend(
                [
                    Building((-0.4, -1.5, 2.4), (0.8, 1.1, 1.8)),
                    Building((0.5, 1.4, 3.0), (0.8, 1.0, 2.4)),
                    Building((0.0, 4.0, 2.4), (0.9, 0.8, 1.8)),
                ]
            )
        if scenario == "ood_dense":
            base.append(Building((4.8, -2.7, 3.2), (0.9, 1.4, 2.6)))
        return base

    def reset(self) -> dict[str, np.ndarray]:
        self.rng = np.random.default_rng(self.seed_value)
        angles = np.linspace(0.0, 2.0 * np.pi, self.num_agents, endpoint=False)
        radii = np.where(np.arange(self.num_agents) % 2 == 0, 7.0, 6.2)
        self.pos = np.stack(
            [radii * np.cos(angles), radii * np.sin(angles), 2.4 + 0.35 * (np.arange(self.num_agents) % 4)],
            axis=1,
        ).astype(np.float32)
        self.pos += self.rng.normal(0.0, 0.18, self.pos.shape).astype(np.float32)
        self.goals = self.pos.copy()
        self.goals[:, :2] *= -1.0
        self.goals[:, 2] = np.clip(5.5 - self.pos[:, 2] * 0.35, 2.2, 5.0)
        self.vel = np.zeros_like(self.pos)
        self.wind = np.zeros(3, dtype=np.float32)
        if self.scenario == "static":
            self.dynamic_pos = np.zeros((0, 3), dtype=np.float32)
            self.dynamic_vel = np.zeros((0, 3), dtype=np.float32)
        else:
            self.dynamic_pos = np.asarray([[0.0, -6.2, 2.6], [1.0, 6.0, 4.0]], dtype=np.float32)
            self.dynamic_vel = np.asarray([[0.0, 0.65, 0.0], [0.0, -0.55, 0.0]], dtype=np.float32)
        self.step_count = 0
        self.done = False
        self.collision = False
        self.collision_type = "none"
        self.path_length = np.zeros(self.num_agents, dtype=np.float64)
        self.energy = np.zeros(self.num_agents, dtype=np.float64)
        self.min_pair_distance = float("inf")
        self.min_dynamic_distance = float("inf")
        self.min_static_clearance = float("inf")
        self.comm_packets = 0
        self.comm_drops = 0
        self.trajectory = [self.pos.copy()]
        return self.observe(privileged=False)

    def _point_static_clearance(self, point: np.ndarray) -> float:
        boundary = float(np.min(np.concatenate((point - self.world_low, self.world_high - point))))
        clearances = [boundary]
        for building in self.buildings:
            center = np.asarray(building.center, dtype=np.float32)
            extent = np.asarray(building.half_extent, dtype=np.float32)
            delta = np.abs(point - center) - extent
            outside = np.maximum(delta, 0.0)
            if np.any(delta > 0.0):
                clearances.append(float(np.linalg.norm(outside)))
            else:
                clearances.append(float(np.max(delta)))
        return min(clearances)

    def _closest_static_vectors(self, point: np.ndarray, limit: int = 6) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for building in self.buildings:
            center = np.asarray(building.center, dtype=np.float32)
            extent = np.asarray(building.half_extent, dtype=np.float32)
            closest = np.clip(point, center - extent, center + extent)
            vector = closest - point
            if np.linalg.norm(vector) < 1e-6:
                delta = point - center
                axis = int(np.argmax(np.abs(delta / np.maximum(extent, 1e-6))))
                face = center.copy()
                face[axis] += np.sign(delta[axis] or 1.0) * extent[axis]
                vector = face - point
            vectors.append(vector.astype(np.float32))
        vectors.sort(key=lambda x: float(np.linalg.norm(x)))
        return vectors[:limit]

    @staticmethod
    def _node(rel_pos: np.ndarray, rel_vel: np.ndarray, goal: np.ndarray, node_type: int, received: float, age: float) -> np.ndarray:
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[node_type] = 1.0
        return np.concatenate(
            [
                np.clip(rel_pos / 10.0, -1.0, 1.0),
                np.clip(rel_vel / 2.0, -1.0, 1.0),
                np.clip(goal / 20.0, -1.0, 1.0),
                one_hot,
                np.asarray([received, age], dtype=np.float32),
            ]
        ).astype(np.float32)

    def observe(self, privileged: bool) -> dict[str, np.ndarray]:
        nodes = np.zeros((self.num_agents, self.MAX_NODES, self.NODE_DIM), dtype=np.float32)
        masks = np.zeros((self.num_agents, self.MAX_NODES), dtype=np.float32)
        adjacency = np.zeros((self.num_agents, self.MAX_NODES, self.MAX_NODES), dtype=np.float32)
        for i in range(self.num_agents):
            row = [self._node(np.zeros(3), self.vel[i], self.goals[i] - self.pos[i], 0, 1.0, 0.0)]
            candidates = [j for j in range(self.num_agents) if j != i]
            candidates.sort(key=lambda j: float(np.linalg.norm(self.pos[j] - self.pos[i])))
            for j in candidates:
                distance = float(np.linalg.norm(self.pos[j] - self.pos[i]))
                if not privileged and distance > self.sensor_radius:
                    continue
                self.comm_packets += int(not privileged)
                received = privileged or self.rng.random() >= self.comm_dropout
                self.comm_drops += int(not privileged and not received)
                if not received:
                    continue
                row.append(self._node(self.pos[j] - self.pos[i], self.vel[j] - self.vel[i], np.zeros(3), 1, 1.0, 0.0))
            for obstacle, velocity in zip(self.dynamic_pos, self.dynamic_vel):
                rel = obstacle - self.pos[i]
                if privileged or np.linalg.norm(rel) <= self.sensor_radius:
                    row.append(self._node(rel, velocity - self.vel[i], np.zeros(3), 2, 1.0, 0.0))
            for rel in self._closest_static_vectors(self.pos[i]):
                if privileged or np.linalg.norm(rel) <= self.sensor_radius:
                    row.append(self._node(rel, -self.vel[i], np.zeros(3), 3, 1.0, 0.0))
            row = row[: self.MAX_NODES]
            count = len(row)
            nodes[i, :count] = np.asarray(row)
            masks[i, :count] = 1.0
            adjacency[i, 0, :count] = 1.0
            adjacency[i, :count, 0] = 1.0
            adjacency[i, np.arange(count), np.arange(count)] = 1.0
        global_state = np.concatenate(
            [self.pos.reshape(-1) / 10.0, self.vel.reshape(-1) / self.max_speed, self.goals.reshape(-1) / 10.0]
        ).astype(np.float32)
        return {"nodes": nodes, "mask": masks, "adjacency": adjacency, "global_state": global_state}

    def _collision_state(self) -> tuple[bool, str]:
        if self.num_agents > 1:
            delta = self.pos[:, None, :] - self.pos[None, :, :]
            matrix = np.linalg.norm(delta, axis=-1)
            pair = matrix[np.triu_indices(self.num_agents, 1)]
            self.min_pair_distance = min(self.min_pair_distance, float(np.min(pair)))
            if np.any(pair < self.collision_distance):
                return True, "uav_uav"
        if len(self.dynamic_pos):
            distances = np.linalg.norm(self.pos[:, None, :] - self.dynamic_pos[None, :, :], axis=-1)
            self.min_dynamic_distance = min(self.min_dynamic_distance, float(np.min(distances)))
            if np.any(distances < self.collision_distance):
                return True, "dynamic_obstacle"
        static = [self._point_static_clearance(point) for point in self.pos]
        self.min_static_clearance = min(self.min_static_clearance, min(static))
        if min(static) < 0.0:
            return True, "static_or_boundary"
        return False, "none"

    def step(self, actions: np.ndarray, privileged_observation: bool = False) -> tuple[dict[str, np.ndarray], np.ndarray, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("step called after terminal state")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_agents, self.CONTROL_DIM):
            raise ValueError(f"expected {(self.num_agents, self.CONTROL_DIM)}, got {actions.shape}")
        actions = np.clip(actions, -1.0, 1.0)
        before = self.pos.copy()
        goal_before = np.linalg.norm(self.goals - before, axis=1)
        self.wind = (0.92 * self.wind + self.rng.normal(0.0, self.wind_scale, 3)).astype(np.float32)
        desired_vel = actions * self.max_speed
        delta_v = np.clip(desired_vel - self.vel, -self.max_accel * self.dt, self.max_accel * self.dt)
        self.vel = np.clip(self.vel + delta_v, -self.max_speed, self.max_speed)
        self.pos = self.pos + (self.vel + 0.18 * self.wind) * self.dt
        self.dynamic_pos = self.dynamic_pos + self.dynamic_vel * self.dt
        self.step_count += 1
        self.path_length += np.linalg.norm(self.pos - before, axis=1)
        self.energy += np.sum(delta_v**2, axis=1)
        collision, collision_type = self._collision_state()
        self.collision = self.collision or collision
        if collision:
            self.collision_type = collision_type
        goal_after = np.linalg.norm(self.goals - self.pos, axis=1)
        reached = goal_after <= self.goal_radius
        team_success = bool(np.all(reached) and not self.collision)
        timeout = self.step_count >= self.episode_length
        self.done = bool(collision or team_success or timeout)
        reward = 4.0 * (goal_before - goal_after) - 0.02 * np.sum(actions**2, axis=1)
        reward -= 30.0 * float(collision)
        reward += 15.0 * reached.astype(np.float32)
        self.trajectory.append(self.pos.copy())
        info = self.metrics(team_success=team_success, timeout=timeout)
        return self.observe(privileged=privileged_observation), reward.astype(np.float32), self.done, info

    def metrics(self, team_success: bool | None = None, timeout: bool | None = None) -> dict[str, Any]:
        distances = np.linalg.norm(self.goals - self.pos, axis=1)
        if team_success is None:
            team_success = bool(np.all(distances <= self.goal_radius) and not self.collision)
        if timeout is None:
            timeout = bool(self.step_count >= self.episode_length and not team_success and not self.collision)
        return {
            "seed": self.seed_value,
            "num_agents": self.num_agents,
            "scenario": self.scenario,
            "steps": self.step_count,
            "team_success": bool(team_success),
            "collision": bool(self.collision),
            "collision_type": self.collision_type,
            "timeout": bool(timeout),
            "mean_path_length": float(np.mean(self.path_length)),
            "total_energy": float(np.sum(self.energy)),
            "min_pair_distance": float(self.min_pair_distance),
            "min_dynamic_distance": float(self.min_dynamic_distance),
            "min_static_clearance": float(self.min_static_clearance),
            "mean_goal_distance": float(np.mean(distances)),
            "comm_drop_rate": self.comm_drops / max(1, self.comm_packets),
        }

