from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class Building:
    center: tuple[float, float, float]
    half_extent: tuple[float, float, float]


class CityUAVEnv:
    """多无人机城市低空搜索环境，提供官方 on-policy 所需的旧式多智能体接口。"""

    ACTIONS = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        num_agents: int = 3,
        episode_length: int = 128,
        seed: int = 1,
        reward_variant: str = "baseline",
        wind_scale: float = 0.25,
        comm_dropout: float = 0.08,
        enforce_shield: bool = False,
        scenario: str = "dynamic",
        safe_distance: float = 1.2,
    ) -> None:
        if num_agents < 2:
            raise ValueError("num_agents 必须至少为 2")
        if reward_variant not in {"baseline", "safe"}:
            raise ValueError("reward_variant 必须为 baseline 或 safe")
        if scenario not in {"static", "dynamic", "dense", "wind", "dropout"}:
            raise ValueError("不支持的 scenario")
        self.n = self.num_agents = num_agents
        self.episode_length = episode_length
        self.reward_variant = reward_variant
        self.wind_scale = float(wind_scale)
        self.comm_dropout = float(comm_dropout)
        self.enforce_shield = bool(enforce_shield)
        self.scenario = scenario
        self.dt = 0.25
        self.max_speed = 2.0
        self.max_accel = 2.0
        self.safe_distance = float(safe_distance)
        self.collision_distance = 0.55
        self.world_low = np.asarray([-10.0, -8.0, 0.8], dtype=np.float32)
        self.world_high = np.asarray([10.0, 8.0, 8.0], dtype=np.float32)
        self.starts, self.goals = self._make_endpoints(num_agents)
        standard_buildings = (
            Building((-2.4, -2.8, 2.8), (1.3, 1.5, 2.8)),
            Building((-1.0, 3.4, 3.7), (1.6, 1.2, 3.7)),
            Building((3.0, -3.6, 3.1), (1.2, 1.2, 3.1)),
            Building((3.2, 2.4, 4.5), (1.4, 1.6, 4.5)),
        )
        dense_extra = (
            Building((-5.1, 1.2, 2.2), (0.9, 1.0, 2.2)),
            Building((0.8, -0.2, 2.4), (0.8, 1.0, 2.4)),
            Building((5.7, -0.5, 3.3), (0.9, 1.1, 3.3)),
        )
        self.buildings = standard_buildings + (dense_extra if scenario == "dense" else ())
        # 固定两个局部邻居槽，使 N=3 训练策略可在更大机群上零样本评估。
        self.max_observed_neighbors = 2
        self._obs_dim = 3 + 3 + 3 + 3 + self.max_observed_neighbors * 4 + 6 + 1
        obs_box = spaces.Box(-1.0, 1.0, shape=(self._obs_dim,), dtype=np.float32)
        share_box = spaces.Box(
            -1.0, 1.0, shape=(self._obs_dim * self.num_agents,), dtype=np.float32
        )
        self.observation_space = [obs_box for _ in range(self.num_agents)]
        self.share_observation_space = [share_box for _ in range(self.num_agents)]
        self.action_space = [spaces.Discrete(len(self.ACTIONS)) for _ in range(self.num_agents)]
        self.seed(seed)
        self.reset()

    @staticmethod
    def _make_endpoints(num_agents: int) -> tuple[np.ndarray, np.ndarray]:
        if num_agents == 3:
            starts = [[-8.5, -6.0, 1.6], [-8.5, 0.0, 2.3], [-8.5, 6.0, 3.0]]
            goals = [[8.5, 6.0, 2.4], [8.5, 0.0, 3.4], [8.5, -6.0, 4.2]]
            return np.asarray(starts, dtype=np.float32), np.asarray(goals, dtype=np.float32)
        y = np.linspace(-6.0, 6.0, num_agents, dtype=np.float32)
        z = 1.6 + 0.55 * (np.arange(num_agents) % 5)
        starts = np.column_stack((np.full(num_agents, -8.5), y, z))
        goals = np.column_stack((np.full(num_agents, 8.5), y[::-1], 2.2 + z[::-1] * 0.45))
        return starts.astype(np.float32), goals.astype(np.float32)

    def seed(self, seed: int | None = None) -> list[int]:
        self._seed = int(0 if seed is None else seed)
        self.rng = np.random.default_rng(self._seed)
        return [self._seed]

    def reset(self) -> np.ndarray:
        jitter = self.rng.normal(0.0, 0.08, self.starts.shape).astype(np.float32)
        self.pos = self.starts + jitter
        self.vel = np.zeros_like(self.pos)
        self.wind = np.zeros(3, dtype=np.float32)
        if self.scenario == "static":
            self.dynamic_pos = np.empty((0, 3), dtype=np.float32)
            self.dynamic_vel = np.empty((0, 3), dtype=np.float32)
        else:
            self.dynamic_pos = np.asarray(
                [[0.0, -6.2, 2.6], [1.0, 6.0, 4.0]], dtype=np.float32
            )
            self.dynamic_vel = np.asarray(
                [[0.0, 0.65, 0.0], [0.0, -0.55, 0.0]], dtype=np.float32
            )
        self.step_count = 0
        self.success = np.zeros(self.num_agents, dtype=bool)
        self.collision = False
        self.collision_type = "none"
        self.total_energy = 0.0
        self.total_reward = 0.0
        self.cumulative_safety_cost = 0.0
        self.max_safety_cost = 0.0
        self.near_miss_steps = 0
        self.interventions = 0
        self.path_length = np.zeros(self.num_agents, dtype=np.float64)
        self.comm_packets = 0
        self.comm_drops = 0
        self.min_pair_distance = float("inf")
        self.min_dynamic_distance = float("inf")
        self.min_static_clearance = float("inf")
        return self._observations()

    def close(self) -> None:
        return None

    def _decode_actions(self, actions: Iterable[np.ndarray | int]) -> np.ndarray:
        decoded: list[int] = []
        for action in actions:
            arr = np.asarray(action)
            decoded.append(int(arr.item()) if arr.size == 1 else int(np.argmax(arr)))
        out = np.asarray(decoded, dtype=np.int64)
        if np.any((out < 0) | (out >= len(self.ACTIONS))):
            raise ValueError("动作越界")
        return out

    def step(self, actions: Iterable[np.ndarray | int]):
        action_idx = self._decode_actions(actions)
        # 先采样本步扰动，再进行安全投影，保证预测状态与积分状态使用同一风场。
        self.wind = (0.92 * self.wind + self.rng.normal(0.0, self.wind_scale, 3)).astype(
            np.float32
        )
        if self.enforce_shield:
            requested = action_idx.copy()
            action_idx = self.shield_actions(action_idx)
            self.interventions += int(np.count_nonzero(requested != action_idx))
        previous = self.pos.copy()
        previous_distance = np.linalg.norm(self.goals - self.pos, axis=1)

        desired_vel = self.ACTIONS[action_idx] * self.max_speed
        delta_v = np.clip(desired_vel - self.vel, -self.max_accel * self.dt, self.max_accel * self.dt)
        self.vel = np.clip(self.vel + delta_v, -self.max_speed, self.max_speed)
        self.pos = self.pos + (self.vel + 0.18 * self.wind) * self.dt
        if self.enforce_shield:
            self._project_state_safe()
        self.step_count += 1

        if len(self.dynamic_pos):
            self.dynamic_pos += self.dynamic_vel * self.dt
            flip = np.abs(self.dynamic_pos[:, 1]) > 6.4
            self.dynamic_vel[flip, 1] *= -1.0

        segment = np.linalg.norm(self.pos - previous, axis=1)
        self.path_length += segment
        energy = float(np.sum(np.square(self.vel)) * self.dt)
        self.total_energy += energy
        current_distance = np.linalg.norm(self.goals - self.pos, axis=1)
        progress = previous_distance - current_distance

        pair_dist = self._pair_distances()
        dynamic_dist = self._dynamic_distances(self.pos)
        pair_min = float(np.min(pair_dist)) if pair_dist.size else float("inf")
        dynamic_min = float(np.min(dynamic_dist)) if dynamic_dist.size else float("inf")
        static_min = min(self.static_clearance(point) for point in self.pos)
        self.min_pair_distance = min(self.min_pair_distance, pair_min)
        self.min_dynamic_distance = min(self.min_dynamic_distance, dynamic_min)
        self.min_static_clearance = min(self.min_static_clearance, static_min)
        min_clearance = min(pair_min, dynamic_min, static_min)
        safety_cost = max(0.0, self.safe_distance - min_clearance) / self.safe_distance
        self.cumulative_safety_cost += safety_cost
        self.max_safety_cost = max(self.max_safety_cost, safety_cost)
        near_miss = bool(
            (pair_dist.size and pair_min < self.safe_distance)
            or (dynamic_dist.size and dynamic_min < self.safe_distance)
            or static_min < self.safe_distance * 0.60
        )
        self.near_miss_steps += int(near_miss)
        self.collision_type = self._collision_type(pair_dist, dynamic_dist)
        self.collision = self.collision_type != "none"
        newly_success = (current_distance < 0.75) & (~self.success)
        self.success |= current_distance < 0.75

        reward = 1.8 * progress - 0.02 - 0.003 * np.square(self.vel).sum(axis=1)
        reward += newly_success.astype(np.float32) * 12.0
        if self.reward_variant == "safe":
            reward -= 1.2 * float(near_miss)
            reward -= 0.002 * energy
            reward += 0.12 * np.min(progress)
        if self.collision:
            reward -= 25.0 if self.reward_variant == "safe" else 12.0
        self.total_reward += float(np.mean(reward))

        timed_out = self.step_count >= self.episode_length
        done_all = bool(self.collision or np.all(self.success) or timed_out)
        dones = np.full(self.num_agents, done_all, dtype=bool)
        shared_reward = float(0.75 * np.mean(reward) + 0.25 * np.min(reward))
        rewards = np.full((self.num_agents, 1), shared_reward, dtype=np.float32)
        metrics = self.episode_metrics(done_all)
        infos = [
            {
                "individual_reward": float(reward[i]),
                "collision": self.collision,
                "success": bool(self.success[i]),
                "episode_metrics": metrics if done_all else None,
            }
            for i in range(self.num_agents)
        ]
        return self._observations(), rewards, dones, infos

    def episode_metrics(self, terminal: bool = True) -> dict[str, float | int | bool]:
        packet_total = max(1, self.comm_packets)
        finite_cap = float(np.linalg.norm(self.world_high - self.world_low))
        min_pair = min(self.min_pair_distance, finite_cap)
        min_dynamic = min(self.min_dynamic_distance, finite_cap)
        min_static = min(self.min_static_clearance, finite_cap)
        return {
            "terminal": bool(terminal),
            "team_success": bool(np.all(self.success) and not self.collision),
            "collision": bool(self.collision),
            "collision_type": self.collision_type,
            "steps": int(self.step_count),
            "mean_path_length": float(np.mean(self.path_length)),
            "total_energy": float(self.total_energy),
            "episode_return": float(self.total_reward),
            "mean_safety_cost": float(self.cumulative_safety_cost / max(1, self.step_count)),
            "max_safety_cost": float(self.max_safety_cost),
            "near_miss_rate": float(self.near_miss_steps / max(1, self.step_count)),
            "intervention_rate": float(
                self.interventions / max(1, self.step_count * self.num_agents)
            ),
            "comm_drop_rate": float(self.comm_drops / packet_total),
            "min_pair_distance": float(min_pair),
            "min_dynamic_distance": float(min_dynamic),
            "min_static_clearance": float(min_static),
            "min_separation": float(min(min_pair, min_dynamic, min_static)),
            "mean_goal_distance": float(np.mean(np.linalg.norm(self.goals - self.pos, axis=1))),
            "scenario": self.scenario,
            "num_agents": self.num_agents,
        }

    def _hard_collision(self, pair_dist: np.ndarray, dynamic_dist: np.ndarray) -> bool:
        return self._collision_type(pair_dist, dynamic_dist) != "none"

    def _collision_type(self, pair_dist: np.ndarray, dynamic_dist: np.ndarray) -> str:
        if pair_dist.size and np.min(pair_dist) < self.collision_distance:
            return "uav_uav"
        if dynamic_dist.size and np.min(dynamic_dist) < self.collision_distance:
            return "dynamic_obstacle"
        if any(self._point_static_collision(point, margin=0.25) for point in self.pos):
            return "static_or_boundary"
        return "none"

    def _point_static_collision(self, point: np.ndarray, margin: float) -> bool:
        if np.any(point < self.world_low + margin) or np.any(point > self.world_high - margin):
            return True
        for building in self.buildings:
            center = np.asarray(building.center)
            extent = np.asarray(building.half_extent) + margin
            if np.all(np.abs(point - center) <= extent):
                return True
        return False

    def static_clearance(self, point: np.ndarray) -> float:
        """返回点到边界或建筑 AABB 的最小有符号距离，正值表示可行。"""
        boundary = float(np.min(np.concatenate((point - self.world_low, self.world_high - point))))
        clearances = [boundary]
        for building in self.buildings:
            center = np.asarray(building.center)
            extent = np.asarray(building.half_extent) + 0.25
            q = np.abs(point - center) - extent
            signed = float(np.linalg.norm(np.maximum(q, 0.0)) + min(float(np.max(q)), 0.0))
            clearances.append(signed)
        return min(clearances)

    def _dynamic_distances(self, positions: np.ndarray) -> np.ndarray:
        if not len(self.dynamic_pos):
            return np.empty((self.num_agents, 0), dtype=np.float32)
        return np.linalg.norm(positions[:, None, :] - self.dynamic_pos[None, :, :], axis=-1)

    def minimum_clearance(self, positions: np.ndarray) -> float:
        diff = positions[:, None, :] - positions[None, :, :]
        matrix = np.linalg.norm(diff, axis=-1)
        pair = matrix[np.triu_indices(self.num_agents, 1)]
        pair_min = float(np.min(pair)) if pair.size else float("inf")
        dynamic = self._dynamic_distances(positions)
        dynamic_min = float(np.min(dynamic)) if dynamic.size else float("inf")
        static_min = min(self.static_clearance(point) for point in positions)
        return min(pair_min, dynamic_min, static_min)

    def safety_cost(self, positions: np.ndarray) -> float:
        diff = positions[:, None, :] - positions[None, :, :]
        matrix = np.linalg.norm(diff, axis=-1)
        pair = matrix[np.triu_indices(self.num_agents, 1)]
        pair_min = float(np.min(pair)) if pair.size else float("inf")
        dynamic = self._dynamic_distances(positions)
        dynamic_min = float(np.min(dynamic)) if dynamic.size else float("inf")
        static_min = min(self.static_clearance(point) for point in positions)
        static_budget = self.safe_distance * 0.60
        components = [
            (self.safe_distance - pair_min) / self.safe_distance,
            (self.safe_distance - dynamic_min) / self.safe_distance,
            (static_budget - static_min) / static_budget,
        ]
        return float(np.clip(max(components), -1.0, 1.0))

    def _predict_positions(self, action_idx: np.ndarray) -> np.ndarray:
        desired_vel = self.ACTIONS[action_idx] * self.max_speed
        delta_v = np.clip(
            desired_vel - self.vel,
            -self.max_accel * self.dt,
            self.max_accel * self.dt,
        )
        predicted_vel = np.clip(self.vel + delta_v, -self.max_speed, self.max_speed)
        return self.pos + (predicted_vel + 0.18 * self.wind) * self.dt

    def _project_state_safe(self) -> None:
        """动作集无可行制动解时的紧急不变集投影。"""
        margin = 0.27
        lower, upper = self.world_low + margin, self.world_high - margin
        for _ in range(12):
            clipped = np.clip(self.pos, lower, upper)
            boundary_axes = ~np.isclose(clipped, self.pos)
            self.pos = clipped
            self.vel[boundary_axes] = 0.0
            for i in range(self.num_agents):
                for building in self.buildings:
                    center = np.asarray(building.center)
                    extent = np.asarray(building.half_extent) + margin
                    delta = self.pos[i] - center
                    if np.all(np.abs(delta) <= extent):
                        penetration = extent - np.abs(delta)
                        axis = int(np.argmin(penetration))
                        sign = 1.0 if delta[axis] >= 0.0 else -1.0
                        self.pos[i, axis] = center[axis] + sign * (extent[axis] + 1e-3)
                        self.vel[i, axis] = 0.0

            dynamic_future = self.dynamic_pos + self.dynamic_vel * self.dt
            for i in range(self.num_agents):
                for obstacle in dynamic_future:
                    delta = self.pos[i] - obstacle
                    distance = float(np.linalg.norm(delta))
                    if distance < self.collision_distance + 0.02:
                        direction = delta / max(distance, 1e-6)
                        if distance < 1e-6:
                            direction = np.asarray([1.0, 0.0, 0.0])
                        self.pos[i] = obstacle + direction * (self.collision_distance + 0.02)
                        self.vel[i] -= np.dot(self.vel[i], direction) * direction

            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    delta = self.pos[i] - self.pos[j]
                    distance = float(np.linalg.norm(delta))
                    target = self.collision_distance + 0.10
                    if distance < target:
                        direction = delta / max(distance, 1e-6)
                        if distance < 1e-6:
                            direction = np.asarray([1.0, 0.0, 0.0])
                        correction = 0.5 * (target - distance + 1e-3) * direction
                        self.pos[i] += correction
                        self.pos[j] -= correction
                        relative_normal = np.dot(self.vel[i] - self.vel[j], direction)
                        if relative_normal < 0:
                            self.vel[i] -= 0.5 * relative_normal * direction
                            self.vel[j] += 0.5 * relative_normal * direction

    def is_candidate_safe(
        self, agent: int, candidate: np.ndarray, joint_candidates: np.ndarray | None = None
    ) -> bool:
        if self._point_static_collision(candidate, margin=self.safe_distance * 0.60):
            return False
        dynamic_future = self.dynamic_pos + self.dynamic_vel * self.dt
        if len(dynamic_future) and np.min(np.linalg.norm(candidate - dynamic_future, axis=1)) < self.safe_distance:
            return False
        others = self.pos if joint_candidates is None else joint_candidates
        for j in range(self.num_agents):
            if j != agent and np.linalg.norm(candidate - others[j]) < self.safe_distance:
                return False
        return True

    def shield_actions(self, requested: np.ndarray) -> np.ndarray:
        """将策略动作投影到单步可行集；硬安全不依赖 reward 抵消。"""
        selected = np.asarray(requested, dtype=np.int64).copy()
        candidates = self._predict_positions(selected)
        order = np.argsort(np.linalg.norm(self.goals - self.pos, axis=1))
        for _ in range(self.num_agents + 1):
            changed = False
            for agent in order:
                if self.is_candidate_safe(agent, candidates[agent], candidates):
                    continue
                best_action, best_score = 0, -np.inf
                for action_id in range(len(self.ACTIONS)):
                    trial_actions = selected.copy()
                    trial_actions[agent] = action_id
                    trial_positions = self._predict_positions(trial_actions)
                    candidate = trial_positions[agent]
                    if not self.is_candidate_safe(agent, candidate, trial_positions):
                        continue
                    progress = np.linalg.norm(self.goals[agent] - self.pos[agent]) - np.linalg.norm(
                        self.goals[agent] - candidate
                    )
                    dynamic_future = self.dynamic_pos + self.dynamic_vel * self.dt
                    values = [
                        np.linalg.norm(candidate - trial_positions[j])
                        for j in range(self.num_agents)
                        if j != agent
                    ]
                    if len(dynamic_future):
                        values.append(float(np.min(np.linalg.norm(candidate - dynamic_future, axis=1))))
                    values.append(self.static_clearance(candidate))
                    clearance = min(values)
                    score = 8.0 * progress + 0.08 * min(clearance, 4.0) - 0.02 * action_id
                    if score > best_score:
                        best_score, best_action = score, action_id
                if selected[agent] != best_action:
                    changed = True
                selected[agent] = best_action
                candidates = self._predict_positions(selected)
            if not changed:
                break
        return selected

    def _pair_distances(self) -> np.ndarray:
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        matrix = np.linalg.norm(diff, axis=-1)
        return matrix[np.triu_indices(self.num_agents, 1)]

    def _axis_clearance(self, point: np.ndarray) -> np.ndarray:
        clear = np.concatenate((self.world_high - point, point - self.world_low))
        directions = np.vstack((np.eye(3), -np.eye(3)))
        for building in self.buildings:
            center = np.asarray(building.center)
            extent = np.asarray(building.half_extent) + 0.25
            for k, direction in enumerate(directions):
                axis = k % 3
                other = [j for j in range(3) if j != axis]
                if np.all(np.abs(point[other] - center[other]) <= extent[other]):
                    face = center[axis] - np.sign(direction[axis]) * extent[axis]
                    distance = (face - point[axis]) * direction[axis]
                    if distance >= 0:
                        clear[k] = min(clear[k], distance)
        return np.clip(clear / 10.0, 0.0, 1.0).astype(np.float32)

    def _observations(self) -> np.ndarray:
        observations = []
        for i in range(self.num_agents):
            neighbours = []
            candidates = [j for j in range(self.num_agents) if j != i]
            if self.num_agents > 3:
                candidates.sort(key=lambda j: float(np.linalg.norm(self.pos[j] - self.pos[i])))
            for j in candidates[: self.max_observed_neighbors]:
                self.comm_packets += 1
                received = self.rng.random() >= self.comm_dropout
                self.comm_drops += int(not received)
                rel = np.clip((self.pos[j] - self.pos[i]) / 20.0, -1.0, 1.0)
                neighbours.extend(rel.tolist() if received else [0.0, 0.0, 0.0])
                neighbours.append(float(received))
            while len(neighbours) < self.max_observed_neighbors * 4:
                neighbours.extend([0.0, 0.0, 0.0, 0.0])
            obs = np.concatenate(
                [
                    self.pos[i] / np.asarray([10.0, 8.0, 8.0]),
                    self.vel[i] / self.max_speed,
                    np.clip((self.goals[i] - self.pos[i]) / 20.0, -1.0, 1.0),
                    np.clip(self.wind / 3.0, -1.0, 1.0),
                    np.asarray(neighbours, dtype=np.float32),
                    self._axis_clearance(self.pos[i]),
                    np.asarray([1.0 - self.step_count / self.episode_length], dtype=np.float32),
                ]
            )
            observations.append(np.clip(obs, -1.0, 1.0).astype(np.float32))
        return np.asarray(observations, dtype=np.float32)
