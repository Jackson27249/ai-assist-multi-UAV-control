import numpy as np

from city_uav_env import CityUAVEnv
from safe_benchmark import GoalPolicy, build_policies, residual_penalty_sensitivity_variants


def test_shapes_and_ranges():
    env = CityUAVEnv(seed=7)
    obs = env.reset()
    assert obs.shape == (3, env._obs_dim)
    assert np.isfinite(obs).all()
    assert np.max(np.abs(obs)) <= 1.0
    next_obs, rewards, dones, infos = env.step(np.eye(7)[[1, 1, 1]])
    assert next_obs.shape == obs.shape
    assert rewards.shape == (3, 1)
    assert dones.shape == (3,)
    assert len(infos) == 3


def test_deterministic_seed():
    a, b = CityUAVEnv(seed=19), CityUAVEnv(seed=19)
    np.testing.assert_allclose(a.reset(), b.reset())
    for _ in range(5):
        out_a = a.step(np.eye(7)[[1, 1, 1]])
        out_b = b.step(np.eye(7)[[1, 1, 1]])
        np.testing.assert_allclose(out_a[0], out_b[0])
        np.testing.assert_allclose(out_a[1], out_b[1])


def test_building_is_hard_constraint():
    env = CityUAVEnv(seed=3)
    env.pos[0] = np.asarray(env.buildings[0].center)
    assert env._hard_collision(env._pair_distances(), np.full((3, 2), 10.0))


def test_reward_variant_changes_safety_penalty():
    baseline = CityUAVEnv(seed=5, reward_variant="baseline")
    safe = CityUAVEnv(seed=5, reward_variant="safe")
    baseline.pos[1] = baseline.pos[0] + np.asarray([0.8, 0.0, 0.0])
    safe.pos[1] = safe.pos[0] + np.asarray([0.8, 0.0, 0.0])
    actions = np.eye(7)[[0, 0, 0]]
    _, base_reward, _, _ = baseline.step(actions)
    _, safe_reward, _, _ = safe.step(actions)
    assert float(safe_reward[0, 0]) < float(base_reward[0, 0])


def test_safety_shield_rejects_building_entry():
    env = CityUAVEnv(seed=9)
    building = env.buildings[0]
    env.pos[0] = np.asarray(building.center, dtype=float)
    env.pos[0, 0] -= building.half_extent[0] + 0.60
    env.vel[0] = [0.0, 0.0, 0.0]
    requested = np.asarray([1, 0, 0])
    shielded = env.shield_actions(requested)
    predicted = env._predict_positions(shielded)
    assert shielded[0] != requested[0]
    assert env.is_candidate_safe(0, predicted[0], predicted)


def test_shield_prevents_random_policy_collisions():
    rng = np.random.default_rng(77)
    for seed in range(10):
        env = CityUAVEnv(seed=seed, enforce_shield=True, wind_scale=0.40)
        env.reset()
        for _ in range(env.episode_length):
            indices = rng.integers(0, 7, size=3)
            _, _, dones, infos = env.step(np.eye(7)[indices])
            if np.all(dones):
                break
        assert not infos[0]["collision"], f"seed={seed} shield collision"


def test_variable_agent_count_keeps_local_observation_shape():
    for count in (3, 5, 7, 10):
        env = CityUAVEnv(num_agents=count, seed=31)
        obs = env.reset()
        assert obs.shape == (count, 27)
        next_obs, rewards, dones, infos = env.step(np.eye(7)[np.ones(count, dtype=int)])
        assert next_obs.shape == obs.shape
        assert rewards.shape == (count, 1)
        assert dones.shape == (count,)
        assert len(infos) == count


def test_static_scenario_has_finite_serializable_metrics():
    env = CityUAVEnv(num_agents=5, seed=41, scenario="static")
    env.reset()
    for _ in range(3):
        env.step(np.eye(7)[np.ones(5, dtype=int)])
    metrics = env.episode_metrics(terminal=False)
    assert metrics["scenario"] == "static"
    assert np.isfinite(metrics["min_separation"])


def test_collision_type_distinguishes_uav_contact():
    env = CityUAVEnv(seed=43)
    env.pos[1] = env.pos[0] + np.asarray([0.1, 0.0, 0.0])
    dynamic = env._dynamic_distances(env.pos)
    assert env._collision_type(env._pair_distances(), dynamic) == "uav_uav"


def test_safety_cost_sign_and_safe_distance_budget():
    env = CityUAVEnv(seed=47, scenario="static", safe_distance=1.2)
    env.pos[:] = np.asarray([[-8.0, -4.0, 4.0], [-6.0, -4.0, 4.0], [-8.0, -2.0, 4.0]])
    assert env.safety_cost(env.pos) < 0.0
    env.pos[1] = env.pos[0] + np.asarray([0.8, 0.0, 0.0])
    assert env.safety_cost(env.pos) > 0.0


def test_nominal_only_controller_has_no_safety_interventions():
    env = CityUAVEnv(seed=53, scenario="dynamic")
    policy = GoalPolicy()
    policy.reset(env.num_agents)
    actions = policy.act(env, env.reset())
    assert actions.shape == (env.num_agents,)
    assert np.all((0 <= actions) & (actions < len(env.ACTIONS)))
    assert policy.last_interventions == 0


def test_sensitivity_variants_change_one_residual_penalty_parameter_at_a_time():
    variants = residual_penalty_sensitivity_variants()
    assert len(variants) == 11
    reference = variants[0]
    parameters = ("progress_weight", "cbf_weight", "cbf_eps", "deviation_weight", "clearance_weight")
    for variant in variants[1:]:
        changed = [key for key in parameters if getattr(variant, key) != getattr(reference, key)]
        assert len(changed) == 1
        assert getattr(variant, changed[0]) in {
            0.5 * getattr(reference, changed[0]),
            1.5 * getattr(reference, changed[0]),
        }


def test_explicit_policy_selection_includes_nominal_only():
    names = [policy.name for policy in build_policies(["nominal_only", "residual_filter"], None, False)]
    assert names == ["nominal_only", "residual_filter"]
