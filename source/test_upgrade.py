from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from continuous_city_env import ContinuousCityUAVEnv
from models import GraphActorCritic, cbf_filter, observation_batch, policy_apply


def test_continuous_environment_shapes_and_bounds():
    env = ContinuousCityUAVEnv(num_agents=5, seed=7, scenario="dynamic")
    obs = env.observe(privileged=False)
    assert obs["nodes"].shape == (5, env.MAX_NODES, env.NODE_DIM)
    assert obs["adjacency"].shape == (5, env.MAX_NODES, env.MAX_NODES)
    next_obs, reward, done, metrics = env.step(np.zeros((5, 3), dtype=np.float32))
    assert next_obs["nodes"].shape == obs["nodes"].shape
    assert reward.shape == (5,)
    assert isinstance(done, bool)
    assert metrics["num_agents"] == 5
    assert np.all(env.vel <= env.max_speed)


def test_local_observation_contains_no_out_of_range_neighbor():
    env = ContinuousCityUAVEnv(num_agents=3, seed=11, scenario="static", sensor_radius=1.0)
    env.pos = np.asarray([[-8.0, -7.0, 2.0], [0.0, 0.0, 2.0], [8.0, 7.0, 2.0]], dtype=np.float32)
    local = env.observe(privileged=False)
    privileged = env.observe(privileged=True)
    local_types = np.argmax(local["nodes"][0, :, 9:13], axis=-1)
    privileged_types = np.argmax(privileged["nodes"][0, :, 9:13], axis=-1)
    assert np.count_nonzero((local["mask"][0] > 0) & (local_types == 1)) == 0
    assert np.count_nonzero((privileged["mask"][0] > 0) & (privileged_types == 1)) == 2


def test_simultaneous_dynamics_are_agent_order_equivariant():
    actions = np.asarray([[0.2, -0.1, 0.0], [-0.3, 0.4, 0.1], [0.0, -0.2, 0.3]], dtype=np.float32)
    env_a = ContinuousCityUAVEnv(num_agents=3, seed=41, scenario="static")
    env_b = ContinuousCityUAVEnv(num_agents=3, seed=41, scenario="static")
    order = np.asarray([2, 0, 1])
    env_b.pos = env_a.pos[order].copy()
    env_b.vel = env_a.vel[order].copy()
    env_b.goals = env_a.goals[order].copy()
    env_a.step(actions)
    env_b.step(actions[order])
    inverse = np.argsort(order)
    assert np.allclose(env_a.pos, env_b.pos[inverse], atol=1e-6)


def test_graph_model_and_filter_are_finite():
    env = ContinuousCityUAVEnv(num_agents=3, seed=9, scenario="dense")
    obs = env.observe(privileged=False)
    batch = {k: jnp.asarray(v) for k, v in observation_batch(obs).items()}
    model = GraphActorCritic(graph_actor=True)
    params = model.init(jax.random.PRNGKey(1), **batch)["params"]
    actions = np.asarray(policy_apply(model, params, batch, jax.random.PRNGKey(2), True)[0])
    filtered, interventions, infeasible = cbf_filter(actions, obs["nodes"], obs["mask"])
    assert filtered.shape == (3, 3)
    assert np.isfinite(filtered).all()
    assert np.max(np.abs(filtered)) <= 1.0
    assert interventions >= 0 and infeasible >= 0


def test_policy_interface_cannot_receive_environment_object():
    env = ContinuousCityUAVEnv(num_agents=3, seed=13)
    obs = env.observe(privileged=False)
    assert set(obs) == {"nodes", "mask", "adjacency", "global_state"}
    assert all(not hasattr(value, "pos") for value in obs.values())

