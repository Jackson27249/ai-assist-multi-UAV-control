from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

from continuous_city_env import ContinuousCityUAVEnv
from models import load_policy


@dataclass
class VehicleState:
    north: float = 0.0
    east: float = 0.0
    down: float = 0.0
    north_velocity: float = 0.0
    east_velocity: float = 0.0
    down_velocity: float = 0.0
    timestamp: float = 0.0
    received: int = 0


async def monitor_vehicle(drone: System, state: VehicleState, stop: asyncio.Event) -> None:
    try:
        async for item in drone.telemetry.position_velocity_ned():
            state.north = float(item.position.north_m)
            state.east = float(item.position.east_m)
            state.down = float(item.position.down_m)
            state.north_velocity = float(item.velocity.north_m_s)
            state.east_velocity = float(item.velocity.east_m_s)
            state.down_velocity = float(item.velocity.down_m_s)
            state.timestamp = time.monotonic()
            state.received += 1
            if stop.is_set():
                return
    except asyncio.CancelledError:
        return


async def connect_vehicle(instance: int, timeout: float = 20.0) -> System:
    drone = System(port=50051 + instance)
    await drone.connect(system_address=f"udpin://0.0.0.0:{14540 + instance}")

    async def wait_connected() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                return

    try:
        await asyncio.wait_for(wait_connected(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"PX4 instance {instance} did not connect on UDP {14540 + instance}") from exc
    return drone


def start_px4(binary: Path, romfs: Path, count: int, work: Path, instance_offset: int) -> list[subprocess.Popen]:
    processes = []
    for local_instance in range(count):
        instance = instance_offset + local_instance
        instance_dir = work / f"instance_{local_instance}"
        instance_dir.mkdir(parents=True, exist_ok=True)
        stdout = (instance_dir / "stdout.log").open("wb")
        stderr = (instance_dir / "stderr.log").open("wb")
        env = os.environ.copy()
        env.update({"PX4_SIM_MODEL": "none", "PX4_SYS_AUTOSTART": "10040"})
        process = subprocess.Popen(
            [str(binary), "-i", str(instance), "-d", "-w", str(instance_dir), str(romfs)],
            cwd=instance_dir,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        processes.append(process)
    return processes


def stop_px4(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def arrange_world(states: list[VehicleState], starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(
        [[starts[i, 0] + state.east, starts[i, 1] + state.north, starts[i, 2] - state.down] for i, state in enumerate(states)],
        dtype=np.float32,
    )
    velocities = np.asarray(
        [[state.east_velocity, state.north_velocity, -state.down_velocity] for state in states],
        dtype=np.float32,
    )
    return positions, velocities


async def run_episode(
    *,
    binary: Path,
    romfs: Path,
    model_path: Path,
    policy_name: str,
    training_seed: int,
    evaluation_seed: int,
    num_agents: int,
    duration: float,
    output: Path,
    ulg_output: Path,
    instance_offset: int,
) -> dict[str, Any]:
    episode_id = f"{policy_name}_train{training_seed}_n{num_agents}_seed{evaluation_seed}"
    work_root = Path(os.environ.get("PX4_WORK_ROOT", "/tmp"))
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"px4_{episode_id}_", dir=work_root))
    processes: list[subprocess.Popen] = []
    drones: list[System] = []
    monitors: list[asyncio.Task] = []
    stop = asyncio.Event()
    rows: list[dict[str, Any]] = []
    started = time.time()
    error = ""
    try:
        processes = start_px4(binary, romfs, num_agents, work, instance_offset)
        await asyncio.sleep(5.0)
        drones = await asyncio.gather(*(connect_vehicle(instance_offset + i) for i in range(num_agents)))
        states = [VehicleState() for _ in range(num_agents)]
        monitors = [asyncio.create_task(monitor_vehicle(drone, state, stop)) for drone, state in zip(drones, states)]
        for drone in drones:
            try:
                await drone.param.set_param_int("COM_ARM_WO_GPS", 1)
            except Exception:
                pass
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
            await drone.action.arm()
            await drone.offboard.start()
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline and any(state.received == 0 for state in states):
            await asyncio.sleep(0.1)
        if any(state.received == 0 for state in states):
            raise TimeoutError("position_velocity_ned telemetry missing")

        scenario = "dynamic" if num_agents == 3 else "dense"
        reference = ContinuousCityUAVEnv(num_agents=num_agents, seed=evaluation_seed, scenario=scenario)
        starts = reference.pos.copy()
        goals = reference.goals.copy()
        policy = load_policy(model_path, policy_name, graph_actor=(policy_name == "gcbf_local"), use_filter=(policy_name == "gcbf_local"))
        key = jax.random.PRNGKey(evaluation_seed)
        tick = 0
        next_tick = time.monotonic()
        while tick * 0.1 < duration:
            positions, velocities = arrange_world(states, starts)
            reference.pos = positions
            reference.vel = velocities
            reference.step_count = tick
            reference.dynamic_pos = reference.dynamic_pos + reference.dynamic_vel * 0.1
            obs = reference.observe(privileged=False)
            key, subkey = jax.random.split(key)
            actions, interventions, infeasible = policy.act(obs, subkey)
            commands = actions * reference.max_speed
            send_start = time.monotonic()
            await asyncio.gather(
                *(
                    drone.offboard.set_velocity_ned(
                        VelocityNedYaw(float(command[1]), float(command[0]), float(-command[2]), 0.0)
                    )
                    for drone, command in zip(drones, commands)
                )
            )
            send_ms = 1000.0 * (time.monotonic() - send_start)
            pair = positions[:, None, :] - positions[None, :, :]
            pair_distance = np.linalg.norm(pair, axis=-1)[np.triu_indices(num_agents, 1)]
            timestamp = tick * 0.1
            for index, (state, position, velocity, command) in enumerate(zip(states, positions, velocities, commands)):
                rows.append(
                    {
                        "episode_id": episode_id,
                        "policy": policy_name,
                        "training_seed": training_seed,
                        "evaluation_seed": evaluation_seed,
                        "num_agents": num_agents,
                        "time": timestamp,
                        "agent": index,
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "z": float(position[2]),
                        "vx": float(velocity[0]),
                        "vy": float(velocity[1]),
                        "vz": float(velocity[2]),
                        "cmd_vx": float(command[0]),
                        "cmd_vy": float(command[1]),
                        "cmd_vz": float(command[2]),
                        "min_pair_distance": float(np.min(pair_distance)),
                        "send_latency_ms": send_ms,
                        "telemetry_age_ms": 1000.0 * max(0.0, time.monotonic() - state.timestamp),
                        "interventions": interventions,
                        "qp_infeasible": infeasible,
                    }
                )
            tick += 1
            next_tick += 0.1
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))

        for drone in drones:
            try:
                await drone.offboard.stop()
            except OffboardError:
                pass
            try:
                await drone.action.disarm()
            except Exception:
                pass
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        for task in monitors:
            task.cancel()
        if monitors:
            await asyncio.gather(*monitors, return_exceptions=True)
        stop_px4(processes)
        # PX4 may fork logger/mavlink children outside the parent process group.
        subprocess.run(["pkill", "-TERM", "-f", str(work)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for instance in range(instance_offset, instance_offset + num_agents):
            subprocess.run(["pkill", "-TERM", "-f", f"mavsdk_server.*:{14540 + instance}"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await asyncio.sleep(0.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    ulg_output.mkdir(parents=True, exist_ok=True)
    ulg_files = []
    for path in work.rglob("*.ulg"):
        destination = ulg_output / f"{episode_id}_instance{len(ulg_files)}.ulg"
        shutil.copy2(path, destination)
        ulg_files.append(destination.name)
    stdout_tails = []
    for path in sorted(work.glob("instance_*/stdout.log")):
        stdout_tails.append(path.read_text(errors="replace")[-1200:])
    shutil.rmtree(work, ignore_errors=True)
    if rows:
        frame = rows
        min_distance = min(float(row["min_pair_distance"]) for row in frame)
        max_tracking_error = max(
            float(np.linalg.norm(np.asarray([row["vx"] - row["cmd_vx"], row["vy"] - row["cmd_vy"], row["vz"] - row["cmd_vz"]])))
            for row in frame
        )
        success = error == "" and min_distance >= 0.55
    else:
        min_distance = float("nan")
        max_tracking_error = float("nan")
        success = False
    return {
        "episode_id": episode_id,
        "policy": policy_name,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "num_agents": num_agents,
        "duration": duration,
        "rows": len(rows),
        "success": success,
        "collision": bool(rows and min_distance < 0.55),
        "min_pair_distance": min_distance,
        "max_tracking_error": max_tracking_error,
        "error": error,
        "ulg_files": ulg_files,
        "telemetry_file": output.name if rows else "",
        "stdout_tails": stdout_tails,
        "elapsed_seconds": time.time() - started,
    }


async def async_main(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    telemetry = args.output / "telemetry"
    ulg = args.output / "ulg"
    telemetry.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "episode_summary.jsonl"
    count = 0
    with summary_path.open("w", encoding="utf-8") as handle:
        for policy_name in args.policies:
            for training_seed in args.training_seeds:
                checkpoint = args.models_root / policy_name / f"seed_{training_seed}" / "best.msgpack"
                for num_agents in args.agent_counts:
                    for offset in range(args.episodes_per_model):
                        evaluation_seed = args.seed_start + offset
                        result = await run_episode(
                            binary=args.px4_binary,
                            romfs=args.px4_romfs,
                            model_path=checkpoint,
                            policy_name=policy_name,
                            training_seed=training_seed,
                            evaluation_seed=evaluation_seed,
                            num_agents=num_agents,
                            duration=args.duration,
                            output=telemetry / f"{policy_name}_train{training_seed}_n{num_agents}_seed{evaluation_seed}.csv",
                            ulg_output=ulg,
                            instance_offset=args.instance_offset,
                        )
                        handle.write(json.dumps(result, allow_nan=True) + "\n")
                        handle.flush()
                        count += 1
                        print(f"PX4_PROGRESS={count} success={result['success']} error={result['error']}", flush=True)
    print(f"PX4_EXPERIMENT_RESULT=PASS rows={count} output={summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--px4-binary", type=Path, required=True)
    parser.add_argument("--px4-romfs", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", default=["gcbf_local", "mappo_local"])
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[1101, 1102, 1103, 1104, 1105])
    parser.add_argument("--agent-counts", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--episodes-per-model", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=41000)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--instance-offset", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
