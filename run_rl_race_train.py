"""
RL smoke test for PteroSim (spawn + gates + actuators + IMU).

Tensorboard: From project folder, second terminal after PPO training starts:
  .venv\Scripts\Activate.ps1
  pip install tensorboard
  tensorboard --logdir tensorboard_logs
  # http://localhost:6006

1. Unreal: Play (simulation must already be listening on gRPC).
2. Run from project root:
   python scripts/rl_examples/run_rl_race_train.py --mode random
   python scripts/rl_examples/run_rl_race_train.py --mode ppo --timesteps 5000
   # continue from checkpoint:
   python scripts/rl_examples/run_rl_race_train.py --mode ppo --timesteps 10000 --load checkpoints/ppo_pterorace_smoke.zip

--mode random  Random actions only, reward logging (pipeline smoke test).
--mode ppo      Stable-Baselines3 PPO (requires torch + stable-baselines3).

Example: C:/Users/Yollnahkriin/Documents/Unreal_Projects/PteroSim/.venv/Scripts/python.exe run_rl_race_train.py --mode ppo --timesteps 2000 --load checkpoints/ppo_pterorace_smoke.zip --save checkpoints/ppo_pterorace_smoke  
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from pterosim import PteroSim
from pterosim.aircraft import Aircraft

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

DEFAULT_SIM_ADDRESS = "localhost:10010"
DEFAULT_AIRCRAFT_CLASS = "F450"

DRONE_SPAWN = {"x": 0.0, "y": 0.0, "z": 300.0, "yaw": 0.0}

GATE_POSITIONS = [
    {"x": 5000.0, "y": 0.0, "z": 300.0, "yaw": 0.0},
    {"x": 10000.0, "y": 2000.0, "z": 300.0, "yaw": 45.0},
    {"x": 15000.0, "y": 0.0, "z": 500.0, "yaw": 0.0},
]

OBS_DIM = 19
MAX_EPISODE_STEPS = 500
# UE cm: if farther from next gate center than this, episode fails (reset).
MAX_DIST_FROM_NEXT_GATE_CM = 7500.0
# Clip observation for stable PPO (replace nan/inf to avoid NaN action params).
OBS_CLIP = float(1e5)

# Set once before sim.start(); engine rejects changes while Running.
PHYSICS_HZ = 1000.0
SIM_TIME_SCALE = 10.0
def wait_for_aircraft_status(
    sim: PteroSim,
    instance_id: int | None = None,
    timeout_s: float = 2.0,
    poll_interval_s: float = 0.05,
) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        statuses = sim.aircraft_status()
        if instance_id is None and statuses:
            return statuses[0]
        if instance_id is not None:
            for status in statuses:
                if status.instance_id == instance_id:
                    return status
        time.sleep(poll_interval_s)
    if instance_id is None:
        raise RuntimeError("No aircraft status received before timeout")
    raise RuntimeError(f"Aircraft {instance_id} not found before timeout")


def build_controls(sim: PteroSim, instance_id: int) -> list[float]:
    cfg = sim.get_actuator_configuration(instance_id)
    controls = [0.55] * min(4, cfg.channel_count)
    while len(controls) < cfg.channel_count:
        controls.append(0.0)
    return controls


def remove_all_aircraft(sim: PteroSim) -> None:
    # Remove every existing aircraft before spawning a fresh one.
    for status in list(sim.aircraft_status()):
        Aircraft(sim, status.instance_id).remove()


def get_observation(sim: PteroSim, instance_id: int) -> dict[str, Any]:
    status = wait_for_aircraft_status(sim, instance_id=instance_id)
    race = sim.get_race_state(instance_id)
    gate = sim.get_next_gate_pose(instance_id)
    imu = sim.get_imu(instance_id)

    drone_pos = np.array([status.x, status.y, status.z], dtype=np.float64)
    gate_pos = np.array([gate.x, gate.y, gate.z], dtype=np.float64)
    dist = float(np.linalg.norm(gate_pos - drone_pos))

    return {
        "x": status.x,
        "y": status.y,
        "z": status.z,
        "yaw": status.yaw,
        "pitch": status.pitch,
        "roll": status.roll,
        "ax": imu.acceleration[0],
        "ay": imu.acceleration[1],
        "az": imu.acceleration[2],
        "wx": imu.angular_velocity[0],
        "wy": imu.angular_velocity[1],
        "wz": imu.angular_velocity[2],
        "gate_x": gate.x,
        "gate_y": gate.y,
        "gate_z": gate.z,
        "gate_fwd_x": gate.forward_x,
        "gate_fwd_y": gate.forward_y,
        "gate_fwd_z": gate.forward_z,
        "dist_to_next_gate": dist,
        "gates_passed": race.gates_passed,
        "crashed": status.crashed,
    }


def obs_to_vector(obs: dict) -> np.ndarray:
    keys = [
        "x", "y", "z", "yaw", "pitch", "roll",
        "ax", "ay", "az", "wx", "wy", "wz",
        "gate_x", "gate_y", "gate_z",
        "gate_fwd_x", "gate_fwd_y", "gate_fwd_z",
        "dist_to_next_gate",
    ]
    v = np.array([obs[k] for k in keys], dtype=np.float32)
    v = np.nan_to_num(v, nan=0.0, posinf=OBS_CLIP, neginf=-OBS_CLIP)
    return np.clip(v, -OBS_CLIP, OBS_CLIP)


def gate_approach_shaping(delta_dist: float, closest_cm: float) -> float:
    """Shaping from distance change toward next gate (UE cm). closest_cm = min(prev, cur)."""
    if delta_dist == 0.0:
        return 0.0
    if delta_dist < 0.0:
        if closest_cm <= 500.0:
            scale = 0.01
        elif closest_cm >= 2000.0:
            scale = 0.04
        else:
            t = (closest_cm - 500.0) / (2000.0 - 500.0)
            scale = 0.01 + t * (0.04 - 0.01)
        return delta_dist * scale
    # delta_dist > 0: bonus * 0.01, multiplier interpolated 4→3→2→1 by distance bands
    if closest_cm <= 500.0:
        mult = 4.0
    elif closest_cm <= 1000.0:
        t = (closest_cm - 500.0) / (1000.0 - 500.0)
        mult = 4.0 + t * (3.0 - 4.0)
    elif closest_cm <= 2000.0:
        t = (closest_cm - 1000.0) / (2000.0 - 1000.0)
        mult = 3.0 + t * (2.0 - 3.0)
    else:
        mult = 1.0
    return delta_dist * 0.01 * mult


def compute_reward(
    obs: dict,
    prev_obs: dict | None,
    done: bool,
    reason: str,
) -> float:
    if done and reason == "crash":
        return -10.0
    if done and reason == "too_far":
        return -100.0
    if done and reason == "timeout":
        return -50.0
    reward = 0.0
    if prev_obs is not None:
        delta_dist = prev_obs["dist_to_next_gate"] - obs["dist_to_next_gate"]
        closest_cm = min(prev_obs["dist_to_next_gate"], obs["dist_to_next_gate"])
        reward += gate_approach_shaping(delta_dist, closest_cm)
    if prev_obs is None:
        gp_prev = 0
    else:
        gp_prev = prev_obs["gates_passed"]
    if obs["gates_passed"] > gp_prev:
        reward += 100.0 * (obs["gates_passed"] - gp_prev)
    return reward


def apply_action(sim: PteroSim, instance_id: int, action: np.ndarray, controls_buffer: list[float]) -> None:
    a = np.asarray(action, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
    a = np.clip(a, -1.0, 1.0)
    th = ((a + 1.0) * 0.5).clip(0.0, 1.0)
    controls = list(controls_buffer)
    for i in range(min(4, len(controls))):
        controls[i] = float(th[i])
    sim.set_actuator_controls(instance_id, controls)


def reset_race_session(sim: PteroSim, aircraft_class: str) -> tuple[int, int, list[float]]:
    sim.stop()
    remove_all_aircraft(sim)
    drone = sim.spawn(aircraft_class, **DRONE_SPAWN)
    drone_id = drone.instance_id
    sim.set_track_gates(GATE_POSITIONS)
    sim.reset_all_races()
    sim.reset_race(drone_id)
    sim.start()
    wait_for_aircraft_status(sim, instance_id=drone_id)
    track = sim.get_track_info()
    controls = build_controls(sim, drone_id)
    return drone_id, track.gate_count, controls


def run_random(
    sim_addr: str,
    aircraft: str,
    episodes: int,
    max_dist_gate_cm: float,
    seed: int | None,
) -> None:
    with PteroSim(sim_addr) as sim:
        sim.set_physics_frequency(PHYSICS_HZ)
        sim.set_time_scale(SIM_TIME_SCALE)

        drone_id, total_gates, controls = reset_race_session(sim, aircraft)

        rng = np.random.default_rng(seed)

        for ep in range(episodes):
            obs_dict = get_observation(sim, drone_id)
            obs_vec = obs_to_vector(obs_dict)
            total_r = 0.0
            reason = "timeout"

            for t in range(MAX_EPISODE_STEPS):
                action = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
                apply_action(sim, drone_id, action, controls)
                sim.step_once()

                next_dict = get_observation(sim, drone_id)
                done = False
                reason = ""
                if next_dict["crashed"]:
                    done = True
                    reason = "crash"
                elif next_dict["dist_to_next_gate"] > max_dist_gate_cm:
                    done = True
                    reason = "too_far"
                elif next_dict["gates_passed"] >= total_gates:
                    done = True
                    reason = "success"
                elif t >= MAX_EPISODE_STEPS - 1:
                    done = True
                    reason = "timeout"

                r = compute_reward(next_dict, obs_dict, done, reason)
                total_r += r
                obs_dict = next_dict
                if done:
                    break

            print(f"[random EP {ep}] {reason=} gates={obs_dict['gates_passed']}/{total_gates} R={total_r:.1f}")

            drone_id, total_gates, controls = reset_race_session(sim, aircraft)

        sim.stop()


if gym is not None and spaces is not None:

    class PteroRaceEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(
            self,
            sim_address: str = DEFAULT_SIM_ADDRESS,
            aircraft_class: str = DEFAULT_AIRCRAFT_CLASS,
            max_dist_from_next_gate_cm: float = MAX_DIST_FROM_NEXT_GATE_CM,
        ):
            super().__init__()
            self.sim_address = sim_address
            self.aircraft_class = aircraft_class
            self.max_dist_from_next_gate_cm = max_dist_from_next_gate_cm
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            )
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
            )
            self._sim: PteroSim | None = None
            self._drone_id = 0
            self._controls: list[float] = []
            self._total_gates = 0
            self._step_count = 0
            self._prev_obs_dict: dict | None = None

        def _connect(self) -> None:
            if self._sim is not None:
                return
            self._sim = PteroSim(self.sim_address)
            self._sim.set_physics_frequency(PHYSICS_HZ)
            self._sim.set_time_scale(SIM_TIME_SCALE)

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict | None = None,
        ):
            super().reset(seed=seed)
            self._connect()
            assert self._sim is not None
            self._drone_id, self._total_gates, self._controls = reset_race_session(
                self._sim, self.aircraft_class
            )
            self._step_count = 0
            self._prev_obs_dict = None
            obs_dict = get_observation(self._sim, self._drone_id)
            return obs_to_vector(obs_dict), {}

        def step(self, action):
            assert self._sim is not None
            self._step_count += 1
            apply_action(self._sim, self._drone_id, action, self._controls)
            self._sim.step_once()
            obs_dict = get_observation(self._sim, self._drone_id)
            obs = obs_to_vector(obs_dict)

            terminated = False
            truncated = False
            reason = ""
            if obs_dict["crashed"]:
                terminated = True
                reason = "crash"
            elif obs_dict["dist_to_next_gate"] > self.max_dist_from_next_gate_cm:
                terminated = True
                reason = "too_far"
            elif obs_dict["gates_passed"] >= self._total_gates:
                terminated = True
                reason = "success"
            elif self._step_count >= MAX_EPISODE_STEPS:
                truncated = True
                reason = "timeout"

            reward = compute_reward(obs_dict, self._prev_obs_dict, terminated or truncated, reason)
            self._prev_obs_dict = obs_dict
            return obs, float(reward), terminated, truncated, {}

        def close(self):
            if self._sim is not None:
                try:
                    self._sim.stop()
                except Exception:
                    pass
                self._sim.close()
                self._sim = None


def run_ppo(
    sim_addr: str,
    aircraft: str,
    timesteps: int,
    max_dist_gate_cm: float,
    tensorboard_log: str | None,
    run_name: str,
    load_path: str | None,
    save_path: str,
    seed: int | None,
) -> None:
    if gym is None or spaces is None or "PteroRaceEnv" not in globals():
        raise SystemExit("Install gymnasium: pip install gymnasium")
    from pathlib import Path

    from stable_baselines3 import PPO

    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    env = PteroRaceEnv(
        sim_address=sim_addr,
        aircraft_class=aircraft,
        max_dist_from_next_gate_cm=max_dist_gate_cm,
    )
    env.reset(seed=seed)
    try:
        load_kwargs: dict = {"env": env, "verbose": 1}
        if tensorboard_log:
            Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
            load_kwargs["tensorboard_log"] = tensorboard_log

        if load_path:
            lp = Path(load_path)
            if not lp.is_file() and lp.suffix != ".zip":
                lp_zip = lp.with_suffix(".zip")
                if lp_zip.is_file():
                    lp = lp_zip
            if not lp.is_file():
                raise SystemExit(f"Checkpoint not found: {load_path}")
            model = PPO.load(str(lp), **load_kwargs)
            print(f"Loaded policy from {load_path}, continuing for {timesteps} timesteps")
            model.learn(
                total_timesteps=timesteps,
                tb_log_name=run_name,
                reset_num_timesteps=False,
            )
        else:
            ppo_kwargs: dict = {
                "policy": "MlpPolicy",
                "env": env,
                "verbose": 1,
                "seed": seed,
            }
            if tensorboard_log:
                ppo_kwargs["tensorboard_log"] = tensorboard_log
            model = PPO(**ppo_kwargs)
            model.learn(total_timesteps=timesteps, tb_log_name=run_name)

        model.save(save_path)
        print(f"Saved policy to {save_path}")
        if tensorboard_log:
            print(
                f"TensorBoard: tensorboard --logdir {tensorboard_log}  "
                    "(PowerShell, second window) -> http://localhost:6006"
            )
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", default=DEFAULT_SIM_ADDRESS)
    parser.add_argument("--aircraft", default=DEFAULT_AIRCRAFT_CLASS)
    parser.add_argument("--mode", choices=["random", "ppo"], default="random")
    parser.add_argument("--episodes", type=int, default=3, help="For random mode")
    parser.add_argument("--timesteps", type=int, default=5000, help="For PPO mode")
    parser.add_argument(
        "--max-dist-gate",
        type=float,
        default=MAX_DIST_FROM_NEXT_GATE_CM,
        help="Fail episode if distance to next gate exceeds this (UE cm)",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="tensorboard_logs",
        help="Folder for TensorBoard (PPO only). Empty string disables.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="PPO_race",
        help="Subfolder name inside --tensorboard-log (PPO only)",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help="PPO: continue training from this .zip (e.g. checkpoints/ppo_pterorace_smoke.zip)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="checkpoints/ppo_pterorace_smoke",
        help="PPO: save path without extension (SB3 adds .zip)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (random mode and PPO)",
    )
    args = parser.parse_args()

    if args.mode == "random":
        run_random(args.sim, args.aircraft, args.episodes, args.max_dist_gate, args.seed)
    else:
        tb = args.tensorboard_log.strip() or None
        load_p = args.load.strip() or None
        run_ppo(
            args.sim,
            args.aircraft,
            args.timesteps,
            args.max_dist_gate,
            tensorboard_log=tb,
            run_name=args.run_name,
            load_path=load_p,
            save_path=args.save,
            seed=args.seed,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
