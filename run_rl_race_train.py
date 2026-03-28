"""
RL drone racing trainer for PteroSim with C++ attitude rate controller.

RL agent outputs attitude commands (roll/pitch angles + yaw rate + throttle).
C++ QuadXAttitudeController (Crazyflie-style cascaded PID) runs at 1000 Hz
on the physics thread, converting attitude commands to motor throttles.

Architecture:
  RL Agent (10 Hz, via time_scale=100)
    → attitude command (roll, pitch, yaw_rate, throttle)
    → gRPC set_attitude_command
    → C++ PID at 1000 Hz → JSBSim motors
    → step_once (100 physics ticks)

Modes:
  --mode random   Random actions (pipeline smoke test)
  --mode train    SB3 training (headless recommended: -nullrhi)
  --mode play     Inference with real-time rendering
  --mode optuna   Optuna hyperparameter search

Usage:
  python run_rl_race_train.py --mode train --timesteps 500000
  python run_rl_race_train.py --mode train --algo ppo --timesteps 500000
  python run_rl_race_train.py --mode train --load checkpoints/sac_pterorace.zip
  python run_rl_race_train.py --mode play --load checkpoints/sac_pterorace.zip
  python run_rl_race_train.py --mode optuna --optuna-trials 50 --optuna-timesteps 30000

TensorBoard:
  tensorboard --logdir tensorboard_logs  # http://localhost:6006
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

try:
    from stable_baselines3.common.callbacks import BaseCallback

    class ProgressCallback(BaseCallback):
        """Prints ETA, ep count, avg reward, avg ep_len every `print_freq` steps."""

        def __init__(self, total_timesteps: int, print_freq: int = 2048, verbose: int = 1):
            super().__init__(verbose)
            self._total = total_timesteps
            self._print_freq = print_freq
            self._start_time: float = 0.0
            self._ep_count = 0
            self._last_print = 0

        def _on_training_start(self) -> None:
            self._start_time = time.monotonic()

        def _on_step(self) -> bool:
            dones = self.locals.get("dones", self.locals.get("done", None))
            if dones is not None:
                if hasattr(dones, "__len__"):
                    self._ep_count += sum(dones)
                elif dones:
                    self._ep_count += 1

            if self.num_timesteps - self._last_print >= self._print_freq:
                self._last_print = self.num_timesteps
                elapsed = time.monotonic() - self._start_time
                remaining = self._total - self.num_timesteps
                fps = self.num_timesteps / max(elapsed, 1e-6)
                eta_s = remaining / max(fps, 1e-6)
                eta_min = eta_s / 60.0

                ep_rew = self.logger.name_to_value.get("rollout/ep_rew_mean", float("nan"))
                ep_len = self.logger.name_to_value.get("rollout/ep_len_mean", float("nan"))

                pct = 100.0 * self.num_timesteps / self._total
                print(
                    f"[{pct:5.1f}%] steps={self.num_timesteps}/{self._total} "
                    f"eps={self._ep_count} fps={fps:.0f} "
                    f"ep_rew={ep_rew:.1f} ep_len={ep_len:.1f} "
                    f"ETA={eta_min:.1f}min"
                )
            return True

except ImportError:
    BaseCallback = None
    ProgressCallback = None

# --- Config ---

DEFAULT_SIM_ADDRESS = "localhost:10010"
DEFAULT_AIRCRAFT_CLASS = "F450"

DRONE_SPAWN = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}  # JSBSim always starts on ground

GATE_POSITIONS = [
    {"x": 1500.0, "y": 0.0, "z": 300.0, "yaw": 0.0},
    {"x": 3000.0, "y": 1000.0, "z": 300.0, "yaw": 30.0},
    {"x": 5000.0, "y": 0.0, "z": 400.0, "yaw": 0.0},
]

OBS_DIM = 16  # att(3) + imu_accel(3) + imu_gyro(3) + gate_rel_body(3) + gate_fwd(3) + dist_norm(1)
MAX_EPISODE_STEPS = 500
MAX_DIST_FROM_NEXT_GATE_CM = 7500.0
OBS_CLIP = 10.0  # clip normalized obs to [-10, 10]

# Normalization constants (match physical ranges)
NORM_ANGLE_DEG = 180.0     # degrees
NORM_ACCEL = 20.0          # m/s²
NORM_GYRO = 5.0            # rad/s
NORM_DIST = 7500.0         # cm (max gate distance)

PHYSICS_HZ = 1000.0
DEFAULT_TIME_SCALE = 100.0  # 100 physics ticks per step_once → agent at 10 Hz

GRACE_STEPS = 10


# --- Action mapping ---

HOVER_THROTTLE = 0.425

def action_to_commands(action: np.ndarray) -> tuple[float, float, float, float]:
    """Map RL action [-1,1]^4 to attitude commands + throttle.

    action[0] -> desired roll angle (rad), scaled to [-0.5, 0.5] (~30 deg)
    action[1] -> desired pitch angle (rad), scaled to [-0.5, 0.5] (~30 deg)
    action[2] -> throttle centered at HOVER_THROTTLE [0.15, 0.70]
    action[3] -> yaw rate (rad/s), scaled to [-1, 1]
    """
    a = np.asarray(action, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
    a = np.clip(a, -1.0, 1.0)

    desired_roll = float(a[0]) * 0.5   # ±30 deg max
    desired_pitch = float(a[1]) * 0.5  # ±30 deg max
    yaw_rate = float(a[3]) * 1.0       # ±1 rad/s
    # Center at hover: action=0 → hover, action=-1 → 0.15, action=+1 → 0.70
    throttle = float(np.clip(HOVER_THROTTLE + float(a[2]) * 0.275, 0.15, 0.70))

    return desired_roll, desired_pitch, yaw_rate, throttle


def send_attitude_command(sim: PteroSim, instance_id: int,
                          roll: float, pitch: float, yaw_rate: float, throttle: float) -> None:
    """Send attitude command to C++ PID controller at 1000 Hz."""
    sim.set_attitude_command(
        instance_id,
        roll_rad=roll,
        pitch_rad=pitch,
        yaw_rate_rad_sec=yaw_rate,
        throttle=throttle,
        enabled=True,
    )


# --- Observation ---

def _crashed_observation() -> dict[str, Any]:
    return {
        "x": 0.0, "y": 0.0, "z": 0.0,
        "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
        "ax": 0.0, "ay": 0.0, "az": 0.0,
        "wx": 0.0, "wy": 0.0, "wz": 0.0,
        "gate_x": 0.0, "gate_y": 0.0, "gate_z": 0.0,
        "gate_fwd_x": 0.0, "gate_fwd_y": 0.0, "gate_fwd_z": 0.0,
        "dist_to_next_gate": 0.0,
        "gates_passed": 0,
        "crashed": True,
    }


def get_observation(sim: PteroSim, instance_id: int) -> dict[str, Any]:
    statuses = sim.aircraft_status()
    status = None
    for s in statuses:
        if s.instance_id == instance_id:
            status = s
            break
    if status is None:
        return _crashed_observation()
    if status.crashed:
        return {**_crashed_observation(), "x": status.x, "y": status.y, "z": status.z,
                "yaw": status.yaw, "pitch": status.pitch, "roll": status.roll}

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


def _rotate_to_body_frame(vec_world: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate a world-frame XY vector into the drone's body frame using yaw only."""
    yaw_rad = np.radians(yaw_deg)
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    bx = cos_y * vec_world[0] + sin_y * vec_world[1]
    by = -sin_y * vec_world[0] + cos_y * vec_world[1]
    bz = vec_world[2]
    return np.array([bx, by, bz], dtype=np.float32)


def obs_to_vector(obs: dict) -> np.ndarray:
    """Build a normalized 16-dim observation vector.

    Components (all scaled to roughly [-1, 1]):
      [0:3]  attitude: yaw, pitch, roll  (normalized by 180 deg)
      [3:6]  IMU acceleration ax, ay, az (normalized by 20 m/s²)
      [6:9]  IMU angular velocity wx, wy, wz (normalized by 5 rad/s)
      [9:12] gate position relative to drone, in drone body frame (normalized by 7500 cm)
      [12:15] gate forward unit vector (already [-1,1], no normalization needed)
      [15]   distance to next gate (normalized by 7500 cm)
    """
    drone_pos = np.array([obs["x"], obs["y"], obs["z"]], dtype=np.float64)
    gate_pos = np.array([obs["gate_x"], obs["gate_y"], obs["gate_z"]], dtype=np.float64)
    gate_rel_world = (gate_pos - drone_pos).astype(np.float32)
    gate_rel_body = _rotate_to_body_frame(gate_rel_world, obs["yaw"])

    att = np.array([obs["yaw"], obs["pitch"], obs["roll"]], dtype=np.float32) / NORM_ANGLE_DEG
    accel = np.array([obs["ax"], obs["ay"], obs["az"]], dtype=np.float32) / NORM_ACCEL
    gyro = np.array([obs["wx"], obs["wy"], obs["wz"]], dtype=np.float32) / NORM_GYRO
    gate_rel_norm = gate_rel_body / NORM_DIST
    gate_fwd = np.array([obs["gate_fwd_x"], obs["gate_fwd_y"], obs["gate_fwd_z"]], dtype=np.float32)
    dist_norm = np.array([obs["dist_to_next_gate"] / NORM_DIST], dtype=np.float32)

    v = np.concatenate([att, accel, gyro, gate_rel_norm, gate_fwd, dist_norm])
    v = np.nan_to_num(v, nan=0.0, posinf=OBS_CLIP, neginf=-OBS_CLIP)
    return np.clip(v, -OBS_CLIP, OBS_CLIP).astype(np.float32)


# --- Reward (with tunable coefficients) ---

# Default reward coefficients — Optuna will override these
REWARD_DEFAULTS = {
    "approach_scale": 0.04,       # multiplier for gate approach shaping
    "gate_bonus": 100.0,          # bonus per gate passed
    "crash_penalty": -50.0,
    "too_far_penalty": -20.0,
    "timeout_penalty": -30.0,     # penalty for running out of time (incentivize speed)
    "time_penalty": -0.05,        # per-step penalty (incentivize finishing fast)
    "att_coef": 0.05,             # attitude penalty coefficient
    "att_threshold_deg": 25.0,    # threshold below which no attitude penalty
    "proximity_bonus": 0.1,       # small alive bonus when near gate (<3000cm)
    "proximity_radius_cm": 3000.0,
}

# Active reward config (mutable — Optuna overwrites before each trial)
reward_config: dict[str, float] = dict(REWARD_DEFAULTS)


def gate_approach_shaping(delta_dist: float, closest_cm: float) -> float:
    if delta_dist == 0.0:
        return 0.0
    scale = reward_config["approach_scale"]
    if delta_dist < 0.0:
        # Moving away — symmetric penalty (prevents approach-retreat farming)
        return delta_dist * scale
    # Moving toward gate — scale up when close
    if closest_cm <= 500.0:
        mult = 4.0
    elif closest_cm <= 1000.0:
        t = (closest_cm - 500.0) / 500.0
        mult = 4.0 - t
    elif closest_cm <= 2000.0:
        t = (closest_cm - 1000.0) / 1000.0
        mult = 3.0 - t
    else:
        mult = 1.0
    return delta_dist * scale * mult


def compute_reward(
    obs: dict,
    prev_obs: dict | None,
    done: bool,
    reason: str,
) -> float:
    if done and reason == "crash":
        return reward_config["crash_penalty"]
    if done and reason == "too_far":
        return reward_config["too_far_penalty"]
    if done and reason == "timeout":
        return reward_config["timeout_penalty"]

    reward = reward_config["time_penalty"]

    # Gate approach shaping
    if prev_obs is not None:
        delta_dist = prev_obs["dist_to_next_gate"] - obs["dist_to_next_gate"]
        closest_cm = min(prev_obs["dist_to_next_gate"], obs["dist_to_next_gate"])
        reward += gate_approach_shaping(delta_dist, closest_cm)

    # Attitude penalty — thresholded, only penalizes past threshold
    att_thresh = np.radians(reward_config["att_threshold_deg"])
    roll_excess = max(0.0, abs(np.radians(obs["roll"])) - att_thresh)
    pitch_excess = max(0.0, abs(np.radians(obs["pitch"])) - att_thresh)
    reward -= reward_config["att_coef"] * (roll_excess ** 2 + pitch_excess ** 2)

    # Proximity bonus — incentivize staying near gate
    if obs["dist_to_next_gate"] < reward_config["proximity_radius_cm"]:
        reward += reward_config["proximity_bonus"]

    # Gate passed bonus
    gp_prev = prev_obs["gates_passed"] if prev_obs is not None else 0
    if obs["gates_passed"] > gp_prev:
        reward += reward_config["gate_bonus"] * (obs["gates_passed"] - gp_prev)

    return float(reward)


# --- Episode checks ---

def check_episode_end(obs_dict: dict, step_count: int, total_gates: int, max_dist_gate_cm: float) -> tuple[bool, bool, str]:
    if obs_dict["crashed"]:
        return True, False, "crash"
    if step_count > GRACE_STEPS and obs_dict["dist_to_next_gate"] > max_dist_gate_cm:
        return True, False, "too_far"
    if obs_dict["gates_passed"] >= total_gates:
        return True, False, "success"
    if step_count >= MAX_EPISODE_STEPS:
        return False, True, "timeout"
    return False, False, ""


# --- Sim helpers ---

def load_checkpoint(load_path: str):
    from pathlib import Path
    lp = Path(load_path)
    if not lp.is_file() and lp.suffix != ".zip":
        lp_zip = lp.with_suffix(".zip")
        if lp_zip.is_file():
            lp = lp_zip
    if not lp.is_file():
        raise SystemExit(f"Checkpoint not found: {load_path}")
    return lp


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


def remove_all_aircraft(sim: PteroSim) -> None:
    for status in list(sim.aircraft_status()):
        Aircraft(sim, status.instance_id).remove()


def setup_race(sim: PteroSim, aircraft_class: str) -> tuple[int, int]:
    """One-time setup: spawn aircraft and gates, enable PID. Returns (drone_id, total_gates)."""
    drone = sim.spawn(aircraft_class, **DRONE_SPAWN)
    drone_id = drone.instance_id
    sim.set_track_gates(GATE_POSITIONS)
    sim.reset_all_races()
    sim.reset_race(drone_id)
    sim.start()
    wait_for_aircraft_status(sim, instance_id=drone_id)
    sim.set_attitude_command(
        drone_id,
        roll_rad=0.0, pitch_rad=0.0, yaw_rate_rad_sec=0.0,
        throttle=HOVER_THROTTLE, enabled=True,
    )

    track = sim.get_track_info()
    return drone_id, track.gate_count


def reset_race_session(sim: PteroSim) -> int:
    """Reset between episodes: stop/start. Controller survives on FDMComponent."""
    sim.stop()
    sim.reset_all_races()
    statuses = sim.aircraft_status()
    assert statuses, "No aircraft after stop"
    drone_id = statuses[0].instance_id
    sim.start()
    wait_for_aircraft_status(sim, instance_id=drone_id)
    return drone_id


# --- Random mode ---

def run_random(
    sim_addr: str,
    aircraft: str,
    episodes: int,
    max_dist_gate_cm: float,
    seed: int | None,
    time_scale: float = DEFAULT_TIME_SCALE,
) -> None:
    with PteroSim(sim_addr) as sim:
        sim.set_physics_frequency(PHYSICS_HZ)
        sim.set_time_scale(time_scale)

        drone_id, total_gates = setup_race(sim, aircraft)
        rng = np.random.default_rng(seed)

        for ep in range(episodes):
            obs_dict = get_observation(sim, drone_id)
            total_r = 0.0
            reason = "timeout"

            for t in range(MAX_EPISODE_STEPS):
                action = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
                des_roll, des_pitch, yaw_rate, throttle = action_to_commands(action)

                send_attitude_command(sim, drone_id, des_roll, des_pitch, yaw_rate, throttle)
                sim.step_once()

                next_dict = get_observation(sim, drone_id)
                terminated, truncated, reason = check_episode_end(next_dict, t + 1, total_gates, max_dist_gate_cm)
                done = terminated or truncated

                r = compute_reward(next_dict, obs_dict, done, reason)
                total_r += r
                obs_dict = next_dict

                if t % 50 == 0:
                    print(f"  [EP{ep} t={t}] pos=({obs_dict['x']:.0f},{obs_dict['y']:.0f},{obs_dict['z']:.0f}) "
                          f"dist_gate={obs_dict['dist_to_next_gate']:.0f}")

                if done:
                    break

            print(f"[random EP {ep}] {reason=} steps={t+1} gates={obs_dict['gates_passed']}/{total_gates} R={total_r:.1f}")
            drone_id = reset_race_session(sim)

        sim.stop()
        remove_all_aircraft(sim)


# --- Gym Env ---

if gym is not None and spaces is not None:

    class PteroRaceEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(
            self,
            sim_address: str = DEFAULT_SIM_ADDRESS,
            aircraft_class: str = DEFAULT_AIRCRAFT_CLASS,
            max_dist_from_next_gate_cm: float = MAX_DIST_FROM_NEXT_GATE_CM,
            time_scale: float = DEFAULT_TIME_SCALE,
        ):
            super().__init__()
            self.sim_address = sim_address
            self.aircraft_class = aircraft_class
            self.max_dist_from_next_gate_cm = max_dist_from_next_gate_cm
            self.time_scale = time_scale
            # Action: roll, pitch, throttle, yaw_rate (all [-1,1])
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            )
            self.observation_space = spaces.Box(
                low=-OBS_CLIP, high=OBS_CLIP, shape=(OBS_DIM,), dtype=np.float32
            )
            self._sim: PteroSim | None = None
            self._drone_id = 0
            self._total_gates = 0
            self._step_count = 0
            self._prev_obs_dict: dict | None = None
            self._setup_done = False

        def _connect(self) -> None:
            if self._sim is not None:
                return
            self._sim = PteroSim(self.sim_address)
            self._sim.set_physics_frequency(PHYSICS_HZ)
            self._sim.set_time_scale(self.time_scale)

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            self._connect()
            assert self._sim is not None

            if not self._setup_done:
                self._drone_id, self._total_gates = setup_race(
                    self._sim, self.aircraft_class
                )
                self._setup_done = True
            else:
                self._drone_id = reset_race_session(self._sim)

            self._step_count = 0
            self._prev_obs_dict = None
            obs_dict = get_observation(self._sim, self._drone_id)
            self._prev_obs_dict = obs_dict
            return obs_to_vector(obs_dict), {}

        def step(self, action):
            assert self._sim is not None
            self._step_count += 1

            des_roll, des_pitch, yaw_rate, throttle = action_to_commands(action)

            send_attitude_command(self._sim, self._drone_id,
                                 des_roll, des_pitch, yaw_rate, throttle)
            self._sim.step_once()

            obs_dict = get_observation(self._sim, self._drone_id)
            obs = obs_to_vector(obs_dict)

            terminated, truncated, reason = check_episode_end(
                obs_dict, self._step_count, self._total_gates, self.max_dist_from_next_gate_cm
            )

            reward = compute_reward(obs_dict, self._prev_obs_dict, terminated or truncated, reason)
            self._prev_obs_dict = obs_dict
            return obs, float(reward), terminated, truncated, {}

        def close(self):
            if self._sim is not None:
                try:
                    self._sim.stop()
                    remove_all_aircraft(self._sim)
                except Exception:
                    pass
                self._sim.close()
                self._sim = None


# --- Training ---

def run_train(
    sim_addr: str,
    aircraft: str,
    timesteps: int,
    max_dist_gate_cm: float,
    tensorboard_log: str | None,
    run_name: str,
    load_path: str | None,
    save_path: str,
    seed: int | None,
    time_scale: float = DEFAULT_TIME_SCALE,
    algo: str = "sac",
) -> None:
    if gym is None or spaces is None:
        raise SystemExit("Install gymnasium: pip install gymnasium")
    from pathlib import Path

    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    algo_cls = {"ppo": PPO, "sac": SAC}[algo]

    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    env = Monitor(PteroRaceEnv(
        sim_address=sim_addr,
        aircraft_class=aircraft,
        max_dist_from_next_gate_cm=max_dist_gate_cm,
        time_scale=time_scale,
    ))
    env.reset(seed=seed)
    try:
        if tensorboard_log:
            Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

        if load_path:
            lp = load_checkpoint(load_path)
            load_kwargs: dict = {"env": env, "verbose": 1}
            if tensorboard_log:
                load_kwargs["tensorboard_log"] = tensorboard_log
            model = algo_cls.load(str(lp), **load_kwargs)
            print(f"Loaded {algo.upper()} policy from {load_path}, continuing for {timesteps} timesteps")
            reset_num = False
        else:
            common_kwargs: dict = {
                "policy": "MlpPolicy",
                "env": env,
                "verbose": 1,
                "seed": seed,
            }
            if tensorboard_log:
                common_kwargs["tensorboard_log"] = tensorboard_log
            if algo == "sac":
                common_kwargs.update(
                    learning_rate=1e-4,
                    buffer_size=200_000,
                    batch_size=256,
                    tau=0.01,
                    gamma=0.99,
                    train_freq=4,
                    gradient_steps=4,
                    learning_starts=2000,
                    ent_coef="auto_0.1",
                    policy_kwargs={"net_arch": [256, 256]},
                )
            model = algo_cls(**common_kwargs)
            reset_num = True

        callbacks = []
        progress_total = timesteps if reset_num else timesteps + model.num_timesteps
        if ProgressCallback:
            callbacks.append(ProgressCallback(progress_total))
        callbacks.append(CheckpointCallback(
            save_freq=10_000,
            save_path="checkpoints/",
            name_prefix=f"{algo}_pterorace",
            save_replay_buffer=False,
        ))
        model.learn(
            total_timesteps=timesteps,
            tb_log_name=run_name,
            reset_num_timesteps=reset_num,
            callback=CallbackList(callbacks),
        )
        model.save(save_path)
        print(f"Saved policy to {save_path}")
        if tensorboard_log:
            print(
                f"TensorBoard: tensorboard --logdir {tensorboard_log}  "
                    "(PowerShell, second window) -> http://localhost:6006"
            )
    finally:
        env.close()


# --- Play ---

def run_play(
    sim_addr: str,
    aircraft: str,
    episodes: int,
    load_path: str,
    max_dist_gate_cm: float,
    time_scale: float = DEFAULT_TIME_SCALE,
) -> None:
    from stable_baselines3 import PPO, SAC

    lp = load_checkpoint(load_path)
    try:
        model = SAC.load(str(lp))
    except Exception:
        model = PPO.load(str(lp))

    with PteroSim(sim_addr) as sim:
        sim.set_physics_frequency(PHYSICS_HZ)
        sim.set_time_scale(time_scale)

        drone_id, total_gates = setup_race(sim, aircraft)

        sim.hold()
        print(f"Drone spawned (id={drone_id}). Press Enter to start playback...")
        input()
        sim.start()

        for ep in range(episodes):
            obs_dict = get_observation(sim, drone_id)
            prev_obs_dict = None
            obs = obs_to_vector(obs_dict)
            total_r = 0.0
            steps = 0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                des_roll, des_pitch, yaw_rate, throttle = action_to_commands(action)

                send_attitude_command(sim, drone_id, des_roll, des_pitch, yaw_rate, throttle)
                sim.step_once()
                time.sleep(time_scale / PHYSICS_HZ)

                prev_obs_dict = obs_dict
                obs_dict = get_observation(sim, drone_id)
                obs = obs_to_vector(obs_dict)
                steps += 1

                terminated, truncated, reason = check_episode_end(
                    obs_dict, steps, total_gates, max_dist_gate_cm
                )
                done = terminated or truncated

                total_r += compute_reward(obs_dict, prev_obs_dict, done, reason)
                if done:
                    break

            print(f"[play EP {ep}] {reason=} steps={steps} gates={obs_dict['gates_passed']}/{total_gates} reward={total_r:.1f}")
            drone_id = reset_race_session(sim)
            sim.hold()

        sim.stop()
        remove_all_aircraft(sim)


# --- Optuna hyperparameter search ---

def run_optuna(
    sim_addr: str,
    aircraft: str,
    n_trials: int,
    timesteps_per_trial: int,
    max_dist_gate_cm: float,
    time_scale: float,
    seed: int | None,
    study_name: str = "pterorace_sac",
    storage: str | None = None,
) -> None:
    """Run Optuna hyperparameter optimization for SAC.

    Each trial trains for `timesteps_per_trial` steps with sampled hyperparameters,
    then reports mean episode reward over the last 20 episodes.
    """
    try:
        import optuna
    except ImportError:
        raise SystemExit("Install optuna: pip install optuna")

    if gym is None or spaces is None:
        raise SystemExit("Install gymnasium: pip install gymnasium")

    from pathlib import Path
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback

    Path("optuna_checkpoints").mkdir(parents=True, exist_ok=True)
    Path("optuna_logs").mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        # --- Sample hyperparameters ---
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        tau = trial.suggest_float("tau", 0.005, 0.05, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        train_freq = trial.suggest_categorical("train_freq", [1, 2, 4, 8])
        gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4, 8])
        ent_coef_init = trial.suggest_float("ent_coef_init", 0.01, 0.5, log=True)
        net_arch_size = trial.suggest_categorical("net_arch_size", [128, 256, 512])
        net_arch_layers = trial.suggest_int("net_arch_layers", 1, 3)

        # --- Sample reward coefficients ---
        reward_config["approach_scale"] = trial.suggest_float("approach_scale", 0.01, 0.2, log=True)
        reward_config["gate_bonus"] = trial.suggest_float("gate_bonus", 20.0, 500.0, log=True)
        reward_config["att_coef"] = trial.suggest_float("att_coef", 0.005, 0.5, log=True)
        reward_config["att_threshold_deg"] = trial.suggest_float("att_threshold_deg", 10.0, 45.0)
        reward_config["proximity_bonus"] = trial.suggest_float("proximity_bonus", 0.0, 0.5)
        reward_config["crash_penalty"] = -trial.suggest_float("crash_penalty_abs", 10.0, 200.0, log=True)
        reward_config["too_far_penalty"] = -trial.suggest_float("too_far_penalty_abs", 5.0, 100.0, log=True)

        net_arch = [net_arch_size] * net_arch_layers

        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: lr={lr:.1e} gamma={gamma:.4f} tau={tau:.4f}")
        print(f"  batch={batch_size} train_freq={train_freq} grad_steps={gradient_steps}")
        print(f"  ent_coef_init={ent_coef_init:.3f} net_arch={net_arch}")
        print(f"  reward: approach={reward_config['approach_scale']:.3f} "
              f"gate_bonus={reward_config['gate_bonus']:.0f} "
              f"att_coef={reward_config['att_coef']:.3f} "
              f"att_thresh={reward_config['att_threshold_deg']:.0f}°")
        print(f"{'='*60}")

        env = Monitor(PteroRaceEnv(
            sim_address=sim_addr,
            aircraft_class=aircraft,
            max_dist_from_next_gate_cm=max_dist_gate_cm,
            time_scale=time_scale,
        ))

        try:
            env.reset(seed=seed)

            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=lr,
                buffer_size=200_000,
                batch_size=batch_size,
                tau=tau,
                gamma=gamma,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                learning_starts=1000,
                ent_coef=f"auto_{ent_coef_init}",
                policy_kwargs={"net_arch": net_arch},
                verbose=0,
                seed=seed,
                tensorboard_log="optuna_logs",
            )

            # Pruning callback — report intermediate results
            class OptunaCallback(BaseCallback):
                def __init__(self, trial: optuna.Trial, eval_freq: int = 2000):
                    super().__init__()
                    self._trial = trial
                    self._eval_freq = eval_freq
                    self._last_eval = 0

                def _on_step(self) -> bool:
                    if self.num_timesteps - self._last_eval >= self._eval_freq:
                        self._last_eval = self.num_timesteps
                        # Get mean reward from Monitor wrapper
                        if len(self.training_env.get_attr("get_episode_rewards")[0]()) > 0:
                            rewards = self.training_env.get_attr("get_episode_rewards")[0]()
                            mean_rew = float(np.mean(rewards[-20:]))
                            self._trial.report(mean_rew, self.num_timesteps)
                            if self._trial.should_prune():
                                raise optuna.TrialPruned()
                    return True

            model.learn(
                total_timesteps=timesteps_per_trial,
                tb_log_name=f"optuna_trial_{trial.number}",
                callback=OptunaCallback(trial),
            )

            # Final score: mean reward over last 20 episodes
            rewards = env.get_episode_rewards()
            if len(rewards) < 5:
                return -1000.0  # not enough episodes
            mean_reward = float(np.mean(rewards[-20:]))

            # Save best trial checkpoint
            model.save(f"optuna_checkpoints/trial_{trial.number}")
            print(f"Trial {trial.number} done: mean_reward={mean_reward:.1f} (last 20 eps, total {len(rewards)} eps)")

            return mean_reward

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return -1000.0
        finally:
            env.close()
            # Reset reward config for next trial
            reward_config.update(REWARD_DEFAULTS)

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5000,
        ),
        load_if_exists=True,
    )

    print(f"Starting Optuna study '{study_name}' with {n_trials} trials, {timesteps_per_trial} steps each")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Print results
    print(f"\n{'='*60}")
    print("OPTUNA RESULTS")
    print(f"{'='*60}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best reward: {study.best_value:.1f}")
    print(f"\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nBest checkpoint: optuna_checkpoints/trial_{study.best_trial.number}.zip")
    print(f"{'='*60}")


# --- Main ---

def main() -> int:
    parser = argparse.ArgumentParser(description="PteroSim race RL trainer (C++ attitude PID)")
    parser.add_argument("--sim", default=DEFAULT_SIM_ADDRESS)
    parser.add_argument("--aircraft", default=DEFAULT_AIRCRAFT_CLASS)
    parser.add_argument("--mode", choices=["random", "train", "play", "optuna"], default="random")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--max-dist-gate",
        type=float,
        default=MAX_DIST_FROM_NEXT_GATE_CM,
    )
    parser.add_argument("--tensorboard-log", type=str, default="tensorboard_logs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--time-scale", type=float, default=DEFAULT_TIME_SCALE)
    # Optuna args
    parser.add_argument("--optuna-trials", type=int, default=50)
    parser.add_argument("--optuna-timesteps", type=int, default=30_000)
    parser.add_argument("--optuna-study-name", type=str, default="pterorace_sac")
    parser.add_argument("--optuna-storage", type=str, default=None,
                        help="Optuna storage URL (e.g. sqlite:///optuna.db). None = in-memory.")
    args = parser.parse_args()

    ts = args.time_scale
    algo = args.algo
    run_name = args.run_name or f"{algo.upper()}_race"
    save_path = args.save or f"checkpoints/{algo}_pterorace"

    if args.mode == "random":
        run_random(args.sim, args.aircraft, args.episodes, args.max_dist_gate, args.seed, time_scale=ts)
    elif args.mode == "play":
        load_p = args.load.strip()
        if not load_p:
            raise SystemExit("--mode play requires --load <checkpoint.zip>")
        run_play(args.sim, args.aircraft, args.episodes, load_p, args.max_dist_gate, time_scale=ts)
    elif args.mode == "optuna":
        run_optuna(
            args.sim, args.aircraft,
            n_trials=args.optuna_trials,
            timesteps_per_trial=args.optuna_timesteps,
            max_dist_gate_cm=args.max_dist_gate,
            time_scale=ts,
            seed=args.seed,
            study_name=args.optuna_study_name,
            storage=args.optuna_storage,
        )
    else:
        tb = args.tensorboard_log.strip() or None
        load_p = args.load.strip() or None
        N_STEPS = 2048
        timesteps = args.timesteps if args.timesteps is not None else args.max_iterations * N_STEPS
        run_train(
            args.sim, args.aircraft, timesteps, args.max_dist_gate,
            tensorboard_log=tb, run_name=run_name, load_path=load_p,
            save_path=save_path, seed=args.seed, time_scale=ts, algo=algo,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
