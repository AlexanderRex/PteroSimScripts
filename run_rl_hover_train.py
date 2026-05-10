"""
Single-drone PPO hover trainer for PteroSim — raw motor control (no PID).

Architecture (free-run):
  Simulation runs continuously at PHYSICS_HZ × time_scale.
  RL agent sends raw motor commands and reads observations via gRPC.
  No step_once — physics never pauses during training.
    → gRPC set_actuator_controls (N channels from aircraft config)
    → JSBSim motors at 1000 Hz (continuous)
    → gRPC get observations (async read)

Action space (4-dim policy output u in [-1, 1]):
  Delta around trim hover throttle (Genesis-style): for each motor i,
    throttle_i = clip(HOVER_THROTTLE + ACTION_DELTA_MAX * u_i, 0, 1)
  Extra actuator channels (if any) are set to 0.
  prev_action in observations is the policy output u (not raw throttle).

Observation (19 dims): unchanged layout; prev_action = last u.

Target: (spawn_x, spawn_y, spawn_z + HOVER_HEIGHT_ABOVE_SPAWN_CM).

Checkpoint note: policies trained with absolute-throttle semantics are incompatible
with delta-throttle semantics — retrain from scratch after this change.
PPO .zip archives are not loadable as SAC and vice versa.

Usage:
  python run_rl_hover_train.py --timesteps 500000
  python run_rl_hover_train.py --timesteps 500000 --load checkpoints/ppo_hover.zip
  python run_rl_hover_train.py --mode play --load checkpoints/ppo_hover.zip

Note: PPO and SAC checkpoints are not interchangeable.

TensorBoard:
  tensorboard --logdir tensorboard_logs
"""

from __future__ import annotations

import argparse
import math
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
        """Prints ETA, ep count, avg reward every print_freq steps."""

        def __init__(self, total_timesteps: int, print_freq: int = 2048, verbose: int = 1):
            super().__init__(verbose)
            self._total = total_timesteps
            self._print_freq = print_freq
            self._start_time: float = 0.0
            self._ep_count = 0
            self._last_print = 0
            self._recent_rewards: list[float] = []
            self._reason_counts: dict[str, int] = {}

        def _on_training_start(self) -> None:
            self._start_time = time.monotonic()
            self._ep_count = 0
            self._last_print = 0
            self._recent_rewards.clear()
            self._reason_counts = {}

        def _on_step(self) -> bool:
            # Track episode rewards from Monitor wrapper infos
            infos = self.locals.get("infos", [])
            for info in infos if isinstance(infos, list) else [infos]:
                if isinstance(info, dict) and "episode" in info:
                    self._recent_rewards.append(info["episode"]["r"])
                    self._ep_count += 1
                    reason = info.get("reason")
                    ep = info.get("episode")
                    if reason is None and isinstance(ep, dict):
                        reason = ep.get("reason")
                    if isinstance(reason, str) and reason:
                        self._reason_counts[reason] = self._reason_counts.get(reason, 0) + 1

            if self.num_timesteps - self._last_print >= self._print_freq:
                self._last_print = self.num_timesteps
                elapsed = time.monotonic() - self._start_time
                remaining = self._total - self.num_timesteps
                fps = self.num_timesteps / max(elapsed, 1e-6)
                eta_min = remaining / max(fps, 1e-6) / 60.0

                if self._recent_rewards:
                    last_n = self._recent_rewards[-20:]
                    ep_rew = sum(last_n) / len(last_n)
                    ep_len_str = f" ep_len={self._recent_rewards[-1]:.0f}" if self._recent_rewards else ""
                else:
                    ep_rew = float("nan")
                    ep_len_str = ""

                reason_str = (
                    " ".join(f"{k}={v}" for k, v in sorted(self._reason_counts.items()))
                    if self._reason_counts
                    else ""
                )

                pct = 100.0 * self.num_timesteps / self._total
                print(
                    f"[{pct:5.1f}%] steps={self.num_timesteps}/{self._total} "
                    f"eps={self._ep_count} fps={fps:.0f} "
                    f"ep_rew={ep_rew:.1f}{ep_len_str} "
                    f"{reason_str} "
                    f"ETA={eta_min:.1f}min"
                )
                self._reason_counts = {}
            return True

except ImportError:
    BaseCallback = None
    ProgressCallback = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_SIM_ADDRESS = "localhost:10010"
DEFAULT_AIRCRAFT_CLASS = "F450"

SPAWN_Z_CM = 100.0
HOVER_HEIGHT_ABOVE_SPAWN_CM = 200.0

DRONE_SPAWN = {"x": 0.0, "y": 0.0, "z": SPAWN_Z_CM, "yaw": 0.0}

PHYSICS_HZ = 1000.0
DEFAULT_TIME_SCALE = 50.0   # sim speed multiplier for training
AGENT_DT_S = 0.05           # 20 Hz agent (in sim-seconds)

# Delta action: throttle_i = clip(HOVER_THROTTLE + ACTION_DELTA_MAX * u_i, 0, 1), u_i in [-1, 1]
HOVER_THROTTLE = 0.4
ACTION_DELTA_MAX = 0.2

# Episode
MAX_EPISODE_STEPS = 400      # 400 * 0.05s = 20 sim-seconds

# Termination
MAX_DIST_FROM_TARGET_CM = 700.0   # 7 m — unrecoverable
TILT_MAX_DEG = 65.0

# Observation
OBS_DIM = 19
NORM_POS_CM = 500.0
NORM_VEL_CMS = 500.0
NORM_ATT_RAD = np.pi
NORM_GYRO = 10.0       # rad/s
NORM_ACCEL = 20.0      # m/s^2
OBS_CLIP = 5.0

# Reward weights (tune here). Terminal keys apply only on the final step of an episode.
# Tiered location shaping: no xy/z penalties. (1) survival_bonus every alive step.
# (2) approach_progress_coef * delta_dist toward target while outside hover_zone_radius_cm.
# (3) hover_zone_bonus per step while dist <= hover_zone_radius_cm.
REWARD_WEIGHTS: dict[str, float] = {
    "survival_bonus": 0.12,
    "approach_progress_coef": 0.002,
    "hover_zone_radius_cm": 60.0,
    "hover_zone_bonus": 0.18,
    "attitude_penalty_scale": 0.32,
    "attitude_sq_cap_rad2": 0.35,
    "yaw_hold_scale": 0.018,
    "gyro_penalty_scale": 0.02,
    "gyro_sq_cap": 12.0,
    "accel_penalty_scale": 0.025,
    "accel_sq_cap": 10.0,
    "action_smooth_coef": 0.09,
    "crash_penalty": -40.0,
    "drift_penalty": -26.0,
    "tilt_penalty": -24.0,
    "timeout_reward": 10.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def angle_wrap_rad(delta_rad: float) -> float:
    return float(math.atan2(math.sin(delta_rad), math.cos(delta_rad)))


def policy_to_actuator_controls(policy_u: np.ndarray, num_channels: int) -> list[float]:
    """Map policy output u in [-1,1]^4 to per-motor throttle; pad/truncate to num_channels."""
    u = np.clip(np.asarray(policy_u, dtype=np.float64), -1.0, 1.0)
    motors = [
        float(np.clip(HOVER_THROTTLE + ACTION_DELTA_MAX * float(u[i]), 0.0, 1.0))
        for i in range(4)
    ]
    n_m = min(4, max(0, num_channels))
    out = motors[:n_m]
    while len(out) < num_channels:
        out.append(0.0)
    return [float(x) for x in out]


def wait_for_aircraft_status(
    sim: PteroSim,
    instance_id: int | None = None,
    timeout_s: float = 2.0,
) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        statuses = sim.aircraft_status()
        if instance_id is None and statuses:
            return statuses[0]
        if instance_id is not None:
            for s in statuses:
                if s.instance_id == instance_id:
                    return s
        time.sleep(0.05)
    raise RuntimeError(f"Aircraft status timeout (instance_id={instance_id})")


def remove_all_aircraft(sim: PteroSim) -> None:
    for s in list(sim.aircraft_status()):
        Aircraft(sim, s.instance_id).remove()


def get_obs_raw(sim: PteroSim, instance_id: int) -> dict[str, Any]:
    """Read position/attitude/IMU for one drone."""
    statuses = sim.aircraft_status()
    status = None
    for s in statuses:
        if s.instance_id == instance_id:
            status = s
            break
    if status is None:
        return {"crashed": True}
    if status.crashed:
        return {"crashed": True, "x": status.x, "y": status.y, "z": status.z}

    imu = sim.get_imu(instance_id)
    return {
        "x": status.x, "y": status.y, "z": status.z,
        "roll_deg": status.roll,
        "pitch_deg": status.pitch,
        "yaw_deg": status.yaw,
        "accel": imu.acceleration,
        "gyro": imu.angular_velocity,
        "crashed": False,
    }


def build_obs(
    raw: dict,
    target: np.ndarray,
    prev_pos: np.ndarray | None,
    prev_action: np.ndarray,
) -> np.ndarray:
    """Build normalized 19-dim observation vector."""
    if raw.get("crashed"):
        return np.zeros(OBS_DIM, dtype=np.float32)

    pos = np.array([raw["x"], raw["y"], raw["z"]], dtype=np.float32)
    pos_err = (target - pos) / NORM_POS_CM

    if prev_pos is not None:
        vel_norm = (pos - prev_pos) / (AGENT_DT_S * NORM_VEL_CMS)
    else:
        vel_norm = np.zeros(3, dtype=np.float32)

    att = np.array([
        np.radians(raw["roll_deg"]),
        np.radians(raw["pitch_deg"]),
        np.radians(raw["yaw_deg"]),
    ], dtype=np.float32) / NORM_ATT_RAD

    gyro_norm = np.array(raw["gyro"], dtype=np.float32) / NORM_GYRO
    accel_norm = np.array(raw["accel"], dtype=np.float32) / NORM_ACCEL

    obs = np.concatenate([pos_err, vel_norm, att, gyro_norm, accel_norm, prev_action])
    obs = np.nan_to_num(obs, nan=0.0, posinf=OBS_CLIP, neginf=-OBS_CLIP)
    return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)


# ---------------------------------------------------------------------------
# Gym Env
# ---------------------------------------------------------------------------

if gym is not None and spaces is not None:

    class PteroHoverEnv(gym.Env):
        """Single-drone hover env with raw motor control, free-run physics."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            sim_address: str = DEFAULT_SIM_ADDRESS,
            aircraft_class: str = DEFAULT_AIRCRAFT_CLASS,
            time_scale: float = DEFAULT_TIME_SCALE,
        ):
            super().__init__()
            self.sim_address = sim_address
            self.aircraft_class = aircraft_class
            self.time_scale = time_scale

            self.observation_space = spaces.Box(
                low=-OBS_CLIP, high=OBS_CLIP, shape=(OBS_DIM,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            )

            self._sim: PteroSim | None = None
            self._drone_id = 0
            self._num_channels = 5
            self._spawn_yaw_rad = float(np.radians(DRONE_SPAWN["yaw"]))
            self._target = np.array(
                [
                    DRONE_SPAWN["x"],
                    DRONE_SPAWN["y"],
                    DRONE_SPAWN["z"] + HOVER_HEIGHT_ABOVE_SPAWN_CM,
                ],
                dtype=np.float32,
            )

            self._step_count = 0
            self._prev_pos: np.ndarray | None = None
            self._prev_action = np.zeros(4, dtype=np.float32)
            self._episode_reward = 0.0
            self._setup_done = False
            self._prev_dist: float | None = None

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
                # First reset: spawn drone
                drone = self._sim.spawn(
                    self.aircraft_class, **DRONE_SPAWN
                )
                self._drone_id = drone.instance_id
                self._sim.start()
                wait_for_aircraft_status(self._sim, self._drone_id)
                try:
                    cfg = self._sim.get_actuator_configuration(self._drone_id)
                    self._num_channels = int(getattr(cfg, "channel_count", self._num_channels))
                except Exception:
                    pass
                # Disable attitude controller — raw motor control
                self._sim.set_attitude_command(
                    self._drone_id,
                    roll_rad=0.0, pitch_rad=0.0,
                    yaw_rate_rad_sec=0.0, throttle=0.0, enabled=False,
                )
                self._setup_done = True
            else:
                # Subsequent resets: stop/start (sim restarts JSBSim at spawn)
                self._sim.stop()
                self._sim.start()
                wait_for_aircraft_status(self._sim, self._drone_id)
                try:
                    cfg = self._sim.get_actuator_configuration(self._drone_id)
                    self._num_channels = int(getattr(cfg, "channel_count", self._num_channels))
                except Exception:
                    pass
                self._sim.set_attitude_command(
                    self._drone_id,
                    roll_rad=0.0, pitch_rad=0.0,
                    yaw_rate_rad_sec=0.0, throttle=0.0, enabled=False,
                )

            neutral_u = np.zeros(4, dtype=np.float32)
            hover_t = policy_to_actuator_controls(neutral_u, self._num_channels)
            self._sim.set_actuator_controls(self._drone_id, hover_t)

            self._step_count = 0
            self._prev_pos = None
            self._prev_action = neutral_u.copy()
            self._episode_reward = 0.0
            self._prev_dist = None

            raw = get_obs_raw(self._sim, self._drone_id)
            obs = build_obs(raw, self._target, None, self._prev_action)
            return obs, {}

        def step(self, action):
            assert self._sim is not None
            self._step_count += 1
            action = np.asarray(action, dtype=np.float32)

            # Send motor commands (policy = delta around HOVER_THROTTLE)
            self._sim.set_actuator_controls(
                self._drone_id,
                policy_to_actuator_controls(action, self._num_channels),
            )

            # Let physics run
            time.sleep(AGENT_DT_S / self.time_scale)

            # Read state
            raw = get_obs_raw(self._sim, self._drone_id)

            # Check termination
            terminated = False
            truncated = False
            reason = ""

            if raw.get("crashed"):
                terminated = True
                reason = "crash"
            else:
                pos = np.array([raw["x"], raw["y"], raw["z"]], dtype=np.float32)
                dist = float(np.linalg.norm(pos - self._target))
                roll_deg = float(raw["roll_deg"])
                pitch_deg = float(raw["pitch_deg"])
                tilt_max = max(abs(roll_deg), abs(pitch_deg))

                if tilt_max > TILT_MAX_DEG:
                    terminated = True
                    reason = "tilt"
                elif dist > MAX_DIST_FROM_TARGET_CM:
                    terminated = True
                    reason = "too_far"

                if not terminated and self._step_count >= MAX_EPISODE_STEPS:
                    truncated = True
                    reason = "timeout"

            # Reward
            done = terminated or truncated
            if reason == "crash":
                reward = REWARD_WEIGHTS["crash_penalty"]
            elif reason == "too_far":
                reward = REWARD_WEIGHTS["drift_penalty"]
            elif reason == "tilt":
                reward = REWARD_WEIGHTS["tilt_penalty"]
            elif reason == "timeout":
                reward = REWARD_WEIGHTS["timeout_reward"]
            else:
                pos = np.array([raw["x"], raw["y"], raw["z"]], dtype=np.float32)
                err = self._target - pos
                dist = float(np.linalg.norm(err))
                r_hover = float(REWARD_WEIGHTS["hover_zone_radius_cm"])
                in_hover = dist <= r_hover

                survival = REWARD_WEIGHTS["survival_bonus"]

                approach_bonus = 0.0
                if not in_hover and self._prev_dist is not None:
                    approach_bonus = REWARD_WEIGHTS["approach_progress_coef"] * max(
                        0.0, float(self._prev_dist) - dist
                    )

                hover_bonus = REWARD_WEIGHTS["hover_zone_bonus"] if in_hover else 0.0

                roll_r = math.radians(float(raw["roll_deg"]))
                pitch_r = math.radians(float(raw["pitch_deg"]))
                att_sq = roll_r**2 + pitch_r**2
                att_sq = min(att_sq, float(REWARD_WEIGHTS["attitude_sq_cap_rad2"]))
                att_pen = -REWARD_WEIGHTS["attitude_penalty_scale"] * att_sq

                yaw_r = np.radians(float(raw["yaw_deg"]))
                yaw_err = angle_wrap_rad(float(yaw_r) - self._spawn_yaw_rad)
                yaw_pen = -REWARD_WEIGHTS["yaw_hold_scale"] * (yaw_err**2)

                gyro = np.asarray(raw["gyro"], dtype=np.float64)
                gyro_sq = float(np.sum(gyro**2))
                gyro_sq = min(gyro_sq, float(REWARD_WEIGHTS["gyro_sq_cap"]))
                gyro_pen = -REWARD_WEIGHTS["gyro_penalty_scale"] * gyro_sq

                accel = np.asarray(raw["accel"], dtype=np.float64)
                accel_sq = float(np.sum(accel**2))
                accel_sq = min(accel_sq, float(REWARD_WEIGHTS["accel_sq_cap"]))
                accel_pen = -REWARD_WEIGHTS["accel_penalty_scale"] * accel_sq

                smooth_pen = -REWARD_WEIGHTS["action_smooth_coef"] * float(
                    np.sum(np.abs(action - self._prev_action))
                )

                reward = (
                    survival
                    + approach_bonus
                    + hover_bonus
                    + att_pen
                    + yaw_pen
                    + gyro_pen
                    + accel_pen
                    + smooth_pen
                )

                self._prev_dist = dist

            self._episode_reward += reward

            # Build obs
            obs = build_obs(raw, self._target, self._prev_pos, self._prev_action)

            # Update state
            if not raw.get("crashed"):
                self._prev_pos = np.array(
                    [raw["x"], raw["y"], raw["z"]], dtype=np.float32
                )
            self._prev_action = action.copy()

            info: dict[str, Any] = {}
            if done:
                info["episode"] = {
                    "r": self._episode_reward,
                    "l": self._step_count,
                    "reason": reason,
                }
                info["reason"] = reason

            return obs, float(reward), terminated, truncated, info

        def close(self):
            if self._sim is not None:
                try:
                    self._sim.stop()
                    remove_all_aircraft(self._sim)
                except Exception:
                    pass
                self._sim.close()
                self._sim = None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_train(
    sim_addr: str,
    aircraft: str,
    timesteps: int,
    tensorboard_log: str | None,
    run_name: str,
    load_path: str | None,
    save_path: str,
    seed: int | None,
    time_scale: float = DEFAULT_TIME_SCALE,
) -> None:
    if gym is None:
        raise SystemExit("Install gymnasium: pip install gymnasium")

    from pathlib import Path
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    # --save path without .zip: same stem/dir for periodic checkpoints and final .zip
    save_base = Path(save_path).expanduser()
    if save_base.suffix.lower() == ".zip":
        save_base = save_base.with_suffix("")
    save_base.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(save_base.parent)
    checkpoint_prefix = save_base.name

    env = Monitor(PteroHoverEnv(
        sim_address=sim_addr,
        aircraft_class=aircraft,
        time_scale=time_scale,
    ))

    model: PPO | None = None
    try:
        if tensorboard_log:
            Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

        if load_path:
            lp = Path(load_path)
            if not lp.is_file():
                lp = lp.with_suffix(".zip")
            if not lp.is_file():
                raise SystemExit(f"Checkpoint not found: {load_path}")

            model = PPO.load(
                str(lp), env=env, verbose=1, tensorboard_log=tensorboard_log or None
            )
            print(f"Loaded from {lp}")
            reset_num = False
        else:
            # n_steps: rollout length per update; smaller => more frequent updates on slow sim
            model = PPO(
                policy="MlpPolicy",
                env=env,
                verbose=1,
                seed=seed,
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs={"net_arch": [256, 256]},
                tensorboard_log=tensorboard_log or None,
            )
            reset_num = True

        progress_total = timesteps if reset_num else timesteps + model.num_timesteps

        callbacks: list = []
        if ProgressCallback:
            callbacks.append(ProgressCallback(progress_total))
        callbacks.append(CheckpointCallback(
            save_freq=10_000,
            save_path=checkpoint_dir,
            name_prefix=checkpoint_prefix,
            save_replay_buffer=False,
        ))

        interrupted = False
        try:
            model.learn(
                total_timesteps=timesteps,
                tb_log_name=run_name,
                reset_num_timesteps=reset_num,
                callback=CallbackList(callbacks),
            )
        except KeyboardInterrupt:
            interrupted = True
            print("KeyboardInterrupt — saving latest policy before exit")
        finally:
            if model is not None:
                final_path = str(save_base)
                model.save(final_path)
                print(f"Saved policy to {final_path}.zip")
            if interrupted:
                raise KeyboardInterrupt

        if tensorboard_log:
            print(f"TensorBoard: tensorboard --logdir {tensorboard_log}")
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Play
# ---------------------------------------------------------------------------

def run_play(
    sim_addr: str,
    aircraft: str,
    load_path: str,
    time_scale: float = 1.0,
) -> None:
    from pathlib import Path
    from stable_baselines3 import PPO

    lp = Path(load_path)
    if not lp.is_file():
        lp = lp.with_suffix(".zip")
    if not lp.is_file():
        raise SystemExit(f"Checkpoint not found: {load_path}")

    env = PteroHoverEnv(
        sim_address=sim_addr,
        aircraft_class=aircraft,
        time_scale=time_scale,
    )
    model = PPO.load(str(lp))

    try:
        while True:
            obs, _ = env.reset()
            total_r = 0.0
            steps = 0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_r += reward
                steps += 1

                if steps % 20 == 0:
                    thr = policy_to_actuator_controls(action, getattr(env, "_num_channels", 5))
                    print(
                        f"  step={steps} r={total_r:.2f} "
                        f"thr=[{thr[0]:.2f},{thr[1]:.2f},{thr[2]:.2f},{thr[3]:.2f}]"
                    )

                if terminated or truncated:
                    reason = info.get("reason", "?")
                    print(f"Episode: {reason}, reward={total_r:.2f}, steps={steps}")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------

def run_optuna(
    sim_addr: str,
    aircraft: str,
    n_trials: int,
    timesteps_per_trial: int,
    time_scale: float,
    seed: int | None,
    study_name: str = "hover_ppo",
    storage: str | None = None,
) -> None:
    try:
        import optuna
    except ImportError:
        raise SystemExit("Install optuna: pip install optuna")

    if gym is None:
        raise SystemExit("Install gymnasium: pip install gymnasium")

    from pathlib import Path
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    Path("optuna_checkpoints").mkdir(parents=True, exist_ok=True)
    Path("tensorboard_logs").mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        # --- Sample PPO hyperparameters ---
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.02)
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 15])
        net_arch_size = trial.suggest_categorical("net_arch_size", [128, 256, 512])

        # --- Sample reward coefficients (writes into global REWARD_WEIGHTS) ---
        REWARD_WEIGHTS["survival_bonus"] = trial.suggest_float("survival_bonus", 0.04, 0.22)
        REWARD_WEIGHTS["approach_progress_coef"] = trial.suggest_float(
            "approach_progress_coef", 0.0004, 0.006, log=True
        )
        REWARD_WEIGHTS["hover_zone_radius_cm"] = trial.suggest_float(
            "hover_zone_radius_cm", 25.0, 120.0
        )
        REWARD_WEIGHTS["hover_zone_bonus"] = trial.suggest_float("hover_zone_bonus", 0.05, 0.45)
        REWARD_WEIGHTS["attitude_penalty_scale"] = trial.suggest_float(
            "attitude_penalty_scale", 0.05, 0.8
        )
        REWARD_WEIGHTS["attitude_sq_cap_rad2"] = trial.suggest_float(
            "attitude_sq_cap_rad2", 0.15, 0.7
        )
        REWARD_WEIGHTS["yaw_hold_scale"] = trial.suggest_float("yaw_hold_scale", 0.005, 0.04)
        REWARD_WEIGHTS["gyro_penalty_scale"] = trial.suggest_float(
            "gyro_penalty_scale", 0.005, 0.08, log=True
        )
        REWARD_WEIGHTS["gyro_sq_cap"] = trial.suggest_float("gyro_sq_cap", 4.0, 30.0)
        REWARD_WEIGHTS["accel_penalty_scale"] = trial.suggest_float(
            "accel_penalty_scale", 0.003, 0.06, log=True
        )
        REWARD_WEIGHTS["accel_sq_cap"] = trial.suggest_float("accel_sq_cap", 4.0, 25.0)
        REWARD_WEIGHTS["crash_penalty"] = trial.suggest_float("crash_penalty", -100.0, -10.0)
        REWARD_WEIGHTS["drift_penalty"] = trial.suggest_float("drift_penalty", -60.0, -5.0)
        REWARD_WEIGHTS["action_smooth_coef"] = trial.suggest_float("action_smooth_coef", 0.01, 0.2)

        print(f"\n--- Trial {trial.number} ---")
        print(f"  lr={lr:.1e} gamma={gamma:.4f} gae_lambda={gae_lambda:.3f} "
              f"clip={clip_range:.2f} ent={ent_coef:.4f} n_steps={n_steps} "
              f"batch={batch_size} n_epochs={n_epochs} arch={net_arch_size}")
        print(
            f"  surv={REWARD_WEIGHTS['survival_bonus']:.3f} "
            f"appr={REWARD_WEIGHTS['approach_progress_coef']:.5f} "
            f"hz_r={REWARD_WEIGHTS['hover_zone_radius_cm']:.0f} "
            f"hz_b={REWARD_WEIGHTS['hover_zone_bonus']:.3f} "
            f"accel_w={REWARD_WEIGHTS['accel_penalty_scale']:.4f} "
            f"crash={REWARD_WEIGHTS['crash_penalty']:.0f} "
            f"smooth={REWARD_WEIGHTS['action_smooth_coef']:.3f}"
        )

        env = Monitor(PteroHoverEnv(
            sim_address=sim_addr,
            aircraft_class=aircraft,
            time_scale=time_scale,
        ))

        try:
            model = PPO(
                policy="MlpPolicy",
                env=env,
                verbose=0,
                seed=seed,
                learning_rate=lr,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                policy_kwargs={"net_arch": [net_arch_size, net_arch_size]},
                tensorboard_log="tensorboard_logs",
            )

            callbacks: list = []
            if ProgressCallback:
                callbacks.append(ProgressCallback(timesteps_per_trial, print_freq=5000))

            model.learn(
                total_timesteps=timesteps_per_trial,
                tb_log_name=f"optuna_trial_{trial.number}",
                callback=callbacks[0] if callbacks else None,
            )

            # Evaluate: run 5 episodes, return mean reward
            ep_rewards = []
            for _ in range(5):
                obs, _ = env.reset()
                total_r = 0.0
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_r += reward
                    if terminated or truncated:
                        break
                ep_rewards.append(total_r)

            mean_r = float(np.mean(ep_rewards))
            print(f"  Trial {trial.number}: mean_reward={mean_r:.1f} "
                  f"lr={lr:.1e} n_steps={n_steps} batch={batch_size} arch={net_arch_size}")

            # Save best
            model.save(f"optuna_checkpoints/trial_{trial.number}")

            return mean_r
        finally:
            env.close()

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    print("\n=== Best trial ===")
    print(f"  Value: {study.best_trial.value:.1f}")
    print(f"  Params: {study.best_trial.params}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="PteroSim PPO hover trainer — raw motor control")
    p.add_argument("--mode", choices=["train", "play", "optuna"], default="train")
    p.add_argument("--sim-address", default=DEFAULT_SIM_ADDRESS)
    p.add_argument("--aircraft", default=DEFAULT_AIRCRAFT_CLASS)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--time-scale", type=float, default=DEFAULT_TIME_SCALE)
    p.add_argument("--load", default=None, help="Path to .zip checkpoint")
    p.add_argument("--save", default="checkpoints/ppo_hover", help="Save path (no .zip)")
    p.add_argument("--tensorboard-log", default="tensorboard_logs")
    p.add_argument("--run-name", default="ppo_hover")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--optuna-trials", type=int, default=30)
    p.add_argument("--optuna-timesteps", type=int, default=30_000)
    p.add_argument("--optuna-storage", default=None, help="Optuna storage URL, e.g. sqlite:///optuna_hover.db")
    args = p.parse_args()

    if args.mode == "train":
        run_train(
            sim_addr=args.sim_address,
            aircraft=args.aircraft,
            timesteps=args.timesteps,
            tensorboard_log=args.tensorboard_log,
            run_name=args.run_name,
            load_path=args.load,
            save_path=args.save,
            seed=args.seed,
            time_scale=args.time_scale,
        )
    elif args.mode == "play":
        if not args.load:
            raise SystemExit("--load required for play mode")
        run_play(
            sim_addr=args.sim_address,
            aircraft=args.aircraft,
            load_path=args.load,
            time_scale=args.time_scale,
        )
    elif args.mode == "optuna":
        run_optuna(
            sim_addr=args.sim_address,
            aircraft=args.aircraft,
            n_trials=args.optuna_trials,
            timesteps_per_trial=args.optuna_timesteps,
            time_scale=args.time_scale,
            seed=args.seed,
            storage=args.optuna_storage,
        )


if __name__ == "__main__":
    main()
