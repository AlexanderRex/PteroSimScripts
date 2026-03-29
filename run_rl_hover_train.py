"""
Single-drone SAC hover trainer for PteroSim — raw motor control (no PID).

Architecture (free-run):
  Simulation runs continuously at PHYSICS_HZ × time_scale.
  RL agent sends raw motor commands and reads observations via gRPC.
  No step_once — physics never pauses during training.
    → gRPC set_actuator_controls (4 motors + 1 pad)
    → JSBSim motors at 1000 Hz (continuous)
    → gRPC get observations (async read)

Action space:
  4 motors, each [-1, 1] mapped to throttle [0, 1]
  action_to_throttles(a) = (a + 1) / 2

Observation (19 dims):
  pos_err(3): (target - drone_pos) in cm, normalized by 500 cm
  vel(3):     velocity in cm/s, approx as delta_pos / agent_dt, normalized by 500 cm/s
  att(3):     roll, pitch, yaw in radians, normalized by pi
  gyro(3):    angular velocity rad/s from IMU, normalized by 10 rad/s
  accel(3):   specific force m/s^2 from IMU, normalized by 20 m/s^2
  prev_action(4): last motor command [-1, 1]

Target: hover at TARGET_Z_CM above the spawn point.

Usage:
  python run_rl_hover_train.py --timesteps 500000
  python run_rl_hover_train.py --timesteps 500000 --load checkpoints/sac_hover.zip
  python run_rl_hover_train.py --mode play --load checkpoints/sac_hover.zip

TensorBoard:
  tensorboard --logdir tensorboard_logs
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
        """Prints ETA, ep count, avg reward every print_freq steps."""

        def __init__(self, total_timesteps: int, print_freq: int = 2048, verbose: int = 1):
            super().__init__(verbose)
            self._total = total_timesteps
            self._print_freq = print_freq
            self._start_time: float = 0.0
            self._ep_count = 0
            self._last_print = 0
            self._recent_rewards: list[float] = []

        def _on_training_start(self) -> None:
            self._start_time = time.monotonic()

        def _on_step(self) -> bool:
            # Track episode rewards from Monitor wrapper infos
            infos = self.locals.get("infos", [])
            for info in infos if isinstance(infos, list) else [infos]:
                if isinstance(info, dict) and "episode" in info:
                    self._recent_rewards.append(info["episode"]["r"])
                    self._ep_count += 1

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

                pct = 100.0 * self.num_timesteps / self._total
                print(
                    f"[{pct:5.1f}%] steps={self.num_timesteps}/{self._total} "
                    f"eps={self._ep_count} fps={fps:.0f} "
                    f"ep_rew={ep_rew:.1f}{ep_len_str} "
                    f"ETA={eta_min:.1f}min"
                )
            return True

except ImportError:
    BaseCallback = None
    ProgressCallback = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_SIM_ADDRESS = "localhost:10010"
DEFAULT_AIRCRAFT_CLASS = "F450"

DRONE_SPAWN = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}

PHYSICS_HZ = 1000.0
DEFAULT_TIME_SCALE = 50.0   # sim speed multiplier for training
AGENT_DT_S = 0.05           # 20 Hz agent (in sim-seconds)

# Target hover: 300 cm above spawn in UE Z
TARGET_Z_CM = 300.0

# Episode
MAX_EPISODE_STEPS = 400      # 400 * 0.05s = 20 sim-seconds
GRACE_STEPS = 10             # don't penalize distance at start

# Termination
MAX_DIST_FROM_TARGET_CM = 1000.0   # 10 m — unrecoverable

# Observation
OBS_DIM = 19
NORM_POS_CM = 500.0
NORM_VEL_CMS = 500.0
NORM_ATT_RAD = np.pi
NORM_GYRO = 10.0       # rad/s
NORM_ACCEL = 20.0      # m/s^2
OBS_CLIP = 5.0

# Reward defaults — Optuna will override these
REWARD_DEFAULTS = {
    "hover_bonus": 1.6450277666478557,
    "target_radius_cm": 70.20363165296166,
    "approach_scale": 0.009159851416012265,
    "dist_penalty_scale": 0.0005064987258652189,
    "crash_penalty": -11.261146126723233,
    "timeout_penalty": -5.339290723709102,
    "action_smooth_coef": 0.07201320586064817,
}

# Active reward config (mutable — Optuna overwrites before each trial)
reward_config: dict[str, float] = dict(REWARD_DEFAULTS)

# Initial motor command. F450 hover ~0.425 throttle => action ~-0.15 via (a+1)/2.
HOVER_ACTION = -0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def action_to_throttles(action: np.ndarray, num_channels: int = 5) -> list[float]:
    """Map [-1, 1]^4 to throttle [0, 1] and pad to num_channels."""
    throttles = ((np.clip(action, -1.0, 1.0) + 1.0) / 2.0).tolist()
    while len(throttles) < num_channels:
        throttles.append(0.0)
    return throttles


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
            self._target = np.array([
                DRONE_SPAWN["x"], DRONE_SPAWN["y"], TARGET_Z_CM
            ], dtype=np.float32)

            self._step_count = 0
            self._prev_pos: np.ndarray | None = None
            self._prev_action = np.full(4, HOVER_ACTION, dtype=np.float32)
            self._episode_reward = 0.0
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
                # First reset: spawn drone
                drone = self._sim.spawn(
                    self.aircraft_class, **DRONE_SPAWN
                )
                self._drone_id = drone.instance_id
                self._sim.start()
                wait_for_aircraft_status(self._sim, self._drone_id)
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
                self._sim.set_attitude_command(
                    self._drone_id,
                    roll_rad=0.0, pitch_rad=0.0,
                    yaw_rate_rad_sec=0.0, throttle=0.0, enabled=False,
                )

            # Seed hover throttle
            hover_t = action_to_throttles(np.full(4, HOVER_ACTION, dtype=np.float32))
            self._sim.set_actuator_controls(self._drone_id, hover_t)

            self._step_count = 0
            self._prev_pos = None
            self._prev_action = np.full(4, HOVER_ACTION, dtype=np.float32)
            self._episode_reward = 0.0

            raw = get_obs_raw(self._sim, self._drone_id)
            obs = build_obs(raw, self._target, None, self._prev_action)
            return obs, {}

        def step(self, action):
            assert self._sim is not None
            self._step_count += 1
            action = np.asarray(action, dtype=np.float32)

            # Send motor commands
            self._sim.set_actuator_controls(
                self._drone_id, action_to_throttles(action)
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

                if self._step_count > GRACE_STEPS and dist > MAX_DIST_FROM_TARGET_CM:
                    terminated = True
                    reason = "too_far"
                elif self._step_count >= MAX_EPISODE_STEPS:
                    truncated = True
                    reason = "timeout"

            # Reward
            done = terminated or truncated
            if reason == "crash":
                reward = reward_config["crash_penalty"]
            elif reason == "too_far":
                reward = reward_config["crash_penalty"]
            elif reason == "timeout":
                reward = reward_config["timeout_penalty"]
            else:
                pos = np.array([raw["x"], raw["y"], raw["z"]], dtype=np.float32)
                dist = float(np.linalg.norm(pos - self._target))

                # Approach reward
                approach = 0.0
                if self._prev_pos is not None:
                    prev_dist = float(np.linalg.norm(self._prev_pos - self._target))
                    approach = (prev_dist - dist) * reward_config["approach_scale"]

                hover_bonus = reward_config["hover_bonus"] if dist < reward_config["target_radius_cm"] else 0.0
                dist_penalty = -reward_config["dist_penalty_scale"] * dist
                smooth_pen = -reward_config["action_smooth_coef"] * float(
                    np.sum(np.abs(action - self._prev_action))
                )
                reward = approach + hover_bonus + dist_penalty + smooth_pen

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
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    env = Monitor(PteroHoverEnv(
        sim_address=sim_addr,
        aircraft_class=aircraft,
        time_scale=time_scale,
    ))

    try:
        if tensorboard_log:
            Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

        if load_path:
            lp = Path(load_path)
            if not lp.is_file():
                lp = lp.with_suffix(".zip")
            if not lp.is_file():
                raise SystemExit(f"Checkpoint not found: {load_path}")

            model = SAC.load(str(lp), env=env, verbose=1,
                             tensorboard_log=tensorboard_log or None)
            print(f"Loaded from {lp}")

            # Load replay buffer if exists
            rb_path = Path(str(lp).replace(".zip", "_replay_buffer.pkl"))
            if rb_path.is_file():
                model.load_replay_buffer(str(rb_path))
                print(f"Loaded replay buffer ({model.replay_buffer.size()} transitions)")
            reset_num = False
        else:
            model = SAC(
                policy="MlpPolicy",
                env=env,
                verbose=1,
                seed=seed,
                learning_rate=2.9873903952202896e-05,
                buffer_size=200_000,
                batch_size=128,
                tau=0.01625829984594192,
                gamma=0.9509454422913205,
                train_freq=8,
                gradient_steps=4,
                learning_starts=2000,
                ent_coef="auto",            # auto entropy tuning
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
            save_path="checkpoints/",
            name_prefix="sac_hover",
            save_replay_buffer=True,
        ))

        model.learn(
            total_timesteps=timesteps,
            tb_log_name=run_name,
            reset_num_timesteps=reset_num,
            callback=CallbackList(callbacks),
        )
        model.save(save_path)
        model.save_replay_buffer(f"{save_path}_replay_buffer")
        print(f"Saved policy to {save_path}.zip")
        print(f"Saved replay buffer to {save_path}_replay_buffer.pkl")
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
    from stable_baselines3 import SAC

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
    model = SAC.load(str(lp))

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
                    thr = action_to_throttles(action)
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
    study_name: str = "hover_sac",
    storage: str | None = None,
) -> None:
    try:
        import optuna
    except ImportError:
        raise SystemExit("Install optuna: pip install optuna")

    if gym is None:
        raise SystemExit("Install gymnasium: pip install gymnasium")

    from pathlib import Path
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    Path("optuna_checkpoints").mkdir(parents=True, exist_ok=True)
    Path("tensorboard_logs").mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        # --- Sample SAC hyperparameters ---
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        tau = trial.suggest_float("tau", 0.005, 0.05, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        train_freq = trial.suggest_categorical("train_freq", [1, 2, 4, 8])
        gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4])
        net_arch_size = trial.suggest_categorical("net_arch_size", [128, 256, 512])

        # --- Sample reward coefficients ---
        reward_config["hover_bonus"] = trial.suggest_float("hover_bonus", 0.1, 5.0)
        reward_config["target_radius_cm"] = trial.suggest_float("target_radius_cm", 20.0, 200.0)
        reward_config["approach_scale"] = trial.suggest_float("approach_scale", 0.0005, 0.01, log=True)
        reward_config["dist_penalty_scale"] = trial.suggest_float("dist_penalty_scale", 0.0005, 0.01, log=True)
        reward_config["crash_penalty"] = trial.suggest_float("crash_penalty", -100.0, -10.0)
        reward_config["timeout_penalty"] = trial.suggest_float("timeout_penalty", -30.0, -1.0)
        reward_config["action_smooth_coef"] = trial.suggest_float("action_smooth_coef", 0.01, 0.2)

        print(f"\n--- Trial {trial.number} ---")
        print(f"  lr={lr:.1e} gamma={gamma:.4f} tau={tau:.4f} batch={batch_size} "
              f"train_freq={train_freq} grad_steps={gradient_steps} arch={net_arch_size}")
        print(f"  hover_bonus={reward_config['hover_bonus']:.2f} "
              f"approach={reward_config['approach_scale']:.4f} "
              f"crash={reward_config['crash_penalty']:.0f} "
              f"smooth={reward_config['action_smooth_coef']:.3f}")

        env = Monitor(PteroHoverEnv(
            sim_address=sim_addr,
            aircraft_class=aircraft,
            time_scale=time_scale,
        ))

        try:
            model = SAC(
                policy="MlpPolicy",
                env=env,
                verbose=0,
                seed=seed,
                learning_rate=lr,
                buffer_size=200_000,
                batch_size=batch_size,
                tau=tau,
                gamma=gamma,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                learning_starts=1000,
                ent_coef="auto",
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
                  f"lr={lr:.1e} gamma={gamma:.4f} tau={tau:.4f} "
                  f"batch={batch_size} arch={net_arch_size}")

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
    p = argparse.ArgumentParser(description="PteroSim SAC hover trainer — raw motor control")
    p.add_argument("--mode", choices=["train", "play", "optuna"], default="train")
    p.add_argument("--sim-address", default=DEFAULT_SIM_ADDRESS)
    p.add_argument("--aircraft", default=DEFAULT_AIRCRAFT_CLASS)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--time-scale", type=float, default=DEFAULT_TIME_SCALE)
    p.add_argument("--load", default=None, help="Path to .zip checkpoint")
    p.add_argument("--save", default="checkpoints/sac_hover", help="Save path (no .zip)")
    p.add_argument("--tensorboard-log", default="tensorboard_logs")
    p.add_argument("--run-name", default="sac_hover")
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
