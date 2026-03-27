"""
RL hover trainer for PteroSim F450.

Trains a policy to hold position with raw motor control (no PID).
Action space: 4 motors [-1,1] mapped to throttle [0,1].

Modes:
  --mode random   Random actions (smoke test)
  --mode train    SAC training (headless recommended: -nullrhi)
  --mode play     Inference with real-time rendering

Usage:
  # Train:
  python run_rl_hover_train.py --mode train --timesteps 500000 --time-scale 10
  # Play back:
  python run_rl_hover_train.py --mode play --load checkpoints/sac_hover.zip --time-scale 10

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
        def __init__(self, total_timesteps: int, print_freq: int = 1000, verbose: int = 1):
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
                fps = self.num_timesteps / max(elapsed, 1e-6)
                eta_min = (self._total - self.num_timesteps) / max(fps, 1e-6) / 60.0
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

# Drone starts on ground, target = 300cm above ground
GROUND_Z = -84.93  # ground level from aircraft_status (UE coords)
HOVER_TARGET = {"x": 0.0, "y": 0.0, "z": GROUND_Z + 300.0}  # ~215cm UE
DRONE_SPAWN = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}  # spawn on ground

PHYSICS_HZ = 1000.0
DEFAULT_TIME_SCALE = 100.0  # agent at 10 Hz (100 physics ticks per decision), ~6x speedup

# Observation: pos(3) + att(3) + gyro(3) + accel(3) + prev_action(4) = 16
OBS_DIM = 16
MAX_EPISODE_STEPS = 2000  # 2000 steps at 10 Hz = 200 sec
MAX_DRIFT_CM = 500.0      # terminate if >5m from target
OBS_CLIP = float(1e5)

# Normalization constants
NORM_POS = 500.0     # cm, max expected drift
NORM_ANGLE = math.pi
NORM_GYRO = 10.0     # rad/s
NORM_ACCEL = 20.0    # m/s^2 (~2g)


# --- Observation ---

def get_hover_observation(sim: PteroSim, instance_id: int, prev_action: np.ndarray) -> dict[str, Any]:
    statuses = sim.aircraft_status()
    status = None
    for s in statuses:
        if s.instance_id == instance_id:
            status = s
            break
    if status is None or status.crashed:
        return {
            "pos_err": np.zeros(3, dtype=np.float32),
            "attitude": np.zeros(3, dtype=np.float32),
            "gyro": np.zeros(3, dtype=np.float32),
            "accel": np.zeros(3, dtype=np.float32),
            "prev_action": prev_action.copy(),
            "crashed": True,
            "drift_cm": 0.0,
        }

    imu = sim.get_imu(instance_id)

    # Position error relative to hover target (UE cm)
    pos_err = np.nan_to_num(np.array([
        status.x - HOVER_TARGET["x"],
        status.y - HOVER_TARGET["y"],
        status.z - HOVER_TARGET["z"],
    ], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    attitude = np.nan_to_num(np.array([
        math.radians(status.roll),
        math.radians(status.pitch),
        math.radians(status.yaw),
    ], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    gyro = np.nan_to_num(np.array([
        imu.angular_velocity[0],
        imu.angular_velocity[1],
        imu.angular_velocity[2],
    ], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    accel = np.nan_to_num(np.array([
        imu.acceleration[0],
        imu.acceleration[1],
        imu.acceleration[2],
    ], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    drift_cm = float(np.linalg.norm(pos_err))

    return {
        "pos_err": pos_err,
        "attitude": attitude,
        "gyro": gyro,
        "accel": accel,
        "prev_action": prev_action.copy(),
        "crashed": False,
        "drift_cm": drift_cm,
    }


def obs_to_vector(obs: dict) -> np.ndarray:
    # Sanitize each component BEFORE division to avoid inf/nan propagation
    pos_err = np.nan_to_num(obs["pos_err"], nan=0.0, posinf=NORM_POS, neginf=-NORM_POS)
    attitude = np.nan_to_num(obs["attitude"], nan=0.0, posinf=NORM_ANGLE, neginf=-NORM_ANGLE)
    gyro = np.nan_to_num(obs["gyro"], nan=0.0, posinf=NORM_GYRO, neginf=-NORM_GYRO)
    accel = np.nan_to_num(obs["accel"], nan=0.0, posinf=NORM_ACCEL, neginf=-NORM_ACCEL)
    prev_action = np.nan_to_num(obs["prev_action"], nan=0.0, posinf=1.0, neginf=-1.0)

    v = np.concatenate([
        pos_err / NORM_POS,
        attitude / NORM_ANGLE,
        gyro / NORM_GYRO,
        accel / NORM_ACCEL,
        prev_action,
    ]).astype(np.float32)
    return np.clip(v, -10.0, 10.0)


# --- Reward ---

def compute_hover_reward(obs: dict, prev_obs: dict | None, action: np.ndarray, done: bool, reason: str) -> float:
    if done and reason == "crash":
        return -100.0
    if done and reason == "drift":
        return -50.0

    # Alive bonus
    reward = 1.0

    # Position error penalty (cm -> normalized)
    pos_err = np.linalg.norm(obs["pos_err"])
    reward -= 2.0 * (pos_err / NORM_POS)

    # Attitude penalty (roll^2 + pitch^2, ignore yaw)
    roll, pitch, _ = obs["attitude"]
    reward -= 3.0 * (roll ** 2 + pitch ** 2)

    # Angular velocity penalty (damp oscillations)
    gyro = obs["gyro"]
    reward -= 0.1 * float(np.sum(gyro ** 2))

    # Action smoothness penalty
    if prev_obs is not None:
        action_delta = np.sum((action - prev_obs["prev_action"]) ** 2)
        reward -= 0.5 * float(action_delta)

    return float(np.clip(reward, -200.0, 10.0))


# --- Episode checks ---

MIN_HEIGHT_CM = 50.0  # below this Z (relative to target) = ground contact
GRACE_STEPS = 10      # skip termination checks for first N steps (stale status after reset)

def check_hover_end(obs: dict, step_count: int) -> tuple[bool, bool, str]:
    if obs["crashed"]:
        return True, False, "crash"
    # Only check position-based termination after stale data clears
    if step_count > GRACE_STEPS:
        if obs["drift_cm"] > MAX_DRIFT_CM:
            return True, False, "drift"
    if step_count >= MAX_EPISODE_STEPS:
        return False, True, "timeout"
    return False, False, ""


# --- Sim helpers ---

def apply_action(sim: PteroSim, instance_id: int, action: np.ndarray, controls_buffer: list[float]) -> None:
    a = np.asarray(action, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
    a = np.clip(a, -1.0, 1.0)
    th = ((a + 1.0) * 0.5).clip(0.0, 1.0)
    controls = list(controls_buffer)
    for i in range(min(4, len(controls))):
        controls[i] = float(th[i])
    sim.set_actuator_controls(instance_id, controls)


def wait_for_aircraft_status(sim: PteroSim, instance_id: int, timeout_s: float = 2.0) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for s in sim.aircraft_status():
            if s.instance_id == instance_id:
                return s
        time.sleep(0.05)
    raise RuntimeError(f"Aircraft {instance_id} not found before timeout")


def build_controls(sim: PteroSim, instance_id: int) -> list[float]:
    cfg = sim.get_actuator_configuration(instance_id)
    controls = [0.55] * min(4, cfg.channel_count)
    while len(controls) < cfg.channel_count:
        controls.append(0.0)
    return controls


def remove_all_aircraft(sim: PteroSim) -> None:
    for status in list(sim.aircraft_status()):
        Aircraft(sim, status.instance_id).remove()


def setup_hover(sim: PteroSim, aircraft_class: str) -> tuple[int, list[float]]:
    drone = sim.spawn(aircraft_class, **DRONE_SPAWN)
    drone_id = drone.instance_id
    sim.start()
    wait_for_aircraft_status(sim, drone_id)
    controls = build_controls(sim, drone_id)
    return drone_id, controls


def reset_hover(sim: PteroSim) -> tuple[int, list[float]]:
    sim.stop()
    statuses = sim.aircraft_status()
    assert statuses, "No aircraft after stop"
    drone_id = statuses[0].instance_id
    sim.start()
    wait_for_aircraft_status(sim, drone_id)
    controls = build_controls(sim, drone_id)
    return drone_id, controls


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


# --- Gym Env ---

if gym is not None and spaces is not None:

    class PteroHoverEnv(gym.Env):
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
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32)
            self._sim: PteroSim | None = None
            self._drone_id = 0
            self._controls: list[float] = []
            self._step_count = 0
            self._prev_action = np.zeros(4, dtype=np.float32)
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
            t0 = time.monotonic()
            self._connect()
            assert self._sim is not None
            if not self._setup_done:
                print(f"[HOVER] setup_hover (first reset)...")
                self._drone_id, self._controls = setup_hover(self._sim, self.aircraft_class)
                self._setup_done = True
            else:
                print(f"[HOVER] reset_hover (ep reset)...")
                self._drone_id, self._controls = reset_hover(self._sim)
            self._step_count = 0
            self._prev_action = np.zeros(4, dtype=np.float32)
            obs_dict = get_hover_observation(self._sim, self._drone_id, self._prev_action)
            self._prev_obs_dict = obs_dict
            dt = time.monotonic() - t0
            print(f"[HOVER] reset done in {dt:.2f}s, drone_id={self._drone_id}")
            return obs_to_vector(obs_dict), {}

        def step(self, action):
            assert self._sim is not None
            self._step_count += 1
            if self._step_count == 1:
                print(f"[HOVER] First step_once() call — sim should start advancing now")
            action = np.asarray(action, dtype=np.float32)
            apply_action(self._sim, self._drone_id, action, self._controls)
            self._sim.step_once()
            obs_dict = get_hover_observation(self._sim, self._drone_id, action)
            obs = obs_to_vector(obs_dict)

            terminated, truncated, reason = check_hover_end(obs_dict, self._step_count)
            done = terminated or truncated
            reward = compute_hover_reward(obs_dict, self._prev_obs_dict, action, done, reason)

            # Debug: print first 5 steps and catch NaN
            if self._step_count <= 10 or self._step_count % 500 == 0 or done:
                pe = obs_dict["pos_err"]
                # Get raw status z for debugging
                raw_z = "?"
                for s in self._sim.aircraft_status():
                    if s.instance_id == self._drone_id:
                        raw_z = f"{s.z:.1f}"
                print(f"[DEBUG] step={self._step_count} rew={reward:.2f} z={raw_z} drift={obs_dict['drift_cm']:.0f} "
                      f"att=[{obs_dict['attitude'][0]:.3f},{obs_dict['attitude'][1]:.3f}] "
                      f"gyro=[{obs_dict['gyro'][0]:.2f},{obs_dict['gyro'][1]:.2f},{obs_dict['gyro'][2]:.2f}] "
                      f"crashed={obs_dict['crashed']} reason={reason} "
                      f"thr=[{((action+1)*0.5).clip(0,1)[0]:.2f},{((action+1)*0.5).clip(0,1)[1]:.2f},{((action+1)*0.5).clip(0,1)[2]:.2f},{((action+1)*0.5).clip(0,1)[3]:.2f}]")
            if not np.isfinite(reward):
                print(f"[DEBUG] BAD reward={reward} at step {self._step_count}")
                reward = -100.0

            self._prev_action = action.copy()
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
    tensorboard_log: str | None,
    run_name: str,
    load_path: str | None,
    save_path: str,
    seed: int | None,
    time_scale: float,
) -> None:
    if gym is None or spaces is None:
        raise SystemExit("Install gymnasium: pip install gymnasium")
    from pathlib import Path
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    env = Monitor(PteroHoverEnv(sim_address=sim_addr, aircraft_class=aircraft, time_scale=time_scale))

    try:
        if tensorboard_log:
            Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

        if load_path:
            lp = load_checkpoint(load_path)
            load_kwargs: dict = {"env": env, "verbose": 1}
            if tensorboard_log:
                load_kwargs["tensorboard_log"] = tensorboard_log
            model = SAC.load(str(lp), **load_kwargs)
            print(f"Loaded from {load_path}, continuing for {timesteps} steps")
            reset_num = False
        else:
            sac_kwargs: dict = {
                "policy": "MlpPolicy",
                "env": env,
                "verbose": 1,
                "seed": seed,
                "learning_rate": 3e-4,
                "buffer_size": 500_000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "learning_starts": 1_000,
                "policy_kwargs": {"net_arch": [256, 256]},
            }
            if tensorboard_log:
                sac_kwargs["tensorboard_log"] = tensorboard_log
            model = SAC(**sac_kwargs)
            reset_num = True

        cb = ProgressCallback(timesteps) if ProgressCallback else None
        model.learn(
            total_timesteps=timesteps,
            tb_log_name=run_name,
            reset_num_timesteps=reset_num,
            callback=cb,
        )
        model.save(save_path)
        print(f"Saved to {save_path}")
    finally:
        env.close()


# --- Random ---

def run_random(
    sim_addr: str,
    aircraft: str,
    episodes: int,
    seed: int | None,
    time_scale: float,
) -> None:
    with PteroSim(sim_addr) as sim:
        sim.set_physics_frequency(PHYSICS_HZ)
        sim.set_time_scale(time_scale)
        drone_id, controls = setup_hover(sim, aircraft)
        rng = np.random.default_rng(seed)

        for ep in range(episodes):
            prev_action = np.zeros(4, dtype=np.float32)
            obs_dict = get_hover_observation(sim, drone_id, prev_action)
            total_r = 0.0
            reason = "timeout"

            for t in range(MAX_EPISODE_STEPS):
                action = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
                apply_action(sim, drone_id, action, controls)
                sim.step_once()
                obs_dict = get_hover_observation(sim, drone_id, action)
                terminated, truncated, reason = check_hover_end(obs_dict, t + 1)
                done = terminated or truncated
                total_r += compute_hover_reward(obs_dict, None, action, done, reason)
                prev_action = action
                if done:
                    break

            print(f"[random EP {ep}] {reason} drift={obs_dict['drift_cm']:.0f}cm R={total_r:.1f}")
            drone_id, controls = reset_hover(sim)

        sim.stop()
        remove_all_aircraft(sim)


# --- Play ---

def run_play(
    sim_addr: str,
    aircraft: str,
    episodes: int,
    load_path: str,
    time_scale: float,
) -> None:
    from stable_baselines3 import SAC

    lp = load_checkpoint(load_path)
    model = SAC.load(str(lp))
    sim_steps_per_agent_step = max(1, int(time_scale))
    sleep_per_tick = 1.0 / PHYSICS_HZ

    with PteroSim(sim_addr) as sim:
        sim.set_physics_frequency(PHYSICS_HZ)
        sim.set_time_scale(1.0)

        drone_id, controls = setup_hover(sim, aircraft)
        sim.hold()
        print(f"Drone spawned (id={drone_id}). Press Enter to start...")
        input()

        for ep in range(episodes):
            prev_action = np.zeros(4, dtype=np.float32)
            obs_dict = get_hover_observation(sim, drone_id, prev_action)
            obs = obs_to_vector(obs_dict)
            total_r = 0.0
            steps = 0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
                apply_action(sim, drone_id, action, controls)
                for _ in range(sim_steps_per_agent_step):
                    sim.step_once()
                    time.sleep(sleep_per_tick)

                obs_dict = get_hover_observation(sim, drone_id, action)
                obs = obs_to_vector(obs_dict)
                steps += 1
                prev_action = action

                terminated, truncated, reason = check_hover_end(obs_dict, steps)
                done = terminated or truncated
                total_r += compute_hover_reward(obs_dict, None, action, done, reason)
                if done:
                    break

            print(f"[play EP {ep}] {reason} steps={steps} drift={obs_dict['drift_cm']:.0f}cm R={total_r:.1f}")
            drone_id, controls = reset_hover(sim)
            sim.hold()

        sim.stop()
        remove_all_aircraft(sim)


# --- Main ---

def main() -> int:
    parser = argparse.ArgumentParser(description="PteroSim hover RL trainer")
    parser.add_argument("--sim", default=DEFAULT_SIM_ADDRESS)
    parser.add_argument("--aircraft", default=DEFAULT_AIRCRAFT_CLASS)
    parser.add_argument("--mode", choices=["random", "train", "play"], default="random")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--tensorboard-log", type=str, default="tensorboard_logs")
    parser.add_argument("--run-name", type=str, default="SAC_hover")
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--save", type=str, default="checkpoints/sac_hover")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--time-scale", type=float, default=DEFAULT_TIME_SCALE)
    args = parser.parse_args()

    ts = args.time_scale

    if args.mode == "random":
        run_random(args.sim, args.aircraft, args.episodes, args.seed, ts)
    elif args.mode == "play":
        load_p = args.load.strip()
        if not load_p:
            raise SystemExit("--mode play requires --load <checkpoint.zip>")
        run_play(args.sim, args.aircraft, args.episodes, load_p, ts)
    else:
        tb = args.tensorboard_log.strip() or None
        load_p = args.load.strip() or None
        run_train(args.sim, args.aircraft, args.timesteps, tb, args.run_name, load_p, args.save, args.seed, ts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
