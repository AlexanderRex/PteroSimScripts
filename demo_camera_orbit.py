#!/usr/bin/env python3
"""Orbit demo — spawn F450, add a camera, shoot a race gate from four sides."""

from __future__ import annotations

import math
import struct
import time
import zlib
from pathlib import Path

import numpy as np
from pterosim import PteroSim

SPAWN = (0.0, 0.0, 0.0)
GATE = {"x": 1500.0, "y": 0.0, "z": 0.0}
RADIUS_CM = 600.0
STABLE_S = 0.5
OUT_DIR = Path(__file__).resolve().parent / "camera_orbit_shots"

# yaw 0 faces +X. Each tuple is (filename stem, offset x-sign, offset y-sign, yaw looking at the gate).
SIDES: tuple[tuple[str, int, int, float], ...] = (
    ("from_plus_x", 1, 0, 180.0),
    ("from_plus_y", 0, 1, 270.0),
    ("from_minus_x", -1, 0, 0.0),
    ("from_minus_y", 0, -1, 90.0),
)


def xyz(sim: PteroSim, instance_id: int) -> tuple[float, float, float]:
    """Return the current (x, y, z) position of the given instance."""
    for st in sim.aircraft_status():
        if st.instance_id == instance_id:
            return st.x, st.y, st.z
    raise RuntimeError(f"No status for instance_id={instance_id}")


def wait_near(
    sim: PteroSim,
    instance_id: int,
    target: tuple[float, float, float],
    radius_cm: float,
    timeout_s: float,
    label: str,
) -> None:
    """Block until the instance holds within radius_cm of target, or time out."""
    deadline = time.time() + timeout_s
    stable: float | None = None
    while time.time() < deadline:
        pos = xyz(sim, instance_id)
        err = math.dist(pos, target)
        if err <= radius_cm:
            stable = time.time() if stable is None else stable
            if time.time() - stable >= STABLE_S:
                print(f"{label}: ok ({err:.0f} cm)")
                return
        else:
            stable = None
        time.sleep(0.1)
    raise TimeoutError(f"{label}: timeout at {xyz(sim, instance_id)}")


def wait_ready(sim: PteroSim, instance_id: int, timeout_s: float = 15.0) -> None:
    """Block until the aircraft is running physics and has not crashed."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if any(
            st.instance_id == instance_id and st.actual_frequency_hz > 0 and not st.crashed
            for st in sim.aircraft_status()
        ):
            return
        time.sleep(0.2)
    raise TimeoutError("Aircraft not ready")


def save_png(path: Path, bgr: np.ndarray) -> None:
    """Write a BGR uint8 HxWx3 array as a PNG (no extra image dependency)."""
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    """Spawn an F450 with a camera, orbit a race gate, save four side photos."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    with PteroSim("localhost:10010") as sim:
        try:
            sim.stop()
        except Exception:
            pass

        drone = sim.spawn("F450", x=SPAWN[0], y=SPAWN[1], z=SPAWN[2])
        cam = drone.add_sensor("camera", "Camera")
        print(f"camera: {cam}")

        # SDK has no generic prop/box spawn; a race gate is the visible stand-in.
        track = sim.set_track_gates([GATE])
        gx, gy, gz = track.gates[0].x, track.gates[0].y, track.gates[0].z
        print(f"gate center: ({gx:.0f}, {gy:.0f}, {gz:.0f}) cm")

        sim.start()
        wait_ready(sim, drone.instance_id)

        for name, sx, sy, yaw in SIDES:
            target = (gx + sx * RADIUS_CM, gy + sy * RADIUS_CM, gz)
            print(f"-> {name}")
            drone.go_to(*target, yaw=yaw, acceptance_radius_cm=80)
            wait_near(sim, drone.instance_id, target, 80, 90, name)

            frame = drone.camera(cam)
            path = OUT_DIR / f"{name}.png"
            save_png(path, frame.image)
            saved.append(path)
            print(f"{name}: captured {frame.width}x{frame.height}")

    print("saved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
