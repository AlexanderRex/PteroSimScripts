#!/usr/bin/env python3
"""Air stunt demo — spawn F450, fly a show path, land."""

import math
import time

from pterosim import PteroSim

SPAWN = (0.0, 0.0, 0.0)
CRUISE_Z = 400.0
LEG = 1000.0
STABLE_S = 0.5


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


def main() -> None:
    """Spawn an F450, fly a square-and-diagonal show path, then land."""
    with PteroSim("localhost:10010") as sim:
        try:
            sim.stop()
        except Exception:
            pass

        drone = sim.spawn("F450", x=SPAWN[0], y=SPAWN[1], z=SPAWN[2])
        hx, hy, hz = xyz(sim, drone.instance_id)
        z = hz + CRUISE_Z

        sim.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            if any(
                st.instance_id == drone.instance_id and st.actual_frequency_hz > 0 and not st.crashed
                for st in sim.aircraft_status()
            ):
                break
            time.sleep(0.2)
        else:
            raise TimeoutError("Aircraft not ready")

        waypoints = [
            ("takeoff", (hx, hy, z), 0),
            ("square +X", (hx + LEG, hy, z), 0),
            ("square +Y", (hx + LEG, hy + LEG, z), 90),
            ("diagonal", (hx - LEG, hy + LEG, z), 135),
            ("above home", (hx, hy, z), 0),
        ]

        for label, target, yaw in waypoints:
            print(f"-> {label}")
            drone.go_to(*target, yaw=yaw, acceptance_radius_cm=80)
            wait_near(sim, drone.instance_id, target, 80, 90, label)

        print("-> land")
        drone.land()

        deadline = time.time() + 60
        stable: float | None = None
        while time.time() < deadline:
            pos = xyz(sim, drone.instance_id)
            if abs(pos[2] - hz) <= 1:
                stable = time.time() if stable is None else stable
                if time.time() - stable >= STABLE_S:
                    print(f"land: ok Z={pos[2]:.0f}")
                    break
            else:
                stable = None
            time.sleep(0.1)
        else:
            raise TimeoutError(f"land: timeout at Z={xyz(sim, drone.instance_id)[2]:.0f}")

    print("done")


if __name__ == "__main__":
    main()
