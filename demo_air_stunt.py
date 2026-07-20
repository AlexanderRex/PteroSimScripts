#!/usr/bin/env python3
"""
Air stunt demo — spawn F450, fly a show path, land.

Usage (simulator running, gRPC on localhost:10010):
    python demo_air_stunt.py
"""

from __future__ import annotations

import math
import time

from pterosim import PteroSim

SPAWN = (0.0, 0.0, 0.0)
CRUISE_Z_OFFSET = 400.0  # 4 m
LEG = 1000.0  # 10 m

GO_TO_ACCEPTANCE_CM = 80.0
LAND_ACCEPTANCE_CM = 1.0
LAND_Z_TOLERANCE_CM = 1.0
GO_TO_TIMEOUT_S = 90.0
LAND_TIMEOUT_S = 60.0
ARRIVAL_STABLE_S = 0.5
SIM_READY_TIMEOUT_S = 15.0
BETWEEN_WP_S = 0.3
GRPC_ADDRESS = "localhost:10010"


def _dist(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def aircraft_xyz(sim: PteroSim, instance_id: int) -> tuple[float, float, float]:
    for st in sim.aircraft_status():
        if st.instance_id == instance_id:
            return (st.x, st.y, st.z)
    raise RuntimeError(f"No status for instance_id={instance_id}")


def wait_sim_ready(sim: PteroSim, instance_id: int) -> None:
    deadline = time.time() + SIM_READY_TIMEOUT_S
    while time.time() < deadline:
        for st in sim.aircraft_status():
            if st.instance_id == instance_id and st.actual_frequency_hz > 0 and not st.crashed:
                print(f"       Sim ready ({st.actual_frequency_hz:.0f} Hz)")
                return
        time.sleep(0.2)
    raise TimeoutError("Aircraft not ready")


def wait_go_to(sim: PteroSim, instance_id: int, target, *, label: str) -> None:
    deadline = time.time() + GO_TO_TIMEOUT_S
    stable_since = None
    while time.time() < deadline:
        pos = aircraft_xyz(sim, instance_id)
        err = _dist(pos, target)
        if err <= GO_TO_ACCEPTANCE_CM:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= ARRIVAL_STABLE_S:
                print(f"       {label}: arrived err={err:.1f} cm")
                return
        else:
            stable_since = None
        time.sleep(0.1)
    pos = aircraft_xyz(sim, instance_id)
    raise TimeoutError(f"{label}: timeout pos={pos} target={target}")


def wait_landed(sim: PteroSim, instance_id: int, home_z: float) -> None:
    """Poll until Z is at spawn height after land()."""
    deadline = time.time() + LAND_TIMEOUT_S
    stable_since = None
    while time.time() < deadline:
        pos = aircraft_xyz(sim, instance_id)
        if abs(pos[2] - home_z) <= LAND_Z_TOLERANCE_CM:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= ARRIVAL_STABLE_S:
                print(f"       land: on ground Z={pos[2]:.0f} cm (home Z={home_z:.0f})")
                return
        else:
            stable_since = None
        time.sleep(0.1)
    pos = aircraft_xyz(sim, instance_id)
    raise TimeoutError(
        f"land: timeout — still at Z={pos[2]:.0f} cm, expected ~{home_z:.0f} cm "
        f"(tolerance {LAND_Z_TOLERANCE_CM:.0f})"
    )


def fly_to(sim, drone, target, yaw: float, label: str) -> None:
    print(f"       -> {label} ({target[0]:.0f}, {target[1]:.0f}, {target[2]:.0f}) yaw={yaw:.0f}")
    drone.go_to(*target, yaw=yaw, acceptance_radius_cm=GO_TO_ACCEPTANCE_CM)
    wait_go_to(sim, drone.instance_id, target, label=label)
    time.sleep(BETWEEN_WP_S)


def build_stunt_waypoints(home: tuple[float, float, float]):
    hx, hy, hz = home
    z = hz + CRUISE_Z_OFFSET
    z_high = hz + CRUISE_Z_OFFSET * 2.0
    return [
        ("takeoff", (hx, hy, z), 0.0),
        ("square +X", (hx + LEG, hy, z), 0.0),
        ("square +Y", (hx + LEG, hy + LEG, z), 90.0),
        ("square -X", (hx - LEG, hy + LEG, z), 180.0),
        ("square -Y", (hx - LEG, hy - LEG, z), 270.0),
        ("square close", (hx + LEG, hy - LEG, z), 0.0),
        ("diagonal", (hx - LEG, hy + LEG, z), 135.0),
        ("high hover", (hx, hy, z_high), 0.0),
        ("above home", (hx, hy, z), 0.0),
    ]


def main() -> int:
    print("=== Air Stunt ===")

    with PteroSim(GRPC_ADDRESS) as sim:
        try:
            sim.stop()
        except Exception:
            pass
        time.sleep(0.2)

        print("\n[1/3] Spawn F450")
        drone = sim.spawn("F450", x=SPAWN[0], y=SPAWN[1], z=SPAWN[2])
        home = aircraft_xyz(sim, drone.instance_id)
        print(f"       id={drone.instance_id} home=({home[0]:.0f}, {home[1]:.0f}, {home[2]:.0f})")

        print("\n[2/3] Start + stunt path")
        sim.start()
        wait_sim_ready(sim, drone.instance_id)

        for label, target, yaw in build_stunt_waypoints(home):
            fly_to(sim, drone, target, yaw, label)

        print("\n[3/3] Land")
        drone.land(yaw=0.0, acceptance_radius_cm=LAND_ACCEPTANCE_CM)
        wait_landed(sim, drone.instance_id, home[2])

        print("=== Done ===")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
