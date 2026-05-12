"""Quick test: zero attitude commands + hover throttle through C++ PID controller."""

from pterosim import PteroSim
from run_rl_race_train import wait_for_aircraft_status

SIM_ADDR = "localhost:10010"

sim = PteroSim(SIM_ADDR)
sim.stop()
sim.set_physics_frequency(1000.0)
sim.set_time_scale(10.0)  # 10 ticks per step_once

statuses = sim.aircraft_status()
if not statuses:
    drone = sim.spawn("F450", x=0, y=0, z=0, yaw=0)
    drone_id = drone.instance_id
else:
    drone_id = statuses[0].instance_id

sim.start()
wait_for_aircraft_status(sim, drone_id)

THROTTLE = 0.425

sim.set_attitude_command(
    drone_id,
    roll_rad=0.0,
    pitch_rad=0.0,
    yaw_rate_rad_sec=0.0,
    throttle=THROTTLE,
    enabled=True,
)

print(f"C++ PID enabled: hover throttle={THROTTLE}")
print(f"Running 1000 steps = 10 seconds sim time")

for t in range(1000):
    sim.step_once()

    if t % 50 == 0:
        s = sim.aircraft_status()[0]
        print(f"  [{t/100:.1f}s] z={s.z:.0f} att=(r={s.roll:.1f} p={s.pitch:.1f})"
              f"{' CRASHED' if s.crashed else ''}")
        if s.crashed:
            break

sim.stop()
sim.close()
print("Done.")
