import argparse
from copy import deepcopy

import mujoco

from hydrax.algs import CEM, MPPI, Evosax
from evosax.algorithms.distribution_based.cma_es import CMA_ES
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.humanoid_SRB import HumanoidSRB

"""
Run an interactive simulation of the humanoid motion capture tracking task.
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of mocap tracking with the G1."
)
parser.add_argument(
    "--warp",
    action="store_true",
    help="Whether to use the (experimental) MjWarp backend. (default: False)",
    required=False,
)
parser.add_argument(
    "--show_reference",
    action="store_true",
    help="Show the reference trajectory as a 'ghost' in the simulation.",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=1,
    help="Number of CEM iterations.",
)

args = parser.parse_args()

# Define the task (cost and dynamics)
task = HumanoidSRB(
    impl="warp" if args.warp else "jax",
)

# Set up the controller
ctrl = CEM(
    task,
    num_samples=512,
    num_elites=30,
    sigma_start=0.1,
    sigma_min=0.05,
    explore_fraction=0.5,
    plan_horizon=0.5,
    spline_type="cubic",
    num_knots=5,
    iterations=args.iterations,
)
# ctrl = MPPI(
#     task,
#     num_samples=512,
#     noise_level=0.3,
#     temperature=0.1,
#     # num_randomizations=4,
#     plan_horizon=0.4,
#     spline_type="cubic",
#     num_knots=5,
#     iterations=args.iterations,
# )

# ctrl = Evosax(
#     task,
#     CMA_ES,
#     num_samples=512,
#     # num_randomizations=8,
#     plan_horizon=0.4,
#     spline_type="cubic",
#     num_knots=5,
#     iterations=args.iterations,
# )

# Define the model used for simulation
mj_model = deepcopy(task.mj_model)
mj_model.opt.timestep = 0.01
mj_model.opt.iterations = 10
mj_model.opt.ls_iterations = 50
mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

# Set the initial state
mj_data = mujoco.MjData(mj_model)
keyframe = "standing"
key_id = mj_model.key(keyframe).id
qpos_default = mj_model.key_qpos[key_id]
mj_data.qpos[:] = qpos_default  
initial_knots = task.qpos_ref[: ctrl.num_knots, 3:]

if args.show_reference:
    reference = task.qpos_ref
else:
    reference = None

sim_data = run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=100,
    show_traces=False,
    reference=reference,
    reference_fps=task.reference_fps,
    initial_knots=initial_knots,
    max_time=5.0,
)

# Save recorded data as CSV files
import numpy as np
from pathlib import Path

output_dir = Path(__file__).resolve().parent.parent / "results" / "g1"
output_dir.mkdir(parents=True, exist_ok=True)

np.savetxt(output_dir / "time.csv", sim_data["time"], delimiter=",")
np.savetxt(output_dir / "qpos.csv", sim_data["qpos"], delimiter=",")
np.savetxt(output_dir / "qvel.csv", sim_data["qvel"], delimiter=",")
np.savetxt(output_dir / "actuator_force.csv", sim_data["actuator_force"], delimiter=",")

print(f"Saved simulation data to {output_dir}/"
      f" ({sim_data['time'].shape[0]} steps,"
      f" t={sim_data['time'][-1]:.2f}s)")
