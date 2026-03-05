##
#
# View results from simulation
#
##

# standard imports
import numpy as np
import time

# mujoco imports
import mujoco
import mujoco.viewer


#################################################################################
# SETTINGS
#################################################################################

# playback the optimal trajectory in the mujoco viewer
visualize = 1

# which data to load
experiment = "g1"
xml_path = "./hydrax/models/g1/g1_planar.xml"


#################################################################################
# LOAD DATA
#################################################################################

from pathlib import Path

results_dir = Path(__file__).resolve().parent / experiment

# load data from csv files
times = np.loadtxt(results_dir / "time.csv", delimiter=",")
q_opt = np.loadtxt(results_dir / "qpos.csv", delimiter=",")
v_opt = np.loadtxt(results_dir / "qvel.csv", delimiter=",")
tau_opt = np.loadtxt(results_dir / "actuator_force.csv", delimiter=",")

print("Loaded data:")
print(f"  times: {times.shape}")
print(f"  q_opt: {q_opt.shape}")
print(f"  v_opt: {v_opt.shape}")
print(f"  tau_opt: {tau_opt.shape}")


#################################################################################
# MUJOCO VISUALIZATION
#################################################################################

if visualize == 1:

    # load the mujoco model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # launch the viewer
    viewer = mujoco.viewer.launch_passive(model, data)

    # run the visualization
    try:
        t0 = time.time()
        while True:

            if viewer.is_running() == False:
                break

            i = np.searchsorted(times, time.time() - t0)
            i = min(i, len(times) - 1)  # Clamp to valid range

            print(f"Time: {time.time() - t0:.2f}, Index: {i}\r", end="")

            data.qpos[:] = q_opt[i, :]
            data.qvel[:] = v_opt[i, :]
            mujoco.mj_forward(model, data)
            viewer.sync()

            if time.time() - t0 > times[-1]:
                time.sleep(1.0)
                t0 = time.time()

    except KeyboardInterrupt:
        print("\nClosed visualization.")

    viewer.close()
