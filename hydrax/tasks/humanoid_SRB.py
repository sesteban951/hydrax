from typing import Dict
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from huggingface_hub import hf_hub_download
from mujoco import mjx
from mujoco.mjx._src.math import quat_sub

from hydrax import ROOT
from hydrax.task_base import Task


class HumanoidSRB(Task):
    """The Unitree G1 humanoid tracks a reference from motion capture.

    Retargeted motion capture data comes from the LocoMuJoCo dataset:
    https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.
    """

    def __init__(
        self,
        reference_filename: str = "Lafan1/mocap/UnitreeG1/walk1_subject1.npz",
        impl: str = "jax",
    ) -> None:
        """Load the MuJoCo model and set task parameters.

        The list of available reference files can be found at
        https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.
        """

        # mujoco model
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/g1/g1_planar.xml"
        )

        # parent class
        super().__init__(
            mj_model,
            impl=impl,
        )

        # get model properties
        self._get_model_properties(mj_model)

        # get the SRB reference trajectory
        self._make_reference_trajectory(reference_filename)

        # base weights
        w_pos_x = 1.0
        w_pos_z = 20.0
        w_theta = 5.0

        w_vel_x = 1.0
        w_vel_z = 5.0
        w_omega = 1.0

        # joint weights
        w_pos_hip  = 0.001
        w_pos_knee = 0.001
        w_pos_ankle    = 0.001
        w_pos_shoulder = 0.001
        w_pos_elbow    = 0.001

        w_vel_hip   = 0.0001
        w_vel_knee  = 0.0001
        w_vel_ankle = 0.0001
        w_vel_shoulder = 0.0001
        w_vel_elbow    = 0.0001

        self.q_weights = jnp.array([
            w_pos_x, w_pos_z, w_theta,
            w_pos_hip, w_pos_knee, w_pos_ankle,
            w_pos_hip, w_pos_knee, w_pos_ankle,
            w_pos_shoulder, w_pos_elbow,
            w_pos_shoulder, w_pos_elbow,
        ])
        self.v_weights = jnp.array([
            w_vel_x, w_vel_z, w_omega,
            w_vel_hip, w_vel_knee, w_vel_ankle,
            w_vel_hip, w_vel_knee, w_vel_ankle,
            w_vel_shoulder, w_vel_elbow,
            w_vel_shoulder, w_vel_elbow,
        ])

        # input weights
        self.w_u = 1

        # terminal weights
        terminal_scaling = 10.0
        self.qf_weights = terminal_scaling * self.q_weights
        self.vf_weights = terminal_scaling * self.v_weights

        print("Initialized Humanoid SRB tracking task.")


    def _get_model_properties(self, mj_model):

        # simulation time step
        self.sim_dt = mj_model.opt.timestep

        # get the default state
        keyframe = "standing"
        key_id = mj_model.key(keyframe).id
        self.qpos_default = jnp.array(mj_model.key_qpos[key_id])
        self.qvel_default = jnp.array(mj_model.key_qvel[key_id])
        self.qpos_joints_ref = self.qpos_default[3:]
        self.qvel_joints_ref = self.qvel_default[3:]

        print(f"Model properties:\n"
              f"    Time step (dt): {self.dt}\n"
              f"    Reference keyframe: '{keyframe}'\n"
              f"    qpos_default: {self.qpos_default}\n"
              f"    qvel_default: {self.qvel_default}\n")


    def _make_reference_trajectory(self, traj_path):

        # Load trajectory CSV files from a stable package-relative location.
        traj_path = Path(ROOT) / "trajectories" / "srb_jump_2d"
        time_path = traj_path / "time.csv"
        q_opt_path = traj_path / "q_opt.csv"
        v_opt_path = traj_path / "v_opt.csv"
        a_opt_path = traj_path / "a_opt.csv"
        c_opt_path = traj_path / "contact_schedule.csv"

        t_SRB = np.loadtxt(time_path, delimiter=",")
        q_SRB = np.loadtxt(q_opt_path, delimiter=",")
        q_SRB[:, 1] += 0.08  # Raise CoM z only
        v_SRB = np.loadtxt(v_opt_path, delimiter=",")
        a_SRB = np.loadtxt(a_opt_path, delimiter=",")
        c_SRB = np.loadtxt(c_opt_path, delimiter=",")

        dt_SRB = t_SRB[1] - t_SRB[0]
        # Common zero velocity/acceleration vectors
        v_SRB_zero = np.zeros_like(v_SRB[0])
        a_SRB_zero = np.zeros_like(a_SRB[0])
        c_SRB_first = float(np.asarray(c_SRB).reshape(-1)[0])
        c_SRB_last = float(np.asarray(c_SRB).reshape(-1)[-1])

        # pre and post segments to hold initial and final poses
        T_pre = 1.0
        T_post = 1.0

        # Prepend initial state (hold initial pose)
        t_SRB_pre = np.arange(0.0, T_pre, dt_SRB)

        q_SRB_pre = np.tile(q_SRB[0], (len(t_SRB_pre), 1))
        v_SRB_pre = np.tile(v_SRB_zero, (len(t_SRB_pre), 1))
        a_SRB_pre = np.tile(a_SRB_zero, (len(t_SRB_pre), 1))
        c_SRB_pre = np.full((len(t_SRB_pre),), c_SRB_first)

        # Shift main time array so it starts after the pre segment
        t_SRB_main = t_SRB + t_SRB_pre[-1] + dt_SRB

        # Append final state (hold final pose)
        t_SRB_post = np.arange(0.0, T_post, dt_SRB) + t_SRB_main[-1] + dt_SRB

        q_SRB_post = np.tile(q_SRB[-1], (len(t_SRB_post), 1))
        v_SRB_post = np.tile(v_SRB_zero, (len(t_SRB_post), 1))
        a_SRB_post = np.tile(a_SRB_zero, (len(t_SRB_post), 1))
        c_SRB_post = np.full((len(t_SRB_post),), c_SRB_last)

        # Concatenate pre, main, and post segments
        t_SRB = np.concatenate([t_SRB_pre, t_SRB_main, t_SRB_post], axis=0)
        q_SRB = np.concatenate([q_SRB_pre, q_SRB, q_SRB_post], axis=0)
        v_SRB = np.concatenate([v_SRB_pre, v_SRB, v_SRB_post], axis=0)
        a_SRB = np.concatenate([a_SRB_pre, a_SRB, a_SRB_post], axis=0)
        c_SRB = np.concatenate([c_SRB_pre, c_SRB, c_SRB_post], axis=0)

        p_com_ref = q_SRB[:, :2]  # CoM x and z positions
        v_com_ref = v_SRB[:, :2]  # CoM x and z velocities
        a_com_ref = a_SRB[:, :2]  # CoM x and z accelerations
        theta_ref = q_SRB[:, 2]   # torso pitch angle
        omega_ref = v_SRB[:, 2]   # torso pitch angular velocity
        alpha_ref = a_SRB[:, 2]   # torso pitch angular acceleration

        # Build full reference trajectory
        n_nodes = q_SRB.shape[0]
        theta_col = (-theta_ref)[:, None]
        omega_col = (-omega_ref)[:, None]
        qpos_joint_ref = np.asarray(self.qpos_joints_ref)[None, :].repeat(n_nodes, axis=0)
        qvel_joint_ref = np.asarray(self.qvel_joints_ref)[None, :].repeat(n_nodes, axis=0)
        self.qpos_ref = jnp.array(np.hstack([p_com_ref, theta_col, qpos_joint_ref]))
        self.qvel_ref = jnp.array(np.hstack([v_com_ref, omega_col, qvel_joint_ref]))

        # reference HZ
        self.reference_fps = 1.0 / dt_SRB

        print(f"Initialized SRB tracking Task:\n"
              f"    Loaded trajectory from: {traj_path}\n"
              f"    time: {t_SRB[0]} - {t_SRB[-1]}\n"
              f"    p_com_ref shape: {p_com_ref.shape}\n"
              f"    v_com_ref shape: {v_com_ref.shape}\n"
              f"    a_com_ref shape: {a_com_ref.shape}\n"
              f"    theta_ref shape: {theta_ref.shape}\n"
              f"    omega_ref shape: {omega_ref.shape}\n"
              f"    alpha_ref shape: {alpha_ref.shape}\n")
              

    def _get_reference_configuration(self, t: jax.Array) -> jax.Array:
        """Get the reference position (q) at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.qpos_ref.shape[0] - 1)
        return self.qpos_ref[i, :], self.qvel_ref[i, :]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        q_ref, v_ref = self._get_reference_configuration(state.time)
        q_err = q_ref - state.qpos
        v_err = v_ref - state.qvel

        configuration_cost = jnp.sum(self.q_weights * jnp.square(q_err))
        velocity_cost = jnp.sum(self.v_weights * jnp.square(v_err))
        control_cost = self.w_u * jnp.sum(jnp.square(control - q_ref[3:]))

        return configuration_cost + velocity_cost + control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        q_ref, v_ref = self._get_reference_configuration(state.time)
        q_err = q_ref - state.qpos
        v_err = v_ref - state.qvel

        configuration_cost = jnp.sum(self.qf_weights * jnp.square(q_err))
        velocity_cost = jnp.sum(self.vf_weights * jnp.square(v_err))

        return self.dt(
            configuration_cost +
            velocity_cost
        )

    # def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
    #     """Randomize the friction parameters."""
    #     n_geoms = self.model.geom_friction.shape[0]
    #     multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
    #     new_frictions = self.model.geom_friction.at[:, 0].set(
    #         self.model.geom_friction[:, 0] * multiplier
    #     )
    #     return {"geom_friction": new_frictions}

    # def domain_randomize_data(
    #     self, data: mjx.Data, rng: jax.Array
    # ) -> Dict[str, jax.Array]:
    #     """Randomly perturb the measured base position and velocities."""
    #     rng, q_rng, v_rng = jax.random.split(rng, 3)
    #     q_err = 0.01 * jax.random.normal(q_rng, (3,))
    #     v_err = 0.01 * jax.random.normal(v_rng, (3,))

    #     qpos = data.qpos.at[0:3].set(data.qpos[0:3] + q_err)
    #     qvel = data.qvel.at[0:3].set(data.qvel[0:3] + v_err)

    #     return {"qpos": qpos, "qvel": qvel}

    # def make_data(self) -> mjx.Data:
    #     """Create a new state object with extra constraints allocated."""
    #     return super().make_data(naconmax=20000, njmax=200)
