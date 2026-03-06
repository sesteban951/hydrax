from typing import Dict
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class HumanoidSRB(Task):
    """The Unitree G1 humanoid tracks a reference from SRb data.
    """

    def __init__(
        self,
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

        # make the reference trajectory
        self._make_reference_trajectory()

        # base weights
        w_pos_x = 1.0
        w_pos_z = 10.0
        w_theta = 0.1

        w_vel_x = 0.1
        w_vel_z = 1.0
        w_omega = 0.01

        # joint weights
        w_pos_hip      = 0.0
        w_pos_knee     = 0.0
        w_pos_ankle    = 0.0
        w_pos_shoulder = 0.01
        w_pos_elbow    = 0.01

        w_vel_hip   = 0.0
        w_vel_knee  = 0.0
        w_vel_ankle = 0.0
        w_vel_shoulder = 0.001
        w_vel_elbow    = 0.001

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
        self.w_u = 1e-6

        # centroidal angular momentum weight
        self.w_L = 0.0

        # foot weights
        self.w_pos_foot_x = 0.01
        self.w_pos_foot_theta = 0.1

        # symmetry weights
        self.w_pos_symmetry = 0.1
        self.w_vel_symmetry = 0.01
        self.symmetry_joint_pairs = jnp.array([
            [3, 6],    # hip
            [4, 7],    # knee
            [5, 8],    # ankle
            [9, 11],   # shoulder
            [10, 12],  # elbow
        ], dtype=jnp.int32)

        # terminal weights
        terminal_scaling = 10.0
        self.qf_weights = terminal_scaling * self.q_weights
        self.vf_weights = terminal_scaling * self.v_weights
        self.wf_pos_foot_theta = terminal_scaling * self.w_pos_foot_theta
        self.wf_pos_foot_x = terminal_scaling * self.w_pos_foot_x
        self.wf_pos_symmetry = terminal_scaling * self.w_pos_symmetry
        self.wf_vel_symmetry = terminal_scaling * self.w_vel_symmetry
        self.wf_L = terminal_scaling * self.w_L

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

        # Get sensor IDs
        left_foot_pos_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_position"        )
        left_foot_quat_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_orientation")
        right_foot_pos_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_position")
        right_foot_quat_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_orientation")
        self.left_foot_pos_addr = self.mj_model.sensor_adr[left_foot_pos_id]
        self.left_foot_quat_addr = self.mj_model.sensor_adr[left_foot_quat_id]
        self.right_foot_pos_addr = self.mj_model.sensor_adr[right_foot_pos_id]
        self.right_foot_quat_addr = self.mj_model.sensor_adr[right_foot_quat_id]

        print(f"Model properties:\n"
              f"    Time step (dt): {self.dt}\n"
              f"    Reference keyframe: '{keyframe}'\n"
              f"    qpos_default: {self.qpos_default}\n"
              f"    qvel_default: {self.qvel_default}\n")


    def _make_reference_trajectory(self):

        # Load trajectory CSV files from a stable package-relative location.
        traj_path = Path(ROOT) / "trajectories" / "srb_jump_2d"
        time_path = traj_path / "time.csv"
        q_opt_path = traj_path / "q_opt.csv"
        v_opt_path = traj_path / "v_opt.csv"
        a_opt_path = traj_path / "a_opt.csv"
        c_opt_path = traj_path / "contact_schedule.csv"

        t_SRB = np.loadtxt(time_path, delimiter=",")
        q_SRB = np.loadtxt(q_opt_path, delimiter=",")
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
        T_pre = 0.5
        T_post = 0.5

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
        theta_col = (-theta_ref)[:, None] # negative because planar y has negative axis
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
        return self.qpos_ref[i, :], self.qvel_ref[i, :] # (nq,), (nv,)
    

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        q_ref, v_ref = self._get_reference_configuration(state.time)

        # base tracking
        q_actual = state.qpos  # (nq,)
        v_actual = state.qvel  # (nv,)
        q_err = q_ref - q_actual
        v_err = v_ref - v_actual

        # # COM tracking
        com_pos, com_vel = self._get_com_state(state)
        # q_actual = jnp.concatenate([com_pos, state.qpos[2:]])
        # v_actual = jnp.concatenate([com_vel, state.qvel[2:]])
        # q_err = q_ref - q_actual
        # v_err = v_ref - v_actual

        configuration_cost = jnp.sum(self.q_weights * jnp.square(q_err))
        velocity_cost = jnp.sum(self.v_weights * jnp.square(v_err))
        control_cost = self.w_u * jnp.sum(jnp.square(control - q_ref[3:]))
        position_symmetry_cost = self.w_pos_symmetry * self._joint_symmetry_cost(q_actual)
        velocity_symmetry_cost = self.w_vel_symmetry * self._joint_symmetry_cost(v_actual)

        # Foot costs: orientation (flat) and x position (near 0)
        left_pos, right_pos, left_quat, right_quat = self._get_foot_pose(state) 
        foot_orientation_cost = self.w_pos_foot_theta * (
              jnp.square(self._foot_pitch(left_quat))
            + jnp.square(self._foot_pitch(right_quat))
        )
        foot_x_cost = self.w_pos_foot_x * (
              jnp.square(left_pos[0] - com_pos[0])
            + jnp.square(right_pos[0] - com_pos[0])
        )

        # angular momentum
        L = self._centroidal_ang_mom(state)
        ang_mom_cost = self.w_L * jnp.square(L[1])

        return (
              configuration_cost
            + velocity_cost
            + control_cost
            + foot_orientation_cost
            + foot_x_cost
            + position_symmetry_cost
            + velocity_symmetry_cost
            + ang_mom_cost
        )


    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        q_ref, v_ref = self._get_reference_configuration(state.time)

        # base tracking
        q_actual = state.qpos  # (nq,)
        v_actual = state.qvel  # (nv,)
        q_err = q_ref - q_actual
        v_err = v_ref - v_actual

        # # COM tracking
        com_pos, com_vel = self._get_com_state(state)
        # q_actual = jnp.concatenate([com_pos, state.qpos[2:]])
        # v_actual = jnp.concatenate([com_vel, state.qvel[2:]])
        # q_err = q_ref - q_actual
        # v_err = v_ref - v_actual

        configuration_cost = jnp.sum(self.qf_weights * jnp.square(q_err))
        velocity_cost = jnp.sum(self.vf_weights * jnp.square(v_err))
        position_symmetry_cost = self.wf_pos_symmetry * self._joint_symmetry_cost(q_actual)
        velocity_symmetry_cost = self.wf_vel_symmetry * self._joint_symmetry_cost(v_actual)

        left_pos, right_pos, left_quat, right_quat = self._get_foot_pose(state)
        foot_orientation_cost = self.wf_pos_foot_theta * (
              jnp.square(self._foot_pitch(left_quat))
            + jnp.square(self._foot_pitch(right_quat))
        )
        foot_x_cost = self.wf_pos_foot_x * (
              jnp.square(left_pos[0] - com_pos[0])
            + jnp.square(right_pos[0] - com_pos[0])
        )

        # angular momentum
        L = self._centroidal_ang_mom(state)
        ang_mom_cost = self.wf_L * jnp.square(L[1])

        return self.dt * (
              configuration_cost
            + velocity_cost
            + foot_orientation_cost
            + foot_x_cost
            + position_symmetry_cost
            + velocity_symmetry_cost
            + ang_mom_cost
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

    def make_data(self) -> mjx.Data:
        """Create a new state object with extra constraints allocated."""
        return super().make_data(naconmax=20000, njmax=200)

    # ===============================================================
    # UTILS
    # ===============================================================

    def _get_foot_pose(self, data: mjx.Data):
        """Get the current foot positions and orientations."""
        left_pos = data.sensordata[self.left_foot_pos_addr : self.left_foot_pos_addr + 3]        # (3,)
        right_pos = data.sensordata[self.right_foot_pos_addr : self.right_foot_pos_addr + 3]     # (3,)
        left_quat = data.sensordata[self.left_foot_quat_addr : self.left_foot_quat_addr + 4]     # (4,)
        right_quat = data.sensordata[self.right_foot_quat_addr : self.right_foot_quat_addr + 4]  # (4,)
        return left_pos, right_pos, left_quat, right_quat

    def _foot_pitch(self, quat: jax.Array):
        """Extract pitch angle from a quaternion [qw, qx, qy, qz].

        For planar rotation about the y-axis: q = [cos(θ/2), 0, sin(θ/2), 0],
        so θ = 2 * arctan2(qy, qw).
        """
        qw = quat[0]
        qy = quat[2]
        return 2.0 * jnp.arctan2(qy, qw)
    
    def _get_com_state(self, data: mjx.Data):
        """Extract CoM position (x, z) and linear velocity (vx, vz) of the full robot.

        Uses subtree_com[1] and subtree_linvel[1], where body 1 is the root body
        so its subtree spans the entire robot.
        """
        _com_pos = data.subtree_com[0]
        _com_vel = data._impl.subtree_linvel[0]
        com_pos = _com_pos[jnp.array([0, 2])]  # (x, z)
        com_vel = _com_vel[jnp.array([0, 2])]  # (vx, vz)
        return com_pos, com_vel
    
    def _joint_symmetry_cost(self, values: jax.Array) -> jax.Array:
        """Sum squared left-right differences over mirrored joints."""
        left_vals = values[self.symmetry_joint_pairs[:, 0]]
        right_vals = values[self.symmetry_joint_pairs[:, 1]]
        return jnp.sum(jnp.square(left_vals - right_vals))
    
    def _centroidal_ang_mom(self, data: mjx.Data):
        """
        Compute the centroidal angular momentum about the COM in the world frame 
        for a single environment.
        """
        L = data._impl.subtree_angmom[0]
        return L  # (3,)
