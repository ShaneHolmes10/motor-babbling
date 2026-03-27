import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer


class TwoDOFReachingEnv(gym.Env):

    def __init__(self):
        super().__init__()

        xml = """
        <mujoco>
          <option gravity="0 0 -9.81" timestep="0.001"/>

          <visual>
            <global offwidth="1280" offheight="720"/>
            <quality shadowsize="4096"/>
          </visual>

          <statistic center="0 0 1.2" extent="1.5"/>

          <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>

            <camera name="view1" pos="1.5 -1.5 2" xyaxes="1 1 0 0 0 1"/>

            <body name="base" pos="0 0 1.5">
              <geom type="sphere" size="0.05" rgba="0.5 0.5 0.5 1"/>

              <body name="link1">
                <joint name="joint1" type="hinge" axis="0 1 0" damping="0.5"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03" rgba="0.2 0.4 0.8 1"/>

                <body name="link2" pos="0 0 -0.5">
                  <joint name="joint2" type="hinge" axis="0 1 0" damping="0.5"/>
                  <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.03" rgba="0.2 0.8 0.4 1"/>

                  <site name="endeff" pos="0 0 -0.4" size="0.01"/>
                </body>
              </body>
            </body>
          </worldbody>

          <actuator>
            <motor joint="joint1" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
            <motor joint="joint2" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
          </actuator>
        </mujoco>
        """

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Action space: 9 discrete actions (quantized control range)
        self.action_space = spaces.Discrete(9)
        self.torque_values = [-10.0, 0.0, 10.0]

        # Observation: [q1, q2, qd1, qd2, qacc1, qacc2, ee_x, ee_z, target_x, target_z]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )

        # Target position
        self.target = np.array([0.5, 1.0])  # [x, z] in world frame

        # Viewer for rendering
        self.viewer = None
        self.renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Optionally randomize initial joint positions
        if options and options.get("random_init", False):
            self.data.qpos[0] = np.random.uniform(-np.pi, np.pi)
            self.data.qpos[1] = np.random.uniform(-np.pi, np.pi)

        # Optionally randomize target
        if options and options.get("random_target", False):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.3, 0.7)
            self.target = np.array(
                [radius * np.sin(angle), 1.5 - radius * np.cos(angle)]
            )

        # Step forward to stabilize
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        # Map discrete action to quantized torque values
        joint1_torque = self.torque_values[action // 3]
        joint2_torque = self.torque_values[action % 3]

        # Apply torques
        self.data.ctrl[0] = joint1_torque
        self.data.ctrl[1] = joint2_torque

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Compute reward
        reward = self._compute_reward()

        # Episode termination
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment.
        Call this after step() to visualize the current state.
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _get_obs(self):
        """Construct observation vector with all state information."""
        # Joint angles
        q = self.data.qpos[:2].copy()

        # Joint velocities
        qd = self.data.qvel[:2].copy()

        # Joint accelerations
        qacc = self.data.qacc[:2].copy()

        # End-effector position (site named "endeff")
        ee_pos = self.data.site_xpos[0][[0, 2]].copy()  # [x, z]

        # Concatenate: [q1, q2, qd1, qd2, qacc1, qacc2, ee_x, ee_z, target_x, target_z]
        obs = np.concatenate([q, qd, qacc, ee_pos, self.target])

        return obs.astype(np.float32)

    def _compute_reward(self):
        """Compute reward based on distance to target."""
        # Get end-effector position
        ee_pos = self.data.site_xpos[0][[0, 2]].copy()

        # Distance to target
        distance = np.linalg.norm(ee_pos - self.target)

        # Negative distance as reward
        reward = -distance

        return reward

    def get_state_dict(self):
        """
        Get full state as dictionary for debugging/visualization.

        Returns:
            dict with joint_angles, joint_velocities, joint_accelerations,
            end_effector_pos, target_pos
        """
        return {
            "joint_angles": self.data.qpos[:2].copy(),
            "joint_velocities": self.data.qvel[:2].copy(),
            "joint_accelerations": self.data.qacc[:2].copy(),
            "end_effector_pos": self.data.site_xpos[0][[0, 2]].copy(),
            "target_pos": self.target.copy(),
        }
