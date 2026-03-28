import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer


class TwoDOFReachingEnv(gym.Env):

    def __init__(self, num_links=2):
        super().__init__()

        self.num_links = num_links
        self.link_length = 0.5

        self.max_torque = 15.0

        xml = self._generate_xml()

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.quantize_level = 5
        self.torque_values = np.linspace(-1.0, 1.0, self.quantize_level)
        self.action_space = spaces.Discrete(
            (self.quantize_level) ** self.num_links
        )

        # Observation space size depends on number of links
        obs_dim = (
            3 * self.num_links + 4
        )  # (q, qd, qacc per joint) + (ee_x, ee_z, target_x, target_z)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Target position
        self.target = np.array([0.410, 1.213])  # [x, z] in world frame

        self.target_mocap_id = self.model.body("target").mocapid[0]

        # Viewer for rendering
        self.viewer = None
        self.renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Optionally randomize initial joint positions
        if options and options.get("random_init", False):
            for i in range(self.num_links):
                self.data.qpos[i] = np.random.uniform(-np.pi, np.pi)

        # Optionally randomize target
        if options and options.get("random_target", False):
            angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            max_reach = 0.5 * self.num_links  # Total reach of all links
            radius = np.random.uniform(0.2, max_reach)
            self.target = np.array(
                [radius * np.sin(angle), 1.5 - radius * np.cos(angle)]
            )

        # Update target visualization position
        self.data.mocap_pos[self.target_mocap_id] = [
            self.target[0],
            0,
            self.target[1],
        ]

        # Step forward to stabilize
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        # Map discrete action to torques for all joints
        if self.num_links == 1:
            torques = [self.torque_values[action]]
        else:
            torques = []
            remaining = action
            for i in range(self.num_links):
                torques.append(
                    self.torque_values[remaining % self.quantize_level]
                )
                remaining //= self.quantize_level

        # Apply torques to all joints
        for i, torque in enumerate(torques):
            self.data.ctrl[i] = torque

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Keep target visualization in sync
        self.data.mocap_pos[self.target_mocap_id] = [
            self.target[0],
            0,
            self.target[1],
        ]

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
        # Joint angles for all joints
        q = self.data.qpos[: self.num_links].copy()

        # Joint velocities for all joints
        qd = self.data.qvel[: self.num_links].copy()

        # Joint accelerations for all joints
        qacc = self.data.qacc[: self.num_links].copy()

        # End-effector position (site named "endeff")
        ee_pos = self.data.site_xpos[0][[0, 2]].copy()  # [x, z]

        # Concatenate: [q1, ..., qn, qd1, ..., qdn, qacc1, ..., qaccn, ee_x, ee_z, target_x, target_z]
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
            "joint_angles": self.data.qpos[
                : self.num_links
            ].copy(),  # Change [:2] to [:self.num_links]
            "joint_velocities": self.data.qvel[
                : self.num_links
            ].copy(),  # Change [:2] to [:self.num_links]
            "joint_accelerations": self.data.qacc[
                : self.num_links
            ].copy(),  # Change [:2] to [:self.num_links]
            "end_effector_pos": self.data.site_xpos[0][[0, 2]].copy(),
            "target_pos": self.target.copy(),
        }

    def _generate_xml(self):
        """Generate XML string based on number of links."""

        link_length = 0.5

        # Start with header
        xml = """<mujoco>
      <option gravity="0 0 -9.81" timestep="0.001">
        <flag contact="disable"/>
      </option>
      
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
    """

        # Build links recursively
        indent = "      "
        for i in range(self.num_links):
            xml += f'{indent}<body name="link{i+1}">\n'
            xml += f'{indent}  <joint name="joint{i+1}" type="hinge" axis="0 1 0" damping="0.5"/>\n'
            xml += f'{indent}  <geom type="capsule" fromto="0 0 0 0 0 -{link_length}" size="0.03" rgba="0.2 0.4 0.8 1"/>\n'

            # Add end effector site on last link
            if i == self.num_links - 1:
                xml += f'{indent}  <site name="endeff" pos="0 0 -{link_length}" size="0.01"/>\n'
            else:
                # Position next link at end of current link
                xml += f'\n{indent}  <body name="link{i+2}_parent" pos="0 0 -{link_length}">\n'
                indent += "    "

        # Close all link bodies
        for i in range(self.num_links):
            if i < self.num_links - 1:
                xml += "      " + "  " * (self.num_links - i - 1) + "</body>\n"
            xml += "      " + "  " * (self.num_links - i - 1) + "</body>\n"

        # Close base body
        xml += """    </body>
        
        <body name="target" mocap="true" pos="0.3 0 1.2">
          <geom type="sphere" size="0.03" rgba="1 0 0 0.5" contype="0" conaffinity="0"/>
        </body>
        
      </worldbody>
    
      <actuator>
    """

        # Add actuators for each joint
        for i in range(self.num_links):
            xml += f'    <motor joint="joint{i+1}" gear="{self.max_torque}" ctrllimited="true" ctrlrange="-1 1"/>\n'

        xml += """  </actuator>
    </mujoco>
    """

        return xml
