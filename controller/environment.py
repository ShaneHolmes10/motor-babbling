import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer


class RobotReachingEnv(gym.Env):
    """
    Robot arm reaching environment with configurable action space.
    Supports both discrete (quantized) and continuous actions.
    """

    def __init__(self, num_links=1, continuous=True, action_quantization=10):
        """
        Initialize environment.

        Args:
            num_links: Number of links in the robot arm
            continuous: If True, use continuous actions. If False, use discrete actions.
            action_quantization: Number of discrete torque levels per joint (only used if continuous=False)
        """
        super().__init__()

        self.num_links = num_links
        self.link_length = 0.5
        self.max_torque = 20.0
        self.continuous = continuous

        xml = self._generate_xml()

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        if continuous:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_links,), dtype=np.float32
            )
        else:
            self.quantize_level = action_quantization
            self.torque_values = np.linspace(-1.0, 1.0, self.quantize_level)
            self.action_space = spaces.Discrete(
                (self.quantize_level) ** self.num_links
            )

        obs_dim = 3 * self.num_links + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.target = np.array([0.410, 1.213])
        self.target_mocap_id = self.model.body("target").mocapid[0]

        self.viewer = None
        self.renderer = None

        self.steps_at_target = 0
        self.target_radius = 0.05
        self.required_steps_at_target = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        if options and options.get("random_init", False):
            for i in range(self.num_links):
                self.data.qpos[i] = np.random.uniform(-np.pi, np.pi)

        if options and options.get("random_target", False):
            angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            max_reach = self.link_length * self.num_links
            radius = max_reach
            self.target = np.array(
                [radius * np.sin(angle), 1.5 - radius * np.cos(angle)]
            )

        self.data.mocap_pos[self.target_mocap_id] = [
            self.target[0],
            0,
            self.target[1],
        ]

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}

        self.steps_at_target = 0

        return obs, info

    def step(self, action):
        """
        Step the environment with either continuous or discrete actions.

        Args:
            action: For continuous mode, array of normalized torques in [-1, 1].
                   For discrete mode, single integer representing joint torques.
        """
        if self.continuous:
            action = np.clip(action, -1.0, 1.0)
            for i in range(self.num_links):
                self.data.ctrl[i] = action[i]
        else:
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

            for i, torque in enumerate(torques):
                self.data.ctrl[i] = torque

        mujoco.mj_step(self.model, self.data)

        self.data.mocap_pos[self.target_mocap_id] = [
            self.target[0],
            0,
            self.target[1],
        ]

        obs = self._get_obs()
        reward = self._compute_reward()

        ee_pos = self.data.site_xpos[0][[0, 2]].copy()
        current_distance = np.linalg.norm(ee_pos - self.target)

        if current_distance < self.target_radius:
            self.steps_at_target += 1
        else:
            self.steps_at_target = 0

        if self.steps_at_target >= self.required_steps_at_target:
            terminated = True
        else:
            terminated = False

        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
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
        """Construct observation vector."""
        q = self.data.qpos[: self.num_links].copy()
        qd = self.data.qvel[: self.num_links].copy()
        qacc = self.data.qacc[: self.num_links].copy()
        ee_pos = self.data.site_xpos[0][[0, 2]].copy()

        obs = np.concatenate([q, qd, qacc, ee_pos, self.target])

        return obs.astype(np.float32)

    def _compute_reward(self):
        """Compute reward based on distance to target."""
        ee_pos = self.data.site_xpos[0][[0, 2]].copy()
        distance = np.linalg.norm(ee_pos - self.target)
        reward = -distance

        return reward

    def get_state_dict(self):
        """Get full state as dictionary."""
        return {
            "joint_angles": self.data.qpos[: self.num_links].copy(),
            "joint_velocities": self.data.qvel[: self.num_links].copy(),
            "joint_accelerations": self.data.qacc[: self.num_links].copy(),
            "end_effector_pos": self.data.site_xpos[0][[0, 2]].copy(),
            "target_pos": self.target.copy(),
        }

    def _generate_xml(self):
        """Generate XML string based on number of links."""
        link_length = self.link_length

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

        indent = "      "
        for i in range(self.num_links):
            xml += f'{indent}<body name="link{i+1}">\n'
            xml += f'{indent}  <joint name="joint{i+1}" type="hinge" axis="0 1 0" damping="0.5"/>\n'
            xml += f'{indent}  <geom type="capsule" fromto="0 0 0 0 0 -{link_length}" size="0.03" rgba="0.2 0.4 0.8 1"/>\n'

            if i == self.num_links - 1:
                xml += f'{indent}  <site name="endeff" pos="0 0 -{link_length}" size="0.01"/>\n'
            else:
                xml += f'\n{indent}  <body name="link{i+2}_parent" pos="0 0 -{link_length}">\n'
                indent += "    "

        for i in range(self.num_links):
            if i < self.num_links - 1:
                xml += "      " + "  " * (self.num_links - i - 1) + "</body>\n"
            xml += "      " + "  " * (self.num_links - i - 1) + "</body>\n"

        xml += """    </body>
        
        <body name="target" mocap="true" pos="0.3 0 1.2">
          <geom type="sphere" size="0.03" rgba="1 0 0 0.5" contype="0" conaffinity="0"/>
        </body>
        
      </worldbody>
    
      <actuator>
    """

        for i in range(self.num_links):
            xml += f'    <motor joint="joint{i+1}" gear="{self.max_torque}" ctrllimited="true" ctrlrange="-1 1"/>\n'

        xml += """  </actuator>
    </mujoco>
    """

        return xml
