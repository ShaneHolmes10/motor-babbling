import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer


class RobotReachingEnv(gym.Env):
    """
    @brief Robot arm reaching environment with configurable action space.

    Gymnasium environment wrapping a MuJoCo planar arm. The arm must reach
    and hold a target position for a fixed number of consecutive steps to
    terminate the episode. Supports both continuous and discrete action spaces.
    """

    def __init__(self, num_links=1, continuous=True, action_quantization=10):
        """
        @brief Initialize the robot reaching environment.

        Builds the MuJoCo model, defines the action and observation spaces,
        and sets up termination tracking state.

        @param num_links Number of rigid links in the planar robot arm.
        @param continuous If True, actions are continuous normalized torques in [-1, 1].
                          If False, actions are discrete integers indexing torque levels.
        @param action_quantization Number of evenly spaced torque levels per joint
                                   when using discrete actions (ignored in continuous mode).
        """
        super().__init__()

        self.num_links = num_links
        self.link_length = 0.5  # length of each arm segment in meters
        self.max_torque = 60.0  # maximum torque applied to each joint (Nm), scaled by gear ratio in XML
        self.continuous = continuous

        # Build the MuJoCo model from a dynamically generated XML string
        xml = self._generate_xml()

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        if continuous:
            # Normalized torque per joint, will be scaled by max_torque via the actuator gear in XML
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_links,), dtype=np.float32
            )
        else:
            # Evenly spaced torque levels across [-1, 1] for each joint
            self.quantize_level = action_quantization
            self.torque_values = np.linspace(-1.0, 1.0, self.quantize_level)
            # Encode all joint torques as a single integer using mixed-radix representation
            self.action_space = spaces.Discrete(
                (self.quantize_level) ** self.num_links
            )

        # Observation: joint angles + velocities + accelerations (num_links each) + end-effector XZ + target XZ
        obs_dim = 3 * self.num_links + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Default target position [x, z] in world space
        self.target = np.array([0.410, 1.213])
        # MuJoCo mocap body ID for the visual target marker
        self.target_mocap_id = self.model.body("target").mocapid[0]

        self.viewer = None  # passive viewer for interactive rendering
        self.renderer = None  # offscreen renderer (unused by default)

        # Track consecutive steps within target radius for termination condition
        self.steps_at_target = 0
        self.target_radius = (
            0.10  # distance threshold to count as "at target" (meters)
        )
        self.required_steps_at_target = (
            100  # number of consecutive steps needed to terminate successfully
        )

    def reset(self, seed=None, options=None):
        """
        @brief Reset the environment to an initial state.

        Clears all physics state and optionally randomizes joint angles and
        target position based on flags passed through the options dict.

        @param seed Optional RNG seed forwarded to the base Gymnasium class.
        @param options Optional dict with boolean flags:
                       - "random_init": randomize starting joint angles in [-pi, pi].
                       - "random_target": place the target at a random reachable position.
        @return Tuple of (observation, info) where observation is a float32 numpy array
                and info is an empty dict.
        """
        super().reset(seed=seed)

        # Reset all joint positions, velocities, and accelerations to their defaults
        mujoco.mj_resetData(self.model, self.data)

        if options and options.get("random_init", False):
            # Randomize starting joint angles across the full rotation range
            for i in range(self.num_links):
                self.data.qpos[i] = np.random.uniform(-np.pi, np.pi)

        if options and options.get("random_target", False):
            # Sample a reachable target position within the arm's reach envelope
            angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            max_reach = self.link_length * self.num_links
            radius = np.random.uniform(0.1, max_reach)
            # Convert polar coordinates to world-space [x, z], offset from the base at z=1.5
            self.target = np.array(
                [radius * np.sin(angle), 1.5 - radius * np.cos(angle)]
            )

        # Sync the visual target marker with the current target position
        # MuJoCo uses [x, y, z]; y=0 keeps everything in the XZ plane
        self.data.mocap_pos[self.target_mocap_id] = [
            self.target[0],
            0,
            self.target[1],
        ]

        # Propagate the reset state through the physics model before reading observations
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}

        self.steps_at_target = 0

        return obs, info

    def step(self, action):
        """
        @brief Advance the simulation by one timestep.

        Applies joint torques derived from the given action, steps the MuJoCo
        physics, and evaluates the termination condition.

        @param action In continuous mode: float32 array of shape (num_links,) with
                      normalized torques in [-1, 1]. In discrete mode: a single integer
                      whose mixed-radix digits index per-joint torque levels.
        @return Tuple of (obs, reward, terminated, truncated, info).
                obs is a float32 numpy array, reward is a negative scalar distance,
                terminated is True when the arm holds the target for the required steps,
                truncated is always False (handled externally), info is an empty dict.
        """
        if self.continuous:
            action = np.clip(action, -1.0, 1.0)
            for i in range(self.num_links):
                self.data.ctrl[i] = action[i]
        else:
            if self.num_links == 1:
                # Single-joint case: action directly indexes into the torque lookup table
                torques = [self.torque_values[action]]
            else:
                # Multi-joint case: decode the mixed-radix integer into per-joint torque indices
                # Each digit in base `quantize_level` maps to one joint's torque level
                torques = []
                remaining = action
                for i in range(self.num_links):
                    torques.append(
                        self.torque_values[remaining % self.quantize_level]
                    )
                    remaining //= self.quantize_level

            for i, torque in enumerate(torques):
                self.data.ctrl[i] = torque

        # Advance the physics simulation by one timestep
        mujoco.mj_step(self.model, self.data)

        # Re-pin the mocap target each step; MuJoCo physics can drift mocap bodies otherwise
        self.data.mocap_pos[self.target_mocap_id] = [
            self.target[0],
            0,
            self.target[1],
        ]

        obs = self._get_obs()
        reward = self._compute_reward()

        # Extract end-effector XZ position from the named site
        ee_pos = self.data.site_xpos[0][[0, 2]].copy()
        current_distance = np.linalg.norm(ee_pos - self.target)

        # Increment or reset the consecutive-steps counter based on proximity to target
        if current_distance < self.target_radius:
            self.steps_at_target += 1
        else:
            self.steps_at_target = 0

        # Episode terminates successfully once the arm holds position at the target long enough
        if self.steps_at_target >= self.required_steps_at_target:
            terminated = True
        else:
            terminated = False

        truncated = False  # time-limit truncation is handled externally (e.g. TimeLimit wrapper)
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        @brief Render the current simulation state using the passive MuJoCo viewer.

        Launches the viewer on the first call and synchronizes it on every subsequent call.
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

    def close(self):
        """
        @brief Release the viewer and renderer resources.

        Safe to call even if neither has been initialized.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _get_obs(self):
        """
        @brief Construct the observation vector from current simulation state.

        Concatenates joint angles, velocities, accelerations, end-effector XZ
        position, and target XZ position into a single flat array.

        @return float32 numpy array of shape (3 * num_links + 4,).
        """
        q = self.data.qpos[: self.num_links].copy()  # joint angles (rad)
        qd = self.data.qvel[
            : self.num_links
        ].copy()  # joint velocities (rad/s)
        qacc = self.data.qacc[
            : self.num_links
        ].copy()  # joint accelerations (rad/s^2)
        # site_xpos rows are [x, y, z]; index [0,2] extracts [x, z] for the 2D XZ plane
        ee_pos = self.data.site_xpos[0][[0, 2]].copy()

        # Full observation: [q, qd, qacc, end_effector_xz, target_xz]
        obs = np.concatenate([q, qd, qacc, ee_pos, self.target])

        return obs.astype(np.float32)

    def _compute_reward(self):
        """
        @brief Compute the dense reward for the current timestep.

        Uses negative Euclidean distance between the end-effector and the target
        so the agent is rewarded for getting closer at every step.

        @return Scalar float reward equal to the negative distance to the target.
        """
        ee_pos = self.data.site_xpos[0][[0, 2]].copy()
        distance = np.linalg.norm(ee_pos - self.target)
        # Dense negative-distance reward encourages minimizing distance at every step
        reward = -distance

        return reward

    def get_state_dict(self):
        """
        @brief Return a named snapshot of the current simulation state.

        Useful for logging, debugging, or checkpointing outside of the standard
        Gymnasium observation format.

        @return Dict with keys: "joint_angles", "joint_velocities",
                "joint_accelerations", "end_effector_pos", "target_pos".
                All values are float64 numpy arrays.
        """
        return {
            "joint_angles": self.data.qpos[: self.num_links].copy(),
            "joint_velocities": self.data.qvel[: self.num_links].copy(),
            "joint_accelerations": self.data.qacc[: self.num_links].copy(),
            "end_effector_pos": self.data.site_xpos[0][[0, 2]].copy(),
            "target_pos": self.target.copy(),
        }

    def _generate_xml(self):
        """
        @brief Programmatically generate the MuJoCo MJCF XML for the robot arm.

        Builds a kinematic chain of num_links hinge-jointed bodies rooted at z=1.5,
        each driven by a capped motor actuator, plus a mocap sphere for the target.

        @return String containing the complete MJCF XML model definition.
        """
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

        # Build nested body/joint elements for each link in the chain.
        # Each link after the first is parented inside the previous link's tip body.
        indent = "      "
        for i in range(self.num_links):
            xml += f'{indent}<body name="link{i+1}">\n'
            xml += f'{indent}  <joint name="joint{i+1}" type="hinge" axis="0 1 0" damping="0.5"/>\n'
            xml += f'{indent}  <geom type="capsule" fromto="0 0 0 0 0 -{link_length}" size="0.03" rgba="0.2 0.4 0.8 1"/>\n'

            if i == self.num_links - 1:
                # Place the end-effector site at the tip of the last link
                xml += f'{indent}  <site name="endeff" pos="0 0 -{link_length}" size="0.01"/>\n'
            else:
                # Intermediate links: open a child body offset to the tip of this link
                xml += f'\n{indent}  <body name="link{i+2}_parent" pos="0 0 -{link_length}">\n'
                indent += "    "  # increase indent depth to match the extra nesting level

        # Close all open body tags in reverse order, from innermost to outermost
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

        # One motor actuator per joint; gear scales the normalized [-1,1] control input to max_torque
        for i in range(self.num_links):
            xml += f'    <motor joint="joint{i+1}" gear="{self.max_torque}" ctrllimited="true" ctrlrange="-1 1"/>\n'

        xml += """  </actuator>
    </mujoco>
    """

        return xml
