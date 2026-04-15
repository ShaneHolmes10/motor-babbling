"""
@brief Interactive manual control script for the RobotReachingEnv.

Launches a passive MuJoCo viewer with actuator sliders so the user can
apply torques to each joint by hand. Useful for inspecting the model,
tuning physical parameters, and verifying environment behavior before
running an automated policy.

Usage:
    python control_model.py [--num-links N]
"""

import time
import argparse
import mujoco
import mujoco.viewer
from controller.environment import RobotReachingEnv

if __name__ == "__main__":

    # Parse command-line arguments so the arm geometry can be configured at runtime
    parser = argparse.ArgumentParser(description="Control robot arm manually")
    parser.add_argument(
        "--num-links",
        type=int,
        default=2,
        help="Number of links in the robot arm (default: 2)",
    )
    args = parser.parse_args()

    # Build the environment with the requested number of links
    env = RobotReachingEnv(num_links=args.num_links)

    # Reset to the default state and discard the initial observation
    obs, info = env.reset()

    # Print viewer usage instructions for the user
    print("MuJoCo Viewer Controls:")
    print("- Drag sliders to apply torques")
    print("- Use mouse to rotate camera")
    print("- Close window when done")
    print()
    # print(f"Action space: {env.action_space.n} discrete actions")
    # print(f"Torque levels: {env.torque_values}")
    print()

    # Display the initial state so the user has a reference before interacting
    print("Current state:")
    state = env.get_state_dict()
    print(f"  Joint angles: {state['joint_angles']}")
    print(f"  End effector: {state['end_effector_pos']}")
    print(f"  Target: {state['target_pos']}")
    print()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

        # Position the camera for a clear side-on view of the arm in the XZ plane
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 1.2]

        while viewer.is_running():
            # Step physics only; torques are applied directly via the viewer's actuator sliders
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.001)  # throttle to roughly real-time at 1 ms timestep

    env.close()
