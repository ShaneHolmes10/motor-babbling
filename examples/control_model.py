import time
import argparse
import mujoco
import mujoco.viewer
from controller.environment import RobotReachingEnv

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Control robot arm manually")
    parser.add_argument(
        "--num-links",
        type=int,
        default=2,
        help="Number of links in the robot arm (default: 2)",
    )
    args = parser.parse_args()

    env = RobotReachingEnv(num_links=args.num_links)

    obs, info = env.reset()

    print("MuJoCo Viewer Controls:")
    print("- Drag sliders to apply torques")
    print("- Use mouse to rotate camera")
    print("- Close window when done")
    print()
    print(f"Action space: {env.action_space.n} discrete actions")
    print(f"Torque levels: {env.torque_values}")
    print()
    print("Current state:")
    state = env.get_state_dict()
    print(f"  Joint angles: {state['joint_angles']}")
    print(f"  End effector: {state['end_effector_pos']}")
    print(f"  Target: {state['target_pos']}")
    print()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 1.2]

        while viewer.is_running():
            # Just step physics - control via viewer sliders
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.001)

    env.close()
