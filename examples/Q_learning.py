import argparse
import torch
import numpy as np
import time
import mujoco
import mujoco.viewer
from controller.environment import TwoDOFReachingEnv
from model.dqn_agent import DQNAgent


def train(args):
    """Train DQN agent on robot reaching task."""

    # Create environment
    env = TwoDOFReachingEnv()

    # Create agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        device="cpu",
    )

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []

    print("Starting training...")
    print(f"State dim: {env.observation_space.shape[0]}")
    print(f"Action dim: {env.action_space.n}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"GUI: {'Enabled' if args.gui else 'Disabled'}")
    print()

    # Launch viewer if GUI enabled
    viewer = None
    if args.gui:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 1.2]

    try:
        for episode in range(args.episodes):
            # Reset environment with random target
            state, info = env.reset(
                options={"random_target": args.random_targets}
            )
            episode_reward = 0
            episode_loss = []

            for step in range(args.max_steps):
                # Select action (exploration + exploitation)
                action = agent.select_action(state, training=True)

                # Take action in environment
                next_state, reward, terminated, truncated, info = env.step(
                    action
                )
                done = terminated or truncated

                # Store transition in replay buffer
                agent.store_transition(state, action, reward, next_state, done)

                # Train agent (learns from random batch in buffer)
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)

                episode_reward += reward
                state = next_state

                # Update viewer if GUI enabled
                if args.gui and viewer is not None:
                    if not viewer.is_running():
                        print("Viewer closed, stopping training...")
                        raise KeyboardInterrupt
                    viewer.sync()
                    time.sleep(0.001)  # Small delay to see what's happening

                if done:
                    break

            # Record metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step + 1)
            if episode_loss:
                losses.append(np.mean(episode_loss))

            # Print progress
            if (episode + 1) % args.print_freq == 0:
                avg_reward = np.mean(episode_rewards[-args.print_freq :])
                avg_length = np.mean(episode_lengths[-args.print_freq :])
                avg_loss = np.mean(losses[-args.print_freq :]) if losses else 0
                print(f"Episode {episode + 1}/{args.episodes}")
                print(
                    f"  Avg Reward (last {args.print_freq}): {avg_reward:.2f}"
                )
                print(
                    f"  Avg Length (last {args.print_freq}): {avg_length:.1f}"
                )
                print(f"  Avg Loss (last {args.print_freq}): {avg_loss:.4f}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                print()

            # Save checkpoint
            if (episode + 1) % args.save_freq == 0:
                agent.save(args.save_path)
                print(f"Checkpoint saved to {args.save_path}")

        # Final save
        agent.save(args.save_path)
        print("Training complete!")
        print(f"Final model saved to {args.save_path}")

    finally:
        # Clean up
        if viewer is not None:
            viewer.close()
        env.close()


def evaluate(args):
    """Evaluate trained DQN agent."""

    # Create environment
    env = TwoDOFReachingEnv()

    # Create agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device="cpu",
    )

    # Load trained model
    agent.load(args.load_path)
    print(f"Loaded model from {args.load_path}")
    print()

    # Evaluation metrics
    episode_rewards = []
    success_count = 0

    for episode in range(args.eval_episodes):
        state, info = env.reset(options={"random_target": args.random_targets})
        episode_reward = 0

        print(f"Episode {episode + 1}/{args.eval_episodes}")

        if args.gui:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                # Set camera view
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20
                viewer.cam.distance = 3.0
                viewer.cam.lookat[:] = [0, 0, 1.2]

                step = 0
                while viewer.is_running() and step < args.max_steps:
                    # Select action (greedy, no exploration)
                    action = agent.select_action(state, training=False)

                    # Take action
                    next_state, reward, terminated, truncated, info = env.step(
                        action
                    )
                    done = terminated or truncated

                    episode_reward += reward
                    state = next_state

                    # Sync viewer
                    viewer.sync()
                    time.sleep(0.01)

                    step += 1

                    if done:
                        if terminated:  # Success
                            success_count += 1
                            print("  Target reached!")
                        break
        else:
            # No rendering
            for step in range(args.max_steps):
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = env.step(
                    action
                )
                done = terminated or truncated

                episode_reward += reward
                state = next_state

                if done:
                    if terminated:
                        success_count += 1
                        print("  Target reached!")
                    break

        episode_rewards.append(episode_reward)
        print(f"  Reward: {episode_reward:.2f}")
        print()

    # Print summary
    print("Evaluation Summary:")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f}")
    print(
        f"  Success Rate: {success_count}/{args.eval_episodes} ({100*success_count/args.eval_episodes:.1f}%)"
    )

    env.close()

num_episodes = 1000
decay_rate = 0.995

def main():
    parser = argparse.ArgumentParser(description="DQN for 2DOF Robot Arm")
    subparsers = parser.add_subparsers(dest="mode", help="Mode: train or eval")

    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train the agent")
    train_parser.add_argument(
        "--episodes",
        type=int,
        default=num_episodes,
        help="Number of training episodes",
    )
    train_parser.add_argument(
        "--max-steps", type=int, default=500, help="Max steps per episode"
    )
    train_parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    train_parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor"
    )
    train_parser.add_argument(
        "--epsilon-start", type=float, default=1.0, help="Starting epsilon"
    )
    train_parser.add_argument(
        "--epsilon-end", type=float, default=0.01, help="Final epsilon"
    )
    train_parser.add_argument(
        "--epsilon-decay", type=float, default=decay_rate, help="Epsilon decay rate"
    )
    train_parser.add_argument(
        "--buffer-size", type=int, default=10000, help="Replay buffer size"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size"
    )
    train_parser.add_argument(
        "--target-update-freq",
        type=int,
        default=100,
        help="Target network update frequency",
    )
    train_parser.add_argument(
        "--save-path",
        type=str,
        default="data/checkpoints/dqn_robot.pth",
        help="Path to save model",
    )
    train_parser.add_argument(
        "--save-freq", type=int, default=100, help="Save frequency (episodes)"
    )
    train_parser.add_argument(
        "--print-freq", type=int, default=10, help="Print frequency (episodes)"
    )
    train_parser.add_argument(
        "--random-targets",
        action="store_true",
        help="Use random target positions",
    )
    train_parser.add_argument(
        "--gui", action="store_true", help="Show GUI during training"
    )

    # Evaluation arguments
    eval_parser = subparsers.add_parser("eval", help="Evaluate the agent")
    eval_parser.add_argument(
        "--load-path",
        type=str,
        default="data/checkpoints/dqn_robot.pth",
        help="Path to load model",
    )
    eval_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    eval_parser.add_argument(
        "--max-steps", type=int, default=500, help="Max steps per episode"
    )
    eval_parser.add_argument(
        "--gui", action="store_true", help="Show GUI during evaluation"
    )
    eval_parser.add_argument(
        "--random-targets",
        action="store_true",
        help="Use random target positions",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
