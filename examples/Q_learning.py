import argparse
import importlib
import torch
import numpy as np
import time
import sys
import mujoco
import os
import mujoco.viewer
import matplotlib.pyplot as plt
from controller.environment import RobotReachingEnv
import time as time_module

num_links = 1
action_quantization = 10
num_episodes = 500
decay_rate = 0.99999

max_steps = 500
eval_episodes = 1


def get_agent_class(agent_type):
    """
    Dynamically impoXrt agent class.

    Args:
        agent_type: String name of agent (e.g., 'dqn', 'ddpg')

    Returns:
        Agent class
    """
    try:
        agent_module = importlib.import_module(f"model.{agent_type}_agent")
        class_name = f"{agent_type.upper()}Agent"
        return getattr(agent_module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not load agent '{agent_type}': {e}")


def requires_continuous_actions(agent_type):
    """
    Determine if agent requires continuous action space.

    Args:
        agent_type: String name of agent

    Returns:
        True if continuous, False if discrete
    """
    continuous_agents = ["ddpg", "td3", "sac", "ppo"]
    return agent_type.lower() in continuous_agents


def get_agent(agent_type, env, args, mode="train"):
    """
    Create agent with appropriate parameters based on type.

    Args:
        agent_type: String name of agent
        env: Environment instance
        args: Command line arguments
        mode: 'train', 'eval', or 'play' - for eval/play, uses minimal params

    Returns:
        Initialized agent
    """
    AgentClass = get_agent_class(agent_type)

    state_dim = env.observation_space.shape[0]

    if requires_continuous_actions(agent_type):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    if mode in ["eval", "play"]:
        if agent_type == "dqn":
            return AgentClass(
                state_dim=state_dim,
                action_dim=action_dim,
                device="cpu",
            )
        elif agent_type == "ddpg":
            return AgentClass(
                state_dim=state_dim,
                action_dim=action_dim,
                device="cpu",
            )
        else:
            return AgentClass(
                state_dim=state_dim,
                action_dim=action_dim,
                device="cuda",
            )

    common_params = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "device": "cpu",
    }

    if agent_type == "dqn":
        return AgentClass(
            **common_params,
            learning_rate=args.lr,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            target_update_freq=args.target_update_freq,
        )
    elif agent_type == "ddpg":
        return AgentClass(
            **common_params,
            max_action=1.0,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            tau=args.tau,
            noise_std=args.noise_std,
        )
    else:
        try:
            return AgentClass(**common_params)
        except TypeError as e:
            raise ValueError(
                f"Agent '{agent_type}' requires custom parameters. "
                f"Add configuration in get_agent() function. Error: {e}"
            )


def get_loss_structure(agent_type):
    """
    Determine loss structure for plotting.

    Args:
        agent_type: String name of agent

    Returns:
        'single' for one loss value, 'dual' for critic/actor losses
    """
    dual_loss_agents = ["ddpg", "td3", "sac", "ppo"]
    return "dual" if agent_type.lower() in dual_loss_agents else "single"


def plot_training_results(episode_rewards, losses, model_path, agent_type):
    """Generate and save training plots."""
    plot_dir = "data/plots"
    os.makedirs(plot_dir, exist_ok=True)

    plot_prefix = os.path.splitext(os.path.basename(model_path))[0]
    loss_structure = get_loss_structure(agent_type)

    if loss_structure == "single":
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].scatter(
            range(len(episode_rewards)),
            episode_rewards,
            alpha=0.4,
            s=10,
            label="Episode Reward",
        )
        axes[0].plot(episode_rewards, alpha=0.3, linewidth=0.5)
        window = min(50, len(episode_rewards) // 10)
        if window > 0:
            moving_avg = np.convolve(
                episode_rewards, np.ones(window) / window, mode="valid"
            )
            axes[0].plot(
                range(window - 1, len(episode_rewards)),
                moving_avg,
                color="red",
                linewidth=2,
                label=f"{window}-Episode Moving Avg",
            )
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Total Reward")
        axes[0].set_title("Training Rewards over Episodes")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(losses, alpha=0.6, label="Loss")
        if window > 0 and len(losses) > window:
            moving_avg_loss = np.convolve(
                losses, np.ones(window) / window, mode="valid"
            )
            axes[1].plot(
                range(window - 1, len(losses)),
                moving_avg_loss,
                color="red",
                linewidth=2,
                label=f"{window}-Episode Moving Avg",
            )
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training Loss over Episodes")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        data_dict = {
            "episode_rewards": episode_rewards,
            "losses": losses,
        }

    elif loss_structure == "dual":
        critic_losses, actor_losses = losses
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        axes[0].scatter(
            range(len(episode_rewards)),
            episode_rewards,
            alpha=0.4,
            s=10,
            label="Episode Reward",
        )
        axes[0].plot(episode_rewards, alpha=0.3, linewidth=0.5)
        window = min(50, len(episode_rewards) // 10)
        if window > 0:
            moving_avg = np.convolve(
                episode_rewards, np.ones(window) / window, mode="valid"
            )
            axes[0].plot(
                range(window - 1, len(episode_rewards)),
                moving_avg,
                color="red",
                linewidth=2,
                label=f"{window}-Episode Moving Avg",
            )
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Total Reward")
        axes[0].set_title("Training Rewards over Episodes")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            critic_losses, alpha=0.6, label="Critic Loss", color="blue"
        )
        if window > 0 and len(critic_losses) > window:
            moving_avg_loss = np.convolve(
                critic_losses, np.ones(window) / window, mode="valid"
            )
            axes[1].plot(
                range(window - 1, len(critic_losses)),
                moving_avg_loss,
                color="red",
                linewidth=2,
                label=f"{window}-Episode Moving Avg",
            )
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Critic Loss")
        axes[1].set_title("Critic Loss over Episodes")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(
            actor_losses, alpha=0.6, label="Actor Loss", color="green"
        )
        if window > 0 and len(actor_losses) > window:
            moving_avg_actor = np.convolve(
                actor_losses, np.ones(window) / window, mode="valid"
            )
            axes[2].plot(
                range(window - 1, len(actor_losses)),
                moving_avg_actor,
                color="red",
                linewidth=2,
                label=f"{window}-Episode Moving Avg",
            )
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Actor Loss")
        axes[2].set_title("Actor Loss over Episodes")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        data_dict = {
            "episode_rewards": episode_rewards,
            "critic_losses": critic_losses,
            "actor_losses": actor_losses,
        }

    plt.tight_layout()

    plot_path = os.path.join(plot_dir, f"{plot_prefix}_training.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Training plots saved to {plot_path}")

    plot_path_hires = os.path.join(
        plot_dir, f"{plot_prefix}_training_hires.png"
    )
    plt.savefig(plot_path_hires, dpi=300)

    plt.close()

    data_path = os.path.join(plot_dir, f"{plot_prefix}_training_data.npz")
    np.savez(data_path, **data_dict)
    print(f"Training data saved to {data_path}")


def train(args):
    """Train agent on robot reaching task."""

    start_time = time_module.time()

    continuous = requires_continuous_actions(args.agent)
    env = RobotReachingEnv(
        num_links=num_links,
        continuous=continuous,
        action_quantization=action_quantization,
    )

    agent = get_agent(args.agent, env, args)

    if args.resume and os.path.exists(args.save_path):
        print(f"Resuming from checkpoint: {args.save_path}")
        agent.load(args.save_path)
    elif args.resume:
        print(
            f"Warning: --resume specified but no checkpoint found at {args.save_path}"
        )

    episode_rewards = []
    episode_lengths = []

    loss_structure = get_loss_structure(args.agent)
    if loss_structure == "single":
        losses = []
    elif loss_structure == "dual":
        critic_losses = []
        actor_losses = []

    print(f"Starting training with {args.agent.upper()}...")
    print(f"State dim: {env.observation_space.shape[0]}")
    if continuous:
        print(f"Action dim: {env.action_space.shape[0]} (continuous)")
    else:
        print(f"Action dim: {env.action_space.n} (discrete)")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"GUI: {'Enabled' if args.gui else 'Disabled'}")
    print()

    viewer = None
    if args.gui:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 1.2]

    try:
        for episode in range(args.episodes):
            state, info = env.reset(
                options={"random_target": args.random_targets}
            )
            episode_reward = 0
            episode_loss = []

            for step in range(args.max_steps):
                action = agent.select_action(state, training=True)

                next_state, reward, terminated, truncated, info = env.step(
                    action
                )
                done = terminated or truncated

                agent.store_transition(state, action, reward, next_state, done)

                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)

                episode_reward += reward
                state = next_state

                if args.gui and viewer is not None:
                    if not viewer.is_running():
                        print("Viewer closed, stopping training...")
                        raise KeyboardInterrupt
                    viewer.sync()
                    time.sleep(0.001)

                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(step + 1)

            if episode_loss:
                if loss_structure == "single":
                    losses.append(np.mean(episode_loss))
                elif loss_structure == "dual":
                    # Filter out None values before taking mean
                    valid_losses = [
                        x
                        for x in episode_loss
                        if x[0] is not None and x[1] is not None
                    ]
                    if valid_losses:
                        critic_losses.append(
                            np.mean([x[0] for x in valid_losses])
                        )
                        actor_losses.append(
                            np.mean([x[1] for x in valid_losses])
                        )

            if (episode + 1) % args.print_freq == 0:
                avg_reward = np.mean(episode_rewards[-args.print_freq :])
                avg_length = np.mean(episode_lengths[-args.print_freq :])

                print(f"Episode {episode + 1}/{args.episodes}")
                print(
                    f"  Avg Reward (last {args.print_freq}): {avg_reward:.2f}"
                )
                print(
                    f"  Avg Length (last {args.print_freq}): {avg_length:.1f}"
                )

                if loss_structure == "single":
                    avg_loss = (
                        np.mean(losses[-args.print_freq :]) if losses else 0
                    )
                    print(
                        f"  Avg Loss (last {args.print_freq}): {avg_loss:.4f}"
                    )
                elif loss_structure == "dual":
                    avg_critic = (
                        np.mean(critic_losses[-args.print_freq :])
                        if critic_losses
                        else 0
                    )
                    avg_actor = (
                        np.mean(actor_losses[-args.print_freq :])
                        if actor_losses
                        else 0
                    )
                    print(
                        f"  Avg Critic Loss (last {args.print_freq}): {avg_critic:.4f}"
                    )
                    print(
                        f"  Avg Actor Loss (last {args.print_freq}): {avg_actor:.4f}"
                    )

                metrics = agent.get_training_metrics()
                for key, value in metrics.items():
                    if key != "steps":
                        print(f"  {key.capitalize()}: {value:.3f}")
                print()

            if (episode + 1) % args.save_freq == 0:
                agent.save(args.save_path)
                print(f"Checkpoint saved to {args.save_path}")

        agent.save(args.save_path)
        print("Training complete!")
        print(f"Final model saved to {args.save_path}")

    finally:
        if viewer is not None:
            viewer.close()
        env.close()

    print("Generating training plots...")
    if loss_structure == "single":
        plot_training_results(
            episode_rewards, losses, args.save_path, args.agent
        )
    elif loss_structure == "dual":
        plot_training_results(
            episode_rewards,
            (critic_losses, actor_losses),
            args.save_path,
            args.agent,
        )

    # Print total training time
    end_time = time_module.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(
        f"\nTotal training time: {hours}h {minutes}m {seconds}s ({total_time:.2f} seconds)"
    )


def evaluate(args):
    """Evaluate trained agent."""

    continuous = requires_continuous_actions(args.agent)
    env = RobotReachingEnv(
        num_links=num_links,
        continuous=continuous,
        action_quantization=action_quantization,
    )

    agent = get_agent(args.agent, env, args, mode="eval")

    agent.load(args.load_path)
    print(f"Loaded {args.agent.upper()} model from {args.load_path}")
    print()

    episode_rewards = []
    success_count = 0

    for episode in range(args.eval_episodes):
        state, info = env.reset(options={"random_target": args.random_targets})
        episode_reward = 0

        print(f"Episode {episode + 1}/{args.eval_episodes}")

        if args.gui:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -20
                viewer.cam.distance = 3.0
                viewer.cam.lookat[:] = [0, 0, 1.2]

                step = 0
                while viewer.is_running() and step < args.max_steps:
                    action = agent.select_action(state, training=False)

                    next_state, reward, terminated, truncated, info = env.step(
                        action
                    )
                    done = terminated or truncated

                    episode_reward += reward
                    state = next_state

                    viewer.sync()
                    time.sleep(0.01)

                    step += 1

                    if done:
                        if terminated:
                            success_count += 1
                            print("  Target reached!")
                        break
                viewer.close()
        else:
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

    print("Evaluation Summary:")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f}")
    print(
        f"  Success Rate: {success_count}/{args.eval_episodes} ({100*success_count/args.eval_episodes:.1f}%)"
    )

    env.close()

    if args.gui:
        sys.exit(0)


def play(args):
    """Watch trained agent perform indefinitely."""

    continuous = requires_continuous_actions(args.agent)
    env = RobotReachingEnv(
        num_links=num_links,
        continuous=continuous,
        action_quantization=action_quantization,
    )

    agent = get_agent(args.agent, env, args, mode="play")

    agent.load(args.load_path)
    print(f"Loaded {args.agent.upper()} model from {args.load_path}")
    print("Watching trained agent perform...")
    print("Close viewer window to exit")
    print()

    state, info = env.reset(options={"random_target": args.random_targets})

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 1.2]

        while viewer.is_running():
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)

            viewer.sync()
            time.sleep(0.01)

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agents for robot arm control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Q_learning.py --agent dqn train --episodes 1000 --random-targets
  python Q_learning.py --agent ddpg train --episodes 1000 --random-targets
  python Q_learning.py --agent dqn eval --load-path data/checkpoints/dqn.pth --gui
  python Q_learning.py --agent ddpg play --load-path data/checkpoints/ddpg.pth

To add a new agent:
  1. Create {agent_name}_agent.py inheriting from BaseAgent
  2. Add agent to continuous_agents list in requires_continuous_actions() if needed
  3. Add agent configuration in get_agent() if it needs custom parameters
        """,
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Agent type to use (e.g., 'dqn', 'ddpg')",
    )
    subparsers = parser.add_subparsers(
        dest="mode", help="Mode: train, eval, or play"
    )

    train_parser = subparsers.add_parser("train", help="Train the agent")
    train_parser.add_argument(
        "--episodes",
        type=int,
        default=num_episodes,
        help="Number of training episodes",
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=max_steps,
        help="Max steps per episode",
    )
    train_parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor"
    )
    train_parser.add_argument(
        "--buffer-size", type=int, default=50000, help="Replay buffer size"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    train_parser.add_argument(
        "--save-path",
        type=str,
        default="data/checkpoints/agent.pth",
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
    train_parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )

    train_parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (DQN)"
    )
    train_parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Starting epsilon (DQN)",
    )
    train_parser.add_argument(
        "--epsilon-end", type=float, default=0.01, help="Final epsilon (DQN)"
    )
    train_parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=decay_rate,
        help="Epsilon decay rate (DQN)",
    )
    train_parser.add_argument(
        "--target-update-freq",
        type=int,
        default=100,
        help="Target network update frequency (DQN)",
    )

    train_parser.add_argument(
        "--actor-lr",
        type=float,
        default=1e-4,
        help="Actor learning rate (DDPG/TD3/SAC)",
    )
    train_parser.add_argument(
        "--critic-lr",
        type=float,
        default=1e-3,
        help="Critic learning rate (DDPG/TD3/SAC)",
    )
    train_parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Soft update coefficient (DDPG/TD3/SAC)",
    )
    train_parser.add_argument(
        "--noise-std",
        type=float,
        default=0.1,
        help="Exploration noise std (DDPG/TD3)",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate the agent")
    eval_parser.add_argument(
        "--load-path",
        type=str,
        default="data/checkpoints/agent.pth",
        help="Path to load model",
    )
    eval_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=eval_episodes,
        help="Number of evaluation episodes",
    )
    eval_parser.add_argument(
        "--max-steps",
        type=int,
        default=max_steps,
        help="Max steps per episode",
    )
    eval_parser.add_argument(
        "--gui", action="store_true", help="Show GUI during evaluation"
    )
    eval_parser.add_argument(
        "--random-targets",
        action="store_true",
        help="Use random target positions",
    )

    play_parser = subparsers.add_parser("play", help="Watch the trained agent")
    play_parser.add_argument(
        "--load-path",
        type=str,
        default="data/checkpoints/agent.pth",
        help="Path to load model",
    )
    play_parser.add_argument(
        "--random-targets",
        action="store_true",
        help="Use random target positions",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    elif args.mode == "play":
        play(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
