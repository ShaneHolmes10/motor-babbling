"""
@brief Entry point for training, evaluating, and watching RL agents on the robot reaching task.

Supports any agent that follows the BaseAgent interface by dynamically importing
it from the model/ package. Continuous-action agents (DDPG, TD3, SAC, PPO) and
discrete-action agents (DQN) are both supported; the correct environment mode is
selected automatically based on the agent type.

Usage:
    python Q_learning.py --agent dqn  train [options]
    python Q_learning.py --agent ddpg eval  --load-path <path> [--gui]
    python Q_learning.py --agent ddpg play  --load-path <path>
"""

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

# --- Global defaults (overridable via CLI flags) ---
num_links = 2  # number of arm links used by all modes
action_quantization = (
    10  # discrete torque levels per joint for DQN-style agents
)
num_episodes = 1000  # default training episode count
decay_rate = 0.99999  # default per-step epsilon decay rate for DQN

max_steps = 500  # maximum steps allowed per episode before truncation
eval_episodes = 100  # default number of episodes for evaluation mode


def get_agent_class(agent_type):
    """
    @brief Dynamically import and return an agent class from the model package.

    Constructs the module path as model.<agent_type>_agent and the class name
    as <AGENT_TYPE>Agent (e.g., agent_type="dqn" -> model.dqn_agent.DQNAgent).

    @param agent_type Lowercase string identifier for the agent (e.g., 'dqn', 'ddpg').
    @return The agent class object, ready to be instantiated.
    @throws ValueError if the module or class cannot be found.
    """
    try:
        agent_module = importlib.import_module(f"model.{agent_type}_agent")
        class_name = f"{agent_type.upper()}Agent"
        return getattr(agent_module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not load agent '{agent_type}': {e}")


def requires_continuous_actions(agent_type):
    """
    @brief Determine whether an agent requires a continuous action space.

    Used to configure RobotReachingEnv with the correct action mode before
    constructing the agent.

    @param agent_type Lowercase string identifier for the agent.
    @return True if the agent expects a continuous Box action space,
            False if it expects a discrete Discrete action space.
    """
    continuous_agents = ["ddpg", "td3", "sac", "ppo"]
    return agent_type.lower() in continuous_agents


def get_agent(agent_type, env, args, mode="train"):
    """
    @brief Instantiate an agent with parameters appropriate for the given mode.

    In eval/play mode only state_dim, action_dim, and device are passed so that
    the agent can be constructed cheaply for inference. In train mode all
    hyperparameters from args are forwarded.

    @param agent_type Lowercase string identifier for the agent (e.g., 'dqn', 'ddpg').
    @param env        Constructed RobotReachingEnv instance used to infer state/action dims.
    @param args       Parsed argparse namespace containing hyperparameter flags.
    @param mode       One of 'train', 'eval', or 'play'. Eval/play skip hyperparams.
    @return Initialized agent instance ready for select_action() calls.
    @throws ValueError if the agent needs custom parameters not handled here.
    """
    AgentClass = get_agent_class(agent_type)

    state_dim = env.observation_space.shape[0]

    # Continuous agents expose action_space.shape; discrete agents expose action_space.n
    if requires_continuous_actions(agent_type):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    if mode in ["eval", "play"]:
        # Minimal construction: no replay buffer or optimizer needed for inference
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
                device="cpu",
            )

    # Shared hyperparameters common to all agent types
    common_params = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "device": "cpu",
    }

    if agent_type == "dqn":
        # DQN-specific: epsilon-greedy exploration and periodic target-network sync
        return AgentClass(
            **common_params,
            learning_rate=args.lr,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            target_update_freq=args.target_update_freq,
        )
    elif agent_type == "ddpg":
        # DDPG-specific: separate actor/critic learning rates, soft update tau, and OU noise
        return AgentClass(
            **common_params,
            max_action=1.0,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            tau=args.tau,
            noise_std=args.noise_std,
        )
    else:
        # Attempt a generic construction; agent must accept the common_params subset
        try:
            return AgentClass(**common_params)
        except TypeError as e:
            raise ValueError(
                f"Agent '{agent_type}' requires custom parameters. "
                f"Add configuration in get_agent() function. Error: {e}"
            )


def get_loss_structure(agent_type):
    """
    @brief Return the loss structure used by an agent for plotting purposes.

    Actor-critic agents produce separate critic and actor losses; value-based
    agents produce a single TD loss.

    @param agent_type Lowercase string identifier for the agent.
    @return 'dual' for actor-critic agents (DDPG, TD3, SAC, PPO),
            'single' for value-based agents (DQN).
    """
    dual_loss_agents = ["ddpg", "td3", "sac", "ppo"]
    return "dual" if agent_type.lower() in dual_loss_agents else "single"


def plot_training_results(episode_rewards, losses, model_path, agent_type):
    """
    @brief Generate, save, and export training curve plots and raw data.

    Produces a PNG figure with reward and loss subplots, overlaying a moving
    average for trend clarity. Also saves the underlying arrays to an .npz file
    for offline analysis.

    @param episode_rewards List of total rewards, one entry per training episode.
    @param losses          For single-loss agents: list of scalar loss values.
                           For dual-loss agents: tuple of (critic_losses, actor_losses).
    @param model_path      Path used to derive the output file prefix (basename without extension).
    @param agent_type      Lowercase string identifier for the agent; determines plot layout.
    """
    plot_dir = "data/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Derive a consistent filename prefix from the saved model path
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
        # Adaptive window: at most 50 episodes, but never more than 10% of total data
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
        # Unpack the (critic, actor) loss tuple from actor-critic agents
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

    # Save standard and high-resolution versions of the figure
    plot_path = os.path.join(plot_dir, f"{plot_prefix}_training.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Training plots saved to {plot_path}")

    plot_path_hires = os.path.join(
        plot_dir, f"{plot_prefix}_training_hires.png"
    )
    plt.savefig(plot_path_hires, dpi=300)

    plt.close()

    # Persist raw arrays alongside the plots for offline analysis
    data_path = os.path.join(plot_dir, f"{plot_prefix}_training_data.npz")
    np.savez(data_path, **data_dict)
    print(f"Training data saved to {data_path}")


def train(args):
    """
    @brief Run the full training loop for the specified agent.

    Creates the environment and agent, optionally resumes from a checkpoint,
    then iterates over episodes collecting transitions and updating the policy.
    Periodic console logs and model checkpoints are written according to args.
    Training plots and raw data are saved on completion.

    @param args Parsed argparse namespace. Expected fields: agent, episodes,
                max_steps, gamma, buffer_size, batch_size, save_path, save_freq,
                print_freq, random_targets, gui, resume, and agent-specific
                hyperparameter flags (lr, epsilon_*, actor_lr, critic_lr, tau, noise_std).
    """

    start_time = time_module.time()

    # Select continuous vs discrete environment mode based on agent type
    continuous = requires_continuous_actions(args.agent)
    env = RobotReachingEnv(
        num_links=num_links,
        continuous=continuous,
        action_quantization=action_quantization,
    )

    agent = get_agent(args.agent, env, args)

    # Optionally restore weights and replay buffer from a previous run
    if args.resume and os.path.exists(args.save_path):
        print(f"Resuming from checkpoint: {args.save_path}")
        agent.load(args.save_path)
    elif args.resume:
        print(
            f"Warning: --resume specified but no checkpoint found at {args.save_path}"
        )

    episode_rewards = []
    episode_lengths = []

    # Initialize loss tracking containers matched to the agent's output structure
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

    # Launch a passive viewer for visual monitoring during training (optional)
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
            episode_loss = (
                []
            )  # accumulate per-step losses to average at episode end

            for step in range(args.max_steps):
                action = agent.select_action(state, training=True)

                next_state, reward, terminated, truncated, info = env.step(
                    action
                )
                done = terminated or truncated

                # Push the transition into the replay buffer
                agent.store_transition(state, action, reward, next_state, done)

                # Perform one gradient update step (returns None if buffer not yet full)
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

            # Aggregate per-step losses into a single per-episode scalar
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

            # Periodic progress report to stdout
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

                # Print agent-specific metrics (e.g., epsilon for DQN)
                metrics = agent.get_training_metrics()
                for key, value in metrics.items():
                    if key != "steps":
                        print(f"  {key.capitalize()}: {value:.3f}")
                print()

            # Periodic checkpoint save
            if (episode + 1) % args.save_freq == 0:
                agent.save(args.save_path)
                print(f"Checkpoint saved to {args.save_path}")

        agent.save(args.save_path)
        print("Training complete!")
        print(f"Final model saved to {args.save_path}")

    finally:
        # Always release viewer and environment even if training is interrupted
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
    """
    @brief Run the trained agent through a fixed number of evaluation episodes.

    Loads a saved checkpoint, rolls out the policy without exploration noise,
    and reports per-episode rewards and an aggregate success rate. Optionally
    renders each episode in the MuJoCo viewer.

    @param args Parsed argparse namespace. Expected fields: agent, load_path,
                eval_episodes, max_steps, gui, random_targets.
    """

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
    success_count = (
        0  # episodes where the arm held the target for the required steps
    )

    for episode in range(args.eval_episodes):
        state, info = env.reset(options={"random_target": args.random_targets})
        episode_reward = 0

        print(f"Episode {episode + 1}/{args.eval_episodes}")

        if args.gui:
            # GUI path: open a viewer per episode and step until done or viewer closed
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
            # Headless path: run as fast as possible without rendering overhead
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

    # Exit explicitly when using GUI to avoid viewer teardown issues on some platforms
    if args.gui:
        sys.exit(0)


def play(args):
    """
    @brief Watch the trained agent run indefinitely with an interactive target.

    Loads a saved checkpoint and renders the policy in the MuJoCo viewer.
    The user can reposition the target in real time using arrow keys or
    randomize it by pressing 'r'.

    @param args Parsed argparse namespace. Expected fields: agent, load_path,
                random_targets.
    """

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
    print("Controls:")
    print("  Use arrow keys")
    print("Close viewer window to exit")
    print()

    state, info = env.reset(options={"random_target": args.random_targets})

    def key_callback(keycode):
        """
        @brief Handle keyboard input to reposition the target during play mode.

        Arrow keys translate the target by a fixed step; 'r' randomizes it to a
        new reachable position. The mocap body is updated immediately so the
        visual marker moves on the next viewer sync.

        @param keycode Integer GLFW keycode for the pressed key.
        """
        key = chr(keycode).lower()
        move_amount = 0.05  # world-space distance moved per keypress (meters)

        if keycode == 265:  # up arrow: move target upward (+z)
            env.target[1] += move_amount
        elif keycode == 264:  # down arrow: move target downward (-z)
            env.target[1] -= move_amount
        elif keycode == 263:  # left arrow: move target left (-x)
            env.target[0] -= move_amount
        elif keycode == 262:  # right arrow: move target right (+x)
            env.target[0] += move_amount
        elif key == "r":
            # Randomize target to a reachable polar position within the arm's workspace
            angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            max_reach = env.link_length * env.num_links
            radius = np.random.uniform(0.1, max_reach)
            env.target = np.array(
                [radius * np.sin(angle), 1.5 - radius * np.cos(angle)]
            )

        # Push the updated target position into the MuJoCo mocap body
        env.data.mocap_pos[env.target_mocap_id] = [
            env.target[0],
            0,
            env.target[1],
        ]

    with mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=key_callback
    ) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 1.2]

        while viewer.is_running():
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)

            viewer.sync()
            time.sleep(0.01)  # ~100 Hz render rate

    env.close()


def main():
    """
    @brief Parse command-line arguments and dispatch to train, evaluate, or play.

    Defines a two-level argument structure: global flags (--agent) plus
    subcommand-specific flags (train/eval/play). Prints help if no subcommand
    is provided.
    """
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

    # --- Train subcommand ---
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

    # DQN-specific hyperparameters
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

    # DDPG / TD3 / SAC hyperparameters
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

    # --- Eval subcommand ---
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

    # --- Play subcommand ---
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

    # Dispatch to the appropriate mode handler
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
