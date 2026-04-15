"""
@brief Deep Deterministic Policy Gradient (DDPG) agent for continuous robot arm control.

Implements the DDPG algorithm with an actor-critic architecture, experience replay,
and soft target network updates. Gaussian exploration noise is added to actions
during training and removed at evaluation time.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from .agent import BaseAgent
from .replay_buffer import ReplayBuffer


class Actor(nn.Module):
    """
    @brief Policy network that maps observations to deterministic continuous actions.

    A fully connected network with configurable hidden layers. The output is
    passed through Tanh and scaled by max_action so actions always lie in
    [-max_action, max_action].
    """

    def __init__(
        self, state_dim, action_dim, hidden_dims=[256, 256], max_action=1.0
    ):
        """
        @brief Build the actor network.

        @param state_dim   Integer dimension of the observation vector.
        @param action_dim  Integer dimension of the continuous action vector.
        @param hidden_dims List of hidden layer widths (default: two layers of 256).
        @param max_action  Scalar used to scale the Tanh output to the action range.
        """
        super(Actor, self).__init__()

        # Dynamically build hidden layers so the architecture is configurable at construction time
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Final layer maps to action space; Tanh constrains output to [-1, 1] before scaling
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)
        self.max_action = max_action

    def forward(self, state):
        """
        @brief Compute a deterministic action for the given state.

        @param state Float tensor of shape (batch, state_dim).
        @return Float tensor of shape (batch, action_dim) scaled to [-max_action, max_action].
        """
        return self.network(state) * self.max_action


class Critic(nn.Module):
    """
    @brief Q-value network that estimates the expected return for (state, action) pairs.

    Concatenates the state and action vectors before passing them through a
    two-hidden-layer MLP that outputs a single scalar Q-value.
    """

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        """
        @brief Build the critic network.

        @param state_dim   Integer dimension of the observation vector.
        @param action_dim  Integer dimension of the continuous action vector.
        @param hidden_dims List of hidden layer widths (default: two layers of 256).
        """
        super(Critic, self).__init__()

        # Input is the concatenation of state and action, so first layer is wider
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, state, action):
        """
        @brief Estimate the Q-value for a (state, action) pair.

        @param state  Float tensor of shape (batch, state_dim).
        @param action Float tensor of shape (batch, action_dim).
        @return Float tensor of shape (batch, 1) containing Q-value estimates.
        """
        # Concatenate along the feature dimension before feeding into the network
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class DDPGAgent(BaseAgent):
    """
    @brief DDPG agent implementing the BaseAgent interface for continuous control.

    Maintains online and target copies of both the actor and critic. The target
    networks are updated via Polyak averaging (soft updates) after every training
    step to stabilize learning.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=64,
        noise_std=0.1,
        device="cpu",
    ):
        """
        @brief Initialize the DDPG agent, networks, optimizers, and replay buffer.

        @param state_dim   Integer dimension of the observation vector.
        @param action_dim  Integer dimension of the continuous action vector.
        @param max_action  Maximum absolute value for any action component.
        @param actor_lr    Learning rate for the actor (policy) Adam optimizer.
        @param critic_lr   Learning rate for the critic (Q-function) Adam optimizer.
        @param gamma       Discount factor for future rewards.
        @param tau         Soft update coefficient; fraction of online weights copied
                           to the target network each step (typical range: 0.001-0.01).
        @param buffer_size Maximum number of transitions stored in the replay buffer.
        @param batch_size  Number of transitions sampled per training step.
        @param noise_std   Standard deviation of Gaussian exploration noise added
                           to actions during training.
        @param device      PyTorch device string (e.g., 'cpu' or 'cuda').
        """
        super().__init__(state_dim, action_dim, device)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau          # Polyak averaging coefficient for target network updates
        self.batch_size = batch_size
        self.noise_std = noise_std

        # Online actor and a frozen copy used only for computing target actions
        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action=max_action).to(device)
        # Initialize target weights identically to the online network
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Online critic and its target counterpart
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, training=True):
        """
        @brief Query the actor network for an action, optionally adding exploration noise.

        @param state    Float32 numpy array of shape (state_dim,).
        @param training If True, adds zero-mean Gaussian noise with std=noise_std
                        and clips the result to [-max_action, max_action].
        @return Float32 numpy array of shape (action_dim,).
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]

        if training:
            # Gaussian noise encourages exploration of the continuous action space
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        @brief Push a single (s, a, r, s', done) transition into the replay buffer.

        @param state      Float32 numpy array of shape (state_dim,).
        @param action     Float32 numpy array of shape (action_dim,).
        @param reward     Scalar float reward from the environment.
        @param next_state Float32 numpy array of shape (state_dim,).
        @param done       Boolean; True if the episode ended after this transition.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        @brief Sample a minibatch and update both the critic and actor networks.

        Critic update: minimizes MSE between current Q-values and Bellman targets
        computed with the frozen target networks.

        Actor update: maximizes the critic's Q-value estimate for the actor's
        current output (policy gradient via chain rule through the critic).

        Both target networks are then soft-updated toward their online counterparts.

        @return Tuple (critic_loss, actor_loss) as Python floats, or (None, None)
                if the replay buffer does not yet contain a full batch.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Move all sampled arrays to the target device as float tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- Critic update ---
        with torch.no_grad():
            # Bellman target: r + gamma * Q_target(s', pi_target(s')) for non-terminal steps
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        # Gradient ascent on Q(s, pi(s)); negate because optimizers minimize
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Polyak-average both target networks toward their online counterparts
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.steps += 1

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, source, target):
        """
        @brief Perform a Polyak soft update from source network to target network.

        Each target parameter is blended toward the corresponding source parameter:
            target = tau * source + (1 - tau) * target

        @param source nn.Module whose weights are the "online" reference.
        @param target nn.Module whose weights are updated in place.
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filepath):
        """
        @brief Serialize all network weights, optimizer states, and step count to disk.

        Saves both online and target network weights so training can be fully
        resumed from this checkpoint without losing target network state.

        @param filepath String path where the .pth checkpoint file will be written.
        """
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "steps": self.steps,
            },
            filepath,
        )

    def load(self, filepath):
        """
        @brief Restore network weights, optimizer states, and step count from a checkpoint.

        @param filepath String path to the .pth checkpoint file written by save().
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.steps = checkpoint["steps"]

    def get_training_metrics(self):
        """
        @brief Return diagnostic metrics logged by the training loop each print interval.

        @return Dict with keys "steps" (int) and "noise_std" (float).
        """
        return {
            "steps": self.steps,
            "noise_std": self.noise_std,
        }
