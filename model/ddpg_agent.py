import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from .agent import BaseAgent
from .replay_buffer import ReplayBuffer


class Actor(nn.Module):
    """Actor network: maps states to continuous actions."""

    def __init__(
        self, state_dim, action_dim, hidden_dims=[256, 256], max_action=1.0
    ):
        super(Actor, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)
        self.max_action = max_action

    def forward(self, state):
        return self.network(state) * self.max_action


class Critic(nn.Module):
    """Critic network: maps (state, action) pairs to Q-values."""

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class DDPGAgent(BaseAgent):
    """DDPG agent for continuous control."""

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
        super().__init__(state_dim, action_dim, device)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std

        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(
            device
        )
        self.actor_target = Actor(
            state_dim, action_dim, max_action=max_action
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, training=True):
        """
        Select action from actor network.

        Args:
            state: Current state observation
            training: If True, add exploration noise

        Returns:
            Action array
        """
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            )
            action = self.actor(state_tensor).cpu().numpy()[0]

        if training:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.steps += 1

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, source, target):
        """Soft update target network parameters."""
        for target_param, param in zip(
            target.parameters(), source.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filepath):
        """Save model checkpoint."""
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
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(
            checkpoint["actor_target_state_dict"]
        )
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(
            checkpoint["critic_target_state_dict"]
        )
        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"]
        )
        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer_state_dict"]
        )
        self.steps = checkpoint["steps"]

    def get_training_metrics(self):
        """Get current training metrics."""
        return {
            "steps": self.steps,
            "noise_std": self.noise_std,
        }
