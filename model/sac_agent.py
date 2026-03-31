import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from .agent import BaseAgent
from .replay_buffer import ReplayBuffer


class Actor(nn.Module):
    """Stochastic actor network that outputs Gaussian policy."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=[256, 256],
        log_std_min=-20,
        log_std_max=2,
    ):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.mean = nn.Linear(hidden_dims[1], action_dim)
        self.log_std = nn.Linear(hidden_dims[1], action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        """Sample action from the policy."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Compute log prob
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    """Twin Q-networks."""

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(Critic, self).__init__()

        # Q1
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.q1_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.q1_out = nn.Linear(hidden_dims[1], 1)

        # Q2
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.q2_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.q2_out = nn.Linear(hidden_dims[1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        # Q1
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        # Q2
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        return q1, q2


class SACAgent(BaseAgent):
    """SAC agent for continuous control."""

    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=256,
        device="cpu",
    ):
        super().__init__(state_dim, action_dim, device)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, training=True):
        """Select action from actor network."""
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            )
            if training:
                action, _ = self.actor.sample(state_tensor)
            else:
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
            return action.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # Sample batch
        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute alpha (entropy coefficient)
        alpha = self.log_alpha.exp()

        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (entropy coefficient)
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target networks
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
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha,
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
                "steps": self.steps,
            },
            filepath,
        )

    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
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
        self.log_alpha = checkpoint["log_alpha"]
        self.alpha_optimizer.load_state_dict(
            checkpoint["alpha_optimizer_state_dict"]
        )
        self.steps = checkpoint["steps"]

    def get_training_metrics(self):
        """Get current training metrics."""
        return {
            "steps": self.steps,
            "alpha": self.log_alpha.exp().item(),
        }
