"""
@brief Soft Actor-Critic (SAC) agent for continuous robot arm control.

Implements SAC with twin Q-networks, a stochastic Gaussian policy with the
reparameterization trick, and automatic entropy temperature tuning. The entropy
coefficient alpha is learned alongside the policy so the agent self-tunes its
exploration-exploitation trade-off throughout training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from .agent import BaseAgent
from .replay_buffer import ReplayBuffer


class Actor(nn.Module):
    """
    @brief Stochastic Gaussian policy network for SAC.

    Outputs the mean and log standard deviation of an action distribution.
    Actions are sampled via the reparameterization trick and squashed through
    Tanh to enforce the action bounds, with a corresponding log-probability
    correction applied during training.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=[256, 256],
        log_std_min=-20,
        log_std_max=2,
    ):
        """
        @brief Build the stochastic actor network.

        @param state_dim   Integer dimension of the observation vector.
        @param action_dim  Integer dimension of the continuous action vector.
        @param hidden_dims List of hidden layer widths (default: two layers of 256).
        @param log_std_min Lower clamp for the predicted log standard deviation,
                           preventing vanishingly small variance.
        @param log_std_max Upper clamp for the predicted log standard deviation,
                           preventing excessively large variance.
        """
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Two shared hidden layers feed into separate mean and log_std heads
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.mean = nn.Linear(
            hidden_dims[1], action_dim
        )  # mean of the Gaussian
        self.log_std = nn.Linear(
            hidden_dims[1], action_dim
        )  # log std of the Gaussian

    def forward(self, state):
        """
        @brief Compute the mean and clamped log standard deviation for the given state.

        @param state Float tensor of shape (batch, state_dim).
        @return Tuple (mean, log_std), each of shape (batch, action_dim).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        # Clamp to prevent numerical instability from extreme standard deviation values
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        """
        @brief Draw a Tanh-squashed action sample and its log probability.

        Uses the reparameterization trick (rsample) so gradients flow through
        the sample back to the network parameters. The log-probability correction
        accounts for the change of variables introduced by the Tanh squashing.

        @param state Float tensor of shape (batch, state_dim).
        @return Tuple (action, log_prob):
                - action:   Float tensor of shape (batch, action_dim) in [-1, 1].
                - log_prob: Float tensor of shape (batch, 1) with the log probability
                            of the sampled action under the squashed Gaussian.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick: sample = mean + std * noise, preserving gradients
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Log probability of the pre-squash sample under the Gaussian
        log_prob = normal.log_prob(x_t)
        # Correction for the Tanh squashing: subtract log|d(tanh)/dx| per dimension
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    """
    @brief Twin Q-network that estimates Q-values for (state, action) pairs.

    Two independent Q-networks (Q1 and Q2) are evaluated in a single forward
    pass. SAC takes the minimum of their outputs for target computation to
    reduce overestimation bias.
    """

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        """
        @brief Build both Q-networks with identical architecture.

        @param state_dim   Integer dimension of the observation vector.
        @param action_dim  Integer dimension of the continuous action vector.
        @param hidden_dims List of hidden layer widths (default: two layers of 256).
        """
        super(Critic, self).__init__()

        # Q1 network: independent weights from Q2 to reduce correlated overestimation
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.q1_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.q1_out = nn.Linear(hidden_dims[1], 1)

        # Q2 network: same architecture as Q1 but separate parameters
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.q2_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.q2_out = nn.Linear(hidden_dims[1], 1)

    def forward(self, state, action):
        """
        @brief Compute Q1 and Q2 values for a batch of (state, action) pairs.

        @param state  Float tensor of shape (batch, state_dim).
        @param action Float tensor of shape (batch, action_dim).
        @return Tuple (q1, q2), each a float tensor of shape (batch, 1).
        """
        # Concatenate state and action along the feature dimension for both networks
        x = torch.cat([state, action], dim=1)

        # Q1 forward pass
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        # Q2 forward pass
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        return q1, q2


class SACAgent(BaseAgent):
    """
    @brief SAC agent implementing the BaseAgent interface for continuous control.

    Learns a maximum-entropy policy by jointly optimizing actor, twin critics,
    and an adaptive entropy temperature (alpha). The critic target network is
    updated via Polyak soft averaging after every training step.
    """

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
        """
        @brief Initialize the SAC agent, networks, optimizers, and replay buffer.

        @param state_dim  Integer dimension of the observation vector.
        @param action_dim Integer dimension of the continuous action vector.
        @param actor_lr   Learning rate for the actor (policy) Adam optimizer.
        @param critic_lr  Learning rate for the twin critic Adam optimizer.
        @param alpha_lr   Learning rate for the entropy temperature Adam optimizer.
        @param gamma      Discount factor for future rewards.
        @param tau        Soft update coefficient for the critic target network
                          (typical range: 0.001-0.01).
        @param buffer_size Maximum number of transitions in the replay buffer.
        @param batch_size  Number of transitions sampled per training step.
        @param device      PyTorch device string (e.g., 'cpu' or 'cuda').
        """
        super().__init__(state_dim, action_dim, device)

        self.gamma = gamma
        self.tau = (
            tau  # Polyak averaging coefficient for critic target updates
        )
        self.batch_size = batch_size

        # Actor and twin critic (online + target copy for the critic only)
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        # Initialize target weights identically to the online critic
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        # Automatic entropy tuning: target entropy is set to -|A| (heuristic from the SAC paper)
        self.target_entropy = -action_dim
        # log_alpha is learned; exponentiated to get the positive entropy coefficient alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, training=True):
        """
        @brief Query the actor for an action, sampling stochastically or acting greedily.

        During training a full stochastic sample is drawn to encourage exploration.
        During evaluation the mean action (squashed through Tanh) is used for
        deterministic, reproducible behaviour.

        @param state    Float32 numpy array of shape (state_dim,).
        @param training If True, samples from the stochastic policy.
                        If False, uses the deterministic mean action.
        @return Float32 numpy array of shape (action_dim,) in [-1, 1].
        """
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            )
            if training:
                # Stochastic sample includes exploration noise via the Gaussian policy
                action, _ = self.actor.sample(state_tensor)
            else:
                # Deterministic mean action for evaluation; Tanh keeps it in bounds
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
            return action.cpu().numpy()[0]

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
        @brief Sample a minibatch and update the critic, actor, and entropy temperature.

        Update order:
          1. Critic: minimize MSE between current Q-values and soft Bellman targets
             (uses min of twin target Q-values minus entropy bonus).
          2. Actor: minimize (alpha * log_prob - min_Q) to maximize entropy-regularized return.
          3. Alpha: adjust entropy temperature so the policy's entropy tracks target_entropy.
          4. Critic target: Polyak soft update toward the online critic.

        @return Tuple (critic_loss, actor_loss) as Python floats, or (None, None)
                if the replay buffer does not yet contain a full batch.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # Sample a random minibatch from the replay buffer
        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Exponentiate log_alpha to get the current positive entropy coefficient
        alpha = self.log_alpha.exp()

        # --- Critic update ---
        with torch.no_grad():
            # Soft Bellman target: use min of twin Q-targets to reduce overestimation,
            # then subtract the entropy bonus alpha * log_prob
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next

        q1, q2 = self.critic(states, actions)
        # Sum losses for both Q-networks so both are updated in one backward pass
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        # Use the more conservative (min) Q-estimate to prevent policy exploitation
        q_new = torch.min(q1_new, q2_new)

        # Minimize (alpha * log_prob - Q): encourages high entropy and high Q simultaneously
        actor_loss = (alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha (entropy temperature) update ---
        # Adjust alpha so the policy's entropy stays near target_entropy
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Polyak soft update: gradually blend critic target toward online critic
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
        for target_param, param in zip(
            target.parameters(), source.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filepath):
        """
        @brief Serialize all network weights, optimizer states, log_alpha, and step count.

        Saves the critic target weights and log_alpha value so training can be
        fully resumed without loss of target network or temperature state.

        @param filepath String path where the .pth checkpoint file will be written.
        """
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
        """
        @brief Restore all network weights, optimizer states, log_alpha, and step count.

        @param filepath String path to the .pth checkpoint file written by save().
        """
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
        self.log_alpha = checkpoint[
            "log_alpha"
        ]  # restore learned entropy temperature
        self.alpha_optimizer.load_state_dict(
            checkpoint["alpha_optimizer_state_dict"]
        )
        self.steps = checkpoint["steps"]

    def get_training_metrics(self):
        """
        @brief Return diagnostic metrics logged by the training loop each print interval.

        @return Dict with keys "steps" (int) and "alpha" (float, the current
                entropy temperature coefficient).
        """
        return {
            "steps": self.steps,
            "alpha": self.log_alpha.exp().item(),
        }
