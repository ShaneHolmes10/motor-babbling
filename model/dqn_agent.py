"""
@brief Deep Q-Network (DQN) agent for discrete robot arm control.

Implements DQN with experience replay and a periodically hard-copied target
network to stabilize training. Exploration is handled via an epsilon-greedy
policy whose epsilon decays multiplicatively after every training step.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from .agent import BaseAgent
from .replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """
    @brief Fully connected network that estimates Q-values for all discrete actions.

    Takes an observation vector as input and outputs one Q-value per action.
    Architecture is configurable via hidden_dims at construction time.
    """

    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        """
        @brief Build the Q-network with the specified layer widths.

        @param state_dim   Integer dimension of the observation vector.
        @param action_dim  Integer number of discrete actions (output width).
        @param hidden_dims List of hidden layer widths (default: two layers of 128).
        """
        super(QNetwork, self).__init__()

        # Dynamically stack hidden layers so the architecture is configurable
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Final linear layer outputs one Q-value per discrete action (no activation)
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        @brief Compute Q-values for all actions given a batch of states.

        @param state Float tensor of shape (batch, state_dim).
        @return Float tensor of shape (batch, action_dim) with per-action Q-values.
        """
        return self.network(state)


class DQNAgent(BaseAgent):
    """
    @brief DQN agent implementing the BaseAgent interface for discrete control.

    Maintains an online Q-network updated every step and a target network that
    is hard-copied from the online network every target_update_freq steps to
    provide stable Bellman targets.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=50000,
        batch_size=32,
        target_update_freq=100,
        device="cpu",
    ):
        """
        @brief Initialize the DQN agent, networks, optimizer, and replay buffer.

        @param state_dim         Integer dimension of the observation vector.
        @param action_dim        Integer number of discrete actions.
        @param learning_rate     Adam optimizer learning rate for the Q-network.
        @param gamma             Discount factor applied to future Q-value targets.
        @param epsilon_start     Initial epsilon for epsilon-greedy exploration (1.0 = fully random).
        @param epsilon_end       Minimum epsilon; exploration never drops below this floor.
        @param epsilon_decay     Multiplicative decay applied to epsilon after every train step.
        @param buffer_size       Maximum number of transitions stored in the replay buffer.
        @param batch_size        Number of transitions sampled per training step.
        @param target_update_freq Hard-copy the online network to the target every N steps.
        @param device            PyTorch device string (e.g., 'cpu' or 'cuda').
        """
        super().__init__(state_dim, action_dim, device)

        self.gamma = gamma
        self.epsilon = epsilon_start          # current exploration rate, decays over training
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Online network: updated every train step via gradient descent
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        # Target network: held fixed between hard updates to stabilize Bellman targets
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # target network is never trained directly

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, training=True):
        """
        @brief Select an action using the epsilon-greedy policy.

        During training, a random action is chosen with probability epsilon;
        otherwise the action with the highest Q-value is selected. At evaluation
        time (training=False) the policy is fully greedy.

        @param state    Float32 numpy array of shape (state_dim,).
        @param training If True, applies epsilon-greedy exploration.
        @return Integer action index in [0, action_dim).
        """
        if training and random.random() < self.epsilon:
            # Random action: uniform exploration across the discrete action space
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        @brief Push a single (s, a, r, s', done) transition into the replay buffer.

        @param state      Float32 numpy array of shape (state_dim,).
        @param action     Integer action index that was executed.
        @param reward     Scalar float reward from the environment.
        @param next_state Float32 numpy array of shape (state_dim,).
        @param done       Boolean; True if the episode ended after this transition.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        @brief Sample a minibatch and perform one Q-network gradient update.

        Computes the TD error between current Q-values and Bellman targets derived
        from the frozen target network, then minimizes MSE loss. The target network
        is hard-copied from the online network every target_update_freq steps, and
        epsilon is decayed after every call.

        @return Scalar TD loss as a Python float, or None if the replay buffer
                does not yet contain a full batch.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )

        # Convert sampled numpy arrays to typed tensors on the target device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)   # LongTensor required for gather()
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q(s, a): select only the Q-value for the action that was actually taken
        current_q_values = (
            self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():
            # Target: r + gamma * max_a' Q_target(s', a') for non-terminal transitions
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        # Hard-copy online weights to target network at the specified interval
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon toward its floor value after every gradient update
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, filepath):
        """
        @brief Serialize network weights, optimizer state, epsilon, and step count to disk.

        Saves both the online and target network weights so the target network
        state is preserved across training sessions.

        @param filepath String path where the .pth checkpoint file will be written.
        """
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            filepath,
        )

    def load(self, filepath):
        """
        @brief Restore network weights, optimizer state, epsilon, and step count from a checkpoint.

        @param filepath String path to the .pth checkpoint file written by save().
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]   # restore exploration rate from checkpoint
        self.steps = checkpoint["steps"]

    def get_training_metrics(self):
        """
        @brief Return diagnostic metrics logged by the training loop each print interval.

        @return Dict with keys "steps" (int) and "epsilon" (float).
        """
        return {
            "steps": self.steps,
            "epsilon": self.epsilon,
        }
