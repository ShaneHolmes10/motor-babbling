from abc import ABC, abstractmethod
import torch
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for RL agents."""

    def __init__(self, state_dim, action_dim, device="cpu"):
        """
        Initialize base agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension/count of action space
            device: Device to run computations on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.steps = 0

    @abstractmethod
    def select_action(self, state, training=True):
        """
        Select action given current state.

        Args:
            state: Current state observation
            training: Whether agent is in training mode (affects exploration)

        Returns:
            Action to take (discrete int or continuous array)
        """
        pass

    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        pass

    @abstractmethod
    def train_step(self):
        """
        Perform one training step.

        Returns:
            Loss value(s) or None if not enough samples
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """
        Save agent checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        pass

    @abstractmethod
    def load(self, filepath):
        """
        Load agent checkpoint.

        Args:
            filepath: Path to load checkpoint from
        """
        pass

    def get_training_metrics(self):
        """
        Get current training metrics (optional).

        Returns:
            Dictionary of metrics (e.g., epsilon, noise level)
        """
        return {"steps": self.steps}
