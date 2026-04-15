"""
@brief Abstract base class defining the interface all RL agents must implement.

Any agent used with Q_learning.py must inherit from BaseAgent and implement the
five abstract methods. Optional hooks like get_training_metrics() can be overridden
to expose agent-specific diagnostics to the training loop.
"""

from abc import ABC, abstractmethod
import torch
import numpy as np


class BaseAgent(ABC):
    """
    @brief Abstract base class for RL agents.

    Establishes the common interface (select_action, store_transition, train_step,
    save, load) that Q_learning.py relies on to remain agent-agnostic. Concrete
    subclasses implement each method for their specific algorithm.
    """

    def __init__(self, state_dim, action_dim, device="cpu"):
        """
        @brief Initialize shared agent state.

        Stores the dimensions needed to construct networks and tracks the total
        number of environment steps taken across all episodes.

        @param state_dim  Integer dimension of the observation vector.
        @param action_dim Integer count (discrete) or dimension (continuous) of the action space.
        @param device     PyTorch device string for network computations (e.g., 'cpu', 'cuda').
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.steps = 0  # cumulative environment steps; used for decay schedules and logging

    @abstractmethod
    def select_action(self, state, training=True):
        """
        @brief Select an action given the current observation.

        Subclasses should apply exploration (e.g., epsilon-greedy or noise) when
        training=True and act greedily when training=False.

        @param state    Numpy array or tensor representing the current observation.
        @param training Boolean flag; True enables exploration, False runs the greedy policy.
        @return Action to execute: a discrete integer for Discrete spaces, or a
                float32 numpy array for Box spaces.
        """
        pass

    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done):
        """
        @brief Store a single environment transition in the replay buffer.

        Called once per step in the training loop immediately after env.step().
        Implementations should handle buffer overflow (e.g., circular overwrite).

        @param state      Observation before the action was taken.
        @param action     Action that was executed (int or array depending on agent).
        @param reward     Scalar reward returned by the environment.
        @param next_state Observation after the action was taken.
        @param done       Boolean; True if the episode terminated or was truncated.
        """
        pass

    @abstractmethod
    def train_step(self):
        """
        @brief Sample a minibatch and perform one gradient update.

        Should return None (and skip the update) when the replay buffer does not
        yet contain enough samples to form a full batch.

        @return Scalar loss for single-loss agents (e.g., DQN TD loss), a tuple
                (critic_loss, actor_loss) for actor-critic agents, or None if the
                buffer is not yet full enough to train.
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """
        @brief Serialize agent weights and any training state to disk.

        Should save everything needed to resume training or run inference,
        including network weights, optimizer state, and exploration parameters.

        @param filepath String path where the checkpoint file will be written.
        """
        pass

    @abstractmethod
    def load(self, filepath):
        """
        @brief Restore agent weights and training state from a checkpoint file.

        Must be compatible with files written by save(). After loading, the agent
        should be ready for both inference (select_action) and continued training.

        @param filepath String path to the checkpoint file to load.
        """
        pass

    def get_training_metrics(self):
        """
        @brief Return a dict of current diagnostic metrics for console logging.

        The base implementation returns only the step count. Subclasses should
        override this to surface algorithm-specific values such as epsilon (DQN)
        or noise scale (DDPG/TD3).

        @return Dict mapping metric name strings to scalar float values.
                Always includes at least {"steps": self.steps}.
        """
        return {"steps": self.steps}
