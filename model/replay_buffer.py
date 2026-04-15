"""
@brief Fixed-capacity experience replay buffer for off-policy RL agents.

Stores (state, action, reward, next_state, done) transitions in a circular
deque. Once capacity is reached, the oldest transitions are silently overwritten.
"""

import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """
    @brief Circular experience replay buffer shared by DQN, DDPG, and similar agents.

    Transitions are appended in order and sampled uniformly at random, breaking
    the temporal correlation between consecutive training batches.
    """

    def __init__(self, capacity=10000):
        """
        @brief Initialize an empty buffer with a fixed maximum capacity.

        @param capacity Maximum number of transitions to store before old entries
                        are overwritten (default: 10000).
        """
        # deque with maxlen enforces the circular overwrite behaviour automatically
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        @brief Append a single transition to the buffer.

        If the buffer is already at capacity, the oldest transition is dropped
        to make room for the new one.

        @param state      Numpy array representing the observation before the action.
        @param action     Integer index (discrete) or float array (continuous) of the action taken.
        @param reward     Scalar float reward returned by the environment.
        @param next_state Numpy array representing the observation after the action.
        @param done       Boolean; True if the episode ended after this transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        @brief Draw a uniformly random minibatch of transitions from the buffer.

        Samples without replacement and stacks each field into a contiguous
        numpy array for efficient conversion to PyTorch tensors by the caller.

        @param batch_size Number of transitions to sample; must be <= len(self).
        @return Tuple of five numpy arrays (states, actions, rewards, next_states, dones),
                each with leading dimension equal to batch_size.
        """
        batch = random.sample(self.buffer, batch_size)
        # Unzip the list of tuples into separate per-field sequences, then stack
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        """
        @brief Return the current number of transitions stored in the buffer.

        @return Integer count in the range [0, capacity].
        """
        return len(self.buffer)
