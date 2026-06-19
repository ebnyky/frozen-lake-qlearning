"""
agent.py

Q-Learning agent implemented from scratch using NumPy.
"""

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        state_size,
        action_size,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        seed=42,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((state_size, action_size), dtype=float)

    def choose_action(self, state, training=True):
        """
        Epsilon-greedy action selection.

        During training:
            Random action with probability epsilon.
            Otherwise choose the action with the highest Q-value.

        During evaluation:
            Always choose the greedy action.
        """
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))

        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning update:
        Q(s,a) <- Q(s,a) + alpha [r + gamma max Q(s',a') - Q(s,a)]

        For terminal states, the future value term is zero.
        """
        old_value = self.q_table[state, action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state, action] = old_value + self.alpha * (target - old_value)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
