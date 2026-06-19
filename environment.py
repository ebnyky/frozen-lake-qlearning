"""
environment.py

Custom Frozen Lake environment implemented from first principles.
No Gymnasium, OpenAI Gym, Stable Baselines, RLlib or RL frameworks are used.
"""

from dataclasses import dataclass
import numpy as np


DEFAULT_MAP = (
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
)


@dataclass(frozen=True)
class StepResult:
    next_state: int
    reward: float
    done: bool
    info: dict


class FrozenLakeEnv:
    """
    8x8 Frozen Lake environment.

    Actions:
        0 = Left
        1 = Down
        2 = Right
        3 = Up

    State representation:
        Single integer index: state = row * number_of_columns + column

    Reward structure:
        Frozen / Start state: -1
        Hole: -100
        Goal: +100
    """

    ACTIONS = {
        0: (0, -1),   # Left
        1: (1, 0),    # Down
        2: (0, 1),    # Right
        3: (-1, 0),   # Up
    }

    ACTION_NAMES = {
        0: "Left",
        1: "Down",
        2: "Right",
        3: "Up",
    }

    ACTION_SYMBOLS = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑",
    }

    def __init__(self, lake_map=DEFAULT_MAP, hole_reward=-100, goal_reward=100, step_reward=-1):
        self.lake_map = tuple(lake_map)
        self.grid = np.array([list(row) for row in self.lake_map])
        self.rows, self.cols = self.grid.shape

        self.hole_reward = hole_reward
        self.goal_reward = goal_reward
        self.step_reward = step_reward

        self.start_position = self._find_tile("S")
        self.goal_position = self._find_tile("G")
        self.current_position = self.start_position

    @property
    def state_size(self):
        return self.rows * self.cols

    @property
    def action_size(self):
        return len(self.ACTIONS)

    def _find_tile(self, tile):
        locations = np.argwhere(self.grid == tile)
        if len(locations) == 0:
            raise ValueError(f"Map must contain tile {tile}")
        row, col = locations[0]
        return int(row), int(col)

    def position_to_state(self, position):
        row, col = position
        return row * self.cols + col

    def state_to_position(self, state):
        return divmod(int(state), self.cols)

    def reset(self):
        self.current_position = self.start_position
        return self.get_state()

    def get_state(self):
        return self.position_to_state(self.current_position)

    def get_tile(self, state=None):
        if state is None:
            row, col = self.current_position
        else:
            row, col = self.state_to_position(state)
        return self.grid[row, col]

    def is_terminal(self, state=None):
        tile = self.get_tile(state)
        return tile in ("H", "G")

    def _reward(self, tile):
        if tile == "H":
            return self.hole_reward
        if tile == "G":
            return self.goal_reward
        return self.step_reward

    def step(self, action):
        if action not in self.ACTIONS:
            raise ValueError("Action must be one of 0, 1, 2, 3")

        row, col = self.current_position
        d_row, d_col = self.ACTIONS[action]

        new_row = row + d_row
        new_col = col + d_col

        # Enforce movement boundaries.
        if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
            new_row, new_col = row, col

        self.current_position = (new_row, new_col)
        tile = self.grid[new_row, new_col]
        reward = self._reward(tile)
        done = self.is_terminal()

        return StepResult(
            next_state=self.get_state(),
            reward=reward,
            done=done,
            info={
                "position": self.current_position,
                "tile": str(tile),
                "action_name": self.ACTION_NAMES[action],
            },
        )

    def render(self, show_agent=True):
        display = self.grid.astype(str).copy()
        if show_agent:
            row, col = self.current_position
            display[row, col] = "A"
        print(display)

    def render_policy(self, q_table):
        """
        Convert a Q-table into a readable grid policy.
        Terminal states are displayed as H and G.
        """
        policy_grid = []
        for row in range(self.rows):
            policy_row = []
            for col in range(self.cols):
                tile = self.grid[row, col]
                state = self.position_to_state((row, col))

                if tile == "H":
                    policy_row.append("H")
                elif tile == "G":
                    policy_row.append("G")
                else:
                    best_action = int(np.argmax(q_table[state]))
                    policy_row.append(self.ACTION_SYMBOLS[best_action])

            policy_grid.append(policy_row)

        return np.array(policy_grid, dtype=object)
