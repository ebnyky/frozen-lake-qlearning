"""
evaluate.py

Evaluate the trained Q-Learning agent over at least 100 episodes.
"""

import json
from pathlib import Path

import numpy as np

from environment import FrozenLakeEnv


def evaluate_agent(episodes=200, max_steps=200):
    env = FrozenLakeEnv()
    results_dir = Path("results")
    q_table_path = results_dir / "q_table.npy"

    if not q_table_path.exists():
        raise FileNotFoundError(
            "results/q_table.npy not found. Run python train.py before evaluation."
        )

    q_table = np.load(q_table_path)

    total_rewards = []
    successful_runs = 0
    failures = 0

    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        for _ in range(max_steps):
            action = int(np.argmax(q_table[state]))
            result = env.step(action)

            state = result.next_state
            episode_reward += result.reward
            done = result.done

            if done:
                if result.info["tile"] == "G":
                    successful_runs += 1
                else:
                    failures += 1
                break

        if not done:
            failures += 1

        total_rewards.append(float(episode_reward))

    evaluation_results = {
        "evaluation_episodes": episodes,
        "success_rate_percent": successful_runs / episodes * 100,
        "average_reward": float(np.mean(total_rewards)),
        "number_of_successful_runs": successful_runs,
        "number_of_failures": failures,
    }

    with open(results_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4)

    print("Evaluation Results")
    print(json.dumps(evaluation_results, indent=4))

    return evaluation_results


if __name__ == "__main__":
    evaluate_agent()
