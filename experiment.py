"""
experiment.py

Runs a small hyperparameter comparison as required by the assignment.
"""

import json
from pathlib import Path

import numpy as np

from environment import FrozenLakeEnv
from agent import QLearningAgent


def run_single_experiment(alpha, gamma, epsilon_decay, episodes=8000, max_steps=200, seed=42):
    env = FrozenLakeEnv()
    agent = QLearningAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        alpha=alpha,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    successes = 0
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = agent.choose_action(state, training=True)
            result = env.step(action)
            agent.update(state, action, result.reward, result.next_state, result.done)

            state = result.next_state
            total_reward += result.reward

            if result.done:
                if result.info["tile"] == "G":
                    successes += 1
                break

        rewards.append(total_reward)
        agent.decay_epsilon()

    # Greedy evaluation after training.
    eval_success = 0
    eval_rewards = []
    for _ in range(100):
        state = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action = int(np.argmax(agent.q_table[state]))
            result = env.step(action)
            state = result.next_state
            total_reward += result.reward
            if result.done:
                if result.info["tile"] == "G":
                    eval_success += 1
                break
        eval_rewards.append(total_reward)

    return {
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_decay": epsilon_decay,
        "training_success_rate_percent": successes / episodes * 100,
        "training_average_reward": float(np.mean(rewards)),
        "evaluation_success_rate_percent": eval_success,
        "evaluation_average_reward": float(np.mean(eval_rewards)),
    }


def main():
    experiments = [
        (0.10, 0.90, 0.9995),
        (0.10, 0.95, 0.9995),
        (0.10, 0.99, 0.9995),
        (0.20, 0.99, 0.9995),
    ]

    results = []
    for alpha, gamma, decay in experiments:
        results.append(run_single_experiment(alpha, gamma, decay))

    Path("results").mkdir(exist_ok=True)
    with open("results/hyperparameter_experiments.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
