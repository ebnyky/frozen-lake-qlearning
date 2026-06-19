"""
train.py

Train a Q-Learning agent on the custom Frozen Lake environment.
This file records episode rewards, success rate, epsilon history,
learned policy and training performance graphs.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from environment import FrozenLakeEnv
from agent import QLearningAgent


def moving_average(values, window=100):
    values = np.array(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def train_agent(
    episodes=20000,
    max_steps=200,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.9995,
    seed=42,
):
    env = FrozenLakeEnv()
    agent = QLearningAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    episode_rewards = []
    epsilon_history = []
    success_history = []
    successful_episodes = 0

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        success = 0

        for _ in range(max_steps):
            action = agent.choose_action(state, training=True)
            result = env.step(action)

            agent.update(
                state=state,
                action=action,
                reward=result.reward,
                next_state=result.next_state,
                done=result.done,
            )

            state = result.next_state
            total_reward += result.reward

            if result.done:
                if result.info["tile"] == "G":
                    successful_episodes += 1
                    success = 1
                break

        agent.decay_epsilon()
        episode_rewards.append(float(total_reward))
        epsilon_history.append(float(agent.epsilon))
        success_history.append(int(success))

    policy_grid = env.render_policy(agent.q_table)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    np.save(results_dir / "q_table.npy", agent.q_table)

    with open(results_dir / "learned_policy.txt", "w", encoding="utf-8") as f:
        for row in policy_grid:
            f.write(" ".join(row) + "\n")

    success_rate = successful_episodes / episodes * 100

    metrics = {
        "episodes": episodes,
        "max_steps_per_episode": max_steps,
        "alpha": alpha,
        "gamma": gamma,
        "initial_epsilon": epsilon,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "final_epsilon": agent.epsilon,
        "successful_episodes": successful_episodes,
        "training_success_rate_percent": success_rate,
        "average_training_reward": float(np.mean(episode_rewards)),
        "last_100_episode_success_rate_percent": float(np.mean(success_history[-100:]) * 100),
        "last_100_average_reward": float(np.mean(episode_rewards[-100:])),
    }

    with open(results_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(results_dir / "episode_rewards.json", "w", encoding="utf-8") as f:
        json.dump(episode_rewards, f)

    with open(results_dir / "epsilon_history.json", "w", encoding="utf-8") as f:
        json.dump(epsilon_history, f)

    with open(results_dir / "success_history.json", "w", encoding="utf-8") as f:
        json.dump(success_history, f)

    # Bonus Option B: visualize training performance.
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Training Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.tight_layout()
    plt.savefig(results_dir / "reward_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(episode_rewards, 100))
    plt.title("Training Rewards Moving Average (Window = 100)")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.tight_layout()
    plt.savefig(results_dir / "reward_moving_average.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_history)
    plt.title("Epsilon Decay Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.tight_layout()
    plt.savefig(results_dir / "epsilon_decay.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(success_history, 100) * 100)
    plt.title("Success Rate Moving Average (Window = 100)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (%)")
    plt.tight_layout()
    plt.savefig(results_dir / "success_rate_curve.png", dpi=200)
    plt.close()

    print("Training complete.")
    print(json.dumps(metrics, indent=4))
    print("\nLearned Policy:")
    for row in policy_grid:
        print(" ".join(row))

    return agent, env, metrics


if __name__ == "__main__":
    train_agent()
