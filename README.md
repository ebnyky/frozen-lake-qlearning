# Frozen Lake Q-Learning from First Principles

**Student Name:** Nyamekye Emmanuel Barima  
**Student ID:** 22425704  
**Course:** DSCD 614 - Reinforcement Learning  
**Assignment:** Programming Assignment 1: Frozen Lake from First Principles Using Q-Learning

## Introduction

This project implements the Frozen Lake problem from first principles using Python, NumPy and Matplotlib only. No Reinforcement Learning framework such as Gymnasium, OpenAI Gym, Stable Baselines or RLlib is used.

## What is Reinforcement Learning?

Reinforcement Learning is a machine learning approach where an agent learns by interacting with an environment. The agent observes a state, selects an action, receives a reward and uses this feedback to improve future decisions. The aim is to learn a policy that maximizes cumulative reward over time.

## What is Frozen Lake?

Frozen Lake is a grid-world problem. An agent starts at `S` and must reach the goal `G` while avoiding holes `H`. Frozen states `F` are safe. The map used in this assignment is the standard 8 by 8 map:

```text
SFFFFFFF
FFFFFFFF
FFFHFFFF
FFFHFFFF
FFFHFFFF
FHHFFFHF
FHFFHFHF
FFFHFFFG
```

## Environment Design

The environment is implemented in `environment.py` using a custom `FrozenLakeEnv` class.

Required methods implemented:

- `reset()`
- `step(action)`
- `render()`
- `get_state()`
- `is_terminal()`

### State Representation

States are represented as single integer indices:

```text
state = row * number_of_columns + column
```

For an 8 by 8 grid, there are 64 states numbered from 0 to 63.

### Action Representation

| Action | Meaning |
|---|---|
| 0 | Left |
| 1 | Down |
| 2 | Right |
| 3 | Up |

### Reward Structure

| State Type | Reward |
|---|---:|
| Frozen / Start | -1 |
| Hole | -100 |
| Goal | +100 |

The step penalty encourages the agent to find a shorter route, the hole penalty discourages unsafe moves, and the goal reward encourages successful completion.

## Q-Learning Algorithm

Q-Learning is a value-based reinforcement learning algorithm. It learns an action-value function, called the Q-table, that estimates the expected future reward for taking an action in a given state.

The update equation used is:

```text
Q(s,a) <- Q(s,a) + alpha [r + gamma max Q(s',a') - Q(s,a)]
```

Where:

- `alpha` is the learning rate.
- `gamma` is the discount factor.
- `r` is the reward.
- `s'` is the next state.
- `max Q(s',a')` is the best expected future value.

## Exploration Strategy

The agent uses epsilon-greedy exploration. At the beginning, epsilon is high, so the agent explores random actions. As training progresses, epsilon decays, so the agent increasingly exploits the learned Q-table.

## Training Procedure

Main hyperparameters used for the final model:

| Hyperparameter | Value |
|---|---:|
| Episodes | 20000 |
| Maximum steps per episode | 200 |
| Learning rate alpha | 0.1 |
| Discount factor gamma | 0.99 |
| Initial epsilon | 1.0 |
| Minimum epsilon | 0.01 |
| Epsilon decay | 0.9995 |

## Results

### Final Training Results

| Metric | Value |
|---|---:|
| Successful training episodes | 17634 |
| Training success rate | 88.17% |
| Average training reward | 61.99 |
| Last 100 episode success rate | 98.00% |
| Last 100 average reward | 82.97 |
| Final epsilon | 0.0100 |

### Evaluation Results

The trained agent was evaluated over 200 greedy episodes.

| Metric | Value |
|---|---:|
| Success rate | 100.00% |
| Average reward | 87.00 |
| Successful runs | 200 |
| Failures | 0 |

## Learned Policy

Symbols: `←` Left, `↓` Down, `→` Right, `↑` Up, `H` Hole, `G` Goal.

```text
↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
→ → → → ↓ ↓ ↓ ↓
↑ ↑ ↑ H → → ↓ ↓
→ ↑ ↑ H → → → ↓
↑ ↑ ↑ H → → → ↓
↑ H H → → ↑ H ↓
↓ H → ↑ H ↑ H ↓
→ → ↑ H → ↑ → G

```

## Hyperparameter Comparison

A small experiment was conducted using different learning rates and discount factors.

| Alpha | Gamma | Epsilon Decay | Evaluation Success Rate | Evaluation Average Reward |
|---:|---:|---:|---:|---:|
| 0.10 | 0.90 | 0.9995 | 100.00% | 87.00 |
| 0.10 | 0.95 | 0.9995 | 100.00% | 87.00 |
| 0.10 | 0.99 | 0.9995 | 100.00% | 87.00 |
| 0.20 | 0.99 | 0.9995 | 100.00% | 87.00 |

## Bonus Task

Bonus Option B was implemented: training performance visualization using Matplotlib.

The following graphs are saved in the `results/` folder:

- `reward_curve.png`
- `reward_moving_average.png`
- `epsilon_decay.png`
- `success_rate_curve.png`

## Execution Instructions

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the agent:

```bash
python train.py
```

Evaluate the trained agent:

```bash
python evaluate.py
```

Run hyperparameter comparison:

```bash
python experiment.py
```

## Repository Structure

```text
frozen-lake-qlearning/
├── environment.py
├── agent.py
├── train.py
├── evaluate.py
├── experiment.py
├── requirements.txt
├── README.md
├── report.pdf
└── results/
    ├── q_table.npy
    ├── training_metrics.json
    ├── evaluation_results.json
    ├── hyperparameter_experiments.json
    ├── learned_policy.txt
    ├── reward_curve.png
    ├── reward_moving_average.png
    ├── epsilon_decay.png
    └── success_rate_curve.png
```
