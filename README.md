# BlueChips Labs: Ergodic Negotiation Simulator

This repository contains a reinforcement-learning-ready multi-agent negotiation simulator built on top of [NegMAS](https://negmas.readthedocs.io/). The focus is on **ergodic**, time-average decision making: agents optimise the logarithmic growth of their wealth across negotiation rounds rather than expected utility of individual deals.

## Features

- **NegMAS SAO protocol integration** with a custom `RLNegotiator` that exposes compact observations, candidate offer generation, and opponent-tracking telemetry.
- **Gymnasium environment** (`ErgodicNegotiationEnv`) that wraps a negotiation session into an RL-compatible interface with continuous actions controlling acceptance thresholds and concessions.
- **Time-average wealth process** with multiplicative updates, time-cost penalties, and configurable ruin thresholds.
- **Baseline policies and training utilities** including a random rollout driver and an optional Stable-Baselines3 integration hook.
- **Scriptable entrypoint** (`scripts/run_training.py`) to run smoke tests or invoke PPO training when SB3 is installed.

## Getting started

Install dependencies:

```bash
pip install negmas gymnasium numpy
```

Optionally install Stable-Baselines3 to enable PPO training:

```bash
pip install stable-baselines3
```

Run a quick random-agent evaluation:

```bash
python scripts/run_training.py --episodes 10
```

This produces aggregate time-average growth, wealth and agreement statistics using the random policy.

To train with PPO (requires SB3):

```bash
python scripts/run_training.py --train --timesteps 100000 --model-path ppo_negotiator
```

## Library usage

```python
from ergodic_negotiator import (
    ErgodicNegotiationEnv,
    NegotiationEnvConfig,
    DomainParameters,
    evaluate_random,
)

config = NegotiationEnvConfig(
    domain=DomainParameters(price_levels=25),
    max_rounds=60,
)

env = ErgodicNegotiationEnv(config)
obs, info = env.reset()
print("Initial observation shape:", obs.shape)
```

The environment’s action space is a 3-D continuous box `[0,1]^3` representing:

1. Acceptance threshold (normalised utility).
2. Target concession level.
3. Candidate selector among high-utility offers.

Rewards correspond to the change in log-wealth for each round plus an optional ruin penalty when the wealth drops below the configured threshold.

## Tests

The repository ships with lightweight unit tests (`pytest`) covering environment construction, random rollouts, and wealth updates. Run them via:

```bash
pytest
```

## Citation

If you use this simulator in academic or professional work, please cite the repository and the NegMAS project.
