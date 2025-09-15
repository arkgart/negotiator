# Ergodic Negotiation Simulator

This repository packages a reinforcement learning negotiation simulator built on top of [NegMAS](https://negmas.readthedocs.io/). The agent is trained to maximise time-average (ergodic) growth of wealth in multi-issue bilateral negotiations, contrasting with classical expected-utility strategies.

## Features

- **Multi-issue SAO negotiation domain** with price, quantity, and delivery issues.
- **Actor-critic RL negotiator** that controls concessions, offer selection, and acceptance decisions.
- **Ergodic wealth process** with multiplicative dynamics, time penalties, and ruin handling.
- **Scripted opponent league** (time-based conceder, boulware, tit-for-tat) to create a curriculum.
- **Training harness** producing time-average and ensemble-average diagnostics to support ergodicity analysis.

## Project layout

```
negotiator/
  agents.py        # RL negotiator implementation and episode stats
  config.py        # Dataclasses configuring negotiation, wealth, and training
  domain.py        # Domain factory for the multi-issue outcome space
  env.py           # Environment orchestrating sessions vs. opponent league
  metrics.py       # Aggregation utilities for ergodicity diagnostics
  opponents.py     # Scripted heuristic opponents
  policies.py      # Actor-critic policy networks and controller
  training.py      # High-level trainer coordinating batches and evaluation
scripts/
  train.py         # CLI entry point for running training experiments
```

## Getting started

1. Install the package (and dependencies) in a virtual environment:

   ```bash
   pip install -e .
   ```

2. Run a short training session (buyer role by default):

   ```bash
   python scripts/train.py --iterations 5 --batch 4 --eval-episodes 5
   ```

   The script prints periodic training diagnostics and a JSON evaluation summary containing agreement rate, time-average growth, ensemble growth, and ruin probability.

3. Integrate with your research workflow by importing the trainer:

   ```python
   from negotiator import NegotiationTrainer, NegotiationConfig, TrainingConfig

   trainer = NegotiationTrainer(
       negotiation_cfg=NegotiationConfig(role="buyer"),
       training_cfg=TrainingConfig(max_iterations=20, batch_episodes=6),
   )
   history = trainer.train()
   summary = trainer.evaluate(episodes=20)
   print(summary)
   ```

## Running tests

The repository uses `pytest` for unit tests:

```bash
pytest
```

## Citation

If this simulator contributes to your research, please cite the repository along with the NegMAS framework.
