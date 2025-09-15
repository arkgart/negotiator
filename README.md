# Negotiator

A reinforcement-learning negotiation simulator built on [NegMAS](https://negmas.readthedocs.io/) with
explicit support for ergodic time-average rewards. The toolkit allows you to pit RL-powered
negotiators against a league of baseline strategies, compute multiplicative wealth growth, and
contrast expected-utility versus time-average objectives for your ergodicity research.

## Key capabilities

- **NegMAS integration** – uses `SAOMechanism` multi-issue alternating-offers protocols.
- **Policy-gradient negotiators** – PPO actors with Beta-distributed actions drive offer/accept
  decisions while respecting issue bounds.
- **Ergodic rewards** – wealth ledgers apply per-step time costs and multiplicative deal returns,
  reporting log-growth metrics and ruin events.
- **Opponent league** – mix of aspiration, time-based conceder, tit-for-tat, and random negotiators
  for curriculum-style training.
- **Config-driven experiments** – YAML files describe domains, utilities, wealth processes, and RL
  hyperparameters for reproducible runs.

## Project layout

```
configs/              Example YAML configurations
scripts/train.py      Command-line training entry point
src/negotiator/       Python package with agents, policies, trainers, and utilities
```

## Getting started

1. Create and activate a Python 3.10+ environment.
2. Install the project in editable mode:

   ```bash
   pip install -e .
   ```

3. Launch training with the default ergodic setup (TAG agents as buyer and seller):

   ```bash
   ./scripts/train.py --config configs/ergodic_tag.yaml --episodes 10
   ```

   Use `--agent name:role:mode` to add agents. Example: train both TAG and EU agents simultaneously
   against baseline opponents.

   ```bash
   ./scripts/train.py \
       --agent tag_buyer:buyer:tag \
       --agent eu_seller:seller:eu \
       --episodes 50
   ```

The CLI prints per-role averages for cumulative reward, log-growth, and agreement frequency.

## Customising domains and agents

- Edit `configs/ergodic_tag.yaml` to change issue ranges, utility slopes, wealth parameters, or
  opponent pools.
- `src/negotiator/agents/rl_negotiator.py` defines the observation vector and action decoding – extend
  it with additional opponent-model features or acceptance heuristics.
- `src/negotiator/policies/policy.py` implements the Beta-actor/critic PPO core. Adjust network sizes
  or hyperparameters via the config file.
- `src/negotiator/utils/wealth.py` encapsulates multiplicative wealth updates and ruin penalties. Adapt
  the `buyer_cashflow`/`seller_cashflow` formulas to match your economic environment.

## Evaluation ideas

- Compare TAG versus EU reward modes by registering agents with different `mode` strings and tracking
  the `avg_log_growth` statistic.
- Log `EpisodeResult` objects during training to drive dashboards or scientific plots for your paper.
- Extend `NegotiationTrainer` to support self-play leagues or tournaments against Genius-imported
  opponents.

Enjoy exploring ergodic negotiation dynamics!
