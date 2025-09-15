#!/usr/bin/env python3
"""Command-line entry point for training negotiation agents."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from negotiator.config.schema import NegotiationConfig, load_config
from negotiator.training.runner import NegotiationTrainer


def parse_agent(spec: str) -> Tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Agent specification '{spec}' must be in the form name:role:mode (e.g., tag_buyer:buyer:tag)"
        )
    name, role, mode = parts
    return name, role, mode


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-agent negotiation policies")
    parser.add_argument("--config", type=Path, default=Path("configs/ergodic_tag.yaml"), help="Path to YAML config file")
    parser.add_argument("--agent", action="append", type=parse_agent, dest="agents", help="Agent spec: name:role:mode")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of training episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to use")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    config: NegotiationConfig = load_config(args.config)
    set_seed(args.seed)
    trainer = NegotiationTrainer(config, device=args.device)

    if not args.agents:
        # Default to two TAG agents (buyer and seller)
        args.agents = [("buyer_tag", "buyer", "tag"), ("seller_tag", "seller", "tag")]

    for name, role, mode in args.agents:
        trainer.register_agent(name=name, role=role, reward_mode=mode)

    results = trainer.train(total_episodes=args.episodes)

    if not results:
        print("No training episodes executed.")
        return

    summary: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, int] = defaultdict(int)
    for result in results:
        summary[result.role]["reward"] += result.reward_sum
        summary[result.role]["log_growth"] += result.log_growth
        summary[result.role]["agreements"] += 1.0 if result.agreement else 0.0
        counts[result.role] += 1

    print("Training summary:")
    for role, metrics in summary.items():
        count = counts[role]
        avg_reward = metrics["reward"] / count
        avg_log_growth = metrics["log_growth"] / count
        agreement_rate = metrics["agreements"] / count
        print(
            f"  Role={role:7s} episodes={count:4d} avg_reward={avg_reward:8.4f} "
            f"avg_log_growth={avg_log_growth:8.4f} agreement_rate={agreement_rate:0.3f}"
        )


if __name__ == "__main__":
    main()
