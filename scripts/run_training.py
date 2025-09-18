#!/usr/bin/env python
"""Entry point for running quick simulations or SB3 training."""
from __future__ import annotations

import argparse
from pathlib import Path

from ergodic_negotiator import evaluate_random, make_env, train_sb3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ergodic negotiation simulator")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes for evaluation")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train using stable-baselines3 PPO instead of running the random baseline",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Total training timesteps if --train is supplied",
    )
    parser.add_argument("--model-path", type=Path, default=Path("ppo_negotiator"), help="Where to save the trained model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train:
        model = train_sb3(make_env, total_timesteps=args.timesteps)
        model.save(str(args.model_path))
    else:
        stats = evaluate_random(episodes=args.episodes)
        print("Random policy over", args.episodes, "episodes")
        print("  agreement rate:", round(stats.agreement_rate, 3))
        print("  avg steps:", round(stats.average_steps, 2))
        print("  avg log-wealth:", round(stats.reward, 4))
        print("  avg wealth:", round(stats.final_wealth, 4))


if __name__ == "__main__":
    main()
