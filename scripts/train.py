"""Command line entry point for training the ergodic negotiation agent."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from negotiator import (
    EconomicConfig,
    NegotiationConfig,
    NegotiationTrainer,
    TrainingConfig,
    WealthConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ergodic negotiation RL agent")
    parser.add_argument("--iterations", type=int, default=50, help="Training iterations to run")
    parser.add_argument("--role", type=str, default="buyer", choices=["buyer", "seller"], help="Role played by the RL agent")
    parser.add_argument("--batch", type=int, default=8, help="Episodes per policy update")
    parser.add_argument("--log-every", type=int, default=5, help="Logging frequency in iterations")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes after training")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to store a JSON summary")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    negotiation_cfg = NegotiationConfig(role=args.role)
    training_cfg = TrainingConfig(batch_episodes=args.batch, log_every=args.log_every, max_iterations=args.iterations)
    trainer = NegotiationTrainer(
        negotiation_cfg=negotiation_cfg,
        training_cfg=training_cfg,
        wealth_cfg=WealthConfig(),
        economic_cfg=EconomicConfig(),
    )
    trainer.train(iterations=args.iterations)
    summary = trainer.evaluate(episodes=args.eval_episodes)

    print("Final evaluation summary:")
    print(
        json.dumps(
            {
                "episodes": summary.episodes,
                "agreement_rate": summary.agreement_rate,
                "avg_final_wealth": summary.avg_final_wealth,
                "mean_time_average": summary.mean_time_average,
                "ensemble_growth": summary.ensemble_growth,
                "ruin_probability": summary.ruin_probability,
            },
            indent=2,
        )
    )

    if args.output is not None:
        args.output.write_text(json.dumps(summary.__dict__, indent=2))


if __name__ == "__main__":
    main()
