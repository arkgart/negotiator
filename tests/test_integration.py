"""Smoke tests for the negotiation trainer."""

from __future__ import annotations

from pathlib import Path

from negotiator.config.schema import load_config
from negotiator.training.runner import NegotiationTrainer


def test_train_single_iteration(tmp_path: Path) -> None:
    config = load_config(Path("configs/ergodic_tag.yaml"))
    trainer = NegotiationTrainer(config)
    trainer.register_agent(name="buyer_tag", role="buyer", reward_mode="tag")
    trainer.register_agent(name="seller_tag", role="seller", reward_mode="tag")
    results = trainer.train(total_episodes=1)
    assert results, "Training should yield at least one episode result"
    for result in results:
        assert result.role in ("buyer", "seller")
