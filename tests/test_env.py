from __future__ import annotations

import numpy as np

from ergodic_negotiator import (
    DomainParameters,
    ErgodicNegotiationEnv,
    NegotiationEnvConfig,
    evaluate_random,
)


def test_environment_reset_and_step():
    config = NegotiationEnvConfig(domain=DomainParameters(price_levels=7), max_rounds=6)
    env = ErgodicNegotiationEnv(config)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert "wealth" in info
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "state" in info


def test_random_evaluation_runs():
    stats = evaluate_random(episodes=3)
    assert 0.0 <= stats.agreement_rate <= 1.0
    assert stats.average_steps > 0
    assert np.isfinite(stats.reward)
