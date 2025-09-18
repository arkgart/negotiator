"""Training utilities for the ergodic negotiation environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .config import NegotiationEnvConfig
from .env import ErgodicNegotiationEnv, NegotiationResult


@dataclass(slots=True)
class EpisodeStats:
    reward: float
    agreement_rate: float
    average_steps: float
    final_wealth: float


def make_env(config: Optional[NegotiationEnvConfig] = None) -> ErgodicNegotiationEnv:
    """Factory for the default negotiation environment."""

    return ErgodicNegotiationEnv(config=config)


def rollout(env: ErgodicNegotiationEnv, policy: Callable[[np.ndarray], np.ndarray], episodes: int = 10) -> List[NegotiationResult]:
    """Run ``episodes`` episodes using ``policy`` for action selection."""

    results: list[NegotiationResult] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action = np.clip(policy(obs), 0.0, 1.0)
            obs, _, terminated, _, _ = env.step(action)
        results.append(env.negotiation_result())
    return results


def random_policy(_: np.ndarray) -> np.ndarray:
    """Uniform random actions over the action space."""

    return np.random.rand(3).astype(np.float32)


def evaluate_random(config: Optional[NegotiationEnvConfig] = None, episodes: int = 25) -> EpisodeStats:
    """Evaluate the environment with a random policy for smoke-testing."""

    env = make_env(config)
    results = rollout(env, random_policy, episodes=episodes)
    agreements = sum(1 for r in results if r.agreement is not None)
    avg_steps = float(np.mean([r.steps for r in results])) if results else 0.0
    avg_reward = float(np.mean([r.log_wealth for r in results])) if results else 0.0
    avg_wealth = float(np.mean([r.wealth for r in results])) if results else 0.0
    return EpisodeStats(
        reward=avg_reward,
        agreement_rate=agreements / max(len(results), 1),
        average_steps=avg_steps,
        final_wealth=avg_wealth,
    )


def train_sb3(
    env_builder: Callable[[], ErgodicNegotiationEnv],
    total_timesteps: int = 50_000,
    algorithm: str = "PPO",
    policy: str = "MlpPolicy",
    **kwargs,
):
    """Train an SB3 agent if stable-baselines3 is available."""

    try:
        import stable_baselines3 as sb3
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("stable-baselines3 is required for train_sb3") from exc

    env = env_builder()
    algo_cls = getattr(sb3, algorithm)
    model = algo_cls(policy, env, **kwargs)
    model.learn(total_timesteps=total_timesteps)
    return model


__all__ = [
    "EpisodeStats",
    "evaluate_random",
    "make_env",
    "rollout",
    "train_sb3",
]
