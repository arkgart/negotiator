"""Policy abstractions bridging RL and NegMAS negotiators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


class PolicyInterface:
    """Abstract policy used by :class:`~ergodic_negotiator.agents.RLNegotiator`."""

    def act(self, obs: np.ndarray) -> tuple[float, int]:
        raise NotImplementedError

    def act_accept(self, obs: np.ndarray, my_utility: float) -> float:
        raise NotImplementedError


class RandomPolicy(PolicyInterface):
    """Simple baseline policy sampling uniformly random actions."""

    def __init__(self, k_candidates: int) -> None:
        self.k_candidates = k_candidates

    def act(self, obs: np.ndarray) -> tuple[float, int]:  # noqa: D401 - interface
        _ = obs
        return float(np.random.rand()), int(np.random.randint(0, self.k_candidates))

    def act_accept(self, obs: np.ndarray, my_utility: float) -> float:  # noqa: D401 - interface
        _ = obs
        return my_utility - np.random.rand()


@dataclass(slots=True)
class ExternalPolicy(PolicyInterface):
    """Policy controlled externally (e.g. by a Gym environment)."""

    k_candidates: int
    noise: float = 0.0

    def __post_init__(self) -> None:
        self._target: Optional[float] = None
        self._selector: Optional[int] = None
        self._threshold: Optional[float] = None

    def set_next_action(self, action: np.ndarray) -> None:
        action = np.clip(action.astype(np.float32), 0.0, 1.0)
        self._threshold = float(action[0])
        self._target = float(action[1])
        idx = min(self.k_candidates - 1, max(0, int(np.floor(action[2] * self.k_candidates))))
        self._selector = int(idx)

    def _ensure_ready(self) -> None:
        if self._target is None or self._selector is None or self._threshold is None:
            self._target = 0.5
            self._selector = self.k_candidates // 2
            self._threshold = 0.5

    def act(self, obs: np.ndarray) -> tuple[float, int]:
        self._ensure_ready()
        target = float(np.clip(self._target + self.noise * np.random.randn(), 0.0, 1.0))
        selector = self._selector
        self._target = None
        self._selector = None
        return target, selector

    def act_accept(self, obs: np.ndarray, my_utility: float) -> float:
        self._ensure_ready()
        threshold = float(np.clip(self._threshold + self.noise * np.random.randn(), 0.0, 1.0))
        self._threshold = None
        return my_utility - threshold


__all__ = [
    "PolicyInterface",
    "RandomPolicy",
    "ExternalPolicy",
]
