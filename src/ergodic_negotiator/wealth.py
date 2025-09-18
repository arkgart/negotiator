"""Wealth tracking utilities for time-average ergodic rewards."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from .config import WealthParameters

Outcome = Sequence[int | float]


@dataclass(slots=True)
class WealthStep:
    """Result of applying a wealth update."""

    previous: float
    current: float
    log_delta: float
    reason: str


class TimeAverageWealthTracker:
    """Maintain multiplicative wealth with time-costs and deal growth."""

    def __init__(self, params: WealthParameters, growth_fn: Callable[[Outcome, str], float], role: str) -> None:
        self.params = params
        self._growth_fn = growth_fn
        self._role = role
        self._log_wealth = math.log(max(params.initial_wealth, params.min_wealth))
        self._wealth = max(params.initial_wealth, params.min_wealth)
        self.history: list[WealthStep] = []

    @property
    def wealth(self) -> float:
        return self._wealth

    @property
    def log_wealth(self) -> float:
        return self._log_wealth

    def reset(self) -> None:
        self._wealth = max(self.params.initial_wealth, self.params.min_wealth)
        self._log_wealth = math.log(self._wealth)
        self.history.clear()

    def _register(self, new_wealth: float, reason: str) -> WealthStep:
        prev = self._wealth
        self._wealth = max(new_wealth, self.params.min_wealth)
        self._log_wealth = math.log(self._wealth)
        step = WealthStep(previous=prev, current=self._wealth, log_delta=self._log_wealth - math.log(prev), reason=reason)
        self.history.append(step)
        return step

    def apply_time_cost(self) -> WealthStep:
        multiplier = max(0.0, 1.0 - self.params.time_cost)
        return self._register(self._wealth * multiplier, "time")

    def apply_deal(self, outcome: Outcome) -> WealthStep:
        growth = self._growth_fn(outcome, self._role)
        multiplier = max(1e-9, 1.0 + growth)
        return self._register(self._wealth * multiplier, "deal")

    def ruined(self) -> bool:
        return self._wealth <= self.params.ruin_threshold


__all__ = [
    "TimeAverageWealthTracker",
    "WealthStep",
]
