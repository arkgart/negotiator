"""Wealth process utilities implementing time-average rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from ..config.schema import EconomySpec, WealthSpec

__all__ = ["WealthModel", "WealthLedger"]


@dataclass
class WealthModel:
    """Maps negotiation outcomes to wealth dynamics."""

    spec: WealthSpec
    economy: EconomySpec
    issue_names: Sequence[str]

    def as_dict(self, outcome: Iterable[float]) -> Dict[str, float]:
        return {name: float(value) for name, value in zip(self.issue_names, outcome)}

    def buyer_cashflow(self, data: Dict[str, float]) -> float:
        price = data.get("price", 0.0)
        quantity = data.get("quantity", 0.0)
        delivery = data.get("delivery", self.economy.delivery_reference)
        value = self.economy.value_per_unit * quantity
        delivery_bonus = self.economy.delivery_bonus * max(0.0, self.economy.delivery_reference - delivery)
        return (value + delivery_bonus - price) * quantity

    def seller_cashflow(self, data: Dict[str, float]) -> float:
        price = data.get("price", 0.0)
        quantity = data.get("quantity", 0.0)
        delivery = data.get("delivery", self.economy.delivery_reference)
        revenue = price * quantity
        production_cost = self.economy.seller_cost * quantity
        delivery_cost = self.economy.seller_delivery_cost * max(0.0, self.economy.delivery_reference - delivery)
        return revenue - production_cost - delivery_cost

    def cashflow(self, outcome: Iterable[float], role: str) -> float:
        data = self.as_dict(outcome)
        if role.lower().startswith("buy"):
            return self.buyer_cashflow(data)
        return self.seller_cashflow(data)

    def apply_time_penalty(self, wealth: float) -> Tuple[float, float]:
        base = max(wealth, 1e-9)
        new_wealth = max(1e-9, base * max(1e-6, 1.0 - self.spec.time_cost))
        delta_log = float(np.log(new_wealth) - np.log(base))
        return new_wealth, delta_log

    def apply_deal(self, wealth: float, outcome: Iterable[float], role: str) -> Tuple[float, float, float, bool]:
        base = max(wealth, 1e-9)
        cashflow = self.cashflow(outcome, role)
        growth = cashflow / base
        new_wealth = max(1e-9, base * (1.0 + growth))
        delta_log = float(np.log(new_wealth) - np.log(base))
        ruined = new_wealth < self.spec.ruin_threshold
        if ruined:
            new_wealth = max(self.spec.ruin_threshold, 1e-9)
            delta_log += self.spec.ruin_penalty
        return new_wealth, delta_log, cashflow, ruined


@dataclass
class WealthLedger:
    """Tracks the evolution of wealth across a negotiation episode."""

    model: WealthModel
    wealth: float | None = None
    log_wealth: float | None = None
    history: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.wealth is None:
            self.wealth = self.model.spec.initial
        if self.log_wealth is None:
            self.log_wealth = float(np.log(max(self.wealth, 1e-9)))
        if self.history is None:
            self.history = [self.wealth]

    def step_time(self) -> float:
        assert self.wealth is not None
        new_wealth, delta_log = self.model.apply_time_penalty(self.wealth)
        self.wealth = new_wealth
        self.log_wealth = float(np.log(max(new_wealth, 1e-9)))
        self.history.append(new_wealth)
        return delta_log

    def step_deal(self, outcome: Iterable[float], role: str) -> Tuple[float, float, bool]:
        assert self.wealth is not None
        new_wealth, delta_log, cashflow, ruined = self.model.apply_deal(self.wealth, outcome, role)
        self.wealth = new_wealth
        self.log_wealth = float(np.log(max(new_wealth, 1e-9)))
        self.history.append(new_wealth)
        return delta_log, cashflow, ruined
