"""Configuration dataclasses used throughout the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence


@dataclass(slots=True)
class EconomicConfig:
    """Parameters of the stylised economic model used to value outcomes."""

    buyer_value_intercept: float = 16.0
    buyer_value_quantity_scale: float = 0.35
    buyer_delivery_penalty: float = 0.75
    seller_cost_intercept: float = 4.0
    seller_cost_quantity_scale: float = 0.22
    seller_delivery_cost: float = 0.35
    capital_scale: float = 100.0


@dataclass(slots=True)
class WealthConfig:
    """Controls the ergodic wealth process used for rewards."""

    initial_wealth: float = 100.0
    time_penalty: float = 0.002  # 0.2% cost per negotiation step
    ruin_threshold: float = 1.0
    ruin_penalty: float = -5.0
    log_floor: float = -20.0


@dataclass(slots=True)
class NegotiationConfig:
    """Session-level configuration for the negotiation process."""

    n_steps: int = 40
    k_candidates: int = 64
    role: str = "buyer"
    time_cost_reward: float = -0.002
    timeout_penalty: float = -0.1


@dataclass(slots=True)
class TrainingConfig:
    """High-level parameters driving the RL training loop."""

    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    batch_episodes: int = 8
    max_iterations: int = 200
    log_every: int = 10
    opponent_schedule: Sequence[str] = field(
        default_factory=lambda: ["time_based", "boulware", "tit_for_tat"]
    )
    seed: int = 13


@dataclass(slots=True)
class DomainConfig:
    """Defines the negotiation domain (issues and their ranges)."""

    price_levels: int = 21
    quantity_range: tuple[int, int] = (1, 51)
    delivery_levels: int = 11

    def issue_factories(self) -> List[Callable[[], dict]]:
        return []
