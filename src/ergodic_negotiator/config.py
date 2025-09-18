"""Configuration data structures for the ergodic negotiation simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from negmas.sao import SAONegotiator


@dataclass(slots=True)
class DomainParameters:
    """Parameters describing the supply-chain like negotiation domain."""

    price_levels: int = 21
    quantity_range: tuple[int, int] = (1, 51)
    delivery_levels: int = 11
    buyer_price_bias: float = 20.0
    buyer_quantity_slope: float = 0.25
    buyer_delivery_bias: float = 10.0
    buyer_delivery_slope: float = -1.0
    seller_quantity_slope: float = 0.25
    seller_delivery_bias: float = 10.0
    seller_delivery_slope: float = -1.0
    buyer_growth_scale: float = 0.20
    seller_growth_scale: float = 0.18


@dataclass(slots=True)
class WealthParameters:
    """Parameters controlling the wealth/time-average reward process."""

    initial_wealth: float = 1.0
    min_wealth: float = 1e-9
    time_cost: float = 0.01
    ruin_threshold: float = 1e-6
    ruin_penalty: float = -5.0


@dataclass(slots=True)
class NegotiationEnvConfig:
    """Configuration for the :class:`~ergodic_negotiator.env.ErgodicNegotiationEnv`."""

    domain: DomainParameters = field(default_factory=DomainParameters)
    wealth: WealthParameters = field(default_factory=WealthParameters)
    max_rounds: int = 40
    k_candidates: int = 32
    role: Literal["buyer", "seller"] = "buyer"
    opponent_builder: Optional[Callable[[], SAONegotiator]] = None
    action_noise: float = 0.0


__all__ = [
    "DomainParameters",
    "WealthParameters",
    "NegotiationEnvConfig",
]
