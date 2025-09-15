"""Baseline opponent negotiators built on NegMAS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from negmas.sao import (
    AspirationNegotiator,
    NaiveTitForTatNegotiator,
    RandomNegotiator,
    SAONegotiator,
    TimeBasedConcedingNegotiator,
)

from ..config.schema import OpponentSpec

__all__ = ["build_opponent", "OpponentPool"]


_KIND_MAP = {
    "aspiration": AspirationNegotiator,
    "time": TimeBasedConcedingNegotiator,
    "titfortat": NaiveTitForTatNegotiator,
    "random": RandomNegotiator,
}


def build_opponent(spec: OpponentSpec) -> SAONegotiator:
    """Instantiate an opponent negotiator from a configuration specification."""

    kind = spec.kind.lower()
    if kind not in _KIND_MAP:
        raise KeyError(f"Unknown opponent type: {spec.kind}")
    cls = _KIND_MAP[kind]
    return cls(**spec.params)


@dataclass
class OpponentPool:
    """Weighted pool of opponent negotiators used during training."""

    specs: List[OpponentSpec]

    def sample(self, role: str) -> OpponentSpec:
        role_specs = [spec for spec in self.specs if spec.role == role]
        if not role_specs:
            raise ValueError(f"No opponents configured for role {role}")
        total_weight = sum(spec.weight for spec in role_specs)
        if total_weight <= 0:
            raise ValueError(f"Opponent weights for role {role} must sum to > 0")
        import random

        draw = random.random() * total_weight
        cumulative = 0.0
        for spec in role_specs:
            cumulative += spec.weight
            if draw <= cumulative:
                return spec
        return role_specs[-1]
