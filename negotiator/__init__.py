"""Top-level package for the ergodic negotiation simulator."""

from .config import NegotiationConfig, WealthConfig, TrainingConfig, EconomicConfig
from .domain import build_default_domain, NegotiationDomain
from .agents import RLNegotiator
from .training import NegotiationTrainer

__all__ = [
    "NegotiationConfig",
    "WealthConfig",
    "TrainingConfig",
    "EconomicConfig",
    "NegotiationDomain",
    "build_default_domain",
    "RLNegotiator",
    "NegotiationTrainer",
]
