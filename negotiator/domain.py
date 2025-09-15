"""Domain factories for negotiation environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from negmas import SAOMechanism, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun

from .config import DomainConfig, NegotiationConfig


@dataclass(slots=True)
class NegotiationDomain:
    """Container with issues and utility functions for buyer and seller."""

    issues: Sequence
    buyer_utility: LinearAdditiveUtilityFunction
    seller_utility: LinearAdditiveUtilityFunction

    def create_mechanism(self, config: NegotiationConfig) -> SAOMechanism:
        """Create a negotiation mechanism for the domain and configuration."""

        mechanism = SAOMechanism(issues=self.issues, n_steps=config.n_steps)
        return mechanism


def build_default_domain(config: DomainConfig | None = None) -> NegotiationDomain:
    """Construct the baseline multi-issue domain used for training experiments."""

    if config is None:
        config = DomainConfig()

    issues = [
        make_issue(name="price", values=config.price_levels),
        make_issue(name="quantity", values=config.quantity_range),
        make_issue(name="delivery", values=config.delivery_levels),
    ]

    mechanism = SAOMechanism(issues=issues, n_steps=1)
    outcome_space = mechanism.outcome_space

    seller_ufun = LinearAdditiveUtilityFunction(
        values={
            "price": IdentityFun(),
            "quantity": LinearFun(0.25),
            "delivery": AffineFun(-1.0, bias=10.0),
        },
        outcome_space=outcome_space,
    )

    buyer_ufun = LinearAdditiveUtilityFunction(
        values={
            "price": AffineFun(-1.0, bias=float(config.price_levels - 1)),
            "quantity": LinearFun(0.35),
            "delivery": IdentityFun(),
        },
        outcome_space=outcome_space,
    )

    return NegotiationDomain(
        issues=issues,
        buyer_utility=buyer_ufun,
        seller_utility=seller_ufun,
    )
