"""Domain helpers for building negotiation outcome spaces and utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from negmas import Issue, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun

from .config import DomainParameters

GrowthFn = Callable[[Sequence[int | float]], float]


@dataclass(slots=True)
class DomainArtifacts:
    """Artifacts describing a concrete negotiation domain instance."""

    params: DomainParameters
    issues: list[Issue]

    def create_utilities(self, outcome_space) -> tuple[
        LinearAdditiveUtilityFunction,
        LinearAdditiveUtilityFunction,
        GrowthFn,
        GrowthFn,
    ]:
        buyer = LinearAdditiveUtilityFunction(
            values={
                "price": AffineFun(-1.0, bias=self.params.buyer_price_bias),
                "quantity": LinearFun(self.params.buyer_quantity_slope),
                "delivery": AffineFun(self.params.buyer_delivery_slope, bias=self.params.buyer_delivery_bias),
            },
            outcome_space=outcome_space,
        )
        seller = LinearAdditiveUtilityFunction(
            values={
                "price": IdentityFun(),
                "quantity": LinearFun(self.params.seller_quantity_slope),
                "delivery": AffineFun(self.params.seller_delivery_slope, bias=self.params.seller_delivery_bias),
            },
            outcome_space=outcome_space,
        )
        buyer_min, buyer_max = buyer.minmax()
        seller_min, seller_max = seller.minmax()

        def buyer_growth(outcome: Sequence[int | float]) -> float:
            return _normalized_growth(outcome, buyer, buyer_min, buyer_max, self.params.buyer_growth_scale)

        def seller_growth(outcome: Sequence[int | float]) -> float:
            return _normalized_growth(outcome, seller, seller_min, seller_max, self.params.seller_growth_scale)

        return buyer, seller, buyer_growth, seller_growth


def _normalized_growth(
    outcome: Sequence[int | float],
    ufun: LinearAdditiveUtilityFunction,
    min_u: float,
    max_u: float,
    scale: float,
) -> float:
    """Map utility into a multiplicative growth term in ``[-0.95, 1.0]``."""

    if max_u <= min_u:
        return 0.0
    normalized = (float(ufun(outcome)) - min_u) / (max_u - min_u)
    centered = (normalized - 0.5) * 2.0  # [-1, 1]
    growth = np.clip(centered * scale, -0.95, 1.0)
    return float(growth)


def build_domain(params: DomainParameters) -> DomainArtifacts:
    """Create a :class:`DomainArtifacts` instance with issue definitions."""

    price_issue = make_issue(name="price", values=params.price_levels)
    quantity_issue = make_issue(name="quantity", values=params.quantity_range)
    delivery_issue = make_issue(name="delivery", values=params.delivery_levels)
    return DomainArtifacts(params=params, issues=[price_issue, quantity_issue, delivery_issue])


def issue_bounds(issues: Iterable[Issue]) -> list[tuple[float, float]]:
    """Extract ``(min, max)`` bounds for each issue for normalisation."""

    bounds: list[tuple[float, float]] = []
    for issue in issues:
        lo = float(getattr(issue, "min_value", 0.0))
        hi = float(getattr(issue, "max_value", lo))
        if hi <= lo:
            hi = lo + 1.0
        bounds.append((lo, hi))
    return bounds


__all__ = [
    "DomainArtifacts",
    "build_domain",
    "issue_bounds",
]
