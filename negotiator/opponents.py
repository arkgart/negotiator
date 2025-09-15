"""Hand-crafted opponent negotiators for training diversity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from negmas.outcomes import Outcome
from negmas.preferences.inv_ufun import PresortingInverseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType


@dataclass
class HeuristicConfig:
    """Shape parameters for scripted opponents."""

    concession_exponent: float = 1.0
    acceptance_slack: float = 0.05
    candidate_pool: int = 32


class HeuristicNegotiator(SAONegotiator):
    """Simple negotiator that concedes according to a power schedule."""

    def __init__(self, config: HeuristicConfig, name: str | None = None) -> None:
        super().__init__(name=name)
        self.config = config
        self._inv: Optional[PresortingInverseUtilityFunction] = None
        self._min_util: float = 0.0
        self._max_util: float = 1.0
        self._partner_first: Optional[Outcome] = None

    def on_preferences_changed(self, changes=None):  # type: ignore[override]
        super().on_preferences_changed(changes)
        if self.ufun is None or self.nmi is None:
            return
        self._inv = PresortingInverseUtilityFunction(self.ufun, self.nmi.outcome_space)
        best = self._inv.best()
        worst = self._inv.worst()
        if best is not None:
            self._max_util = float(self.ufun(best))
        if worst is not None:
            self._min_util = float(self.ufun(worst))

    def on_negotiation_start(self, state: SAOState):  # type: ignore[override]
        super().on_negotiation_start(state)
        self._partner_first = None

    def respond(
        self,
        state: SAOState,
        offer: Outcome | None = None,
        nid: str | None = None,
        source: str | None = None,
    ) -> ResponseType:  # type: ignore[override]
        offer = offer or state.current_offer
        if offer is None:
            return ResponseType.NO_RESPONSE
        if self._partner_first is None and offer is not None:
            self._partner_first = offer
        if self.ufun is None:
            return ResponseType.REJECT_OFFER
        relative_time = float(getattr(state, "relative_time", 0.0) or 0.0)
        util = float(self.ufun(offer))
        span = max(1e-6, self._max_util - self._min_util)
        normalized = (util - self._min_utility()) / span
        threshold = max(0.0, min(1.0, 1.0 - relative_time - self.config.acceptance_slack))
        return ResponseType.ACCEPT_OFFER if normalized >= threshold else ResponseType.REJECT_OFFER

    def propose(self, state: SAOState) -> Outcome:  # type: ignore[override]
        if self._inv is None:
            raise RuntimeError("Inverse utility function not initialised")
        relative_time = float(getattr(state, "relative_time", 0.0) or 0.0)
        exponent = max(self.config.concession_exponent, 1e-3)
        rel_target = (1.0 - relative_time) ** exponent
        rel_target = max(0.0, min(1.0, rel_target))
        threshold = self._min_utility() + rel_target * (self._max_util - self._min_utility())
        candidates = list(self._inv.outcomes_above(threshold, k=self.config.candidate_pool))
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            best = self._inv.best()
            if best is None:
                raise RuntimeError("No offers available")
            return best
        if self._partner_first is not None:
            candidates.sort(key=lambda o: self._sq_distance(o, self._partner_first))
        return candidates[0]

    def _min_utility(self) -> float:
        return self._min_util

    def _sq_distance(self, a: Outcome, b: Outcome) -> float:
        return float(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def make_time_conceder(name: str = "time-conceder") -> HeuristicNegotiator:
    return HeuristicNegotiator(HeuristicConfig(concession_exponent=1.0, acceptance_slack=0.15), name=name)


def make_boulware(name: str = "boulware") -> HeuristicNegotiator:
    return HeuristicNegotiator(HeuristicConfig(concession_exponent=4.0, acceptance_slack=0.05), name=name)


def make_tit_for_tat(name: str = "tit-for-tat") -> HeuristicNegotiator:
    return HeuristicNegotiator(HeuristicConfig(concession_exponent=2.0, acceptance_slack=0.1), name=name)
