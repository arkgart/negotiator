"""Custom NegMAS negotiators powered by reinforcement learning policies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from negmas.sao import ResponseType, SAONegotiator, SAOState
from negmas.preferences.inv_ufun import PresortingInverseUtilityFunction

from .policies import PolicyInterface


@dataclass(slots=True)
class NegotiatorTelemetry:
    """Light-weight structure capturing negotiator side-information."""

    last_partner_offer: Optional[Sequence[int | float]] = None
    last_partner_utility: float = 0.0
    last_self_offer: Optional[Sequence[int | float]] = None
    last_self_utility: float = 0.0


class RLNegotiator(SAONegotiator):
    """A NegMAS negotiator delegating decisions to an external RL policy."""

    def __init__(
        self,
        policy: PolicyInterface,
        *,
        k_candidates: int = 32,
        max_rounds: int = 40,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.policy = policy
        self.k_candidates = k_candidates
        self.max_rounds = max_rounds
        self._inv: Optional[PresortingInverseUtilityFunction] = None
        self._umin = 0.0
        self._umax = 1.0
        self._bounds: list[tuple[float, float]] = []
        self.telemetry = NegotiatorTelemetry()
        self._partner_first: Optional[Sequence[int | float]] = None
        self._log_wealth: float = 0.0

    # ------------------------------------------------------------------
    # utility helpers
    # ------------------------------------------------------------------
    def _ensure_inverse(self) -> PresortingInverseUtilityFunction:
        if self._inv is None and self.ufun is not None:
            self._inv = PresortingInverseUtilityFunction(self.ufun)
            self._inv.init()
        return self._inv  # type: ignore[return-value]

    def _normalize_utility(self, value: float) -> float:
        rng = self._umax - self._umin
        if rng <= 1e-12:
            return 0.0
        return float(np.clip((value - self._umin) / rng, 0.0, 1.0))

    def _normalize_outcome(self, outcome: Optional[Sequence[int | float]]) -> list[float]:
        if outcome is None:
            return [0.0] * len(self._bounds)
        return [
            0.0 if hi <= lo else float(np.clip((float(v) - lo) / (hi - lo), 0.0, 1.0))
            for v, (lo, hi) in zip(outcome, self._bounds)
        ]

    # ------------------------------------------------------------------
    # hooks from NegMAS
    # ------------------------------------------------------------------
    def update_log_wealth(self, log_wealth: float) -> None:
        self._log_wealth = log_wealth

    def set_issue_bounds(self, bounds: Iterable[tuple[float, float]]) -> None:
        self._bounds = [(float(lo), float(hi)) for lo, hi in bounds]

    def on_preferences_changed(self, changes=None):  # type: ignore[override]
        super().on_preferences_changed(changes)
        if self.ufun is None:
            return
        self._inv = PresortingInverseUtilityFunction(self.ufun)
        self._inv.init()
        self._umin, self._umax = self.ufun.minmax()
        if self.nmi is not None:
            try:
                issues: Iterable = self.nmi.outcome_space.issues  # type: ignore[attr-defined]
            except AttributeError:
                issues = []
            self.set_issue_bounds(
                (
                    float(getattr(issue, "min_value", 0.0)),
                    float(getattr(issue, "max_value", 1.0)),
                )
                for issue in issues
            )
        self.telemetry = NegotiatorTelemetry()
        self._partner_first = None

    def build_observation(self, state: SAOState) -> np.ndarray:
        step_fraction = 0.0 if self.max_rounds <= 0 else state.step / self.max_rounds
        relative_time = getattr(state, "relative_time", step_fraction)
        current_offer = state.current_offer
        current_util = 0.0 if current_offer is None or self.ufun is None else float(self.ufun(current_offer))
        current_norm = self._normalize_utility(current_util)
        last_self_norm = self._normalize_utility(self.telemetry.last_self_utility)
        obs = [
            float(relative_time),
            float(step_fraction),
            current_norm,
            last_self_norm,
            float(self._log_wealth),
        ]
        obs.extend(self._normalize_outcome(current_offer))
        obs.extend(self._normalize_outcome(self.telemetry.last_self_offer))
        obs.extend(self._normalize_outcome(self.telemetry.last_partner_offer))
        return np.asarray(obs, dtype=np.float32)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:  # type: ignore[override]
        _ = source
        offer = state.current_offer
        if offer is not None:
            self.telemetry.last_partner_offer = offer
            if self.ufun is not None:
                self.telemetry.last_partner_utility = float(self.ufun(offer))
            if self._partner_first is None:
                self._partner_first = offer
        obs = self.build_observation(state)
        if offer is None or self.ufun is None:
            return ResponseType.REJECT_OFFER
        my_util = float(self.ufun(offer))
        score = self.policy.act_accept(obs, self._normalize_utility(my_util))
        if score > 0:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _candidate_offers(self, target: float) -> list[Sequence[int | float]]:
        inv = self._ensure_inverse()
        candidates: list[Sequence[int | float]] = []
        idx = 0
        while True:
            outcome = inv.outcome_at(idx)
            if outcome is None:
                break
            if self.ufun is not None:
                value = float(self.ufun(outcome))
                if self._normalize_utility(value) >= target or not candidates:
                    candidates.append(outcome)
            idx += 1
            if len(candidates) >= self.k_candidates or idx >= self.k_candidates * 4:
                break
        if not candidates:
            best = inv.best()
            if best is not None:
                candidates.append(best)
        return candidates

    def propose(self, state: SAOState):  # type: ignore[override]
        obs = self.build_observation(state)
        target, selector = self.policy.act(obs)
        candidates = self._candidate_offers(target)
        if self._partner_first is not None:
            partner = self._partner_first

            def distance(outcome):
                return sum((float(a) - float(b)) ** 2 for a, b in zip(outcome, partner))

            candidates.sort(key=distance)
        if not candidates:
            raise RuntimeError("No candidates available for proposal")
        chosen_idx = int(np.clip(selector, 0, len(candidates) - 1))
        offer = candidates[chosen_idx]
        if self.ufun is not None:
            self.telemetry.last_self_utility = float(self.ufun(offer))
        self.telemetry.last_self_offer = offer
        return offer


__all__ = ["RLNegotiator", "NegotiatorTelemetry"]
