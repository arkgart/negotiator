"""Reinforcement learning powered negotiator that wraps a policy network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from negmas.outcomes import Outcome
from negmas.sao import ResponseType, SAONegotiator, SAOState

from ..policies.policy import Policy, PolicyOutput

__all__ = ["DecisionRecord", "RLNegotiator"]


@dataclass
class DecisionRecord:
    """Single policy decision recorded during a negotiation episode."""

    observation: np.ndarray
    action: np.ndarray
    log_prob: float
    value: float
    step: int
    relative_time: float
    phase: str
    my_utility: float
    offer: Optional[Outcome]
    candidate: Optional[Outcome] = None
    reward: float = 0.0
    done: bool = False
    accepted: Optional[bool] = None

    def set_reward(self, reward: float, done: bool) -> None:
        self.reward = float(reward)
        self.done = done


class RLNegotiator(SAONegotiator):
    """A NegMAS negotiator driven by a reinforcement learning policy."""

    def __init__(
        self,
        policy: Policy,
        role: str,
        k_candidates: int = 32,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.policy = policy
        self.role = role
        self.k_candidates = k_candidates
        self.training = True
        self._records: List[DecisionRecord] = []
        self._partner_first: Optional[Outcome] = None
        self._last_offer: Optional[Outcome] = None
        self._ordered_outcomes: List[Tuple[float, Outcome]] = []
        self._umin: float = 0.0
        self._umax: float = 1.0
        self._utility_range: float = 1.0
        self._issue_bounds: List[Tuple[float, float]] = []
        self._log_wealth: float = 0.0

    # ------------------------------------------------------------------
    # lifecycle hooks
    # ------------------------------------------------------------------
    def set_training(self, training: bool) -> None:
        self.training = training

    def set_log_wealth(self, log_wealth: float) -> None:
        self._log_wealth = float(log_wealth)

    def reset_episode(self) -> None:
        self._records.clear()
        self._partner_first = None
        self._last_offer = None
        if self.policy is not None:
            self.policy.reset_episode()

    def consume_records(self) -> List[DecisionRecord]:
        records = list(self._records)
        self._records.clear()
        return records

    # NegMAS hooks ------------------------------------------------------
    def on_negotiation_start(self, state: SAOState) -> None:  # type: ignore[override]
        super().on_negotiation_start(state)
        self.reset_episode()
        self._prepare_outcome_cache()

    def on_preferences_changed(self, changes=None):  # type: ignore[override]
        super().on_preferences_changed(changes)
        if self.nmi is not None and self.ufun is not None:
            self._prepare_outcome_cache()

    # ------------------------------------------------------------------
    # RL observation helpers
    # ------------------------------------------------------------------
    def _prepare_outcome_cache(self) -> None:
        if self.nmi is None or self.ufun is None:
            return
        outcome_space = self.nmi.outcome_space
        self._ordered_outcomes = []
        for outcome in outcome_space.enumerate():
            utility = float(self.ufun(outcome))
            self._ordered_outcomes.append((utility, outcome))
        if not self._ordered_outcomes:
            raise RuntimeError("Outcome space enumeration returned no outcomes")
        self._ordered_outcomes.sort(key=lambda item: item[0], reverse=True)
        self._umax = self._ordered_outcomes[0][0]
        self._umin = self._ordered_outcomes[-1][0]
        self._utility_range = max(1e-6, self._umax - self._umin)
        self._issue_bounds = []
        for issue in outcome_space.issues:
            lo = getattr(issue, "min_value", 0.0)
            hi = getattr(issue, "max_value", 1.0)
            self._issue_bounds.append((float(lo), float(hi)))

    def _normalize_outcome(self, outcome: Optional[Outcome]) -> List[float]:
        values: List[float] = []
        if outcome is None:
            return [0.0 for _ in self._issue_bounds]
        for value, (lo, hi) in zip(outcome, self._issue_bounds):
            if hi <= lo:
                values.append(0.0)
            else:
                values.append((float(value) - lo) / (hi - lo))
        return values

    def _scale_utility(self, utility: float) -> float:
        return (utility - self._umin) / self._utility_range

    def _utility_from_relative(self, rel: float) -> float:
        return self._umin + float(rel) * self._utility_range

    def _build_observation(
        self,
        state: SAOState,
        phase: str,
        reference_offer: Optional[Outcome],
    ) -> np.ndarray:
        rel_time = float(getattr(state, "relative_time", 0.0))
        my_offer = self._last_offer
        my_utility = float(self.ufun(my_offer)) if (self.ufun and my_offer is not None) else 0.0
        partner_utility = float(self.ufun(reference_offer)) if (self.ufun and reference_offer is not None) else 0.0
        obs = [1.0 if phase == "respond" else 0.0]
        obs.append(rel_time)
        obs.append(self._scale_utility(my_utility))
        obs.append(self._scale_utility(partner_utility))
        obs.append(self._log_wealth)
        obs.extend(self._normalize_outcome(reference_offer))
        obs.extend(self._normalize_outcome(my_offer))
        return np.asarray(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Negotiation decisions
    # ------------------------------------------------------------------
    def _record(
        self,
        state: SAOState,
        phase: str,
        observation: np.ndarray,
        output: PolicyOutput,
        offer: Optional[Outcome],
        candidate: Optional[Outcome],
        my_utility: float,
        accepted: Optional[bool] = None,
    ) -> None:
        record = DecisionRecord(
            observation=observation,
            action=output.action,
            log_prob=output.log_prob,
            value=output.value,
            step=int(getattr(state, "step", len(self._records))),
            relative_time=float(getattr(state, "relative_time", 0.0)),
            phase=phase,
            my_utility=my_utility,
            offer=offer,
            candidate=candidate,
            accepted=accepted,
        )
        self._records.append(record)

    def respond(
        self,
        state: SAOState,
        offer=None,
        nid: Optional[str] = None,
        **_: object,
    ) -> ResponseType:  # type: ignore[override]
        if self.ufun is None:
            return ResponseType.REJECT_OFFER
        if offer is None:
            offer = state.current_offer
        if self._partner_first is None and offer is not None:
            self._partner_first = offer
        observation = self._build_observation(state, "respond", offer)
        output = self.policy.act(observation, training=self.training)
        action = output.action
        accept_rel = float(np.clip(action[2], 0.0, 1.0))
        threshold = self._utility_from_relative(accept_rel)
        my_utility = float(self.ufun(offer)) if offer is not None else 0.0
        decision = ResponseType.ACCEPT_OFFER if my_utility >= threshold else ResponseType.REJECT_OFFER
        self._record(state, "respond", observation, output, offer, offer, my_utility, accepted=decision is ResponseType.ACCEPT_OFFER)
        if decision is ResponseType.ACCEPT_OFFER:
            self._last_offer = offer
        return decision

    def propose(self, state: SAOState, dest: Optional[str] = None, **_: object):  # type: ignore[override]
        if self.ufun is None:
            return None
        observation = self._build_observation(state, "propose", state.current_offer)
        output = self.policy.act(observation, training=self.training)
        action = output.action
        concede_rel = float(np.clip(action[0], 0.0, 1.0))
        choice_rel = float(np.clip(action[1], 0.0, 1.0))
        threshold = self._utility_from_relative(concede_rel)
        candidates = self._select_candidates(threshold)
        proposal = self._choose_candidate(candidates, choice_rel)
        my_utility = float(self.ufun(proposal)) if proposal is not None else 0.0
        self._record(state, "propose", observation, output, proposal, proposal, my_utility)
        self._last_offer = proposal
        return proposal

    def _select_candidates(self, threshold: float) -> List[Outcome]:
        candidates: List[Outcome] = [outcome for utility, outcome in self._ordered_outcomes if utility >= threshold]
        if not candidates:
            candidates = [outcome for _, outcome in self._ordered_outcomes]
        return candidates[: self.k_candidates]

    def _choose_candidate(self, candidates: List[Outcome], choice_rel: float) -> Outcome:
        if not candidates:
            return self._ordered_outcomes[0][1]
        if self._partner_first is not None:
            candidates = sorted(candidates, key=lambda outcome: self._distance_sq(outcome, self._partner_first))
        idx = min(len(candidates) - 1, int(choice_rel * len(candidates)))
        return candidates[idx]

    @staticmethod
    def _distance_sq(a: Outcome, b: Outcome) -> float:
        return float(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
