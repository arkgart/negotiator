"""Negotiator implementations including a reinforcement learning strategist."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from negmas.outcomes import Outcome
from negmas.preferences.inv_ufun import PresortingInverseUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType

from .config import EconomicConfig, NegotiationConfig, TrainingConfig, WealthConfig
from .policies import PolicyAction, PolicyController
from .wealth import WealthManager


@dataclass
class EpisodeStats:
    """Summary of a single negotiation episode."""

    agreement: Optional[Outcome]
    final_wealth: float
    time_average: float
    training_metrics: dict
    log_returns: List[float]
    wealth_path: List[float]


class RLNegotiator(SAONegotiator):
    """A negotiator powered by an actor-critic policy and ergodic rewards."""

    def __init__(
        self,
        negotiation_cfg: NegotiationConfig,
        training_cfg: TrainingConfig,
        wealth_cfg: WealthConfig,
        economic_cfg: EconomicConfig,
        issue_count: int,
        device: Optional[object] = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.neg_cfg = negotiation_cfg
        self.training_cfg = training_cfg
        self.wealth_manager = WealthManager(role=negotiation_cfg.role, economic=economic_cfg, wealth_cfg=wealth_cfg)
        obs_dim = 7 + 2 * issue_count
        self.controller = PolicyController(obs_dim=obs_dim, k_candidates=negotiation_cfg.k_candidates, cfg=training_cfg)
        if device is not None:
            self.controller.to(device)
        self.issue_count = issue_count
        self._issue_bounds: List[Tuple[float, float]] = [(0.0, 1.0)] * issue_count
        self._inv: Optional[PresortingInverseUtilityFunction] = None
        self._partner_first: Optional[Outcome] = None
        self._last_offer_made: Optional[Outcome] = None
        self._last_offer_received: Optional[Outcome] = None
        self._last_offer_step: Optional[int] = None
        self._last_response_step: Optional[int] = None
        self._min_utility: float = 0.0
        self._max_utility: float = 1.0
        self.last_episode: Optional[EpisodeStats] = None
        self._sorted_outcomes: List[Outcome] = []

    # Lifecycle hooks -----------------------------------------------------
    def on_preferences_changed(self, changes=None):  # type: ignore[override]
        super().on_preferences_changed(changes)
        if self.nmi is None or self.ufun is None:
            return
        self._inv = PresortingInverseUtilityFunction(self.ufun, self.nmi.outcome_space)
        best = self._inv.best()
        worst = self._inv.worst()
        if best is not None:
            self._max_utility = float(self.ufun(best))
        if worst is not None:
            self._min_utility = float(self.ufun(worst))

    def on_negotiation_start(self, state: SAOState):  # type: ignore[override]
        super().on_negotiation_start(state)
        self.controller.start_episode()
        self.wealth_manager.reset()
        self._partner_first = None
        self._last_offer_made = None
        self._last_offer_received = state.current_offer
        self._last_offer_step = None
        self._last_response_step = None
        if self.nmi is not None:
            self._issue_bounds = [
                (float(issue.min_value), float(issue.max_value))
                for issue in self.nmi.outcome_space.issues
            ]
            if self.ufun is not None:
                all_outcomes = list(self.nmi.outcome_space.enumerate())
                all_outcomes.sort(key=lambda o: float(self.ufun(o)), reverse=True)
                self._sorted_outcomes = all_outcomes
                if all_outcomes:
                    self._max_utility = float(self.ufun(all_outcomes[0]))
                    self._min_utility = float(self.ufun(all_outcomes[-1]))
        if self._inv is None and self.ufun is not None and self.nmi is not None:
            self._inv = PresortingInverseUtilityFunction(self.ufun, self.nmi.outcome_space)

    def respond(
        self,
        state: SAOState,
        offer: Optional[Outcome] = None,
        nid: str | None = None,
        source: str | None = None,
    ) -> ResponseType:  # type: ignore[override]
        offer = offer or state.current_offer
        self._last_offer_received = offer
        if self._partner_first is None and offer is not None:
            self._partner_first = offer
        observation = self._response_observation(state, offer)
        action, idx = self.controller.act(observation, mode="respond")
        self._last_response_step = idx
        accept = bool(action.accept)
        return ResponseType.ACCEPT_OFFER if accept else ResponseType.REJECT_OFFER

    def propose(self, state: SAOState, dest: Optional[str] = None) -> Outcome:  # type: ignore[override]
        observation = self._offer_observation(state)
        action, idx = self.controller.act(observation, mode="propose")
        self._last_offer_step = idx
        time_reward = self.wealth_manager.apply_time_penalty()
        self.controller.add_reward(idx, time_reward)
        offer = self._select_offer(action, state)
        self._last_offer_made = offer
        return offer

    def on_negotiation_end(self, state: SAOState):  # type: ignore[override]
        super().on_negotiation_end(state)
        agreement = state.agreement
        if agreement and self._last_response_step is not None:
            reward = self.wealth_manager.apply_deal(agreement)
            self.controller.add_reward(self._last_response_step, reward)
        elif not agreement and self._last_offer_step is not None:
            self.controller.add_reward(self._last_offer_step, self.neg_cfg.timeout_penalty)
        metrics = self.controller.finish_episode()
        self.last_episode = EpisodeStats(
            agreement=agreement,
            final_wealth=self.wealth_manager.wealth,
            time_average=self.wealth_manager.time_average_growth(),
            training_metrics=metrics,
            log_returns=list(self.wealth_manager.log_returns),
            wealth_path=list(self.wealth_manager.wealth_history),
        )

    # Internal helpers ----------------------------------------------------
    def _select_offer(self, action: PolicyAction, state: SAOState) -> Outcome:
        rel = max(0.0, min(1.0, action.target if action.target is not None else 0.0))
        threshold = self._min_utility + rel * (self._max_utility - self._min_utility)
        candidates: List[Outcome] = []
        for outcome in self._sorted_outcomes:
            if self.ufun is None:
                continue
            if float(self.ufun(outcome)) >= threshold:
                candidates.append(outcome)
            if len(candidates) >= self.neg_cfg.k_candidates:
                break
        if not candidates and self._sorted_outcomes:
            candidates = self._sorted_outcomes[: max(1, self.neg_cfg.k_candidates)]
        if not candidates and self._inv is not None:
            best = self._inv.best()
            if best is not None:
                candidates = [best]
        if self._partner_first is not None and candidates:
            candidates.sort(key=lambda o: self._sq_distance(o, self._partner_first))
        index = int(np.clip(action.candidate if action.candidate is not None else 0, 0, max(len(candidates) - 1, 0)))
        selected = candidates[index] if candidates else self._inv.best()
        if selected is None:
            raise RuntimeError("Failed to select an offer candidate")
        return selected

    def _sq_distance(self, a: Outcome, b: Outcome) -> float:
        total = 0.0
        for av, bv in zip(a, b):
            total += float(av - bv) ** 2
        return total

    def _offer_observation(self, state: SAOState) -> np.ndarray:
        incoming = state.current_offer
        features = self._base_features(state, incoming)
        features.extend(self._normalised_offer(incoming))
        features.extend(self._normalised_offer(self._last_offer_made))
        return np.asarray(features, dtype=np.float32)

    def _response_observation(self, state: SAOState, offer: Outcome) -> np.ndarray:
        features = self._base_features(state, offer)
        features.extend(self._normalised_offer(offer))
        features.extend(self._normalised_offer(self._last_offer_made))
        return np.asarray(features, dtype=np.float32)

    def _base_features(self, state: SAOState, incoming: Optional[Outcome]) -> List[float]:
        relative_time = float(getattr(state, "relative_time", 0.0) or 0.0)
        log_wealth = math.log(max(self.wealth_manager.wealth, 1e-6))
        scaled_wealth = math.tanh(log_wealth / 5.0)
        incoming_u = self._normalised_utility(incoming)
        own_u = self._normalised_utility(self._last_offer_made)
        diff = incoming_u - own_u
        history = self.wealth_manager.log_returns[-5:]
        avg_history = math.tanh(sum(history) / len(history)) if history else 0.0
        features = [
            relative_time,
            1.0 - relative_time,
            scaled_wealth,
            incoming_u,
            own_u,
            diff,
            avg_history,
        ]
        return features

    def _normalised_offer(self, offer: Optional[Outcome]) -> List[float]:
        if offer is None:
            return [0.0] * self.issue_count
        values: List[float] = []
        for value, (low, high) in zip(offer, self._issue_bounds):
            if high == low:
                values.append(0.0)
            else:
                values.append((float(value) - low) / (high - low))
        return values

    def _normalised_utility(self, offer: Optional[Outcome]) -> float:
        if offer is None or self.ufun is None:
            return 0.0
        span = self._max_utility - self._min_utility
        if span <= 1e-9:
            return 0.0
        util = float(self.ufun(offer))
        return (util - self._min_utility) / span

