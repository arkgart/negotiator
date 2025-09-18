"""Gymnasium environment exposing NegMAS negotiations with ergodic rewards."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from negmas import SAOMechanism
from negmas.sao import SAONegotiator, TimeBasedConcedingNegotiator

from .agents import RLNegotiator
from .config import NegotiationEnvConfig
from .domain import DomainArtifacts, build_domain, issue_bounds
from .policies import ExternalPolicy
from .wealth import TimeAverageWealthTracker


DEFAULT_ACTION_LOW = 0.0
DEFAULT_ACTION_HIGH = 1.0


@dataclass(slots=True)
class NegotiationResult:
    agreement: Optional[tuple[int, ...]]
    steps: int
    wealth: float
    log_wealth: float


class ErgodicNegotiationEnv(gym.Env[np.ndarray, np.ndarray]):
    """An alternating-offers negotiation environment with time-average rewards."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[NegotiationEnvConfig] = None) -> None:
        super().__init__()
        self.config = config or NegotiationEnvConfig()
        self._domain: DomainArtifacts = build_domain(self.config.domain)
        self._action_space = spaces.Box(
            low=DEFAULT_ACTION_LOW,
            high=DEFAULT_ACTION_HIGH,
            shape=(3,),
            dtype=np.float32,
        )
        self.action_space = self._action_space
        obs_dim = 5 + 3 * len(self._domain.issues)
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.observation_space = self._observation_space
        self._mechanism: Optional[SAOMechanism] = None
        self._rl_negotiator: Optional[RLNegotiator] = None
        self._opponent: Optional[SAONegotiator] = None
        self._policy: Optional[ExternalPolicy] = None
        self._wealth: Optional[TimeAverageWealthTracker] = None
        self._last_observation: Optional[np.ndarray] = None
        self._steps = 0
        self._done = False

    # ------------------------------------------------------------------
    # gym interface
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._setup_session()
        assert self._rl_negotiator is not None
        state = self._mechanism.state  # type: ignore[union-attr]
        obs = self._rl_negotiator.build_observation(state)
        self._last_observation = obs
        self._done = False
        self._steps = 0
        info = {
            "state": state,
            "wealth": self._wealth.wealth if self._wealth else math.nan,
            "log_wealth": self._wealth.log_wealth if self._wealth else math.nan,
        }
        return obs, info

    def step(self, action: np.ndarray):  # type: ignore[override]
        if self._done:
            raise RuntimeError("Environment must be reset before stepping again")
        if self._policy is None or self._mechanism is None or self._wealth is None:
            raise RuntimeError("Environment not reset")
        action = np.asarray(action, dtype=np.float32)
        self._policy.set_next_action(action)
        state = self._mechanism.step()
        self._steps += 1

        reward = 0.0
        reward += self._wealth.apply_time_cost().log_delta
        terminated = False
        if state.agreement is not None:
            reward += self._wealth.apply_deal(state.agreement).log_delta
            terminated = True
        elif not state.running:
            terminated = True
        if self._wealth.ruined():
            reward += self.config.wealth.ruin_penalty
            terminated = True

        self._rl_negotiator.update_log_wealth(self._wealth.log_wealth)
        obs = self._rl_negotiator.build_observation(state)
        self._last_observation = obs
        truncated = False
        if self._steps >= self.config.max_rounds and not terminated:
            truncated = True
            terminated = True
        self._done = terminated
        info = {
            "state": state,
            "wealth": self._wealth.wealth,
            "log_wealth": self._wealth.log_wealth,
            "agreement": state.agreement,
        }
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    def _setup_session(self) -> None:
        opponent_factory = self.config.opponent_builder or self._default_opponent
        opponent = opponent_factory()
        policy = ExternalPolicy(k_candidates=self.config.k_candidates, noise=self.config.action_noise)
        mechanism = SAOMechanism(issues=self._domain.issues, n_steps=self.config.max_rounds)
        buyer_ufun, seller_ufun, buyer_growth, seller_growth = self._domain.create_utilities(mechanism.outcome_space)
        if self.config.role == "buyer":
            rl_ufun = buyer_ufun
            opponent_ufun = seller_ufun
            compute_growth = buyer_growth
        else:
            rl_ufun = seller_ufun
            opponent_ufun = buyer_ufun
            compute_growth = seller_growth
        rl_negotiator = RLNegotiator(policy=policy, k_candidates=self.config.k_candidates, max_rounds=self.config.max_rounds, name="rl")
        rl_negotiator.set_issue_bounds(issue_bounds(self._domain.issues))
        opponent.name = opponent.name or "baseline"
        mechanism.add(rl_negotiator, ufun=rl_ufun)
        mechanism.add(opponent, ufun=opponent_ufun)
        wealth_tracker = TimeAverageWealthTracker(
            self.config.wealth,
            lambda outcome, role: compute_growth(outcome),
            role=self.config.role,
        )
        wealth_tracker.reset()
        rl_negotiator.update_log_wealth(wealth_tracker.log_wealth)

        self._mechanism = mechanism
        self._rl_negotiator = rl_negotiator
        self._policy = policy
        self._opponent = opponent
        self._wealth = wealth_tracker

    @staticmethod
    def _default_opponent() -> SAONegotiator:
        return TimeBasedConcedingNegotiator(name="time-based")

    # ------------------------------------------------------------------
    # convenience API
    # ------------------------------------------------------------------
    def last_observation(self) -> np.ndarray:
        if self._last_observation is None:
            raise RuntimeError("Environment not reset")
        return self._last_observation

    def negotiation_result(self) -> NegotiationResult:
        if self._mechanism is None or self._wealth is None:
            raise RuntimeError("Environment not reset")
        state = self._mechanism.state
        return NegotiationResult(
            agreement=state.agreement,
            steps=self._steps,
            wealth=self._wealth.wealth,
            log_wealth=self._wealth.log_wealth,
        )


__all__ = ["ErgodicNegotiationEnv", "NegotiationResult"]
