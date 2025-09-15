"""Training loop coordinating NegMAS sessions and PPO updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..agents.rl_negotiator import DecisionRecord, RLNegotiator
from ..config.schema import NegotiationConfig
from ..domain.builder import DomainFactory
from ..opponents.baselines import OpponentPool, build_opponent
from ..policies.policy import PPOPolicy, Policy
from ..utils.wealth import WealthLedger, WealthModel

__all__ = ["AgentProfile", "NegotiationTrainer", "EpisodeResult"]


@dataclass
class EpisodeResult:
    episode: int
    role: str
    agreement: bool
    reward_sum: float
    log_growth: float
    wealth: float
    ruin: bool


@dataclass
class AgentProfile:
    name: str
    role: str
    reward_mode: str
    policy: Policy
    negotiator: RLNegotiator
    wealth: WealthLedger
    episodes: int = 0

    def ready(self) -> bool:
        return self.policy.ready()


class NegotiationTrainer:
    """Orchestrates multi-agent negotiation training."""

    def __init__(self, config: NegotiationConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        self.domain_factory = DomainFactory(config.domain)
        self.issue_names = [spec.name for spec in config.domain.issues]
        self.obs_dim = 5 + 2 * len(self.issue_names)
        self.action_dim = 3
        self.opponent_pool = OpponentPool(config.opponents)
        self.roles = list(config.roles.keys())
        self.role_agents: Dict[str, List[AgentProfile]] = {role: [] for role in self.roles}
        self.agents: Dict[str, AgentProfile] = {}
        self.wealth_models = {
            role: WealthModel(config.wealth[role], config.economy, self.issue_names)
            for role in self.roles
        }
        self.episode = 0

    def register_agent(self, name: str, role: str, reward_mode: str = "tag") -> AgentProfile:
        if role not in self.roles:
            raise KeyError(f"Unknown role {role}. Known roles: {self.roles}")
        policy = PPOPolicy(self.obs_dim, self.action_dim, self.config.policy, device=self.device)
        negotiator = RLNegotiator(policy=policy, role=role, name=name)
        wealth_model = self.wealth_models[role]
        wealth = WealthLedger(model=wealth_model)
        profile = AgentProfile(
            name=name,
            role=role,
            reward_mode=reward_mode.lower(),
            policy=policy,
            negotiator=negotiator,
            wealth=wealth,
        )
        self.agents[name] = profile
        self.role_agents[role].append(profile)
        return profile

    # ------------------------------------------------------------------
    def train(self, total_episodes: Optional[int] = None) -> List[EpisodeResult]:
        limit = total_episodes or self.config.training.total_episodes
        results: List[EpisodeResult] = []
        for _ in range(limit):
            for role in self.roles:
                for profile in self.role_agents[role]:
                    opponent_role = self._select_opponent_role(role)
                    opponent_spec = self.opponent_pool.sample(opponent_role)
                    result = self._play_episode(profile, opponent_role, opponent_spec)
                    results.append(result)
                    if profile.ready():
                        profile.policy.update()
        return results

    # ------------------------------------------------------------------
    def _select_opponent_role(self, role: str) -> str:
        others = [candidate for candidate in self.roles if candidate != role]
        if not others:
            return role
        return np.random.choice(others)

    def _play_episode(self, profile: AgentProfile, opponent_role: str, opponent_spec) -> EpisodeResult:
        self.episode += 1
        mechanism = self.domain_factory.make_mechanism()
        utilities = self.domain_factory.build_utilities(mechanism, self.config.roles)
        profile.negotiator.set_training(True)
        profile.negotiator.set_log_wealth(profile.wealth.log_wealth or 0.0)
        mechanism.add(profile.negotiator, ufun=utilities[profile.role])
        opponent = build_opponent(opponent_spec)
        mechanism.add(opponent, ufun=utilities[opponent_role])
        final_state = mechanism.run()
        outcome = final_state.agreement
        records = profile.negotiator.consume_records()
        reward_sum, log_growth, ruined = self._assign_rewards(profile, records, outcome)
        profile.episodes += 1
        return EpisodeResult(
            episode=self.episode,
            role=profile.role,
            agreement=outcome is not None,
            reward_sum=reward_sum,
            log_growth=log_growth,
            wealth=profile.wealth.wealth or 0.0,
            ruin=ruined,
        )

    def _assign_rewards(
        self,
        profile: AgentProfile,
        records: List[DecisionRecord],
        outcome,
    ) -> tuple[float, float, bool]:
        total_reward = 0.0
        total_log = 0.0
        ruined = False
        for idx, record in enumerate(records):
            time_log = profile.wealth.step_time()
            reward = time_log
            done = idx == len(records) - 1
            if done:
                if outcome is not None:
                    delta_log, cashflow, ruin_flag = profile.wealth.step_deal(outcome, profile.role)
                    total_log += delta_log
                    if profile.reward_mode == "tag":
                        reward += delta_log
                    else:
                        reward += cashflow
                    ruined = ruined or ruin_flag
                else:
                    ruined = ruined or profile.wealth.wealth < profile.wealth.model.spec.ruin_threshold
            record.set_reward(reward, done)
            profile.policy.store(record)
            total_reward += reward
            total_log += time_log
        profile.negotiator.set_log_wealth(profile.wealth.log_wealth or 0.0)
        return total_reward, total_log, ruined
