"""Environment wrapper around NegMAS sessions for RL training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

from negmas.sao import SAONegotiator

from .agents import EpisodeStats, RLNegotiator
from .config import EconomicConfig, NegotiationConfig, TrainingConfig, WealthConfig
from .domain import NegotiationDomain
from .metrics import EvaluationSummary, summarise
from .opponents import make_boulware, make_time_conceder, make_tit_for_tat


@dataclass
class EpisodeResult:
    stats: EpisodeStats
    opponent: str


class NegotiationEnvironment:
    """High-level helper that schedules episodes against multiple opponents."""

    def __init__(
        self,
        domain: NegotiationDomain,
        negotiation_cfg: NegotiationConfig,
        training_cfg: TrainingConfig,
        wealth_cfg: WealthConfig,
        economic_cfg: EconomicConfig,
    ) -> None:
        self.domain = domain
        self.neg_cfg = negotiation_cfg
        self.training_cfg = training_cfg
        self.wealth_cfg = wealth_cfg
        self.economic_cfg = economic_cfg
        self.agent = RLNegotiator(
            negotiation_cfg=negotiation_cfg,
            training_cfg=training_cfg,
            wealth_cfg=wealth_cfg,
            economic_cfg=economic_cfg,
            issue_count=len(domain.issues),
            name="ergodic-rl",
        )
        self.opponents: Dict[str, Callable[..., SAONegotiator]] = {
            "time_based": make_time_conceder,
            "boulware": make_boulware,
            "tit_for_tat": make_tit_for_tat,
        }
        self.random = random.Random(training_cfg.seed)

    def available_opponents(self) -> Sequence[str]:
        return tuple(self.opponents.keys())

    def run_episode(self, opponent_name: str) -> EpisodeResult:
        opponent_factory = self.opponents.get(opponent_name)
        if opponent_factory is None:
            raise KeyError(f"Unknown opponent '{opponent_name}'")
        opponent = opponent_factory(name=f"opponent-{opponent_name}")
        mechanism = self.domain.create_mechanism(self.neg_cfg)
        if self.neg_cfg.role == "buyer":
            mechanism.add(self.agent, ufun=self.domain.buyer_utility)
            mechanism.add(opponent, ufun=self.domain.seller_utility)
        else:
            mechanism.add(self.agent, ufun=self.domain.seller_utility)
            mechanism.add(opponent, ufun=self.domain.buyer_utility)
        mechanism.run()
        if self.agent.last_episode is None:
            raise RuntimeError("Agent did not record episode stats")
        return EpisodeResult(stats=self.agent.last_episode, opponent=opponent_name)

    def run_batch(self, opponents: Sequence[str] | None = None) -> List[EpisodeResult]:
        opponents = opponents or self.training_cfg.opponent_schedule
        results: List[EpisodeResult] = []
        for name in opponents:
            results.append(self.run_episode(name))
        return results

    def iterate_training(self, iterations: int | None = None) -> Iterable[EvaluationSummary]:
        iterations = iterations or self.training_cfg.max_iterations
        schedule = list(self.training_cfg.opponent_schedule)
        for iteration in range(iterations):
            self.random.shuffle(schedule)
            batch: List[EpisodeResult] = []
            for _ in range(self.training_cfg.batch_episodes):
                opponent = schedule[_ % len(schedule)]
                batch.append(self.run_episode(opponent))
            summary = summarise(result.stats for result in batch)
            yield summary
