"""Training driver tying together the environment and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from .agents import EpisodeStats
from .config import EconomicConfig, NegotiationConfig, TrainingConfig, WealthConfig
from .domain import NegotiationDomain, build_default_domain
from .env import EpisodeResult, NegotiationEnvironment
from .metrics import EvaluationSummary, summarise


@dataclass
class TrainingHistory:
    summaries: List[EvaluationSummary] = field(default_factory=list)
    episodes: List[EpisodeStats] = field(default_factory=list)


class NegotiationTrainer:
    """Co-ordinates self-play and evaluation for the RL negotiator."""

    def __init__(
        self,
        negotiation_cfg: NegotiationConfig | None = None,
        training_cfg: TrainingConfig | None = None,
        wealth_cfg: WealthConfig | None = None,
        economic_cfg: EconomicConfig | None = None,
        domain: NegotiationDomain | None = None,
    ) -> None:
        self.negotiation_cfg = negotiation_cfg or NegotiationConfig()
        self.training_cfg = training_cfg or TrainingConfig()
        self.wealth_cfg = wealth_cfg or WealthConfig()
        self.economic_cfg = economic_cfg or EconomicConfig()
        self.domain = domain or build_default_domain()
        self.environment = NegotiationEnvironment(
            domain=self.domain,
            negotiation_cfg=self.negotiation_cfg,
            training_cfg=self.training_cfg,
            wealth_cfg=self.wealth_cfg,
            economic_cfg=self.economic_cfg,
        )

    def train(self, iterations: int | None = None) -> TrainingHistory:
        history = TrainingHistory()
        iterations = iterations or self.training_cfg.max_iterations
        for idx, summary in enumerate(self.environment.iterate_training(iterations), start=1):
            history.summaries.append(summary)
            if idx % self.training_cfg.log_every == 0:
                print(
                    f"Iter {idx}: agreements={summary.agreement_rate:.2f}, "
                    f"time_avg={summary.mean_time_average:.4f}, ensemble={summary.ensemble_growth:.4f}"
                )
        return history

    def evaluate(self, opponents: Sequence[str] | None = None, episodes: int = 10) -> EvaluationSummary:
        opponents = opponents or self.training_cfg.opponent_schedule
        logs: List[EpisodeStats] = []
        for idx in range(episodes):
            name = opponents[idx % len(opponents)]
            result = self.environment.run_episode(name)
            logs.append(result.stats)
        return summarise(logs)

    def run_batch(self, opponents: Sequence[str] | None = None) -> List[EpisodeResult]:
        return self.environment.run_batch(opponents)
