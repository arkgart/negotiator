"""Evaluation helpers for negotiation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .agents import EpisodeStats
from .wealth import ensemble_average_growth


@dataclass
class EvaluationSummary:
    """Aggregated statistics over multiple negotiation episodes."""

    episodes: int
    agreement_rate: float
    avg_final_wealth: float
    mean_time_average: float
    ensemble_growth: float
    ruin_probability: float


def summarise(episodes: Iterable[EpisodeStats]) -> EvaluationSummary:
    episodes = list(episodes)
    if not episodes:
        return EvaluationSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    agreements = np.array([1.0 if ep.agreement else 0.0 for ep in episodes], dtype=np.float64)
    final_wealth = np.array([ep.final_wealth for ep in episodes], dtype=np.float64)
    time_averages = np.array([ep.time_average for ep in episodes], dtype=np.float64)

    wealth_paths: List[List[float]] = [ep.wealth_path for ep in episodes]
    ensemble = ensemble_average_growth(wealth_paths)

    ruin_events = sum(1 for ep in episodes if any(w < 1.0 for w in ep.wealth_path))
    ruin_probability = ruin_events / len(episodes)

    return EvaluationSummary(
        episodes=len(episodes),
        agreement_rate=float(agreements.mean()),
        avg_final_wealth=float(final_wealth.mean()),
        mean_time_average=float(time_averages.mean()),
        ensemble_growth=ensemble,
        ruin_probability=float(ruin_probability),
    )


def rolling_time_average(stats: EpisodeStats, window: int = 5) -> List[float]:
    """Return the moving average of log-returns to visualise ergodicity."""

    logs = stats.log_returns
    if not logs:
        return []
    if window <= 1 or len(logs) < window:
        cumsum = np.cumsum(logs)
        return (cumsum / np.arange(1, len(logs) + 1)).tolist()
    cumsum = np.cumsum(np.insert(logs, 0, 0.0))
    averages = (cumsum[window:] - cumsum[:-window]) / window
    prefix = [averages[0]] * (window - 1)
    return prefix + averages.tolist()
