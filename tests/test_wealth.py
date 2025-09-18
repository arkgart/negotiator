from __future__ import annotations

from ergodic_negotiator.config import WealthParameters
from ergodic_negotiator.wealth import TimeAverageWealthTracker


def constant_growth(outcome, role):  # noqa: D401 - test helper
    return 0.05 if role == "buyer" else -0.02


def test_time_average_updates():
    params = WealthParameters(initial_wealth=1.0, time_cost=0.01)
    tracker = TimeAverageWealthTracker(params, constant_growth, role="buyer")
    tracker.reset()
    initial = tracker.wealth
    tracker.apply_time_cost()
    tracker.apply_deal((0, 0, 0))
    assert tracker.wealth > initial * 0.9
    assert tracker.history[-1].reason == "deal"
