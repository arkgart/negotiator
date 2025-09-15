from negotiator.config import EconomicConfig, WealthConfig
from negotiator.wealth import WealthManager


def test_wealth_process_time_penalty():
    manager = WealthManager(
        role="buyer",
        economic=EconomicConfig(),
        wealth_cfg=WealthConfig(initial_wealth=100.0, time_penalty=0.01),
    )
    before = manager.wealth
    log_r = manager.apply_time_penalty()
    assert manager.wealth < before
    assert log_r < 0
    assert manager.time_average_growth() <= 0


def test_deal_updates_wealth():
    manager = WealthManager(
        role="seller",
        economic=EconomicConfig(capital_scale=50.0),
        wealth_cfg=WealthConfig(initial_wealth=50.0, time_penalty=0.0),
    )
    outcome = (15, 10, 2)
    log_r = manager.apply_deal(outcome)
    assert manager.wealth != 50.0
    assert manager.wealth_history[-1] == manager.wealth
    assert len(manager.log_returns) >= 1
    assert isinstance(log_r, float)
