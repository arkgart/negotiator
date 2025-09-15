from negotiator.config import EconomicConfig, NegotiationConfig, TrainingConfig, WealthConfig
from negotiator.domain import build_default_domain
from negotiator.env import NegotiationEnvironment
from negotiator.metrics import summarise


def test_single_episode_summary():
    domain = build_default_domain()
    negotiation_cfg = NegotiationConfig(n_steps=10, role="buyer", k_candidates=16)
    training_cfg = TrainingConfig(
        batch_episodes=1,
        max_iterations=1,
        log_every=1,
        opponent_schedule=["time_based"],
    )
    wealth_cfg = WealthConfig(initial_wealth=80.0, time_penalty=0.001)
    economic_cfg = EconomicConfig(capital_scale=40.0)
    env = NegotiationEnvironment(domain, negotiation_cfg, training_cfg, wealth_cfg, economic_cfg)

    result = env.run_episode("time_based")
    assert result.stats.final_wealth > 0
    summary = summarise([result.stats])
    assert summary.episodes == 1
    assert 0.0 <= summary.agreement_rate <= 1.0
