"""The step reward weights each SKU's fill rate by its stockout_penalty."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models import PharmaAction
from server.my_env_environment import PharmaEnvironment
from tasks import SKU_CATALOGUE, TEST_SEEDS, compute_final_score, compute_step_reward, weighted_fill

PENALTIES = {sku: spec["stockout_penalty"] for sku, spec in SKU_CATALOGUE.items()}


def _state(fill, waste=0.0, penalties=PENALTIES):
    skus = {k: SimpleNamespace(stockout_penalty=penalties[k], demand_fulfilled_today=fill.get(k, 1.0))
            for k in penalties}
    return SimpleNamespace(skus=skus, waste_fraction_today=waste)


def test_running_out_of_a_critical_sku_costs_more():
    total = sum(PENALTIES.values())
    insulin_out = compute_step_reward(_state({"insulin": 0.0}))
    vitamins_out = compute_step_reward(_state({"vitamins": 0.0}))
    assert insulin_out == pytest.approx(1 - PENALTIES["insulin"] / total)
    assert vitamins_out == pytest.approx(1 - PENALTIES["vitamins"] / total)
    assert insulin_out < vitamins_out


def test_full_fill_and_waste_bounds():
    assert compute_step_reward(_state({})) == pytest.approx(1.0)
    assert compute_step_reward(_state({k: 0.0 for k in PENALTIES}, waste=1.0)) == pytest.approx(-1.0)
    assert compute_step_reward(_state({}, waste=0.25)) == pytest.approx(0.75)


def test_zero_penalties_fall_back_to_the_plain_mean():
    zero = {k: 0.0 for k in PENALTIES}
    state = _state({"insulin": 0.0, "vitamins": 0.5}, penalties=zero)
    assert weighted_fill(state.skus) == pytest.approx((0.0 + 0.5 + 1 + 1 + 1) / 5)


def test_episode_score_is_the_mean_of_weighted_step_rewards():
    """The score is rebuilt from running totals; it must match the rewards actually paid."""
    env = PharmaEnvironment()
    obs = env.reset(task_name="flu_season", seed=TEST_SEEDS[0])
    rewards = []
    while not obs.done:
        # Order a little paracetamol only: the other SKUs run short, unevenly.
        obs = env.step(PharmaAction(message='{"paracetamol": 200}'))
        rewards.append(obs.reward)
    assert compute_final_score(obs, env._episode_config) == pytest.approx(
        min(1.0, max(0.0, sum(rewards) / env._episode_config.no_of_days)))


def test_every_prompt_says_shortages_are_weighted_by_stockout_penalty():
    import inference

    from test_prompt import SKU_WITH_NUMBER

    for prompt in (inference.SYSTEM_PROMPT, inference.SYSTEM_PROMPT_DAYS, inference.SYSTEM_PROMPT_ADJUST):
        assert "Shortages are weighted by stockout_penalty in your score" in prompt
        assert not SKU_WITH_NUMBER.search(prompt)
    assert "weighted by stockout_penalty" in inference.NEWS_SYSTEM_PROMPT
    env = PharmaEnvironment()
    obs = env.reset(task_name="flu_season", seed=TEST_SEEDS[0], news=1)
    assert '"stockout_penalty": 100.0' in inference.build_news_prompt(obs)
