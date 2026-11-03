"""scenario="random": disrupted SKUs and days drawn per seed, 14 days of storage; the rest as in "fixed"."""

from __future__ import annotations

import pytest

import eval as E
import inference
from models import PharmaAction
from news import find_events
from server.my_env_environment import PharmaEnvironment
from tasks import (RANDOM_STORAGE_DAYS, SKU_IDS, TASK_REGISTRY, TEST_SEEDS, TRAIN_SEEDS,
                   get_task_config)

TASKS = sorted(TASK_REGISTRY)
SEEDS = list(TRAIN_SEEDS[:150]) + list(TEST_SEEDS[:150])


def _events(task, seed, scenario):
    return find_events(get_task_config(task, seed, scenario=scenario))


def _signature(task, seed, scenario):
    return tuple((e["kind"], e["sku_id"], e["start_day"]) for e in _events(task, seed, scenario))


@pytest.mark.parametrize("task", TASKS)
def test_fixed_is_the_default_and_the_same_for_every_seed(task):
    assert get_task_config(task, 7) == get_task_config(task, 7, scenario="fixed")
    # The same SKUs and days for every seed (only the size of each disruption changes).
    assert len({_signature(task, s, "fixed") for s in TEST_SEEDS[:20]}) == 1


@pytest.mark.parametrize("task", TASKS)
def test_random_draws_skus_and_days_per_seed_reproducibly(task):
    assert get_task_config(task, 11, scenario="random") == get_task_config(task, 11, scenario="random")
    signatures = [_signature(task, s, "random") for s in TEST_SEEDS[:60]]
    assert len({min(day for _, _, day in sig) for sig in signatures}) >= 6
    assert {sku for sig in signatures for _, sku, _ in sig} == set(SKU_IDS)


@pytest.mark.parametrize("task", TASKS)
def test_random_changes_only_the_curves_and_the_storage(task):
    """Base demand, lead times, starting stock and the size of each disruption are the same draws."""
    fixed, rand = get_task_config(task, 3), get_task_config(task, 3, scenario="random")
    assert fixed.initial_inventory == rand.initial_inventory
    for k in fixed.skus:
        a, b = fixed.skus[k], rand.skus[k]
        assert (a.base_demand, a.base_lead_time, a.demand_std) == (b.base_demand, b.base_lead_time, b.demand_std)
    sizes = lambda cfg: sorted(round(max(s.lead_time_curve), 9) for s in cfg.skus.values())
    assert sizes(fixed) == sizes(rand)
    ratio = RANDOM_STORAGE_DAYS / 21.0
    assert rand.cold_storage_total_capacity == pytest.approx(fixed.cold_storage_total_capacity * ratio)
    assert rand.ambient_storage_total_capacity == pytest.approx(fixed.ambient_storage_total_capacity * ratio)


def test_random_days_and_skus_stay_in_their_ranges():
    for s in SEEDS:
        delays = [e for e in _events("supply_chain_broken", s, "random")]
        assert len(delays) == 2 and all(e["kind"] == "supplier_delay" for e in delays)
        first, second = delays
        assert first["sku_id"] != second["sku_id"]
        assert 10 <= first["start_day"] <= 20 and 8 <= first["end_day"] - first["start_day"] <= 12
        assert 3 <= second["start_day"] - first["end_day"] <= 10

        flu = {e["kind"]: e for e in _events("flu_season", s, "random")}
        assert flu["demand_surge"]["sku_id"] == flu["supplier_delay"]["sku_id"]
        assert flu["demand_surge"]["start_day"] >= 8 and 24 <= flu["demand_surge"]["peak_day"] <= 38
        assert abs(flu["supplier_delay"]["start_day"] - flu["demand_surge"]["peak_day"]) <= 3

        waves = [e for e in _events("epidemic_two_wave", s, "random") if e["kind"] == "demand_surge"]
        assert len(waves) == 2 and waves[0]["sku_id"] == waves[1]["sku_id"]  # the waves never merge
        assert 13 <= waves[0]["peak_day"] <= 23 and 26 <= waves[1]["peak_day"] - waves[0]["peak_day"] <= 32


def test_news_follows_the_random_skus_and_days():
    cfg = get_task_config("supply_chain_broken", TEST_SEEDS[0], news=1, scenario="random")
    for event in cfg.news:
        curve = cfg.skus[event.sku_id].lead_time_curve
        assert f"{event.sku_id}: orders placed from day {event.start_day} to day {event.end_day}" in event.message
        assert curve[event.start_day] > 1 and curve[event.start_day - 1] == 1


def _report_after(days, scenario):
    env = PharmaEnvironment()
    obs = env.reset(task_name="supply_chain_broken", seed=TEST_SEEDS[1], scenario=scenario)
    for _ in range(days):
        obs = env.step(PharmaAction(message="{}"))
    return inference.observation_to_dict(obs)["storage"]


def test_report_shows_how_many_days_each_pool_holds():
    assert _report_after(0, "fixed")["ambient_days"] is None  # no demand observed yet
    fixed, rand = _report_after(10, "fixed"), _report_after(10, "random")
    assert 17 <= fixed["ambient_days"] <= 25 and 17 <= fixed["cold_days"] <= 25
    assert 11 <= rand["ambient_days"] <= 17 and 11 <= rand["cold_days"] <= 17


def test_prompts_point_to_the_reported_storage_days():
    for prompt in (inference.SYSTEM_PROMPT_DAYS, inference.SYSTEM_PROMPT_ADJUST):
        assert "holds only about 21 days" not in prompt and "cold_days / ambient_days" in prompt
    assert "cold_days and ambient_days" in inference.SYSTEM_PROMPT


def test_environment_and_eval_take_the_scenario():
    env = PharmaEnvironment()
    env.reset(task_name="flu_season", seed=TEST_SEEDS[0], scenario="random")
    assert env._episode_config == get_task_config("flu_season", TEST_SEEDS[0], scenario="random")
    record = E.run_episode("baseline", "flu_season", TEST_SEEDS[0], scenario="random")
    assert record["scenario"] == "random" and 0 < record["score"] <= 1
    assert E.run_episode("baseline", "flu_season", TEST_SEEDS[0])["scenario"] == "fixed"
    with pytest.raises(ValueError, match="scenario"):
        get_task_config("flu_season", 1, scenario="weekly")
