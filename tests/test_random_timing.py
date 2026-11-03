"""timing="random": disruption days drawn per seed; everything else as with fixed timing."""

from __future__ import annotations

import pytest

import eval as E
from news import find_events
from server.my_env_environment import PharmaEnvironment
from tasks import TASK_REGISTRY, TEST_SEEDS, TRAIN_SEEDS, get_task_config

TASKS = sorted(TASK_REGISTRY)


def _starts(task, seed, timing):
    return {(e["kind"], e["sku_id"], i): e["start_day"]
            for i, e in enumerate(find_events(get_task_config(task, seed, timing=timing)))}


@pytest.mark.parametrize("task", TASKS)
def test_fixed_timing_is_the_default_and_unchanged(task):
    assert get_task_config(task, 7) == get_task_config(task, 7, timing="fixed")
    # The same disruption days for every seed (only their size changes).
    assert len({tuple(sorted(_starts(task, s, "fixed").items())) for s in TEST_SEEDS[:20]}) == 1


@pytest.mark.parametrize("task", TASKS)
def test_random_timing_varies_by_seed_and_is_reproducible(task):
    assert get_task_config(task, 11, timing="random") == get_task_config(task, 11, timing="random")
    first = [min(_starts(task, s, "random").values()) for s in TEST_SEEDS[:40]]
    assert len(set(first)) >= 6


@pytest.mark.parametrize("task", TASKS)
def test_random_timing_changes_only_the_curves(task):
    """Base demand, lead times, starting stock and storage come from the same draws as fixed timing."""
    fixed, rand = get_task_config(task, 3), get_task_config(task, 3, timing="random")
    assert fixed.initial_inventory == rand.initial_inventory
    assert fixed.cold_storage_total_capacity == rand.cold_storage_total_capacity
    assert fixed.ambient_storage_total_capacity == rand.ambient_storage_total_capacity
    for k in fixed.skus:
        a, b = fixed.skus[k], rand.skus[k]
        assert (a.base_demand, a.base_lead_time, a.demand_std) == (b.base_demand, b.base_lead_time, b.demand_std)
        # The size of each delay is drawn as before; only its days move.
        assert max(a.lead_time_curve) == pytest.approx(max(b.lead_time_curve))


def test_random_days_stay_in_their_ranges():
    for s in list(TRAIN_SEEDS[:150]) + list(TEST_SEEDS[:150]):
        sc = {e["sku_id"]: e for e in find_events(get_task_config("supply_chain_broken", s, timing="random"))}
        assert 10 <= sc["insulin"]["start_day"] <= 20 and 8 <= sc["insulin"]["end_day"] - sc["insulin"]["start_day"] <= 12
        assert 3 <= sc["paracetamol"]["start_day"] - sc["insulin"]["end_day"] <= 10
        flu = {e["kind"]: e for e in find_events(get_task_config("flu_season", s, timing="random"))}
        assert flu["demand_surge"]["start_day"] >= 8 and 24 <= flu["demand_surge"]["peak_day"] <= 38
        assert abs(flu["supplier_delay"]["start_day"] - flu["demand_surge"]["peak_day"]) <= 3
        ep = [e for e in find_events(get_task_config("epidemic_two_wave", s, timing="random"))
              if e["kind"] == "demand_surge"]
        assert len(ep) == 2  # the two waves never merge into one surge
        assert 13 <= ep[0]["peak_day"] <= 23 and 26 <= ep[1]["peak_day"] - ep[0]["peak_day"] <= 32


def test_news_follows_the_random_days():
    cfg = get_task_config("supply_chain_broken", TEST_SEEDS[0], news=1, timing="random")
    curve = cfg.skus["insulin"].lead_time_curve
    event = next(e for e in cfg.news if e.sku_id == "insulin")
    assert f"from day {event.start_day} to day {event.end_day}" in event.message
    assert curve[event.start_day] > 1 and curve[event.start_day - 1] == 1


def test_environment_and_eval_take_the_timing():
    env = PharmaEnvironment()
    env.reset(task_name="flu_season", seed=TEST_SEEDS[0], timing="random")
    assert env._episode_config == get_task_config("flu_season", TEST_SEEDS[0], timing="random")
    record = E.run_episode("baseline", "flu_season", TEST_SEEDS[0], timing="random")
    assert record["timing"] == "random" and 0 < record["score"] <= 1
    assert E.run_episode("baseline", "flu_season", TEST_SEEDS[0])["timing"] == "fixed"
    with pytest.raises(ValueError, match="timing"):
        get_task_config("flu_season", 1, timing="weekly")
