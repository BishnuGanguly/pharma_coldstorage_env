"""Tests for tasks whose place and time are drawn per seed (world.py: draw, day ranges, duration, storage_days)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

import eval as E
from tasks import get_task_config
from world import WorldError, build_episode, load_world

ROOT = Path(__file__).resolve().parent.parent
GROCERY_FILE = ROOT / "examples" / "worlds" / "grocery.yaml"
GROCERY = load_world(GROCERY_FILE)
PRODUCTS = ["milk", "yogurt", "ice_cream", "rice", "biscuits"]


def world_with(task, **top):
    data = yaml.safe_load(GROCERY_FILE.read_text())
    data["tasks"] = {"t": task}
    data.update(top)
    return load_world(data)


def delayed(config):
    """{sku: (first day, last day, multiplier)} of the supplier delays in an episode."""
    out = {}
    for sku_id, sku in config.skus.items():
        days = [d for d, v in enumerate(sku.lead_time_curve) if v > 1.0]
        if days:
            out[sku_id] = (days[0], days[-1], sku.lead_time_curve[days[0]])
    return out


def surging(config):
    return {k: s.demand_curve.index(max(s.demand_curve)) for k, s in config.skus.items() if max(s.demand_curve) > 1.0}


# ---------------------------------------------------------------------------
# Drawn products
# ---------------------------------------------------------------------------

def test_drawn_products_vary_by_seed_and_different_names_get_different_products():
    struck, popular = set(), set()
    for seed in range(150):
        config = build_episode(GROCERY, "surprise_disruptions", seed)
        (delay_sku,), (surge_sku,) = delayed(config), surging(config)
        assert delay_sku != surge_sku
        assert surge_sku in ("ice_cream", "biscuits", "milk")
        struck.add(delay_sku)
        popular.add(surge_sku)
        assert config == build_episode(GROCERY, "surprise_disruptions", seed)        # a seed fixes the episode
    assert struck == set(PRODUCTS) and popular == {"ice_cream", "biscuits", "milk"}


def test_events_with_the_same_draw_name_hit_the_same_product():
    world = world_with({"draw": {"hit": "any"}, "events": [
        {"type": "demand_wave", "product": "hit", "peak_day": 20, "amplitude": 1.0, "width": 4},
        {"type": "supplier_delay", "product": "hit", "start_day": 18, "end_day": 26, "multiplier": 3},
        {"type": "supplier_delay", "product": "rice", "start_day": 5, "end_day": 9, "multiplier": 2},   # a fixed product still works
    ]})
    for seed in range(30):
        config = build_episode(world, "t", seed)
        (surge_sku,) = surging(config)
        delays = delayed(config)
        assert "rice" in delays and (surge_sku == "rice" or set(delays) == {"rice", surge_sku})
        assert delays[surge_sku][:2] in ((18, 26), (5, 26))          # rice may get both delays


def test_news_follows_the_drawn_product():
    for seed in range(20):
        config = get_task_config("surprise_disruptions", seed, news=1, world=GROCERY)
        assert {e.sku_id for e in config.news if e.kind == "supplier_delay"} == set(delayed(config))


# ---------------------------------------------------------------------------
# Drawn days
# ---------------------------------------------------------------------------

def test_day_ranges_and_durations_are_drawn_within_their_bounds():
    starts, lengths, peaks = set(), set(), set()
    for seed in range(300):
        config = build_episode(GROCERY, "surprise_disruptions", seed)
        (first, last, multiplier), = delayed(config).values()
        starts.add(first)
        lengths.add(last - first + 1)
        peaks.update(surging(config).values())
        assert 2 <= multiplier <= 4
    assert starts == set(range(8, 26)) and lengths == set(range(8, 13)) and peaks == set(range(18, 35))


def test_end_day_can_be_a_range_and_fixed_days_draw_nothing():
    world = world_with({"events": [
        {"type": "demand_step", "product": "rice", "start_day": [3, 6], "end_day": [10, 14], "multiplier": 2},
        {"type": "supplier_delay", "product": "milk", "start_day": 20, "duration": 5, "multiplier": 3},
    ]})
    windows = set()
    for seed in range(60):
        config = build_episode(world, "t", seed)
        rice = [d for d, v in enumerate(config.skus["rice"].demand_curve) if v == 2.0]
        windows.add((rice[0], rice[-1]))
        assert delayed(config)["milk"][:2] == (20, 24)               # duration 5: days 20-24
    assert {w[0] for w in windows} == {3, 4, 5, 6} and {w[1] for w in windows} == {10, 11, 12, 13, 14}


def test_place_and_time_have_their_own_generator():
    # A task and a copy with ranges share every other value of a seed: demand, lead times, event sizes.
    fixed = {"events": [{"type": "supplier_delay", "product": "rice", "start_day": 15, "end_day": 24, "multiplier": [2, 4]}]}
    ranged = {"draw": {"x": ["rice"]}, "events": [
        {"type": "supplier_delay", "product": "x", "start_day": [10, 20], "duration": [8, 12], "multiplier": [2, 4]}]}
    for seed in range(20):
        a, b = build_episode(world_with(fixed), "t", seed), build_episode(world_with(ranged), "t", seed)
        assert a.initial_inventory == b.initial_inventory
        assert {k: (s.base_demand, s.base_lead_time) for k, s in a.skus.items()} == \
               {k: (s.base_demand, s.base_lead_time) for k, s in b.skus.items()}
        assert delayed(a)["rice"][2] == delayed(b)["rice"][2]          # the same multiplier
    assert len({delayed(build_episode(world_with(ranged), "t", s))["rice"][0] for s in range(40)}) > 5


# ---------------------------------------------------------------------------
# Storage per task
# ---------------------------------------------------------------------------

def test_a_task_can_have_its_own_storage_size():
    def days_held(config, pool):
        return config.storage_pools[pool].capacity / sum(s.base_demand for s in config.skus.values() if s.storage_pool == pool)

    for seed in (0, 7):
        tight = build_episode(GROCERY, "surprise_disruptions", seed)
        usual = build_episode(GROCERY, "delivery_strike", seed)
        assert days_held(tight, "shelf") == pytest.approx(14) and days_held(usual, "shelf") == pytest.approx(21)
        assert days_held(tight, "fridge") == pytest.approx(10) and days_held(tight, "freezer") == pytest.approx(14)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

DELAY = {"type": "supplier_delay", "product": "rice", "start_day": 5, "end_day": 9, "multiplier": 2}


@pytest.mark.parametrize("task, message", [
    ({"draw": {"milk": "any"}, "events": []}, "a draw cannot have the name of a product"),
    ({"draw": {"x": ["bread"]}, "events": []}, "unknown product(s) ['bread']"),
    ({"draw": {"x": ["rice"], "y": ["rice"]}, "events": []}, "may already be taken"),
    ({"draw": {"x": "all"}, "events": []}, "draw.x"),
    ({"events": [{**DELAY, "product": "x"}]}, "unknown product 'x'"),
    ({"storage_days": {"cellar": 5}, "events": []}, "unknown storage pool 'cellar'"),
    ({"storage_days": {"shelf": 0}, "events": []}, "must be greater than 0"),
    ({"events": [{**DELAY, "duration": 3}]}, "exactly one of end_day and duration"),
    ({"events": [{k: v for k, v in DELAY.items() if k != "end_day"}]}, "exactly one of end_day and duration"),
    ({"events": [{**DELAY, "start_day": [5, 12]}]}, "start_day must not be after end_day"),
    ({"events": [{**DELAY, "start_day": 2.5}]}, "whole numbers"),
    ({"events": [{**DELAY, "start_day": [9, 5]}]}, "low > high"),
    ({"events": [{**DELAY, "start_day": -1}]}, "must be 0 or more"),
    ({"events": [{"type": "supplier_delay", "product": "rice", "start_day": [30, 40], "duration": [3, 6], "multiplier": 2}]},
     "day 45 is past the end of the episode"),
    ({"events": [{"type": "supplier_delay", "product": "rice", "start_day": 3, "duration": 0, "multiplier": 2}]},
     "must be 1 or more"),
    ({"events": [{"type": "demand_wave", "product": "rice", "peak_day": [40, 50], "amplitude": 1, "width": 3}]},
     "day 50 is past the end of the episode"),
])
def test_bad_draws_and_days_are_explained(task, message):
    with pytest.raises(WorldError) as info:
        world_with(task)
    assert message in str(info.value)


def test_any_is_reserved():
    data = yaml.safe_load(GROCERY_FILE.read_text())
    data["products"]["any"] = copy.deepcopy(data["products"]["rice"])
    with pytest.raises(WorldError, match="a product cannot be called 'any'"):
        load_world(data)


# ---------------------------------------------------------------------------
# Agents on a drawn task
# ---------------------------------------------------------------------------

def test_reading_the_news_pays_on_the_drawn_task():
    seeds = range(10000, 10012)

    def mean(agent):
        return sum(E.run_episode(agent, "surprise_disruptions", s, news=2, world=GROCERY)["score"] for s in seeds) / 12

    nothing, baseline, plan, oracle = mean("nothing"), mean("baseline"), mean("news_plan"), mean("oracle")
    assert nothing < 0.1 < baseline < plan < oracle
    assert plan - baseline > 0.01
