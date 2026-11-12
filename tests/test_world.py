"""Tests for world files (world.py, worlds/, examples/worlds/). They build configs only: no episode is played."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

import eval as E
from server.my_env_environment import PharmaEnvironment
from tasks import (PHARMA_WORLD, SKU_CATALOGUE, TASK_REGISTRY, TEST_SEEDS, TRAIN_SEEDS, flat_curve,
                   gaussian_curve, get_task_config)
from world import (DemandWave, SupplierDelay, WorldError, build_episode, builtin_worlds, load_world)

ROOT = Path(__file__).resolve().parent.parent
GROCERY = ROOT / "examples" / "worlds" / "grocery.yaml"


def grocery_data():
    return yaml.safe_load(GROCERY.read_text())


def with_change(data, path, value):
    data = copy.deepcopy(data)
    *parents, last = path.split(".")
    target = data
    for key in parents:
        target = target[key]
    target[last] = value
    return data


# ---------------------------------------------------------------------------
# The pharma world is the original environment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY))
def test_pharma_world_builds_exactly_the_original_tasks(task_name):
    task_fn = TASK_REGISTRY[task_name]
    for seed in list(TRAIN_SEEDS[:200]) + list(TEST_SEEDS[:200]):
        assert build_episode(PHARMA_WORLD, task_name, seed) == task_fn(seed)


def test_pharma_is_built_in_and_is_the_default():
    assert "pharma" in builtin_worlds()
    assert load_world() == load_world("pharma") == PHARMA_WORLD
    assert PHARMA_WORLD.task_names == list(TASK_REGISTRY)
    assert get_task_config("flu_season", 3) == get_task_config("flu_season", 3, world="pharma")


def test_catalogue_comes_from_the_world_file():
    assert list(SKU_CATALOGUE) == list(PHARMA_WORLD.products)
    insulin = SKU_CATALOGUE["insulin"]
    assert insulin["cold_storage_required"] is True and insulin["stockout_penalty"] == 100.0
    assert insulin["base_demand"] == (8.0, 12.0)
    assert not SKU_CATALOGUE["vitamins"]["cold_storage_required"]


# ---------------------------------------------------------------------------
# Building episodes from any world
# ---------------------------------------------------------------------------

def test_grocery_example_builds_every_task():
    world = load_world(GROCERY)
    assert world.name == "grocery" and world.task_names == ["normal_weeks", "holiday_rush", "delivery_strike"]
    for task in world.task_names:
        config = build_episode(world, task, 7)
        assert config.no_of_days == 45 and sorted(config.skus) == ["biscuits", "milk", "rice", "yogurt"]
        assert config == build_episode(world, task, 7)           # same seed, same episode
        assert config != build_episode(world, task, 8)


def test_storage_capacity_follows_the_pools():
    world = load_world(GROCERY)
    config = build_episode(world, "normal_weeks", 0)
    fridge = 10 * sum(config.skus[s].base_demand for s in ("milk", "yogurt"))
    shelf = 21 * sum(config.skus[s].base_demand for s in ("rice", "biscuits"))
    assert config.cold_storage_total_capacity == pytest.approx(fridge)
    assert config.ambient_storage_total_capacity == pytest.approx(shelf)
    assert config.skus["milk"].cold_storage_required and not config.skus["rice"].cold_storage_required
    # A world with no cold-chain pool keeps a token cold capacity of 1 unit.
    no_fridge = with_change(grocery_data(), "storage", {"shelf": {"capacity_days": 21}})
    for name in ("milk", "yogurt"):
        no_fridge["products"][name]["storage"] = "shelf"
    assert build_episode(load_world(no_fridge), "normal_weeks", 0).cold_storage_total_capacity == 1.0


def test_events_shape_the_curves():
    world = load_world(GROCERY)
    rush = build_episode(world, "holiday_rush", 3)
    biscuits = rush.skus["biscuits"].demand_curve
    assert biscuits[19] == biscuits[28] == 1.0 and len(set(biscuits[20:28])) == 1 and 1.8 <= biscuits[20] <= 2.2
    milk = rush.skus["milk"].demand_curve
    assert max(milk) == milk[24] and 1.5 <= milk[24] <= 2.0
    strike = build_episode(world, "delivery_strike", 3)
    rice = strike.skus["rice"].lead_time_curve
    assert rice[14] == rice[25] == 1.0 and len(set(rice[15:25])) == 1 and 2 <= rice[15] <= 4
    assert strike.skus["yogurt"].lead_time_curve == flat_curve(45)


def test_demand_events_multiply_and_delays_overwrite():
    data = with_change(grocery_data(), "tasks", {"t": {"events": [
        {"type": "demand_wave", "product": "milk", "peak_day": 10, "amplitude": 1.0, "width": 3},
        {"type": "demand_wave", "product": "milk", "peak_day": 30, "amplitude": 2.0, "width": 3},
        {"type": "demand_step", "product": "milk", "start_day": 0, "end_day": 44, "multiplier": 0.5},
        {"type": "supplier_delay", "product": "rice", "start_day": 5, "end_day": 15, "multiplier": 2},
        {"type": "supplier_delay", "product": "rice", "start_day": 10, "end_day": 20, "multiplier": 3},
    ]}})
    config = build_episode(load_world(data), "t", 0)
    waves = [a * b for a, b in zip(gaussian_curve(10, 1.0, 3, 45), gaussian_curve(30, 2.0, 3, 45))]
    assert config.skus["milk"].demand_curve == pytest.approx([0.5 * w for w in waves])
    assert config.skus["rice"].lead_time_curve == [1.0] * 5 + [2.0] * 5 + [3.0] * 11 + [1.0] * 24


def test_ranges_are_drawn_in_the_documented_order():
    # Events first, in order and field order; then products, in order.
    world = load_world(GROCERY)
    rush = build_episode(world, "holiday_rush", 11)
    from random import Random
    rng = Random(11)
    step, amplitude = rng.uniform(1.8, 2.2), rng.uniform(0.5, 1.0)
    milk_demand = rng.uniform(80, 120)
    assert rush.skus["biscuits"].demand_curve[20] == step
    assert rush.skus["milk"].demand_curve[24] == pytest.approx(1 + amplitude)
    assert rush.skus["milk"].base_demand == milk_demand


def test_fixed_numbers_draw_nothing():
    data = grocery_data()
    for product in data["products"].values():
        product.update(base_demand=50, base_lead_time=2, starting_inventory=10)
    config = build_episode(load_world(data), "normal_weeks", 0)
    assert {s.base_demand for s in config.skus.values()} == {50.0}
    assert set(config.initial_inventory.values()) == {10.0}


def test_news_works_for_any_world():
    config = get_task_config("holiday_rush", 2, news=1, world=load_world(GROCERY))
    assert {e.sku_id for e in config.news} == {"biscuits", "milk"}


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path, value, message", [
    ("products.milk.base_demand", [120, 80], "low > high"),
    ("products.milk.base_demand", 0, "greater than 0"),
    ("products.milk.base_lead_time", [1, 2, 3], "[low, high]"),
    ("products.milk.stokout_penalty", 5, "milk.stokout_penalty"),
    ("products.milk.storage", "cellar", "storage 'cellar', which is not defined"),
    ("storage.freezer", {"capacity_days": 5, "cold_chain": True}, "one cold-chain pool"),
    ("days", 0, "days"),
    ("tasks.normal_weeks.events", [{"type": "supplier_delay", "product": "bread", "start_day": 1,
                                    "end_day": 2, "multiplier": 2}], "unknown product 'bread'"),
    ("tasks.normal_weeks.events", [{"type": "demand_wave", "product": "milk", "peak_day": 45,
                                    "amplitude": 1, "width": 3}], "past the end of the episode"),
    ("tasks.normal_weeks.events", [{"type": "demand_step", "product": "milk", "start_day": 9,
                                    "end_day": 2, "multiplier": 2}], "start_day must not be after end_day"),
    ("tasks.normal_weeks.events", [{"type": "meteor", "product": "milk"}], "meteor"),
    ("tasks", {}, "tasks"),
])
def test_invalid_worlds_are_explained(path, value, message):
    with pytest.raises(WorldError) as info:
        load_world(with_change(grocery_data(), path, value))
    assert message in str(info.value)


def test_file_problems_are_explained(tmp_path):
    duplicate = tmp_path / "dup.yaml"
    duplicate.write_text("name: a\nname: b\n")
    with pytest.raises(WorldError, match="duplicate key 'name'"):
        load_world(duplicate)
    broken = tmp_path / "broken.yaml"
    broken.write_text("name: [unclosed\n")
    with pytest.raises(WorldError, match="not valid YAML"):
        load_world(broken)
    with pytest.raises(WorldError, match="cannot read world file"):
        load_world(tmp_path / "missing.yaml")
    with pytest.raises(WorldError, match="no built-in world named 'nope'"):
        load_world("nope")
    listed = tmp_path / "list.yaml"
    listed.write_text("- a\n- b\n")
    with pytest.raises(WorldError, match="must contain a mapping"):
        load_world(listed)


def test_json_world_files_load(tmp_path):
    import json
    path = tmp_path / "grocery.json"
    path.write_text(json.dumps(grocery_data()))
    assert load_world(path) == load_world(GROCERY)


def test_unknown_task_and_random_scenario_are_refused():
    world = load_world(GROCERY)
    with pytest.raises(WorldError, match="has no task 'flu_season'"):
        build_episode(world, "flu_season", 0)
    with pytest.raises(ValueError, match="Unknown task 'flu_season' in world 'grocery'"):
        get_task_config("flu_season", 0, world=world)
    with pytest.raises(WorldError, match="only available for the built-in pharma world"):
        get_task_config("holiday_rush", 0, scenario="random", world=world)


def test_event_types_are_exported():
    events = PHARMA_WORLD.tasks["epidemic_two_wave"].events
    assert [type(e) for e in events] == [DemandWave, DemandWave, DemandWave, SupplierDelay, SupplierDelay]


# ---------------------------------------------------------------------------
# Environment and command line (reset and argument checks only)
# ---------------------------------------------------------------------------

def test_environment_reset_accepts_worlds_but_not_file_paths():
    env = PharmaEnvironment()
    world = load_world(GROCERY)
    obs = env.reset(world=world, task_name="delivery_strike", seed=1)
    assert sorted(obs.skus) == ["biscuits", "milk", "rice", "yogurt"] and obs.current_date == 0
    env.reset(world=grocery_data(), seed=1)
    assert env._episode_config.task_name == "normal_weeks"          # no task: the first one
    env.reset(world="pharma", task_name="flu_season", seed=5)
    assert env._episode_config == get_task_config("flu_season", 5)
    with pytest.raises(WorldError, match="built-in name"):
        env.reset(world=str(GROCERY), task_name="holiday_rush", seed=1)


def test_command_line_rejects_bad_worlds_and_tasks(capsys):
    for argv, message in [
        (["--world", "nope"], "no built-in world named 'nope'"),
        (["--world", str(GROCERY), "--tasks", "flu_season"], "unknown task(s) ['flu_season'] in world 'grocery'"),
        (["--world", str(GROCERY), "--random-scenarios"], "only available for the built-in pharma world"),
    ]:
        with pytest.raises(SystemExit):
            E.main(["--agent", "baseline", *argv])
        assert message in capsys.readouterr().err


def test_an_edited_copy_of_pharma_is_its_own_world(tmp_path, monkeypatch):
    data = yaml.safe_load((ROOT / "worlds" / "pharma.yaml").read_text())
    data["products"]["insulin"]["stockout_penalty"] = 1      # still named "pharma"
    path = tmp_path / "pharma.yaml"
    path.write_text(yaml.safe_dump(data))
    seen = {}
    monkeypatch.setattr(E, "evaluate", lambda *args: seen.setdefault("world", args[9]) and [])
    monkeypatch.setattr(E, "write_jsonl", lambda *a: None)
    monkeypatch.setattr(E, "print_summary", lambda *a: None)
    E.main(["--agent", "baseline", "--world", str(path), "--episodes", "1"])
    assert seen["world"] is not None and seen["world"].products["insulin"].stockout_penalty == 1
    with pytest.raises(SystemExit):
        E.main(["--agent", "baseline", "--world", str(path), "--random-scenarios"])
