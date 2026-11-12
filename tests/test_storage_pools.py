"""Tests for named storage pools (any number). They check configs, reset and the allocation step directly: no episode is played."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import inference
from models import EpisodeConfig, SKUEpisodeConfig, StoragePoolConfig
from server.my_env_environment import PharmaEnvironment
from world import load_world

GROCERY = Path(__file__).resolve().parent.parent / "examples" / "worlds" / "grocery.yaml"


def sku(sku_id, **kwargs):
    return SKUEpisodeConfig(sku_id=sku_id, no_of_days=5, **kwargs)


def config(**kwargs):
    return EpisodeConfig(task_name="t", no_of_days=5, **kwargs)


# ---------------------------------------------------------------------------
# EpisodeConfig
# ---------------------------------------------------------------------------

def test_original_two_capacity_form_becomes_two_named_pools():
    cfg = config(cold_storage_total_capacity=100.0, ambient_storage_total_capacity=1000.0,
                 skus={"a": sku("a", cold_storage_required=True), "b": sku("b")})
    assert cfg.storage_pools == {"cold": StoragePoolConfig(capacity=100.0, cold_chain=True),
                                 "ambient": StoragePoolConfig(capacity=1000.0, cold_chain=False)}
    assert (cfg.skus["a"].storage_pool, cfg.skus["b"].storage_pool) == ("cold", "ambient")
    assert EpisodeConfig.model_validate(cfg.model_dump()) == cfg          # survives HTTP round trips
    assert EpisodeConfig.model_validate(cfg.model_dump(mode="json")) == cfg


def test_named_pools_set_the_totals_and_the_cold_flag():
    cfg = config(storage_pools={"fridge": StoragePoolConfig(capacity=50.0, cold_chain=True),
                                "freezer": StoragePoolConfig(capacity=20.0, cold_chain=True),
                                "shelf": StoragePoolConfig(capacity=300.0)},
                 skus={"milk": sku("milk", storage_pool="fridge"),
                       "ice": sku("ice", storage_pool="freezer", cold_storage_required=False),
                       "rice": sku("rice")})
    assert cfg.cold_storage_total_capacity == 70.0 and cfg.ambient_storage_total_capacity == 300.0
    assert cfg.skus["ice"].cold_storage_required is True          # the pool decides
    assert cfg.skus["rice"].storage_pool == "shelf"               # the only ambient pool
    # Given totals are replaced by the pools' sums.
    assert config(storage_pools={"shelf": StoragePoolConfig(capacity=9.0)}, ambient_storage_total_capacity=5.0,
                  skus={"x": sku("x")}).ambient_storage_total_capacity == 9.0


@pytest.mark.parametrize("kwargs, message", [
    (dict(skus={"x": sku("x")}), "give storage_pools"),
    (dict(storage_pools={"a": StoragePoolConfig(capacity=1, cold_chain=True),
                         "b": StoragePoolConfig(capacity=1, cold_chain=True)},
          skus={"x": sku("x", cold_storage_required=True)}), "not exactly one cold-chain pool"),
    (dict(storage_pools={"shelf": StoragePoolConfig(capacity=1)},
          skus={"x": sku("x", storage_pool="cellar")}), "storage pool 'cellar', which is not defined"),
])
def test_bad_pool_assignments_are_explained(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        config(**kwargs)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def grocery_env(seed=3):
    env = PharmaEnvironment()
    obs = env.reset(world=load_world(GROCERY), task_name="holiday_rush", seed=seed)
    return env, obs


def test_reset_reports_every_pool():
    env, obs = grocery_env()
    assert list(obs.storage) == ["fridge", "freezer", "shelf"]
    stock = {k: s.inventory_on_hand for k, s in obs.skus.items()}
    assert obs.storage["fridge"].current_capacity == pytest.approx(stock["milk"] + stock["yogurt"])
    assert obs.storage["freezer"].current_capacity == pytest.approx(stock["ice_cream"])
    assert obs.storage["shelf"].current_capacity == pytest.approx(stock["rice"] + stock["biscuits"])
    assert obs.cold_storage_current_capacity == pytest.approx(stock["milk"] + stock["yogurt"] + stock["ice_cream"])
    assert obs.storage["freezer"].total_capacity == env._episode_config.storage_pools["freezer"].capacity
    assert {k: s.storage_pool for k, s in obs.skus.items()}["ice_cream"] == "freezer"
    assert not obs.has_classic_storage()


def test_overflow_is_decided_pool_by_pool():
    env, obs = grocery_env()
    free = {name: pool.total_capacity - pool.current_capacity for name, pool in obs.storage.items()}
    arriving = [("ice_cream", free["freezer"] + 50.0), ("rice", 10.0), ("milk", 5.0)]
    accepted, wasted = env._handle_inbound_overflow(arriving, free, {k: 1.0 for k in obs.skus})
    # The freezer overflows by 50 units; the fridge and the shelf, with room to spare, take everything.
    assert wasted["ice_cream"] == pytest.approx(50.0) and accepted["ice_cream"] == pytest.approx(free["freezer"])
    assert (accepted["rice"], wasted["rice"]) == (10.0, 0.0)
    assert (accepted["milk"], wasted["milk"]) == (5.0, 0.0)


def test_recompute_storage_follows_stock():
    env, obs = grocery_env()
    obs.skus["ice_cream"].inventory_on_hand = 7.0
    obs.skus["rice"].inventory_on_hand = 3.0
    obs.skus["biscuits"].inventory_on_hand = 4.0
    env._recompute_storage()
    assert obs.storage["freezer"].current_capacity == 7.0 and obs.storage["shelf"].current_capacity == 7.0
    assert obs.storage["shelf"].ratio == pytest.approx(7.0 / obs.storage["shelf"].total_capacity)


# ---------------------------------------------------------------------------
# LLM report
# ---------------------------------------------------------------------------

def test_report_lists_every_pool_when_the_layout_is_not_cold_plus_ambient():
    _, obs = grocery_env()
    report = inference.observation_to_dict(obs)
    assert list(report["storage"]) == ["fridge", "freezer", "shelf"]
    assert set(report["storage"]["freezer"]) == {"total", "used", "ratio", "days", "cold_chain"}
    assert report["storage"]["freezer"]["days"] is None                 # no demand seen yet
    assert report["inventory"]["ice_cream"]["storage_pool"] == "freezer"


def test_report_keeps_the_original_layout_for_one_cold_and_one_ambient_pool():
    pharma = PharmaEnvironment().reset(task_name="flu_season", seed=0)
    report = inference.observation_to_dict(pharma)
    assert set(report["storage"]) == {"cold_total", "cold_used", "cold_ratio", "ambient_total", "ambient_used",
                                      "ambient_ratio", "cold_days", "ambient_days"}
    assert "storage_pool" not in report["inventory"]["insulin"]
    # Any names: a fridge and a shelf are the same layout.
    two = load_world(GROCERY).model_dump()
    del two["storage"]["freezer"]
    two["products"]["ice_cream"]["storage"] = "fridge"
    obs = PharmaEnvironment().reset(world=two, task_name="normal_weeks", seed=0)
    assert obs.has_classic_storage() and "cold_total" in inference.observation_to_dict(obs)["storage"]


def test_pool_days_divide_capacity_by_recent_demand():
    _, obs = grocery_env()
    obs.skus["ice_cream"].avg_demand_per_day = 10.0
    obs.skus["ice_cream"].avg_demand_last_5_days = 20.0
    assert inference.pool_days(obs, "freezer") == round(obs.storage["freezer"].total_capacity / 20.0, 1)
