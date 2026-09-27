"""Unit tests for the simulation core (no server required)."""

from __future__ import annotations

import pytest

from models import EpisodeConfig, PharmaAction, SKUEpisodeConfig
from server.my_env_environment import PharmaEnvironment
from tasks import TASK_REGISTRY, compute_final_score, get_task_config


def _deterministic_config(lead_time: float = 3.0, days: int = 10) -> EpisodeConfig:
    """Single-SKU episode with no noise so arrivals and demand are exact."""
    return EpisodeConfig(
        task_name="deterministic",
        no_of_days=days,
        cold_storage_total_capacity=100.0,
        ambient_storage_total_capacity=1000.0,
        initial_inventory={"insulin": 50.0},
        skus={
            "insulin": SKUEpisodeConfig(
                sku_id="insulin",
                no_of_days=days,
                base_demand=10.0,
                base_lead_time=lead_time,
                stockout_penalty=100.0,
            ),
        },
    )


def test_reset_accepts_config_dict():
    """Over HTTP/WS the config arrives as plain JSON, not as a model instance."""
    env = PharmaEnvironment()
    cfg = get_task_config("flu_season", seed=1)
    state = env.reset(episode_config=cfg.model_dump(mode="json"), seed=1)
    assert set(state.skus) == set(cfg.skus)


def test_reset_by_task_name_is_reproducible():
    a = PharmaEnvironment().reset(task_name="epidemic_two_wave", seed=7)
    b = PharmaEnvironment().reset(task_name="epidemic_two_wave", seed=7)
    assert a.model_dump() == b.model_dump()


def test_reset_unknown_task_name():
    with pytest.raises(ValueError, match="Unknown task"):
        PharmaEnvironment().reset(task_name="nope")


@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY))
def test_tasks_are_seeded(task_name):
    assert get_task_config(task_name, seed=3) == get_task_config(task_name, seed=3)
    assert get_task_config(task_name, seed=3) != get_task_config(task_name, seed=4)


@pytest.mark.parametrize(
    "message",
    [
        '{"Insulin": 50}',                     # wrong case
        '{"unobtainium": 50}',                 # unknown SKU
        '{"insulin": -5, "paracetamol": "x"}', # invalid quantities
        '{"insulin": NaN}',                    # non-finite
        'no json here',
    ],
)
def test_bad_actions_do_not_crash(message):
    env = PharmaEnvironment()
    env.reset(seed=0)
    state = env.step(PharmaAction(message=message))
    assert state.current_date == 1


def test_sku_names_are_case_insensitive_and_strings_accepted():
    env = PharmaEnvironment()
    env.reset(episode_config=_deterministic_config())
    state = env.step(PharmaAction(message='Here you go: {"INSULIN": "40"}'))
    assert [(s, q) for s, q, _ in state.expected_inbound_orders] == [("insulin", 40.0)]


def test_same_day_arrivals_are_not_lost():
    env = PharmaEnvironment()
    cfg = _deterministic_config(lead_time=3.0)
    env.reset(episode_config=cfg)
    env.step(PharmaAction(orders={"insulin": 100.0}))   # day 0, arrives day 3
    cfg.skus["insulin"].base_lead_time = 2.0
    env._episode_config = cfg
    env.step(PharmaAction(orders={"insulin": 100.0}))   # day 1, arrives day 3
    env.step(PharmaAction())
    state = env.step(PharmaAction())                    # day 3: both arrive
    # 50 start + 200 delivered - 4 days * 10 demand
    assert state.skus["insulin"].inventory_on_hand == pytest.approx(210.0)
    assert state.inventory_excess_today == 0.0


def test_lead_time_revealed_only_on_delivery():
    env = PharmaEnvironment()
    env.reset(episode_config=_deterministic_config(lead_time=3.0))
    state = env.step(PharmaAction(orders={"insulin": 10.0}))
    assert state.skus["insulin"].lead_time_last3_orders == []
    env.step(PharmaAction())
    env.step(PharmaAction())
    state = env.step(PharmaAction())                    # delivered on day 3
    assert state.skus["insulin"].lead_time_last3_orders == [3.0]
    assert state.actual_inbound_orders == []


def test_overflow_is_recorded_as_waste():
    env = PharmaEnvironment()
    env.reset(episode_config=_deterministic_config(lead_time=1.0))
    env.step(PharmaAction(orders={"insulin": 5000.0}))  # capacity is 1000
    state = env.step(PharmaAction())
    assert state.inventory_excess_today > 0
    assert state.skus["insulin"].inventory_on_hand <= 1000.0
    assert state.reward < 1.0


def test_full_episode_scores_in_unit_range():
    env = PharmaEnvironment()
    cfg = get_task_config("supply_chain_broken", seed=0)
    state = env.reset(episode_config=cfg, seed=0)
    steps = 0
    while not state.done:
        orders = {
            sku_id: sku.avg_demand_per_day * 6
            for sku_id, sku in state.skus.items()
            if sku.stockout_days_if_no_reorder < 6
        }
        state = env.step(PharmaAction(orders=orders))
        steps += 1
        assert -1.0 < state.reward <= 1.0
    assert steps == cfg.no_of_days
    assert 0.0 <= compute_final_score(state, cfg) <= 1.0
    with pytest.raises(RuntimeError):
        env.step(PharmaAction())
