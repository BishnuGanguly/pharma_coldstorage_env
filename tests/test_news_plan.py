"""Tests for the news plan reference (agents.news_plan_orders, NewsPlanAgent)."""

from __future__ import annotations

import json
import statistics

import pytest

import eval as E
from agents import NewsPlanAgent, baseline_policy, news_plan_orders
from models import InventoryState, NewsEvent, PharmaAction, SKUState, StoragePoolState
from server.my_env_environment import PharmaEnvironment
from tasks import TASK_REGISTRY, TEST_SEEDS

TASKS = sorted(TASK_REGISTRY)


def state(day=10, demand=10.0, stock=0.0, pool_capacity=100_000.0, history=None):
    """One SKU 'a' (lead time 5, steady demand) and an untouched SKU 'b', in one pool."""
    history = [demand] * day if history is None else history
    skus = {
        name: SKUState(sku_id=name, avg_demand_per_day=demand, avg_demand_last_5_days=demand, avg_lead_time=5.0,
                       inventory_on_hand=stock, storage_pool="shelf", stockout_penalty=1.0)
        for name in ("a", "b")
    }
    return InventoryState(
        current_date=day, skus=skus,
        storage={"shelf": StoragePoolState(total_capacity=pool_capacity)},
        cold_storage_total_capacity=1.0, ambient_storage_total_capacity=pool_capacity,
        demand_history={"a": list(history), "b": [demand] * day}, lead_time_history={"a": [], "b": []},
    )


def delay(start, end, multiplier=4.0, announce=0, sku="a"):
    return NewsEvent(kind="supplier_delay", sku_id=sku, start_day=start, end_day=end, peak_day=start,
                     multiplier=multiplier, announce_day=announce, message="x")


def surge(start, peak, end, multiplier=3.0, announce=0, sku="a"):
    return NewsEvent(kind="demand_surge", sku_id=sku, start_day=start, end_day=end, peak_day=peak,
                     multiplier=multiplier, announce_day=announce, message="x")


LEAD = {"a": 5.0, "b": 5.0}


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_without_published_news_the_plan_is_the_baseline():
    obs = state()
    assert news_plan_orders(obs, [], LEAD) == baseline_policy(obs)
    # An event announced later is not known yet.
    assert news_plan_orders(obs, [delay(20, 30, announce=11)], LEAD) == baseline_policy(obs)


def test_skus_without_news_are_ordered_as_the_baseline_orders_them():
    obs = state()
    orders = news_plan_orders(obs, [delay(11, 20)], LEAD)
    assert orders["b"] == baseline_policy(obs)["b"]


def test_cover_before_a_delay_lasts_until_the_first_order_after_it_can_arrive():
    # Today 10; orders placed on days 11-20 take 4 x 5 = 20 days, an order on day 21 takes 5.
    # Earliest later arrival: day 26 (ordered day 21), i.e. 15 days after tomorrow: cover 15 + 3 safety.
    orders = news_plan_orders(state(day=10), [delay(11, 20)], LEAD)
    assert orders["a"] == round(10.0 * (15 + 3))
    # A long delay: waiting for its end (day 41 + 5) is slower than ordering inside it (11 + 20).
    orders = news_plan_orders(state(day=10), [delay(11, 40)], LEAD)
    assert orders["a"] == round(10.0 * (20 + 3))


def test_cover_returns_to_normal_once_the_delay_has_passed():
    orders = news_plan_orders(state(day=21), [delay(11, 20)], LEAD)
    assert orders["a"] == round(10.0 * (5 + 3))             # the baseline's lead time + 3


def test_forecast_rises_before_a_surge_and_falls_after_its_peak():
    # Flat sales so far, a surge to 3x from day 12 (peak 20, end 28).
    before = news_plan_orders(state(day=10), [surge(12, 20, 28)], LEAD)["a"]
    assert before > baseline_policy(state(day=10))["a"]
    # Late in the wave, sales of 20 a day are known to be a 2x multiple: demand is falling to 10.
    late = state(day=27, history=[10.0] * 12 + [20.0] * 15, demand=20.0)
    assert news_plan_orders(late, [surge(12, 20, 28)], LEAD)["a"] < baseline_policy(late)["a"]


def test_normal_demand_is_sales_with_the_surge_divided_out():
    # Sales doubled exactly as the published surge says: normal demand is still 10 a day.
    event = surge(0, 5, 10, multiplier=2.0)
    from agents import _surge_profile
    history = [10.0 * _surge_profile(event, day) for day in range(30)]
    orders = news_plan_orders(state(day=30, history=history, demand=15.0), [event], LEAD)
    assert orders["a"] == round(10.0 * (5 + 3))


def test_target_is_capped_by_the_skus_share_of_its_pool():
    # Pool of 200 units shared equally by a and b: 100 for a, plus the 50 used while today's order travels.
    orders = news_plan_orders(state(day=10, pool_capacity=200.0), [delay(11, 20)], LEAD)
    assert orders["a"] == round(100.0 + 10.0 * 5)
    # Stock on hand and inbound orders count against the target.
    obs = state(day=10, stock=30.0)
    obs.expected_inbound_orders = [("a", 20.0, 12)]
    assert news_plan_orders(obs, [delay(11, 20)], LEAD)["a"] == round(10.0 * 18 - 30 - 20)


# ---------------------------------------------------------------------------
# The agent in the environment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_name", TASKS)
def test_agent_without_news_plays_exactly_like_the_baseline(task_name):
    assert (E.run_episode("news_plan", task_name, TEST_SEEDS[0], save_steps=True)["steps"]
            == E.run_episode("baseline", task_name, TEST_SEEDS[0], save_steps=True)["steps"])


def test_agent_keeps_the_first_reported_lead_time_as_normal():
    env = PharmaEnvironment()
    obs = env.reset(task_name="supply_chain_broken", seed=TEST_SEEDS[0], news=1)
    agent = NewsPlanAgent()
    agent.start_episode(env)
    first = {k: s.avg_lead_time for k, s in obs.skus.items()}
    while not obs.done:
        reply, error = agent.act(obs, [])
        assert error is None
        obs = env.step(PharmaAction(message=reply))
    assert agent._normal_lead_time == first
    assert obs.skus["insulin"].avg_lead_time > first["insulin"]      # delayed deliveries raised the average
    agent.start_episode(env)
    assert agent._normal_lead_time == {}


@pytest.mark.parametrize("scenario", ["fixed", "random"])
def test_plan_beats_the_baseline_and_the_extra_days_rule(scenario):
    def mean(agent):
        return statistics.fmean(E.run_episode(agent, t, s, news=2, scenario=scenario)["score"]
                                for t in TASKS for s in TEST_SEEDS[:6])
    plan, rule, baseline = mean("news_plan"), mean("baseline_news"), mean("baseline")
    assert plan > rule > baseline
    assert plan - baseline > 0.02


def test_plan_wastes_less_than_the_extra_days_rule():
    def wasted(agent):
        return sum(E.run_episode(agent, t, s, news=2, scenario="random")["units_wasted"]
                   for t in TASKS for s in TEST_SEEDS[:6])
    assert wasted("news_plan") < wasted("baseline_news")


def test_orders_are_valid_json_for_the_environment():
    env = PharmaEnvironment()
    obs = env.reset(task_name="epidemic_two_wave", seed=TEST_SEEDS[1], news=2, scenario="random")
    agent = E.make_agent("news_plan")
    agent.start_episode(env)
    for _ in range(25):
        reply, _ = agent.act(obs, [])
        orders = json.loads(reply)
        assert all(sku in obs.skus and qty >= 1 for sku, qty in orders.items())
        obs = env.step(PharmaAction(message=reply))
