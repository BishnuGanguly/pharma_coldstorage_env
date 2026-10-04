"""Tests for the LLM prompt built in inference.py."""

from __future__ import annotations

import json
import re

import pytest

import inference
from agents import BaselineAgent
from models import PharmaAction
from server.my_env_environment import PharmaEnvironment
from tasks import TEST_SEEDS

# An SKU name followed by a number, e.g. `"insulin": 100` or `insulin (100)`:
# the kind of example a small model copies instead of reading the report.
SKU_WITH_NUMBER = re.compile(r"(insulin|paracetamol|bp_medication|vitamins|hydroxychloroquine)\W{0,4}\d")


def _played(days: int, seed: int = TEST_SEEDS[0]):
    env = PharmaEnvironment()
    obs = env.reset(task_name="flu_season", seed=seed)
    agent, history = BaselineAgent(), []
    for _ in range(days):
        day = obs.current_date
        reply, _ = agent.act(obs, history)
        obs = env.step(PharmaAction(message=reply))
        history.append((day, reply, inference.build_feedback(obs)))
    return env, obs, history


def test_instructions_contain_no_example_quantities():
    """Regression: a 1.5B model replied {"insulin": 100} every day, copying the prompt's example."""
    assert not SKU_WITH_NUMBER.search(inference.SYSTEM_PROMPT)
    _, obs, history = _played(5)
    user = inference.build_user_prompt(inference.observation_to_dict(obs), history, step=6)
    instructions = user[user.index("--- YOUR PROCUREMENT DECISION ---"):]
    assert not SKU_WITH_NUMBER.search(instructions)


def test_history_shows_results_but_never_the_models_own_replies():
    _, obs, history = _played(6)
    history = [(day, '{"insulin": 4242}', feedback) for day, _, feedback in history]
    user = inference.build_user_prompt(inference.observation_to_dict(obs), history, step=7)
    assert "4242" not in user
    block = user[user.index("--- RESULTS OF THE LAST 3 DAYS ---"):user.index("--- TODAY'S INVENTORY REPORT ---")]
    entries = json.loads(block.split("\n", 1)[1])
    assert [e["day"] for e in entries] == [3, 4, 5]
    assert {"fill_rate", "short_skus", "waste_fraction", "reward"} <= set(entries[0]["result"])


def test_report_includes_inbound_units_and_days_of_cover():
    _, obs, _ = _played(10)
    report = inference.observation_to_dict(obs)["inventory"]
    for sku_id, sku in obs.skus.items():
        inbound = sum(q for k, q, _ in obs.expected_inbound_orders if k == sku_id)
        demand = max(sku.avg_demand_last_5_days, sku.avg_demand_per_day)
        assert report[sku_id]["inbound_units"] == pytest.approx(inbound, abs=0.05)
        assert report[sku_id]["days_of_cover"] == pytest.approx((sku.inventory_on_hand + inbound) / demand, abs=0.05)


def test_days_of_cover_is_null_before_any_demand_is_seen():
    env = PharmaEnvironment()
    obs = env.reset(task_name="flu_season", seed=TEST_SEEDS[0])
    report = inference.observation_to_dict(obs)
    assert all(sku["days_of_cover"] is None for sku in report["inventory"].values())
    assert "null" in inference.build_user_prompt(report, [], step=1)


def test_environment_prompt_is_the_same_prompt():
    env, obs, _ = _played(3)
    prompt = env.to_llm_prompt()
    assert prompt.startswith(inference.SYSTEM_PROMPT)
    assert prompt.endswith(inference.build_user_prompt(inference.observation_to_dict(obs), [], step=4))
