"""The "days of stock" action: the model picks days per SKU, code turns them into units."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import eval as E
import inference
from agents import BaselineAgent, LLMAgent
from models import PharmaAction
from server.my_env_environment import PharmaEnvironment
from tasks import TEST_SEEDS

from test_prompt import SKU_WITH_NUMBER


def _played(days: int):
    env = PharmaEnvironment()
    obs = env.reset(task_name="flu_season", seed=TEST_SEEDS[0])
    agent, history = BaselineAgent(), []
    for _ in range(days):
        day = obs.current_date
        reply, _ = agent.act(obs, history)
        obs = env.step(PharmaAction(message=reply))
        history.append((day, reply, inference.build_feedback(obs)))
    return obs, history


def _stub_client(text):
    def create(model, messages, **kwargs):
        create.system = messages[0]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), create


def _expected_units(obs, sku_id, days):
    sku = obs.skus[sku_id]
    inbound = sum(q for k, q, _ in obs.expected_inbound_orders if k == sku_id)
    demand = max(sku.avg_demand_last_5_days, sku.avg_demand_per_day)
    return days * demand - sku.inventory_on_hand - inbound


def test_days_to_units_subtracts_stock_and_inbound():
    obs, _ = _played(10)
    orders = inference.days_to_units(obs, {"paracetamol": 12, "Insulin": "9.5"})
    for sku_id, days in (("paracetamol", 12), ("insulin", 9.5)):
        expected = _expected_units(obs, sku_id, days)
        if expected >= 1:
            assert orders[sku_id] == round(expected)
        else:
            assert sku_id not in orders


def test_days_are_clipped_and_bad_values_ignored():
    obs, _ = _played(10)
    capped = inference.days_to_units(obs, {"paracetamol": 1000})
    assert capped == inference.days_to_units(obs, {"paracetamol": inference.MAX_DAYS_OF_STOCK})
    junk = {"paracetamol": -5, "vitamins": "lots", "insulin": True, "aspirin": 10,
            "bp_medication": None, "hydroxychloroquine": float("nan")}
    assert inference.days_to_units(obs, junk) == {}


def test_no_order_before_any_demand_is_seen():
    obs = PharmaEnvironment().reset(task_name="flu_season", seed=TEST_SEEDS[0])
    assert inference.days_to_units(obs, {sku: 10 for sku in obs.skus}) == {}


def test_parse_json_object():
    assert inference.parse_json_object('Sure.\n```json\n{"insulin": 9}\n```') == {"insulin": 9}
    assert inference.parse_json_object('{"orders": {"insulin": 9}}') == {"insulin": 9}
    assert inference.parse_json_object("no json here") == {}
    assert inference.parse_json_object("[1, 2]") == {}
    assert inference.parse_json_object("{not json}") == {}


def test_days_prompt_asks_for_days_without_example_quantities():
    assert "DAYS OF STOCK" in inference.SYSTEM_PROMPT_DAYS
    assert "DAYS OF STOCK" not in inference.SYSTEM_PROMPT
    assert inference.system_prompt("days") == inference.SYSTEM_PROMPT_DAYS
    assert inference.system_prompt("units") == inference.SYSTEM_PROMPT
    assert not SKU_WITH_NUMBER.search(inference.SYSTEM_PROMPT_DAYS)
    obs, history = _played(5)
    user = inference.build_user_prompt(inference.observation_to_dict(obs), history, step=6, action="days")
    assert "days of stock" in user and not SKU_WITH_NUMBER.search(user[user.index("--- YOUR DECISION ---"):])


def test_llm_agent_in_days_mode_sends_converted_units():
    obs, history = _played(10)
    client, create = _stub_client('Plan:\n{"paracetamol": 15, "insulin": 0}')
    agent = LLMAgent(model="stub", api_key="x", base_url="x", client=client, action="days")
    reply, error = agent.act(obs, history)
    assert error is None and create.system == inference.SYSTEM_PROMPT_DAYS
    assert agent.last_model_reply == 'Plan:\n{"paracetamol": 15, "insulin": 0}'
    assert json.loads(reply) == inference.days_to_units(obs, {"paracetamol": 15})
    with pytest.raises(ValueError):
        LLMAgent(model="stub", api_key="x", base_url="x", client=client, action="weeks")


def test_eval_logs_model_reply_and_units_sent(monkeypatch):
    client, _ = _stub_client('{"insulin": 9, "paracetamol": 9, "bp_medication": 9, '
                             '"vitamins": 9, "hydroxychloroquine": 9}')
    monkeypatch.setattr(E, "make_agent",
                        lambda name, llm=None: LLMAgent(model="stub", api_key="x", base_url="x", client=client,
                                                       action="days"))
    record = E.run_episode("llm", "flu_season", TEST_SEEDS[0], save_steps=True)
    assert record["action_format"] == "days" and record["parse_failures"] == 0
    assert record["score"] > 0.7  # 9 days of cover every day is a sensible plan
    later = record["steps"][20]
    assert later["reply"].startswith('{"insulin": 9') and later["orders_sent"] != later["reply"]


def test_days_prompt_targets_steady_cover_within_storage():
    """Regression: Gemma 2B answered 30 (the only number in the old format line) and echoed
    days_of_cover, so stock ran down, then a 30-day jump overflowed storage."""
    rules = inference.SYSTEM_PROMPT_DAYS[len(inference.SYSTEM_PROMPT_DAYS) - len(inference._DAYS_RULES):]
    assert "30" not in rules and "ALREADY HAVE" in rules and "between 10 and 14 days" in rules
    assert inference.MAX_DAYS_OF_STOCK == 21
    obs, _ = _played(10)
    assert inference.days_to_units(obs, {"vitamins": 30}) == inference.days_to_units(obs, {"vitamins": 21})
