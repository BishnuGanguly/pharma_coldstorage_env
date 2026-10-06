"""The "adjust" action: the baseline orders a default, the model answers adjustment days."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import eval as E
import inference
from agents import BaselineAgent, LLMAgent, baseline_policy
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


def _agent(text=None, error=None):
    def create(model, messages, **kwargs):
        create.messages = messages
        if error:
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return LLMAgent(model="stub", api_key="x", base_url="x", client=client), create


def test_adjust_is_the_default_and_its_prompt_has_no_example_quantities():
    agent, create = _agent("{}")
    assert agent.action == "adjust"
    obs, history = _played(10)
    agent.act(obs, history)
    system, user = create.messages[0]["content"], create.messages[1]["content"]
    assert system == inference.SYSTEM_PROMPT_ADJUST and "ADJUSTMENT DAYS" in system
    assert not SKU_WITH_NUMBER.search(system)
    assert not SKU_WITH_NUMBER.search(user[user.index("--- YOUR DECISION ---"):])


def test_report_leaves_out_days_of_cover_in_adjust_mode():
    """Regression: Qwen 2.5 3B copied days_of_cover as its answer until its stock ran out."""
    obs, history = _played(10)
    agent, create = _agent("{}")
    agent.act(obs, history)
    assert "days_of_cover" not in create.messages[1]["content"]
    assert "days_of_cover" in inference.build_user_prompt(inference.observation_to_dict(obs), history, step=11)


@pytest.mark.parametrize("text", ["{}", "no json at all", '{"insulin": 0, "paracetamol": 0}', '{"aspirin": 5}'])
def test_no_answer_or_zero_plays_like_the_baseline(text):
    obs, history = _played(10)
    agent, _ = _agent(text)
    reply, error = agent.act(obs, history)
    assert error is None and json.loads(reply) == baseline_policy(obs)
    assert agent.last_model_reply == text


def test_adjustment_days_are_added_to_the_default_and_clipped():
    obs, history = _played(10)
    agent, _ = _agent('{"paracetamol": 4, "Vitamins": 2, "hcq": 100, "insulin": "lots"}')
    reply, _ = agent.act(obs, history)
    expected = baseline_policy(obs, extra_days={"paracetamol": 4, "vitamins": 2, "hydroxychloroquine": 10})
    assert json.loads(reply) == expected
    assert inference.adjust_days(obs, {"paracetamol": -50, "vitamins": 50}) == {"paracetamol": 0, "vitamins": 10}


def test_negative_answers_keep_the_default():
    """Keeping less than the default never helped, so negative answers are clipped to 0."""
    obs, history = _played(10)
    agent, _ = _agent('{"insulin": -3, "paracetamol": -3, "bp_medication": -3, "vitamins": -3, "hydroxychloroquine": -3}')
    reply, _ = agent.act(obs, history)
    assert json.loads(reply) == baseline_policy(obs)


def test_more_days_order_more():
    obs, _ = _played(10)
    sku = "paracetamol"
    totals = [baseline_policy(obs, extra_days={sku: d}).get(sku, 0) for d in (0, 2, 4, 10)]
    assert totals == sorted(totals) and totals[0] < totals[-1]


def test_failed_call_still_orders_the_default():
    obs, history = _played(10)
    agent, _ = _agent(error=TimeoutError("slow"))
    reply, error = agent.act(obs, history)
    assert "TimeoutError" in error and json.loads(reply) == baseline_policy(obs)


def test_news_agent_clipping_is_unchanged():
    obs, _ = _played(10)
    assert baseline_policy(obs, extra_days={"paracetamol": -3}) == baseline_policy(obs)


def test_eval_runs_adjust_by_default(monkeypatch):
    agent, _ = _agent('{"paracetamol": 2}')
    monkeypatch.setattr(E, "make_agent", lambda name, llm=None: agent)
    record = E.run_episode("llm", "flu_season", TEST_SEEDS[0], save_steps=True)
    assert record["action_format"] == "adjust" and record["parse_failures"] == 0
    assert record["score"] > 0.85
    assert record["steps"][20]["reply"] == '{"paracetamol": 2}'
