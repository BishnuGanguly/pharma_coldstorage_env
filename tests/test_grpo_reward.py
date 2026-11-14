"""The GRPO reward (training/grpo_reward.py): every rule in training/REWARD.md."""

from __future__ import annotations

import copy
import json
from random import Random
from types import SimpleNamespace

import pytest

import inference
import training.grpo_reward as G
from agents import LLMAgent
from models import PharmaAction
from server.my_env_environment import PharmaEnvironment
from tasks import TASK_REGISTRY, TEST_SEEDS, TRAIN_SEEDS

TASKS = sorted(TASK_REGISTRY)


@pytest.fixture(scope="module")
def prompts():
    return G.news_day_prompts(TASKS, TRAIN_SEEDS[:4])


def _news_skus(prompt):
    env, _, _ = G.build_state(prompt)
    return sorted({e.sku_id for e in env._episode_config.news if e.announce_day <= prompt.day <= e.end_day})


# --- the environment hook ---------------------------------------------------

def test_resample_future_noise_redraws_from_today_on_only():
    env = PharmaEnvironment()
    obs = env.reset(task_name="flu_season", seed=1, scenario="random")
    for _ in range(10):
        obs = env.step(PharmaAction(message="{}"))
    before = copy.deepcopy((env._demand_z, env._lead_time_z))
    env.resample_future_noise(Random(5))
    for old, new in zip(before, (env._demand_z, env._lead_time_z)):
        for sku in old:
            assert old[sku][:10] == new[sku][:10]          # the past is kept
            assert old[sku][10:] != new[sku][10:]          # today (day 10) and later are redrawn


# --- prompts --------------------------------------------------------------------

def test_prompts_are_news_days_of_the_random_scenario(prompts):
    assert prompts and all(p.scenario == "random" and p.news == 2 for p in prompts)
    for p in prompts[::25]:
        _, obs, _ = G.build_state(p)
        assert obs.current_date == p.day and obs.news
    assert G.news_day_prompts(TASKS[:1], TRAIN_SEEDS[:2], news=0) == []


def test_state_is_rebuilt_exactly(prompts):
    p = prompts[len(prompts) // 2]
    (_, a, ha), (_, b, hb) = G.build_state(p), G.build_state(p)
    assert a == b and ha == hb


def test_prompt_is_what_the_llm_agent_sends(prompts):
    """An adjust-mode LLM that answers {} plays exactly like the baseline, so its messages on that day must match."""
    p = prompts[len(prompts) // 3]
    sent = []

    def create(model, messages, **kwargs):
        sent.append(messages)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

    agent = LLMAgent("stub", "x", "x", client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    env = PharmaEnvironment()
    obs = env.reset(task_name=p.task, seed=p.seed, news=p.news, scenario=p.scenario)
    history = []
    while obs.current_date <= p.day:
        day = obs.current_date
        reply, _ = agent.act(obs, history)
        obs = env.step(PharmaAction(message=reply))
        history.append((day, reply, inference.build_feedback(obs)))
    assert sent[p.day] == G.prompt_messages(p)
    assert "days_of_cover" not in sent[p.day][1]["content"]


def test_dataset_rows_carry_the_columns_the_reward_reads(prompts):
    row = G.dataset_rows(prompts[:1])[0]
    assert set(row) == {"prompt", "task", "seed", "day", "scenario", "news", "states"}
    assert row["states"] == "baseline"
    assert row["prompt"][0]["role"] == "system"


# --- parsing and penalties ------------------------------------------------------

@pytest.mark.parametrize("text, extra, penalty", [
    ("I would order more.", {}, G.PENALTY_NO_JSON),
    ("", {}, G.PENALTY_NO_JSON),
    ('{"paracetamol": 4}', {"paracetamol": 4.0}, 0.0),
    ('```json\n{"Paracetamol": 4, "hcq": "lots"}\n```', {"paracetamol": 4.0}, 0.0),
    ('{"aspirin": 4}', {}, 0.0),
    ('{"paracetamol": 25, "insulin": -3}', {"paracetamol": 10.0, "insulin": 0.0}, G.PENALTY_OUT_OF_RANGE),
])
def test_parse_answer(prompts, text, extra, penalty):
    _, obs, _ = G.build_state(prompts[0])
    got, pen = G.parse_answer(obs, text)
    assert {k: v for k, v in got.items() if v or k in extra} == extra and pen == penalty


# --- scoring --------------------------------------------------------------------

def test_answering_zero_scores_exactly_zero(prompts):
    reward = G.GRPOReward()
    for p in prompts[::40]:
        for text in ("{}", '{"insulin": 0, "paracetamol": 0}', "  {}  "):
            assert reward.score_answer(p, text)["reward"] == 0.0


def test_reward_is_the_gain_over_answering_zero_per_day_in_points(prompts):
    p = next(q for q in prompts if q.task == "supply_chain_broken")
    sku = _news_skus(p)[0]
    env, obs, _ = G.build_state(p)
    expected = (G.score(env, obs, p, {sku: 7.0}) - G.score(env, obs, p, {})) * 100 / G.HORIZON
    assert G.GRPOReward().score_answer(p, json.dumps({sku: 7}))["reward"] == pytest.approx(expected)


def test_rewards_are_deterministic_and_answers_share_futures(prompts):
    """Two separate reward objects, two orders of evaluation: the same answer, the same reward."""
    sample = prompts[::30]
    answers = [json.dumps({s: v for s in _news_skus(p)}) for p in sample for v in (2, 7)]
    targets = [p for p in sample for _ in (2, 7)]
    a = [G.GRPOReward().score_answer(p, t)["reward"] for p, t in zip(targets, answers)]
    r = G.GRPOReward()
    b = [r.score_answer(p, t)["reward"] for p, t in reversed(list(zip(targets, answers)))][::-1]
    assert a == b


def test_score_does_not_depend_on_the_episodes_real_future(prompts):
    p = prompts[len(prompts) // 2]
    env, obs, _ = G.build_state(p)
    sku = _news_skus(p)[0]
    before = G.score(env, obs, p, {sku: 7.0})
    env.resample_future_noise(Random(123))      # change the real episode's future
    assert G.score(env, obs, p, {sku: 7.0}) == before


def test_scoring_leaves_the_state_untouched(prompts):
    p = prompts[5]
    env, obs, _ = G.build_state(p)
    snapshot = copy.deepcopy((env._inventory_state, env._open_orders, env._demand_z))
    G.score(env, obs, p, {_news_skus(p)[0]: 10.0})
    assert (env._inventory_state, env._open_orders, env._demand_z) == snapshot


def test_answer_applies_today_only(prompts, monkeypatch):
    calls = []
    real = G.baseline_policy
    monkeypatch.setattr(G, "baseline_policy", lambda obs, extra_days=None: calls.append(extra_days) or real(obs, extra_days=extra_days))
    p = prompts[3]
    env, obs, _ = G.build_state(p)
    G.play_forward(copy.deepcopy(env), copy.deepcopy(obs), {"insulin": 5.0})
    assert calls[0] == {"insulin": 5.0} and all(c is None for c in calls[1:]) and len(calls) <= G.HORIZON


def test_the_reward_separates_answers(prompts):
    """Across news days some answers beat 0 and some lose to it: there is something to learn."""
    r = G.GRPOReward()
    rewards = [r.score_answer(p, json.dumps({s: v for s in _news_skus(p)}))["reward"]
               for p in prompts[::15] for v in (2, 10)]
    assert max(rewards) > 0.3 and min(rewards) < -0.3


def test_trl_style_call_with_chat_completions(prompts):
    p = prompts[0]
    completions = [[{"role": "assistant", "content": "{}"}], [{"role": "assistant", "content": "no idea"}], "{}"]
    out = G.GRPOReward()(completions, task=[p.task] * 3, seed=[p.seed] * 3, day=[p.day] * 3,
                         scenario=[p.scenario] * 3, news=[p.news] * 3, prompts=None)
    assert out == [0.0, G.PENALTY_NO_JSON, 0.0]


def test_train_and_test_seeds_never_mix():
    assert all(p.seed in TRAIN_SEEDS for p in G.news_day_prompts(TASKS[:1], TRAIN_SEEDS[:2]))
    assert not set(TRAIN_SEEDS[:2]) & set(TEST_SEEDS)
