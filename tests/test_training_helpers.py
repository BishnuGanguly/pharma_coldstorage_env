"""The GPU-free helpers of the Kaggle GRPO notebook: probe exam, batched episodes, training monitor."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import eval as E
import inference
from agents import LLMAgent
from models import PharmaAction
from server.my_env_environment import PharmaEnvironment
from tasks import TASK_REGISTRY, TEST_SEEDS, TRAIN_SEEDS
from training import episodes as EP
from training import probe as P
from training.grpo_reward import GRPOReward, PENALTY_NO_JSON
from training.monitor import MonitoredReward

TASKS = sorted(TASK_REGISTRY)


@pytest.fixture(scope="module")
def items():
    return P.build_probe(seeds=P.PROBE_SEEDS[:3], count=12)


def _news_answer(item, days):
    return json.dumps({k: days for k in item.skus})


# --- probe ----------------------------------------------------------------------

def test_probe_is_deterministic_balanced_and_held_out(items):
    assert len(items) == 12 and [i.prompt for i in items] == [i.prompt for i in P.build_probe(P.PROBE_SEEDS[:3], 12)]
    assert {i.prompt.task for i in items} == set(TASKS)
    assert all(i.prompt.seed in P.PROBE_SEEDS for i in items)
    assert all(i.skus and i.reference[0] == 0.0 and i.best >= 0 for i in items)
    assert items[0].messages[0]["role"] == "system"


def test_answering_zero_scores_zero_and_holds_back(items):
    results = P.score_probe(items, ["{}"] * len(items))
    summary = P.summarize_probe(items, results)
    assert summary["reward"] == 0 and summary["pct_of_best"] == 0 and summary["all_zero"] == 1
    assert all(r["held_back"] for r in results if r["held_back"] is not None)


def test_best_reference_answers_score_100_percent(items):
    texts = [_news_answer(i, max(i.reference, key=i.reference.get)) for i in items]
    summary = P.summarize_probe(items, P.score_probe(items, texts))
    assert summary["pct_of_best"] == pytest.approx(100.0)
    assert summary["right_product"] == 1.0


def test_behaviour_checks(items):
    texts = ['{"insulin": 10}'] * len(items)
    results = P.score_probe(items, texts)
    s = P.summarize_probe(items, results)
    no_insulin = [i for i in items if "insulin" not in i.skus]
    assert no_insulin and s["insulin_habit"] == 1.0
    rambling = P.summarize_probe(items, P.score_probe(items, ["no idea"] * len(items)))
    assert rambling["no_json"] == 1.0 and rambling["reward"] == PENALTY_NO_JSON


def test_compare_probes_is_paired(items):
    zero = P.score_probe(items, ["{}"] * len(items))
    best = P.score_probe(items, [_news_answer(i, max(i.reference, key=i.reference.get)) for i in items])
    diff = P.compare_probes(best, zero)
    assert diff["mean"] == pytest.approx(sum(i.best for i in items) / len(items)) and diff["worse"] == 0


# --- batched episodes -----------------------------------------------------------

def test_empty_answers_play_like_the_baseline():
    calls = []

    def generate(batch):
        calls.append(len(batch))
        return ["{}"] * len(batch)

    seeds = TEST_SEEDS[:2]
    records = EP.run_llm_episodes(TASKS, seeds, generate, news=2, scenario="random")
    baseline = [E.run_episode("baseline", t, s, news=2, scenario="random") for t in TASKS for s in seeds]
    assert [r["score"] for r in records] == [b["score"] for b in baseline]
    assert calls[0] == len(TASKS) * len(seeds) and len(calls) == 60    # one batched call per day
    assert E.paired_comparison(records, baseline)["ALL"]["mean"] == 0


def test_batched_episodes_match_the_llm_agent():
    """A fixed answer through run_llm_episodes == the same answer through agents.LLMAgent."""
    answer = '{"insulin": 4, "paracetamol": 2}'
    stub = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]))))
    agent = LLMAgent("stub", "x", "x", client=stub)
    env = PharmaEnvironment()
    obs = env.reset(task_name="flu_season", seed=TEST_SEEDS[0], news=2, scenario="random")
    history = []
    while not obs.done:
        day = obs.current_date
        reply, _ = agent.act(obs, history)
        obs = env.step(PharmaAction(message=reply))
        history.append((day, reply, inference.build_feedback(obs)))
    expected = E.compute_final_score(obs, env._episode_config)
    got = EP.run_llm_episodes(["flu_season"], TEST_SEEDS[:1], lambda b: [answer] * len(b), save_replies=True)[0]
    assert got["score"] == round(expected, 6) and got["parse_failures"] == 0 and len(got["replies"]) == 60


def test_parse_failures_are_counted():
    rec = EP.run_llm_episodes(["flu_season"], TEST_SEEDS[:1], lambda b: ["hmm"] * len(b))[0]
    assert rec["parse_failures"] == 60


def test_flat_extra_agent_and_rule_runner():
    plus0 = EP.run_rule_episodes(EP.FlatExtraAgent(0), "plus0", TASKS[:1], TEST_SEEDS[:2])
    baseline = [E.run_episode("baseline", TASKS[0], s, news=2, scenario="random") for s in TEST_SEEDS[:2]]
    assert [r["score"] for r in plus0] == [b["score"] for b in baseline]
    plus3 = EP.run_rule_episodes(EP.FlatExtraAgent(3), "plus3", TASKS[:1], TEST_SEEDS[:2])
    assert [r["score"] for r in plus3] != [r["score"] for r in plus0]


# --- monitor --------------------------------------------------------------------

def test_monitored_reward_returns_the_same_rewards_and_health_numbers(items):
    item = items[0]
    p = item.prompt
    texts = ["{}", "rambling", _news_answer(item, 7), '{"vitamins": 30}'] * 2
    cols = dict(task=[p.task] * 8, seed=[p.seed] * 8, day=[p.day] * 8, scenario=[p.scenario] * 8, news=[p.news] * 8)
    m = MonitoredReward()
    assert m(texts, **cols) == GRPOReward()(texts, **cols)
    stats = m.pop_stats()
    assert stats["health/no_json"] == 0.25 and stats["health/all_zero"] == 0.25
    assert stats["health/out_of_range"] == 0.25 and stats["health/zero_std_groups"] == 0.0
    assert m.pop_stats() == {}
    m(["{}"] * 8, **cols)
    assert m.pop_stats()["health/zero_std_groups"] == 1.0


def test_probe_seeds_are_train_seeds():
    assert set(P.PROBE_SEEDS) <= set(TRAIN_SEEDS) and not set(P.PROBE_SEEDS) & set(TEST_SEEDS)
