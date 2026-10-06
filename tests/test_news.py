"""Tests for news (news.py), the news-aware agents (agents.py) and their prompt (inference.py)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import eval as E
import inference
from agents import MAX_COVER_DAYS, NewsLLMAgent, NewsRuleAgent, baseline_lead_time, baseline_policy
from models import PharmaAction
from news import ANNOUNCE_DAYS_AHEAD, add_news, find_events, visible_news
from server.my_env_environment import PharmaEnvironment
from tasks import TASK_REGISTRY, TEST_SEEDS, get_task_config

SEED = TEST_SEEDS[0]
TASKS = sorted(TASK_REGISTRY)


def fake_client(text="{}", fail=False):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if fail:
            raise TimeoutError("model too slow")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


# ---------------------------------------------------------------------------
# Events and text
# ---------------------------------------------------------------------------

def test_events_follow_the_task_curves():
    kinds = {t: sorted((e["kind"], e["sku_id"], e["start_day"], e["end_day"])
                       for e in find_events(get_task_config(t, SEED))) for t in TASKS}
    assert kinds["supply_chain_broken"] == [("supplier_delay", "insulin", 15, 25),
                                            ("supplier_delay", "paracetamol", 30, 40)]
    assert ("demand_surge", "paracetamol", 14, 46) in kinds["flu_season"]
    # The epidemic's two waves are two separate surges.
    waves = [k for k in kinds["epidemic_two_wave"] if k[:2] == ("demand_surge", "hydroxychloroquine")]
    assert len(waves) == 2 and waves[0][3] < waves[1][2]


@pytest.mark.parametrize("task_name", TASKS)
@pytest.mark.parametrize("level", [1, 2])
def test_news_is_published_5_to_10_days_ahead(task_name, level):
    events = get_task_config(task_name, SEED, news=level).news
    assert events
    for e in events:
        ahead = e.start_day - e.announce_day
        assert e.announce_day == 0 or ANNOUNCE_DAYS_AHEAD[0] <= ahead <= ANNOUNCE_DAYS_AHEAD[1]


def test_level_1_is_exact_and_level_2_is_varied_but_deterministic():
    exact = get_task_config("supply_chain_broken", SEED, news=1).news[0]
    assert "insulin" in exact.message and "day 15" in exact.message and "day 25" in exact.message
    varied = [e.message for t in TASKS for s in TEST_SEEDS[:5] for e in get_task_config(t, s, news=2).news]
    assert not any("day 15" in m for m in varied)          # timing is relative, not exact
    assert len({m.split(":")[0].split(" ")[0] for m in varied}) > 2   # several templates
    assert varied == [e.message for t in TASKS for s in TEST_SEEDS[:5] for e in get_task_config(t, s, news=2).news]


def test_news_level_0_and_bad_levels():
    config = get_task_config("flu_season", SEED)
    assert config.news == [] and add_news(config, 0, SEED) is config
    with pytest.raises(ValueError):
        add_news(config, 3, SEED)


@pytest.mark.parametrize("task_name", TASKS)
def test_news_never_changes_the_world(task_name):
    plain = get_task_config(task_name, SEED)
    with_news = get_task_config(task_name, SEED, news=2)
    assert plain.model_dump(exclude={"news"}) == with_news.model_dump(exclude={"news"})
    # The baseline ignores news, so it scores exactly the same.
    assert (E.run_episode("baseline", task_name, SEED)["score"]
            == E.run_episode("baseline", task_name, SEED, news=2)["score"])


def test_observation_shows_news_from_publication_to_end():
    env = PharmaEnvironment()
    obs = env.reset(task_name="supply_chain_broken", seed=SEED, news=1)
    event = env._episode_config.news[0]
    seen = {}
    while not obs.done:
        seen[obs.current_date] = list(obs.news)
        obs = env.step(PharmaAction(message="{}"))
    for day, items in seen.items():
        shown = any(item == f"Day {event.announce_day}: {event.message}" for item in items)
        assert shown == (event.announce_day <= day <= event.end_day)
    assert visible_news(env._episode_config, 0) == []


def test_report_includes_news_only_when_there_is_some():
    env = PharmaEnvironment()
    assert "news" not in inference.observation_to_dict(env.reset(task_name="flu_season", seed=SEED))
    obs = env.reset(task_name="epidemic_two_wave", seed=SEED, news=2)
    assert inference.observation_to_dict(obs)["news"] == obs.news


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def test_extra_days_are_capped_for_storage():
    env = PharmaEnvironment()
    obs = env.reset(task_name="flu_season", seed=SEED)
    for _ in range(5):
        obs = env.step(PharmaAction(message="{}"))
    plain = baseline_policy(obs)
    huge = baseline_policy(obs, extra_days={"paracetamol": 1000, "vitamins": -5})
    sku = obs.skus["paracetamol"]
    demand = max(sku.avg_demand_last_5_days, sku.avg_demand_per_day)
    cover = max(MAX_COVER_DAYS, baseline_lead_time(sku) + 3.0)
    assert huge["paracetamol"] == round(demand * cover - sku.inventory_on_hand)
    assert huge["vitamins"] == plain["vitamins"]       # negative extra is ignored


@pytest.mark.parametrize("task_name", TASKS)
def test_news_rule_agent_beats_the_baseline(task_name):
    baseline = E.run_episode("baseline", task_name, SEED, news=1)["score"]
    assert E.run_episode("baseline_news", task_name, SEED, news=1)["score"] > baseline
    # Without news it is the baseline.
    assert E.run_episode("baseline_news", task_name, SEED)["score"] == E.run_episode("baseline", task_name, SEED)["score"]


def test_news_llm_agent_only_calls_the_model_on_news_days(monkeypatch):
    client, calls = fake_client('{"insulin": 6, "Amlodipine": 2, "HCQ": "lots", "aspirin": 9}')
    monkeypatch.setattr(E, "make_agent", lambda name, llm=None: NewsLLMAgent("stub", "x", "x", client=client))
    record = E.run_episode("custom", "supply_chain_broken", SEED, news=2, save_steps=True)
    news_days = [s["day"] for s in record["steps"] if s["news"]]
    assert len(calls) == len(news_days) > 0
    assert record["parse_failures"] == 0 and record["llm_errors"] == 0
    prompt = calls[0]["messages"][1]["content"]
    assert f"Today is day {news_days[0]}" in prompt and "PRODUCTS:" in prompt
    assert calls[0]["messages"][0]["content"] == inference.NEWS_SYSTEM_PROMPT


def test_news_llm_agent_without_news_is_the_baseline():
    client, calls = fake_client()
    env = PharmaEnvironment()
    obs = env.reset(task_name="flu_season", seed=SEED)
    for _ in range(4):
        obs = env.step(PharmaAction(message="{}"))
    agent = NewsLLMAgent("stub", "x", "x", client=client)
    reply, error = agent.act(obs, [])
    assert json.loads(reply) == baseline_policy(obs) and error is None and calls == []


def test_news_llm_agent_applies_the_reply_and_survives_failures():
    env = PharmaEnvironment()
    obs = env.reset(task_name="supply_chain_broken", seed=SEED, news=1)
    while not obs.news:
        obs = env.step(PharmaAction(message=json.dumps(baseline_policy(obs))))
    client, _ = fake_client('Insulin needs more: {"insulin": 5, "paracetamol": 0}')
    reply, error = NewsLLMAgent("stub", "x", "x", client=client).act(obs, [])
    assert error is None and json.loads(reply) == baseline_policy(obs, extra_days={"insulin": 5})
    client, _ = fake_client(fail=True)
    reply, error = NewsLLMAgent("stub", "x", "x", client=client).act(obs, [])
    assert "TimeoutError" in error and json.loads(reply) == baseline_policy(obs)


def test_parse_extra_days_accepts_aliases_and_skips_junk():
    obs = PharmaEnvironment().reset(task_name="flu_season", seed=SEED)
    reply = {"Insulin": 3, "amlodipine": "2.5", "HCQ": 4, "vitamins": True, "paracetamol": "nan", "x": 1}
    assert inference.parse_extra_days(obs, reply) == {"insulin": 3.0, "bp_medication": 2.5, "hydroxychloroquine": 4.0}


def test_rule_agent_reads_events_from_the_environment():
    env = PharmaEnvironment()
    env.reset(task_name="flu_season", seed=SEED, news=2)
    agent = NewsRuleAgent()
    agent.start_episode(env)
    assert agent._events == env._episode_config.news


def test_command_line_news_flag(tmp_path):
    out = tmp_path / "news.jsonl"
    E.main(["--agent", "baseline_news", "--news", "2", "--episodes", "1", "--tasks", "flu_season",
            "--workers", "1", "--out", str(out)])
    assert json.loads(out.read_text())["news"] == 2


def test_news_prompt_has_no_example_answer():
    """Regression: the prompt ended with an all-zero example answer. A model that copies it
    plays exactly like the baseline every day, so every sampled answer gets the same
    reward and GRPO has nothing to learn from."""
    from test_prompt import SKU_WITH_NUMBER

    assert not SKU_WITH_NUMBER.search(inference.NEWS_SYSTEM_PROMPT)
    assert '{"<product_name>": <extra_days>' in inference.NEWS_SYSTEM_PROMPT
    env = PharmaEnvironment()
    obs = env.reset(task_name="flu_season", seed=TEST_SEEDS[0], news=2)
    while not obs.news:
        obs = env.step(PharmaAction(message="{}"))
    prompt = inference.build_news_prompt(obs)
    assert not SKU_WITH_NUMBER.search(prompt[prompt.index("Extra days"):])
