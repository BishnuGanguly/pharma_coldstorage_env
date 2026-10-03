"""Tests for the evaluation script (eval.py) and the reference agents (agents.py)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import eval as E
from agents import Agent, LLMAgent
from tasks import TASK_REGISTRY, TEST_SEEDS, TRAIN_SEEDS

TEST_SEED = TEST_SEEDS[0]


class FixedReplyAgent(Agent):
    def __init__(self, reply, error=None):
        self.reply, self.error = reply, error

    def act(self, obs, history):
        return self.reply, self.error


def use_agent(monkeypatch, agent_factory):
    original = E.make_agent
    monkeypatch.setattr(E, "make_agent", lambda name, llm=None: agent_factory() if name == "custom" else original(name, llm))


def test_seed_splits_do_not_overlap():
    assert set(TRAIN_SEEDS).isdisjoint(TEST_SEEDS)
    assert len(TRAIN_SEEDS) == len(TEST_SEEDS) == 10_000


def test_episode_record_is_complete_and_deterministic():
    first = E.run_episode("baseline", "flu_season", TEST_SEED, split="test", save_steps=True)
    second = E.run_episode("baseline", "flu_season", TEST_SEED, split="test", save_steps=True)
    assert first == second
    for key in ("agent", "task", "seed", "split", "score", "avg_fill", "stockout_sku_days",
                "overflow_days", "units_wasted", "parse_failures", "llm_errors", "days", "steps"):
        assert key in first
    assert first["days"] == len(first["steps"]) == 60
    json.dumps(first)   # serialisable


@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY))
def test_floor_baseline_and_ceiling_are_ordered(task_name):
    nothing = E.run_episode("nothing", task_name, TEST_SEED)["score"]
    baseline = E.run_episode("baseline", task_name, TEST_SEED)["score"]
    oracle = E.run_episode("oracle", task_name, TEST_SEED)
    assert nothing < 0.02 < baseline < oracle["score"]
    assert oracle["overflow_days"] == 0


def test_replies_without_json_are_counted(monkeypatch):
    use_agent(monkeypatch, lambda: FixedReplyAgent("I would rather not order today."))
    assert E.run_episode("custom", "flu_season", TEST_SEED)["parse_failures"] == 60
    use_agent(monkeypatch, lambda: FixedReplyAgent("Nothing needed: {}"))
    assert E.run_episode("custom", "flu_season", TEST_SEED)["parse_failures"] == 0


def test_llm_failing_from_the_start_stops_the_run(monkeypatch):
    use_agent(monkeypatch, lambda: FixedReplyAgent("{}", error="AuthenticationError: 401"))
    with pytest.raises(RuntimeError, match="401"):
        E.run_episode("custom", "flu_season", TEST_SEED)


def test_llm_agent_runs_through_eval(monkeypatch):
    def create(model, messages, **kwargs):
        message = SimpleNamespace(content='Restocking.\n```json\n{"paracetamol": 400}\n```')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    use_agent(monkeypatch, lambda: LLMAgent(model="stub", api_key="x", base_url="x", client=client))
    record = E.run_episode("custom", "flu_season", TEST_SEED, save_steps=True)
    assert record["parse_failures"] == 0 and record["llm_errors"] == 0
    assert record["score"] > 0.02


def test_parallel_and_serial_runs_match():
    serial = E.evaluate("baseline", ["epidemic_two_wave"], "test", episodes=3, workers=1)
    parallel = E.evaluate("baseline", ["epidemic_two_wave"], "test", episodes=3, workers=2)
    assert serial == parallel
    assert [r["seed"] for r in serial] == list(TEST_SEEDS[:3])


def test_paired_comparison_and_gap_closed():
    def records(scores):
        return [{"task": "t", "seed": s, "score": v} for s, v in enumerate(scores)]
    reference = records([0.80, 0.70, 0.90])
    candidate = records([0.85, 0.74, 0.92])
    oracle = records([0.90, 0.80, 0.94])
    result = E.paired_comparison(candidate, reference, oracle)
    assert result["ALL"]["n"] == 3
    assert result["ALL"]["mean"] == pytest.approx((0.05 + 0.04 + 0.02) / 3)
    assert result["ALL"]["gap_closed"] == pytest.approx(0.11 / 0.24)
    # Seeds missing from the reference are ignored.
    assert E.paired_comparison(records([0.5, 0.5, 0.5, 0.5]), reference)["ALL"]["n"] == 3


def test_command_line_writes_results(tmp_path, capsys):
    out = tmp_path / "nothing.jsonl"
    E.main(["--agent", "nothing", "--episodes", "2", "--tasks", "flu_season", "--workers", "1", "--out", str(out)])
    lines = out.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["seed"] == TEST_SEEDS[0]
    assert "flu_season" in capsys.readouterr().out
    E.main(["--compare", str(out), str(out)])
    assert "+0.000" in capsys.readouterr().out
