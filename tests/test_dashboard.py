"""Tests for the live dashboard (server/dashboard.py). No server or network needed."""

from __future__ import annotations

from types import SimpleNamespace

import gradio as gr
import pytest

from server.dashboard import (
    MAX_CONSECUTIVE_LLM_ERRORS,
    BaselineAgent,
    LLMAgent,
    build_dashboard,
    decisions_table,
    kpi_html,
    render_detail,
    reward_figure,
    run_episode,
    score_so_far,
    unmet_heatmap,
)
from tasks import TASK_REGISTRY, compute_final_score


class StubCompletions:
    """Mimics client.chat.completions.create from the openai package."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.prompts = reply, error, []

    def create(self, model, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def stub_llm(reply=None, error=None, action="units") -> LLMAgent:
    client = SimpleNamespace(chat=SimpleNamespace(completions=StubCompletions(reply, error)))
    return LLMAgent(model="stub", api_key="unused", base_url="unused", client=client, action=action)


def last(frames):
    trace = None
    for trace in frames:
        pass
    return trace


@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY))
def test_baseline_plays_full_episode(task_name):
    trace = last(run_episode(task_name, 42, BaselineAgent()))
    assert trace.finished and not trace.stopped_reason
    assert len(trace.rows) == trace.config.no_of_days
    assert score_so_far(trace) == pytest.approx(compute_final_score(trace.state, trace.config))


def test_llm_free_text_reply_is_parsed():
    agent = stub_llm(reply='Stocking up.\n```json\n{"Insulin": 30, "paracetamol": "500"}\n```')
    trace = last(run_episode("flu_season", 1, agent))
    assert trace.finished and trace.llm_errors == 0
    assert trace.rows[0]["orders"] == {"insulin": 30.0, "paracetamol": 500.0}
    prompts = agent.client.chat.completions.prompts
    assert len(prompts) == trace.config.no_of_days
    # From day 2 the prompt carries the previous day's results.
    assert "RESULTS OF THE LAST 1 DAY ---" in prompts[1]


def test_llm_failures_stop_the_run():
    agent = stub_llm(error=RuntimeError("401 invalid token"))
    trace = last(run_episode("flu_season", 1, agent))
    assert not trace.finished
    assert len(trace.rows) == MAX_CONSECUTIVE_LLM_ERRORS
    assert "401 invalid token" in trace.stopped_reason
    assert all(r["orders"] == {} for r in trace.rows)


def test_rendering_works_before_and_after_a_run():
    frames = run_episode("epidemic_two_wave", 3, BaselineAgent())
    empty = next(frames)
    finished = last(frames)
    for trace in (empty, finished):
        for sku_id in trace.config.skus:
            stock, lead, orders, table = render_detail(trace, sku_id)
            assert len(table) == len(trace.rows)
        reward_figure(trace)
        unmet_heatmap(trace)
        assert len(decisions_table(trace)) == len(trace.rows)
        assert "Score so far" in kpi_html(trace)


def test_build_dashboard_returns_blocks():
    assert isinstance(build_dashboard(None, None, None, False, "pharma_env", ""), gr.Blocks)
