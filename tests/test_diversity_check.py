"""local_llm/diversity_check.py against fake Ollama-style servers (fixed, varied and broken answers)."""

from __future__ import annotations

import json
import random
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local_llm"))
import diversity_check as D  # noqa: E402
from test_local_llm import FakeOllama  # noqa: E402
from tasks import TRAIN_SEEDS  # noqa: E402
from training.grpo_reward import GRPOReward  # noqa: E402

MODEL = "pharma-phi4-mini:latest"


class FixedOllama(FakeOllama):
    models = [MODEL]

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._reply(request, self.answer(request))

    def answer(self, request):
        return '{"insulin": 0, "paracetamol": 0, "bp_medication": 0, "vitamins": 0, "hydroxychloroquine": 0}'


class VariedOllama(FixedOllama):
    """Varies its extra days at temperature 0.9 and above, like a sampling model."""
    rng = random.Random(0)

    def answer(self, request):
        if request.get("temperature", 0) < 0.9:
            return super().answer(request)
        skus = ["insulin", "paracetamol", "bp_medication", "vitamins", "hydroxychloroquine"]
        return "```json\n" + json.dumps({k: self.rng.choice([0, 0, 2, 4, 7, 10]) for k in skus}) + "\n```"


class RamblingOllama(FixedOllama):
    def answer(self, request):
        return "I would keep a bit more stock of everything this week."


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


@pytest.fixture
def run(tmp_path, capsys):
    servers = []

    def _run(handler, *extra):
        server, url = _serve(handler)
        servers.append(server)
        D.main(["--base-url", url, "--server-model", MODEL, "--results-dir", str(tmp_path), *extra])
        out = capsys.readouterr().out
        return out, json.loads((tmp_path / "diversity_pharma-phi4-mini-latest.json").read_text())

    yield _run
    for server in servers:
        server.shutdown()


def test_picks_days_where_the_answer_matters_across_tasks():
    reward = GRPOReward()
    prompts = D.pick_prompts(reward, 3, TRAIN_SEEDS[:40])
    assert len({p.task for p in prompts}) == 3
    assert prompts == D.pick_prompts(GRPOReward(), 3, TRAIN_SEEDS[:40])     # deterministic
    for p in prompts:
        refs = D.reference_rewards(reward, p)
        assert max(refs.values()) - min(refs.values()) >= D.MIN_SPREAD
        assert p.seed in TRAIN_SEEDS and p.scenario == "random" and p.news == 2


def test_identical_answers_are_reported_as_no_variety(run):
    out, report = run(FixedOllama)
    results = report["runs"][0]["results"]
    assert len(results) == 3 and all(r["distinct_answers"] == 1 and r["reward_sd"] == 0 for r in results)
    assert "the same answer every time" in out and "24 model calls" in out


def test_varied_answers_are_scored_and_give_a_go(run):
    out, report = run(VariedOllama, "--temperatures", "0.5,1.0")
    low, high = report["runs"]
    assert all(r["distinct_answers"] == 1 for r in low["results"])
    assert all(r["distinct_answers"] >= D.MIN_DISTINCT for r in high["results"])
    assert "enough variety for GRPO" in high["verdict"] and "same answer every time" in low["verdict"]
    first = high["results"][0]["answers"][0]
    assert set(first) >= {"text", "reward", "gain_points", "penalty", "extra_days"}
    assert "48 model calls" in out


def test_answers_without_json_are_flagged(run):
    out, report = run(RamblingOllama)
    results = report["runs"][0]["results"]
    assert all(r["no_json"] == 8 and r["reward_max"] == -1.0 for r in results)
    assert "had no JSON" in out


def test_bad_temperatures_are_rejected():
    with pytest.raises(SystemExit):
        D.main(["--temperatures", "warm"])
