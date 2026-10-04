"""End-to-end test of local_llm/run_local_llm.py against a fake Ollama-style server."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local_llm"))
import run_local_llm as R  # noqa: E402


class FakeOllama(BaseHTTPRequestHandler):
    """Serves /v1/models and /v1/chat/completions like Ollama's OpenAI-compatible API."""

    def log_message(self, *args):
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"object": "list", "data": [
            {"id": "pharma-gemma2-2b:latest", "object": "model", "created": 0, "owned_by": "library"}]})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = request["messages"][-1]["content"]
        start = prompt.index("{", prompt.index("TODAY'S INVENTORY REPORT"))
        report, _ = json.JSONDecoder().raw_decode(prompt[start:])
        orders = {
            sku: round(max(d["avg_demand_per_day"], 1) * (d["avg_lead_time"] + 3) - d["inventory_on_hand"])
            for sku, d in report["inventory"].items()
        }
        text = "Restocking.\n```json\n" + json.dumps({k: v for k, v in orders.items() if v > 0}) + "\n```"
        self._send({"id": "x", "object": "chat.completion", "created": 0, "model": request["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})


@pytest.fixture
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_full_run_against_a_local_server(fake_server, tmp_path, capsys):
    R.main(["--base-url", fake_server, "--tasks", "flu_season", "--episodes", "1",
            "--show", "3", "--results-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Mean score per task" in out and "LLM minus baseline" in out
    assert "Replies without any JSON object: 0/60" in out
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ["baseline_test_flu_season_1.jsonl", "llm_pharma-gemma2-2b_test_flu_season_1.jsonl",
                       "oracle_test_flu_season_1.jsonl"]
    llm = json.loads((tmp_path / "llm_pharma-gemma2-2b_test_flu_season_1.jsonl").read_text())
    assert llm["days"] == 60 and llm["score"] > 0.5


def test_missing_model_gives_setup_instructions(fake_server):
    with pytest.raises(SystemExit, match="--setup"):
        R.check_server(fake_server, "some-other-model")


def test_unreachable_server_says_to_start_ollama():
    with pytest.raises(SystemExit, match="ollama serve"):
        R.check_server("http://127.0.0.1:9/v1", "pharma-gemma2-2b")


def test_default_is_gemma2_2b_with_a_4k_context():
    assert R.DEFAULT_BASE_MODEL == "gemma2:2b" and R.model_name("gemma2:2b") == "pharma-gemma2-2b"
    assert R.model_name("qwen2.5:3b") == "pharma-qwen2-5-3b"
    assert R.modelfile_text("gemma2:2b") == "FROM gemma2:2b\nPARAMETER num_ctx 4096\n"


def test_missing_other_model_suggests_its_setup_command(fake_server, capsys):
    with pytest.raises(SystemExit, match="--setup --base-model qwen2.5:3b"):
        R.main(["--base-url", fake_server, "--base-model", "qwen2.5:3b"])


def test_call_timeout_reaches_the_model_client():
    import eval as E
    agent = E.make_agent("llm", {"model": "m", "base_url": "http://127.0.0.1:9/v1", "api_key": "x", "timeout": 600})
    assert agent.client.timeout == 600
    assert E.make_agent("llm", {"model": "m", "base_url": "http://127.0.0.1:9/v1", "api_key": "x"}).client.timeout == 60
