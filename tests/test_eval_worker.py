"""training/eval_worker.py with a fake model loader (no GPU)."""

from __future__ import annotations

import json
import threading
import time

import eval as E
from tasks import TEST_SEEDS
from training import eval_worker as W
from training import probe as P


class FakeLoader:
    """Answers '{}' for the base model, and +4 insulin for any adapter folder."""

    def __init__(self):
        self.loaded, self.released = [], []

    def __call__(self, name):
        self.loaded.append(name)
        answer = "{}" if name == "base-model" else '{"insulin": 4}'
        return (lambda batch: [answer] * len(batch)), name

    def release(self, handle):
        self.released.append(handle)


def _spec(tmp_path, jobs):
    return {"model": "base-model", "out": str(tmp_path), "max_seq_length": 0, "gen_batch": 8,
            "max_new_tokens": 8, "tag": "test", "jobs": jobs}


def test_probe_and_episode_jobs_write_files_and_reuse_the_model(tmp_path):
    loader = FakeLoader()
    seeds = list(P.PROBE_SEEDS[:2])
    jobs = [
        {"kind": "probe", "name": "base_probe", "seeds": seeds, "count": 6},
        {"kind": "episodes", "name": "base_eps", "tasks": ["flu_season"], "seeds": TEST_SEEDS[:2], "news": 2,
         "scenario": "random"},
        {"kind": "episodes", "name": "lora_eps", "adapter": "/x/lora", "tasks": ["flu_season"],
         "seeds": TEST_SEEDS[:1]},
    ]
    done = W.run(_spec(tmp_path, jobs), load=loader, release=loader.release)
    assert done == ["base_probe", "base_eps", "lora_eps"]
    assert loader.loaded == ["base-model", "/x/lora"] and loader.released == ["base-model", "/x/lora"]
    probe = json.loads((tmp_path / "base_probe.json").read_text())
    assert probe["summary"]["reward"] == 0 and len(probe["results"]) == 6
    base = E.read_jsonl(tmp_path / "base_eps.jsonl")
    baseline = [E.run_episode("baseline", "flu_season", s, news=2, scenario="random") for s in TEST_SEEDS[:2]]
    assert [r["score"] for r in base] == [b["score"] for b in baseline]      # '{}' plays like the baseline
    assert json.loads((tmp_path / "worker_test_status.json").read_text())["done"] == done


def test_watch_probes_each_checkpoint_until_training_is_done(tmp_path):
    loader = FakeLoader()
    ckpts, flag = tmp_path / "checkpoints", tmp_path / "training_done"

    def trainer():
        for step in (25, 50):
            d = ckpts / f"checkpoint-{step}"
            d.mkdir(parents=True)
            time.sleep(0.05)
            (d / "trainer_state.json").write_text("{}")      # a checkpoint counts once this exists
            time.sleep(0.2)
        (ckpts / "checkpoint-75").mkdir()                     # unfinished: never probed
        flag.write_text("done")

    thread = threading.Thread(target=trainer)
    thread.start()
    job = {"kind": "watch", "checkpoints": str(ckpts), "done_flag": str(flag), "seeds": list(P.PROBE_SEEDS[:2]),
           "count": 4, "history": "history.jsonl"}
    W.run(_spec(tmp_path, [job]), load=loader, release=loader.release, poll_seconds=0.05, settle_seconds=0)
    thread.join()
    rows = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [25, 50]
    assert loader.loaded == [str(ckpts / "checkpoint-25"), str(ckpts / "checkpoint-50")]
    assert "pct_of_best" in rows[0]
