"""
Evaluation on a second GPU, in its own process, while the notebook trains on the first.

    CUDA_VISIBLE_DEVICES=1 python training/eval_worker.py jobs.json

jobs.json:
    {"model": "unsloth/Phi-4-mini-instruct", "out": "/kaggle/working/grpo_run",
     "max_seq_length": 3168, "gen_batch": 8, "max_new_tokens": 96,
     "jobs": [
        {"kind": "probe", "name": "base_probe", "seeds": [9000, ...], "count": 100, "salt": "probe"},
        {"kind": "episodes", "name": "base_random_news2", "tasks": [...], "seeds": [...],
         "news": 2, "scenario": "random", "label": "phi4-mini-base"},
        {"kind": "watch", "checkpoints": ".../checkpoints", "done_flag": ".../training_done",
         "seeds": [9000, ...], "count": 100, "salt": "probe", "history": "probe_history.jsonl"}
     ]}

Each job may name an "adapter" (a LoRA folder); without one it uses the base model.

    probe     -> <out>/<name>.json   {"summary": ..., "results": [...]}
    episodes  -> <out>/<name>.jsonl  one record per episode (as eval.run_episode)
    watch     -> probes every checkpoint-<step> the trainer saves, until done_flag exists,
                 appending {"step": ..., **summary} to <out>/<history>

Progress goes to stdout (the notebook sends it to a log file) and the finished job names to
<out>/worker_<tag>_status.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval as E  # noqa: E402
from training import episodes as EP  # noqa: E402
from training import probe as P  # noqa: E402

Generate = Callable[[List[List[Dict[str, str]]]], List[str]]
# load(model name or adapter folder) -> (generate, handle to free)
Loader = Callable[[str], Tuple[Generate, Any]]
CHECKPOINT = re.compile(r"checkpoint-(\d+)$")


def gpu_loader(spec: Dict[str, Any]) -> Loader:
    from training.gpu import load_model, make_generate

    def load(name_or_path: str) -> Tuple[Generate, Any]:
        model, tokenizer = load_model(name_or_path, spec["max_seq_length"])
        return make_generate(model, tokenizer, spec["gen_batch"], spec["max_new_tokens"]), (model, tokenizer)

    return load


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def probe_job(job: Dict[str, Any], generate: Generate, out: Path) -> Dict[str, Any]:
    items = P.build_probe(seeds=job["seeds"], count=job["count"], salt=job.get("salt", "probe"))
    results = P.score_probe(items, generate([i.messages for i in items]))
    summary = P.summarize_probe(items, results)
    if job.get("name"):
        (out / f"{job['name']}.json").write_text(json.dumps({"summary": summary, "results": results}, default=float))
    return summary


def episodes_job(job: Dict[str, Any], generate: Generate, out: Path) -> List[Dict[str, Any]]:
    records = EP.run_llm_episodes(job["tasks"], job["seeds"], generate, news=job.get("news", 2),
                                  scenario=job.get("scenario", "random"), model=job.get("label", ""),
                                  save_replies=job.get("save_replies", False))
    E.write_jsonl(out / f"{job['name']}.jsonl", records)
    return records


def checkpoints(folder: Path) -> List[Tuple[int, Path]]:
    """Finished checkpoints (trainer_state.json is written last), oldest first."""
    found = []
    for path in folder.glob("checkpoint-*") if folder.exists() else []:
        m = CHECKPOINT.search(path.name)
        if m and (path / "trainer_state.json").exists():
            found.append((int(m.group(1)), path))
    return sorted(found)


def watch_job(job: Dict[str, Any], load: Loader, release: Callable[[Any], None], out: Path,
              poll_seconds: float = 60.0, settle_seconds: float = 20.0) -> None:
    folder, done_flag = Path(job["checkpoints"]), Path(job["done_flag"])
    history = out / job.get("history", "probe_history.jsonl")
    seen = set()
    while True:
        finished = done_flag.exists()   # read before listing, so a last checkpoint saved with the flag is seen
        new = [(s, p) for s, p in checkpoints(folder) if s not in seen]
        for step, path in new:
            time.sleep(settle_seconds)
            log(f"probe exam on {path.name}")
            generate, handle = load(str(path))
            summary = probe_job({**job, "name": None}, generate, out)
            release(handle)
            with history.open("a") as f:
                f.write(json.dumps({"step": step, **summary}, default=float) + "\n")
            log(f"step {step}: reward {summary['reward']:+.3f}, {summary['pct_of_best']:.1f}% of best")
            seen.add(step)
        if finished and not new:
            return
        if not new:
            time.sleep(poll_seconds)


def run(spec: Dict[str, Any], load: Optional[Loader] = None, release: Optional[Callable[[Any], None]] = None,
        **watch_kwargs: Any) -> List[str]:
    """Run the jobs in order; returns the names of the finished jobs."""
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    if load is None:
        load = gpu_loader(spec)
    if release is None:
        from training.gpu import free as release
    status = out / f"worker_{spec.get('tag', 'gpu1')}_status.json"
    done: List[str] = []
    loaded_key, generate, handle = None, None, None
    for job in spec["jobs"]:
        name = job.get("name", job["kind"])
        started = time.time()
        log(f"start {name}")
        if job["kind"] == "watch":
            if handle is not None:
                release(handle)
                loaded_key, generate, handle = None, None, None
            watch_job(job, load, release, out, **watch_kwargs)
        else:
            key = job.get("adapter") or spec["model"]
            if key != loaded_key:
                if handle is not None:
                    release(handle)
                generate, handle = load(key)
                loaded_key = key
            if job["kind"] == "probe":
                s = probe_job(job, generate, out)
                log(f"{name}: reward {s['reward']:+.3f}, {s['pct_of_best']:.1f}% of best")
            elif job["kind"] == "episodes":
                records = episodes_job(job, generate, out)
                log(f"{name}: {len(records)} episodes, mean score {E.summarize(records)['ALL']['mean']:.4f}")
            else:
                raise ValueError(f"unknown job kind {job['kind']!r}")
        done.append(name)
        status.write_text(json.dumps({"done": done, "running": None}))
        log(f"finished {name} in {(time.time() - started) / 60:.1f} min")
    if handle is not None:
        release(handle)
    return done


def main(argv: Optional[List[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    spec = json.loads(Path(argv[0]).read_text())
    try:
        run(spec)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
