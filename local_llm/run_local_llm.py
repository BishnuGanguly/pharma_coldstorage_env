"""
Run a local LLM (served by Ollama) on the Pharma environment and compare it with
the baseline and the oracle on exactly the same seeds.

One-time setup (install Ollama first: https://ollama.com/download):

    uv run python local_llm/run_local_llm.py --setup

That downloads qwen2.5:1.5b and creates `pharma-qwen15`, a copy with a context
window large enough for this environment's prompts (see local_llm/Modelfile).

Then, with Ollama running:

    uv run python local_llm/run_local_llm.py                           # 1 flu_season episode
    uv run python local_llm/run_local_llm.py --tasks all --episodes 3  # 3 seeds of every task

Each episode is 60 model calls. The script prints progress, a score table, the
per-seed comparison with the baseline, and the model's first replies. Results
are written to results/ in the same format as eval.py, so they can be compared
later with `python eval.py --compare ...`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval as E  # noqa: E402
from tasks import SEED_SPLITS, TASK_REGISTRY  # noqa: E402

BASE_MODEL = "qwen2.5:1.5b"
DEFAULT_MODEL = "pharma-qwen15"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
MODELFILE = Path(__file__).resolve().parent / "Modelfile"


# ---------------------------------------------------------------------------
# Setup and checks
# ---------------------------------------------------------------------------

def setup(model: str) -> None:
    """Download the base model and create the large-context copy with the Ollama CLI."""
    if shutil.which("ollama") is None:
        sys.exit("The `ollama` command was not found. Install Ollama from https://ollama.com/download, "
                 "make sure it is running, then run this again.")
    for command in (["ollama", "pull", BASE_MODEL], ["ollama", "create", model, "-f", str(MODELFILE)]):
        print("$", " ".join(command), flush=True)
        if subprocess.run(command).returncode != 0:
            sys.exit(f"`{' '.join(command)}` failed. Is the Ollama app or `ollama serve` running?")
    print(f"\nModel `{model}` is ready.\n", flush=True)


def check_server(base_url: str, model: str, api_key: str = "not-needed") -> None:
    """Fail early, with instructions, if the server is down or the model is missing."""
    from openai import OpenAI

    try:
        available = [m.id for m in OpenAI(base_url=base_url, api_key=api_key, timeout=10, max_retries=0).models.list()]
    except Exception as exc:
        sys.exit(f"Cannot reach a model server at {base_url} ({type(exc).__name__}).\n"
                 "Start Ollama (open the app, or run `ollama serve` in another terminal) and try again.")
    names = {m for m in available} | {m.split(":")[0] for m in available}
    if model not in names:
        listed = ", ".join(sorted(available)) or "none"
        sys.exit(f"Model `{model}` is not available on {base_url} (available: {listed}).\n"
                 f"Run: uv run python local_llm/run_local_llm.py --setup")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_llm(tasks: Sequence[str], seeds: Sequence[int], split: str, llm: Dict[str, str]) -> List[Dict[str, Any]]:
    records = []
    total = len(tasks) * len(seeds)
    for i, (task, seed) in enumerate(((t, s) for t in tasks for s in seeds), start=1):
        started = time.time()
        print(f"[{i}/{total}] {task}, seed {seed}: ", end="", flush=True)

        def progress(day: int, reply: str) -> None:
            if (day + 1) % 10 == 0:
                print(f"day {day + 1} ", end="", flush=True)

        record = E.run_episode("llm", task, seed, split=split, llm=llm, save_steps=True, on_day=progress)
        print(f"-> score {record['score']:.3f}, {record['parse_failures']} replies without JSON "
              f"({time.time() - started:.0f}s)", flush=True)
        records.append(record)
    return records


def print_report(llm_records, baseline_records, oracle_records, show: int) -> None:
    print("\n=== Mean score per task ===")
    print(f"{'task':22s} {'LLM':>7s} {'baseline':>9s} {'oracle':>7s}")
    summaries = [E.summarize(r) for r in (llm_records, baseline_records, oracle_records)]
    for task in summaries[0]:
        print(f"{task:22s} " + " ".join(f"{s[task]['mean']:{w}.3f}" for s, w in zip(summaries, (7, 9, 7))))

    print("\n=== LLM minus baseline, seed by seed ===")
    E.print_comparison(E.paired_comparison(llm_records, baseline_records, oracle_records), ("LLM", "baseline"))

    replies = sum(r["days"] for r in llm_records)
    failures = sum(r["parse_failures"] for r in llm_records)
    errors = sum(r["llm_errors"] for r in llm_records)
    print(f"\nReplies without any JSON object: {failures}/{replies} ({failures / max(replies, 1):.1%}); "
          f"failed model calls: {errors}")

    if show > 0 and llm_records:
        first = llm_records[0]
        print(f"\n=== First {show} replies ({first['task']}, seed {first['seed']}) ===")
        for step in first["steps"][:show]:
            reply = " ".join(step["reply"].split())
            print(f"day {step['day']:2d}  reward {step['reward']:+.2f}  {reply[:110]}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", action="store_true", help="download and create the model with Ollama first")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model name on the server (default {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible endpoint")
    parser.add_argument("--tasks", default="flu_season", help="comma-separated task names, or 'all'")
    parser.add_argument("--episodes", type=int, default=1, help="seeds per task")
    parser.add_argument("--split", choices=sorted(SEED_SPLITS), default="test")
    parser.add_argument("--show", type=int, default=10, help="how many of the model's replies to print")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    tasks = list(TASK_REGISTRY) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    unknown = [t for t in tasks if t not in TASK_REGISTRY]
    if unknown:
        parser.error(f"unknown task(s) {unknown}; choose from {sorted(TASK_REGISTRY)}")
    seeds = list(SEED_SPLITS[args.split][:args.episodes])

    if args.setup:
        setup(args.model)
    check_server(args.base_url, args.model)
    llm = {"model": args.model, "base_url": args.base_url, "api_key": "not-needed"}

    print(f"Model {args.model} at {args.base_url}: {len(tasks)} task(s) x {len(seeds)} seed(s) "
          f"= {len(tasks) * len(seeds) * 60} model calls\n", flush=True)
    llm_records = run_llm(tasks, seeds, args.split, llm)
    baseline_records = [E.run_episode("baseline", t, s, split=args.split) for t in tasks for s in seeds]
    oracle_records = [E.run_episode("oracle", t, s, split=args.split) for t in tasks for s in seeds]

    stem = f"{args.split}_{'-'.join(tasks) if args.tasks != 'all' else 'all'}_{len(seeds)}"
    model_name = "".join(c if c.isalnum() or c == "." else "-" for c in args.model)
    paths = {
        f"llm_{model_name}": llm_records,
        "baseline": baseline_records,
        "oracle": oracle_records,
    }
    for name, records in paths.items():
        E.write_jsonl(args.results_dir / f"{name}_{stem}.jsonl", records)

    print_report(llm_records, baseline_records, oracle_records, args.show)
    print(f"\nResults written to {args.results_dir}/ (*_{stem}.jsonl)")


if __name__ == "__main__":
    main()
