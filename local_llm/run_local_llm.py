"""
Run a local LLM (served by Ollama) on the Pharma environment and compare it with
the baseline and the oracle on exactly the same seeds.

One-time setup (install Ollama first: https://ollama.com/download):

    uv run python local_llm/run_local_llm.py --setup                             # Gemma 2 2B (default)
    uv run python local_llm/run_local_llm.py --setup --base-model qwen2.5:3b     # any other Ollama model

That downloads the base model and creates `pharma-<base-model>` (e.g.
`pharma-gemma2-2b`), a copy with a 4,096-token context window: the prompts are
about 1,800 tokens, and Ollama may cut longer prompts from the start, losing the
rules and output format.

Then, with Ollama running:

    uv run python local_llm/run_local_llm.py                           # 1 flu_season episode
    uv run python local_llm/run_local_llm.py --tasks all --episodes 3  # 3 seeds of every task
    uv run python local_llm/run_local_llm.py --base-model qwen2.5:3b   # another model

Each episode is 60 model calls. The script prints progress, a score table, the
per-seed comparison with the baseline, and the model's first replies. Results
are written to results/ in the same format as eval.py, so they can be compared
later with `python eval.py --compare ...`.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval as E  # noqa: E402
from tasks import SEED_SPLITS, TASK_REGISTRY  # noqa: E402

# Gemma 2 2B (about 2.6 billion parameters, ~1.6 GB download). Any Ollama model
# tag works with --base-model, e.g. qwen2.5:1.5b, qwen2.5:3b, qwen2.5:7b, gemma2:9b;
# larger models follow the rules better but are slower.
DEFAULT_BASE_MODEL = "gemma2:2b"
CONTEXT_TOKENS = 4096
DEFAULT_BASE_URL = "http://localhost:11434/v1"
# Seconds allowed per model call. A 3B+ model reading a ~1,800-token prompt on a
# laptop CPU can take well over a minute, so the default is generous.
DEFAULT_TIMEOUT = 600.0


def model_name(base_model: str) -> str:
    """Name of the large-context copy, e.g. gemma2:2b -> pharma-gemma2-2b."""
    return "pharma-" + re.sub(r"[^a-z0-9]+", "-", base_model.lower()).strip("-")


def modelfile_text(base_model: str) -> str:
    """Ollama Modelfile: the base model with a large enough context window."""
    return f"FROM {base_model}\nPARAMETER num_ctx {CONTEXT_TOKENS}\n"


# ---------------------------------------------------------------------------
# Setup and checks
# ---------------------------------------------------------------------------

def setup(base_model: str, model: str) -> None:
    """Download `base_model` and create `model` from it with a larger context, using the Ollama CLI."""
    if shutil.which("ollama") is None:
        sys.exit("The `ollama` command was not found. Install Ollama from https://ollama.com/download, "
                 "make sure it is running, then run this again.")
    with tempfile.TemporaryDirectory() as tmp:
        modelfile = Path(tmp) / "Modelfile"
        modelfile.write_text(modelfile_text(base_model))
        for command in (["ollama", "pull", base_model], ["ollama", "create", model, "-f", str(modelfile)]):
            print("$", " ".join(command), flush=True)
            if subprocess.run(command).returncode != 0:
                sys.exit(f"`{' '.join(command)}` failed. Is the Ollama app or `ollama serve` running?")
    print(f"\nModel `{model}` is ready.\n", flush=True)


def check_server(base_url: str, model: str, setup_hint: str = "--setup", api_key: str = "not-needed") -> None:
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
                 f"Run: uv run python local_llm/run_local_llm.py {setup_hint}")


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
            # A dot per day, the day number every 10: large models on a CPU are slow.
            print(f"{day + 1}" if (day + 1) % 10 == 0 else ".", end="", flush=True)

        record = E.run_episode("llm", task, seed, split=split, llm=llm, save_steps=True, on_day=progress)
        print(f" -> score {record['score']:.3f}, {record['parse_failures']} replies without JSON "
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
            line = f"day {step['day']:2d}  reward {step['reward']:+.2f}  model: {reply[:90]}"
            if step.get("orders_sent") and step["orders_sent"] != step["reply"]:
                line += f"\n{'':21s}units sent: {step['orders_sent'][:90]}"
            print(line)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", action="store_true", help="download and create the model with Ollama first")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help=f"Ollama model to use (default {DEFAULT_BASE_MODEL}), e.g. qwen2.5:3b")
    parser.add_argument("--model", default=None,
                        help="model name on the server, to skip the pharma-<base-model> copy (e.g. a non-Ollama server)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds allowed per model call")
    parser.add_argument("--action", choices=("days", "units"), default="days",
                        help="the model answers in days of stock per SKU (converted to units; default) or in units")
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

    model = args.model or model_name(args.base_model)
    if args.setup:
        setup(args.base_model, model)
    hint = "--setup" + ("" if args.base_model == DEFAULT_BASE_MODEL else f" --base-model {args.base_model}")
    check_server(args.base_url, model, setup_hint=hint)
    llm = {"model": model, "base_url": args.base_url, "api_key": "not-needed",
           "timeout": args.timeout, "action": args.action}

    print(f"Model {model} at {args.base_url}, answering in {args.action}: {len(tasks)} task(s) x {len(seeds)} seed(s) "
          f"= {len(tasks) * len(seeds) * 60} model calls\n", flush=True)
    llm_records = run_llm(tasks, seeds, args.split, llm)
    baseline_records = [E.run_episode("baseline", t, s, split=args.split) for t in tasks for s in seeds]
    oracle_records = [E.run_episode("oracle", t, s, split=args.split) for t in tasks for s in seeds]

    stem = f"{args.split}_{'-'.join(tasks) if args.tasks != 'all' else 'all'}_{len(seeds)}"
    safe_model = "".join(c if c.isalnum() or c == "." else "-" for c in model)
    paths = {
        f"llm_{safe_model}": llm_records,
        "baseline": baseline_records,
        "oracle": oracle_records,
    }
    for name, records in paths.items():
        E.write_jsonl(args.results_dir / f"{name}_{stem}.jsonl", records)

    print_report(llm_records, baseline_records, oracle_records, args.show)
    print(f"\nResults written to {args.results_dir}/ (*_{stem}.jsonl)")


if __name__ == "__main__":
    main()
