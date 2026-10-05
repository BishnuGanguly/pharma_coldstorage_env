"""
Run a local LLM (served by Ollama) on the Pharma environment and compare it with
the baseline and the oracle on exactly the same seeds.

Install Ollama first (https://ollama.com/download) and keep it running. Then:

    uv run python local_llm/run_local_llm.py                                  # Gemma 2 2B (default), 1 flu_season episode
    uv run python local_llm/run_local_llm.py --model qwen2.5:3b               # any Ollama model
    uv run python local_llm/run_local_llm.py --model qwen2.5:3b --seed 10000 --tasks all
    uv run python local_llm/run_local_llm.py --tasks all --episodes 3         # 3 seeds of every task
    uv run python local_llm/run_local_llm.py --model qwen2.5:3b --agent llm_news --news 2 --tasks all
                                                     # hybrid: the model reads news, the baseline orders

The first time a model is used, the script downloads it (`ollama pull`) and
creates `pharma-<model>` (e.g. `pharma-qwen2-5-3b`), a copy with a 4,096-token
context window: the prompts are about 1,800 tokens, and Ollama may cut longer
prompts from the start, losing the rules and output format. `--setup` forces
that step again.

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


def available_models(base_url: str, api_key: str = "not-needed") -> List[str]:
    """Model names on the server; exits with instructions if the server cannot be reached."""
    from openai import OpenAI

    try:
        return [m.id for m in OpenAI(base_url=base_url, api_key=api_key, timeout=10, max_retries=0).models.list()]
    except Exception as exc:
        sys.exit(f"Cannot reach a model server at {base_url} ({type(exc).__name__}).\n"
                 "Start Ollama (open the app, or run `ollama serve` in another terminal) and try again.")


def has_model(available: Sequence[str], model: str) -> bool:
    return model in set(available) | {m.split(":")[0] for m in available}


def check_server(base_url: str, model: str, setup_hint: str = "--setup", api_key: str = "not-needed") -> None:
    """Fail early, with instructions, if the server is down or the model is missing."""
    available = available_models(base_url, api_key)
    if not has_model(available, model):
        listed = ", ".join(sorted(available)) or "none"
        sys.exit(f"Model `{model}` is not available on {base_url} (available: {listed}).\n"
                 f"Run: uv run python local_llm/run_local_llm.py {setup_hint}")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_llm(tasks: Sequence[str], seeds: Sequence[int], split: str, llm: Dict[str, Any],
            base_model: Optional[str] = None, agent: str = "llm", news: int = 0) -> List[Dict[str, Any]]:
    records = []
    total = len(tasks) * len(seeds)
    for i, (task, seed) in enumerate(((t, s) for t in tasks for s in seeds), start=1):
        started = time.time()
        print(f"[{i}/{total}] {task}, seed {seed}: ", end="", flush=True)

        def progress(day: int, reply: str) -> None:
            # A dot per day, the day number every 10: large models on a CPU are slow.
            print(f"{day + 1}" if (day + 1) % 10 == 0 else ".", end="", flush=True)

        record = E.run_episode(agent, task, seed, split=split, llm=llm, save_steps=True, on_day=progress, news=news)
        if base_model:
            record = {"agent": record["agent"], "model": record["model"], "base_model": base_model,
                      **{k: v for k, v in record.items() if k not in ("agent", "model")}}
        print(f" -> score {record['score']:.3f}, {record['parse_failures']} replies without JSON "
              f"({time.time() - started:.0f}s)", flush=True)
        records.append(record)
    return records


def print_report(llm_records, baseline_records, oracle_records, show: int, news_rule_records=None) -> None:
    name = (llm_records[0].get("base_model") or llm_records[0]["model"]) if llm_records else "LLM"
    print(f"\n=== Mean score per task: {name} ===")
    columns = [("LLM", llm_records), ("baseline", baseline_records)]
    if news_rule_records:
        columns.append(("baseline_news", news_rule_records))
    columns.append(("oracle", oracle_records))
    print(f"{'task':22s} " + " ".join(f"{title:>{max(len(title), 7)}s}" for title, _ in columns))
    summaries = [(max(len(title), 7), E.summarize(r)) for title, r in columns]
    for task in summaries[0][1]:
        print(f"{task:22s} " + " ".join(f"{s[task]['mean']:{w}.3f}" for w, s in summaries))

    print(f"\n=== {name} minus baseline, seed by seed ===")
    E.print_comparison(E.paired_comparison(llm_records, baseline_records, oracle_records), (name, "baseline"))

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
    parser.add_argument("--model", "--base-model", dest="base_model", default=DEFAULT_BASE_MODEL,
                        help=f"Ollama model to run (default {DEFAULT_BASE_MODEL}), e.g. qwen2.5:3b, qwen2.5:1.5b, gemma2:9b; "
                             "downloaded and set up automatically the first time")
    parser.add_argument("--setup", action="store_true",
                        help="download the model and create its pharma-<model> copy again, even if it exists")
    parser.add_argument("--server-model", default=None,
                        help="exact model name on the server, to skip the pharma-<model> copy (e.g. a non-Ollama server)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds allowed per model call")
    parser.add_argument("--action", choices=("days", "units"), default="days",
                        help="the model answers in days of stock per SKU (converted to units; default) or in units")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible endpoint")
    parser.add_argument("--agent", choices=("llm", "llm_news"), default="llm",
                        help="llm: the model decides every order; llm_news: the baseline orders and the model "
                             "only reads the news and adds extra safety days (needs --news 1 or 2)")
    parser.add_argument("--news", type=int, choices=E.NEWS_LEVELS, default=0,
                        help="announce disruptions ahead: 0 no news (default), 1 exact template, 2 varied wording")
    parser.add_argument("--tasks", default="flu_season", help="comma-separated task names, or 'all'")
    parser.add_argument("--episodes", type=int, default=1, help="seeds per task")
    parser.add_argument("--seed", type=int, default=None,
                        help="run exactly this one seed per task instead of the first --episodes seeds of --split "
                             "(test seeds are 10000-19999, train seeds 0-9999)")
    parser.add_argument("--split", choices=sorted(SEED_SPLITS), default="test")
    parser.add_argument("--show", type=int, default=10, help="how many of the model's replies to print")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    if args.agent == "llm_news" and args.news == 0:
        parser.error("--agent llm_news reads the news: add --news 1 or --news 2")
    tasks = list(TASK_REGISTRY) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    unknown = [t for t in tasks if t not in TASK_REGISTRY]
    if unknown:
        parser.error(f"unknown task(s) {unknown}; choose from {sorted(TASK_REGISTRY)}")
    if args.seed is not None:
        if args.seed < 0:
            parser.error("--seed must be 0 or more")
        seeds = [args.seed]
        # The split only labels the results; pick the one the seed belongs to.
        args.split = next((name for name, split in SEED_SPLITS.items() if args.seed in split), "custom")
    else:
        seeds = list(SEED_SPLITS[args.split][:args.episodes])

    model = args.server_model or model_name(args.base_model)
    if args.server_model:
        check_server(args.base_url, model, setup_hint="--server-model <a name listed above>")
    else:
        if args.setup:
            setup(args.base_model, model)
        elif not has_model(available_models(args.base_url), model) and shutil.which("ollama"):
            print(f"First run of {args.base_model}: setting it up as `{model}`.\n", flush=True)
            setup(args.base_model, model)
        check_server(args.base_url, model, setup_hint=f"--setup --model {args.base_model}")
    llm = {"model": model, "base_url": args.base_url, "api_key": "not-needed",
           "timeout": args.timeout, "action": args.action}

    shown = model if args.server_model else f"{args.base_model} (as {model})"
    if args.agent == "llm_news":
        print(f"Model {shown} at {args.base_url}, reading level-{args.news} news (called only on days with news): "
              f"{len(tasks)} task(s) x {len(seeds)} seed(s)\n", flush=True)
    else:
        print(f"Model {shown} at {args.base_url}, answering in {args.action}: {len(tasks)} task(s) x {len(seeds)} "
              f"seed(s) = {len(tasks) * len(seeds) * 60} model calls\n", flush=True)
    llm_records = run_llm(tasks, seeds, args.split, llm, base_model=None if args.server_model else args.base_model,
                          agent=args.agent, news=args.news)
    runs = {name: [E.run_episode(name, t, s, split=args.split, news=args.news) for t in tasks for s in seeds]
            for name in ("baseline", "oracle") + (("baseline_news",) if args.news else ())}

    count = f"seed{args.seed}" if args.seed is not None else str(len(seeds))
    news = f"_news{args.news}" if args.news else ""
    stem = f"{args.split}_{'-'.join(tasks) if args.tasks != 'all' else 'all'}_{count}"
    safe_model = "".join(c if c.isalnum() or c == "." else "-" for c in model)
    paths = {f"{args.agent}_{safe_model}{news}": llm_records,
             **{f"{name}{news if name == 'baseline_news' else ''}": r for name, r in runs.items()}}
    for name, records in paths.items():
        E.write_jsonl(args.results_dir / f"{name}_{stem}.jsonl", records)

    print_report(llm_records, runs["baseline"], runs["oracle"], args.show, runs.get("baseline_news"))
    print(f"\nResults written to {args.results_dir}/ (*_{stem}.jsonl)")


if __name__ == "__main__":
    main()
