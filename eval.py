"""
eval.py
=======
Evaluate an agent on the Pharma environment over many seeded episodes, and
compare agents fairly.

Run an agent (results go to results/<agent>_<split>_<n>.jsonl, one line per episode):

    python eval.py --agent baseline --split test --episodes 50
    python eval.py --agent oracle   --split test --episodes 50
    python eval.py --agent llm --model qwen2.5:3b --base-url http://localhost:11434/v1 \\
                   --split test --episodes 10 --save-steps

Compare two result files seed by seed (optionally against the oracle's):

    python eval.py --compare results/llm_test_10.jsonl results/baseline_test_50.jsonl \\
                   --oracle results/oracle_test_50.jsonl

Agents (see agents.py): nothing (floor), baseline (order-up-to rule),
oracle (perfect foresight, the ceiling), llm (any OpenAI-compatible server).

Seeds: --split train uses tasks.TRAIN_SEEDS, --split test uses tasks.TEST_SEEDS,
always the first N. Tune and train on train seeds only; report test seeds.

The same seed gives the same world for every agent (noise is drawn at reset),
so --compare looks at per-seed differences, which cancels out each seed's luck
and gives much tighter confidence intervals than comparing two averages.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import inference  # noqa: E402
from agents import Agent, BaselineAgent, LLMAgent, NothingAgent, OracleAgent  # noqa: E402
from models import PharmaAction  # noqa: E402
from server.my_env_environment import PharmaEnvironment  # noqa: E402
from tasks import SEED_SPLITS, TASK_REGISTRY, compute_final_score  # noqa: E402

AGENTS = ("nothing", "baseline", "oracle", "llm")
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
# An LLM that fails this many calls in a row from the very first day is
# misconfigured (bad token, wrong model, server down): stop instead of
# spending the whole run on errors.
MAX_INITIAL_LLM_ERRORS = 3


# ---------------------------------------------------------------------------
# Running episodes
# ---------------------------------------------------------------------------

def make_agent(name: str, llm: Optional[Dict[str, str]] = None) -> Agent:
    if name == "nothing":
        return NothingAgent()
    if name == "baseline":
        return BaselineAgent()
    if name == "oracle":
        return OracleAgent()
    if name == "llm":
        llm = llm or {}
        return LLMAgent(model=llm["model"], api_key=llm["api_key"], base_url=llm["base_url"])
    raise ValueError(f"Unknown agent '{name}'. Choose from {AGENTS}.")


def run_episode(
    agent_name: str,
    task_name: str,
    seed: int,
    split: str = "",
    llm: Optional[Dict[str, str]] = None,
    save_steps: bool = False,
) -> Dict[str, Any]:
    """Play one full episode and return a JSON-serialisable record of it."""
    agent = make_agent(agent_name, llm)
    env = PharmaEnvironment()
    obs = env.reset(task_name=task_name, seed=seed)
    agent.start_episode(env)

    history: List[Tuple[int, str, str]] = []
    steps: List[Dict[str, Any]] = []
    fill_total = 0.0
    stockout_sku_days = overflow_days = parse_failures = llm_errors = 0
    units_wasted = 0.0

    while not obs.done:
        day = obs.current_date
        reply, error = agent.act(obs, history)
        if not JSON_OBJECT.search(reply):
            parse_failures += 1
        if error:
            llm_errors += 1
            if llm_errors == day + 1 and llm_errors >= MAX_INITIAL_LLM_ERRORS:
                raise RuntimeError(f"The LLM failed its first {llm_errors} calls; last error: {error}")

        obs = env.step(PharmaAction(message=reply))
        fills = [sku.demand_fulfilled_today for sku in obs.skus.values()]
        fill_total += sum(fills) / len(fills)
        stockout_sku_days += sum(1 for f in fills if f < 0.999)
        overflow_days += obs.inventory_excess_today > 0
        units_wasted += obs.inventory_excess_today
        history.append((day, reply, inference.build_feedback(obs)))
        if save_steps:
            steps.append({
                "day": day,
                "reply": reply,
                "error": error,
                "reward": round(obs.reward, 4),
                "fill": {k: round(s.demand_fulfilled_today, 3) for k, s in obs.skus.items()},
                "waste_fraction": round(obs.waste_fraction_today, 4),
            })

    days = len(history)
    record: Dict[str, Any] = {
        "agent": agent_name,
        "model": (llm or {}).get("model") if agent_name == "llm" else None,
        "task": task_name,
        "seed": seed,
        "split": split,
        "score": round(compute_final_score(obs, env._episode_config), 6),
        "avg_fill": round(fill_total / days, 6) if days else 0.0,
        "stockout_sku_days": stockout_sku_days,
        "overflow_days": overflow_days,
        "units_wasted": round(units_wasted, 2),
        "parse_failures": parse_failures,
        "llm_errors": llm_errors,
        "days": days,
    }
    if save_steps:
        record["steps"] = steps
    return record


def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    return run_episode(**job)


def evaluate(
    agent_name: str,
    task_names: Sequence[str],
    split: str,
    episodes: int,
    workers: int = 1,
    llm: Optional[Dict[str, str]] = None,
    save_steps: bool = False,
) -> List[Dict[str, Any]]:
    """Run `episodes` seeds (the first N of the split) on each task."""
    seeds = list(SEED_SPLITS[split][:episodes])
    jobs = [
        dict(agent_name=agent_name, task_name=t, seed=s, split=split, llm=llm, save_steps=save_steps)
        for t in task_names for s in seeds
    ]
    if workers <= 1:
        return [_run_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_job, jobs, chunksize=max(1, len(jobs) // (workers * 4))))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def describe(values: Sequence[float]) -> Dict[str, float]:
    """Mean, sample std, 95% confidence half-width of the mean, min and max."""
    n = len(values)
    mean = statistics.fmean(values) if n else float("nan")
    std = statistics.stdev(values) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "ci95": 1.96 * std / math.sqrt(n) if n else float("nan"),
            "min": min(values) if n else float("nan"), "max": max(values) if n else float("nan")}


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_task: Dict[str, List[float]] = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r["score"])
    summary = {t: describe(v) for t, v in sorted(by_task.items())}
    summary["ALL"] = describe([r["score"] for r in records])
    return summary


def paired_comparison(
    candidate: Sequence[Dict[str, Any]],
    reference: Sequence[Dict[str, Any]],
    oracle: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Per-seed differences candidate - reference, on the (task, seed) pairs both
    files contain. With oracle results, also the share of the reference-to-oracle
    gap the candidate closes: 0 = as good as the reference, 1 = perfect foresight.
    """
    ref = {(r["task"], r["seed"]): r["score"] for r in reference}
    orc = {(r["task"], r["seed"]): r["score"] for r in oracle} if oracle else {}
    diffs: Dict[str, List[float]] = {}
    gaps: Dict[str, List[Tuple[float, float]]] = {}
    for r in candidate:
        key = (r["task"], r["seed"])
        if key not in ref:
            continue
        for group in (r["task"], "ALL"):
            diffs.setdefault(group, []).append(r["score"] - ref[key])
            if key in orc:
                gaps.setdefault(group, []).append((r["score"] - ref[key], orc[key] - ref[key]))
    result: Dict[str, Dict[str, float]] = {}
    for group in sorted(diffs, key=lambda g: (g == "ALL", g)):
        stats = describe(diffs[group])
        if group in gaps:
            gained = sum(g for g, _ in gaps[group])
            possible = sum(p for _, p in gaps[group])
            stats["gap_closed"] = gained / possible if possible > 0 else float("nan")
        result[group] = stats
    return result


# ---------------------------------------------------------------------------
# Files and printing
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def print_summary(records: Sequence[Dict[str, Any]]) -> None:
    summary = summarize(records)
    print(f"{'task':22s} {'n':>4s} {'mean':>7s} {'±95%':>7s} {'std':>7s} {'min':>7s} {'max':>7s}")
    for task, s in summary.items():
        print(f"{task:22s} {s['n']:4d} {s['mean']:7.3f} {s['ci95']:7.3f} {s['std']:7.3f} {s['min']:7.3f} {s['max']:7.3f}")
    failures = sum(r["parse_failures"] for r in records)
    errors = sum(r["llm_errors"] for r in records)
    replies = sum(r["days"] for r in records)
    if records and records[0]["agent"] == "llm":
        print(f"\nreplies without any JSON object: {failures}/{replies} ({failures / max(replies, 1):.1%}); "
              f"failed LLM calls: {errors}")


def print_comparison(result: Dict[str, Dict[str, float]], names: Tuple[str, str]) -> None:
    print(f"{names[0]}  minus  {names[1]}  (per-seed differences)")
    has_gap = any("gap_closed" in s for s in result.values())
    print(f"{'task':22s} {'n':>4s} {'mean diff':>10s} {'±95%':>7s}" + (f" {'gap closed':>11s}" if has_gap else ""))
    for task, s in result.items():
        line = f"{task:22s} {s['n']:4d} {s['mean']:+10.3f} {s['ci95']:7.3f}"
        if has_gap:
            line += f" {s['gap_closed']:11.1%}" if "gap_closed" in s else f" {'':>11s}"
        print(line)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=AGENTS)
    parser.add_argument("--split", choices=sorted(SEED_SPLITS), default="test")
    parser.add_argument("--episodes", type=int, default=20, help="seeds per task (first N of the split)")
    parser.add_argument("--tasks", default="all", help="comma-separated task names, or 'all'")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel processes (default: CPU count for rule agents, 1 for llm)")
    parser.add_argument("--save-steps", action="store_true", help="also store every day's reply and reward")
    parser.add_argument("--out", type=Path, help="output .jsonl path")
    parser.add_argument("--model", default=inference.MODEL_NAME, help="llm: model name")
    parser.add_argument("--base-url", default=inference.API_BASE_URL, help="llm: OpenAI-compatible endpoint")
    parser.add_argument("--api-key", default=None, help="llm: API key (default: HF_TOKEN / API_KEY env var)")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("CANDIDATE", "REFERENCE"),
                        help="compare two result files seed by seed instead of running")
    parser.add_argument("--oracle", type=Path, help="with --compare: oracle results, to report gap closed")
    args = parser.parse_args(argv)

    if args.compare:
        candidate, reference = (read_jsonl(p) for p in args.compare)
        oracle = read_jsonl(args.oracle) if args.oracle else None
        print_comparison(paired_comparison(candidate, reference, oracle), (args.compare[0].stem, args.compare[1].stem))
        return
    if not args.agent:
        parser.error("--agent is required unless --compare is used")

    task_names = list(TASK_REGISTRY) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    unknown = [t for t in task_names if t not in TASK_REGISTRY]
    if unknown:
        parser.error(f"unknown task(s) {unknown}; choose from {sorted(TASK_REGISTRY)}")

    llm = None
    if args.agent == "llm":
        # Local servers such as Ollama ignore the key, but the client needs a non-empty one.
        api_key = args.api_key or os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "not-needed"
        llm = {"model": args.model, "base_url": args.base_url, "api_key": api_key}
    workers = args.workers or (1 if args.agent == "llm" else min(os.cpu_count() or 1, 8))

    records = evaluate(args.agent, task_names, args.split, args.episodes, workers, llm, args.save_steps)
    name = args.agent if args.agent != "llm" else "llm_" + re.sub(r"[^A-Za-z0-9.]+", "-", args.model)
    out = args.out or ROOT / "results" / f"{name}_{args.split}_{args.episodes}.jsonl"
    write_jsonl(out, records)
    print_summary(records)
    print(f"\n{len(records)} episodes written to {out}")


if __name__ == "__main__":
    main()
