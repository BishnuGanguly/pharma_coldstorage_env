"""
Diversity check before GRPO: does the model give different answers to the same prompt,
and do those answers get different rewards?

GRPO learns only from differences between the answers to one prompt: if all 8 answers
are the same, or all score the same, that prompt teaches nothing. This script picks a few
days where the answer matters (one per task by default, from train seeds, random scenario,
level-2 news), asks the model 8 times per day at a sampling temperature, scores every
answer with the GRPO reward (training/grpo_reward.py), and says whether there is enough
variety to train on.

    uv run python local_llm/diversity_check.py                                   # Phi-4-mini, 3 days x 8 answers = 24 calls
    uv run python local_llm/diversity_check.py --model phi4-mini --temperatures 0.7,1.0,1.2

The model is set up with Ollama the first time, as in run_local_llm.py. Results are
written to results/diversity_<model>.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "local_llm"))

import inference  # noqa: E402
import run_local_llm as R  # noqa: E402
from tasks import TASK_REGISTRY, TRAIN_SEEDS  # noqa: E402
from training.grpo_reward import DayPrompt, GRPOReward, build_state, news_day_prompts, prompt_messages  # noqa: E402

DEFAULT_MODEL = "phi4-mini"
REFERENCE_DAYS = (0, 2, 4, 7, 10)  # answers tried on a prompt's news SKUs to see whether its answer matters
MIN_SPREAD = 0.5                   # points between the best and worst reference answer: "the answer matters"
# Verdict thresholds per prompt: at least this many different answers, and this spread of rewards.
MIN_DISTINCT = 3
MIN_REWARD_SD = 0.1


def news_skus(prompt: DayPrompt) -> List[str]:
    env, _, _ = build_state(prompt)
    return sorted({e.sku_id for e in env._episode_config.news if e.announce_day <= prompt.day <= e.end_day})


def reference_rewards(reward: GRPOReward, prompt: DayPrompt) -> Dict[int, float]:
    """The reward of answering 0, +2, +4, +7 and +10 days for the SKUs in the news."""
    skus = news_skus(prompt)
    return {v: reward.score_answer(prompt, json.dumps({k: v for k in skus}))["reward"] for v in REFERENCE_DAYS}


def pick_prompts(reward: GRPOReward, count: int, seeds: Sequence[int]) -> List[DayPrompt]:
    """
    Days where the answer matters (reference answers differ by at least MIN_SPREAD points),
    taking the tasks in turn so the days cover different tasks. Deterministic.
    """
    tasks = sorted(TASK_REGISTRY)
    candidates = {t: news_day_prompts([t], seeds) for t in tasks}
    picked: List[DayPrompt] = []
    position = {t: 0 for t in tasks}
    while len(picked) < count and any(position[t] < len(candidates[t]) for t in tasks):
        for t in tasks:
            while position[t] < len(candidates[t]):
                prompt = candidates[t][position[t]]
                position[t] += 7   # step through the episode rather than taking consecutive days
                refs = reference_rewards(reward, prompt)
                if max(refs.values()) - min(refs.values()) >= MIN_SPREAD:
                    picked.append(prompt)
                    break
            if len(picked) == count:
                break
    return picked


def sample_answers(client: Any, model: str, prompt: DayPrompt, samples: int, temperature: float) -> List[str]:
    messages = prompt_messages(prompt)
    answers = []
    for _ in range(samples):
        completion = client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                    max_tokens=inference.MAX_TOKENS)
        answers.append((completion.choices[0].message.content or "").strip())
    return answers


def analyse(reward: GRPOReward, prompt: DayPrompt, answers: List[str]) -> Dict[str, Any]:
    scored = [reward.score_answer(prompt, a) for a in answers]
    rewards = [s["reward"] for s in scored]
    distinct = {json.dumps({k: v for k, v in sorted(s["extra_days"].items()) if v}) if s["penalty"] != -1.0 else "no JSON"
                for s in scored}
    return {
        "prompt": prompt.__dict__,
        "news": build_state(prompt)[1].news,
        "answers": [{"text": a, **s} for a, s in zip(answers, scored)],
        "distinct_answers": len(distinct),
        "no_json": sum(s["penalty"] == -1.0 for s in scored),
        "reward_mean": st.mean(rewards),
        "reward_sd": st.pstdev(rewards),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "reference": reference_rewards(reward, prompt),
    }


def verdict(results: List[Dict[str, Any]]) -> str:
    useful = [r for r in results if r["distinct_answers"] >= MIN_DISTINCT and r["reward_sd"] >= MIN_REWARD_SD]
    no_json = sum(r["no_json"] for r in results) / max(sum(len(r["answers"]) for r in results), 1)
    lines = [f"{len(useful)} of {len(results)} days give GRPO something to learn from "
             f"(at least {MIN_DISTINCT} different answers and a reward spread of {MIN_REWARD_SD}+ points)."]
    if len(useful) * 3 >= len(results) * 2:
        lines.append("Verdict: enough variety for GRPO at this temperature.")
    elif all(r["distinct_answers"] == 1 for r in results):
        lines.append("Verdict: the model gives the same answer every time. Try a higher temperature "
                     "(e.g. --temperatures 1.2,1.5); if that does not help, start with a short supervised "
                     "fine-tune so the model learns to vary its extra days.")
    else:
        lines.append("Verdict: too little variety on most days. Try a higher temperature, or more samples per "
                     "prompt in GRPO (e.g. 16 instead of 8).")
    if no_json > 0.25:
        lines.append(f"Warning: {no_json:.0%} of the answers had no JSON; GRPO's format penalty will spend "
                     "its first steps on that.")
    return "\n".join(lines)


def print_prompt(result: Dict[str, Any], show: int) -> None:
    p = result["prompt"]
    print(f"\n--- {p['task']}, seed {p['seed']}, day {p['day']} ---")
    for item in result["news"]:
        print(f"  news: {item[:150]}")
    ref = "  ".join(f"+{k}: {v:+.2f}" for k, v in result["reference"].items())
    print(f"  reference answers for the SKUs in the news: {ref}")
    for a in result["answers"][:show]:
        shown = {k: v for k, v in a["extra_days"].items() if v} if a["penalty"] != -1.0 else "no JSON"
        print(f"  {a['reward']:+6.2f}  {shown}")
    print(f"  -> {result['distinct_answers']} different answers; rewards {result['reward_min']:+.2f} .. "
          f"{result['reward_max']:+.2f} (sd {result['reward_sd']:.2f}); {result['no_json']} without JSON")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", "--base-model", dest="base_model", default=DEFAULT_MODEL,
                        help=f"Ollama model (default {DEFAULT_MODEL}, the model to train); set up the first time")
    parser.add_argument("--server-model", default=None, help="exact model name on the server (non-Ollama servers)")
    parser.add_argument("--base-url", default=R.DEFAULT_BASE_URL, help="OpenAI-compatible endpoint")
    parser.add_argument("--timeout", type=float, default=R.DEFAULT_TIMEOUT, help="seconds allowed per model call")
    parser.add_argument("--prompts", type=int, default=3, help="days to test (default 3: one per task)")
    parser.add_argument("--samples", type=int, default=8, help="answers per day, as in a GRPO group (default 8)")
    parser.add_argument("--temperatures", default="1.0", help="comma-separated sampling temperatures (default 1.0)")
    parser.add_argument("--show", type=int, default=8, help="answers to print per day")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    try:
        temperatures = [float(t) for t in args.temperatures.split(",")]
    except ValueError:
        parser.error("--temperatures takes numbers separated by commas, e.g. 0.7,1.0")

    model = args.server_model or R.model_name(args.base_model)
    if not args.server_model and not R.has_model(R.available_models(args.base_url), model) and shutil.which("ollama"):
        print(f"First run of {args.base_model}: setting it up as `{model}`.\n", flush=True)
        R.setup(args.base_model, model)
    R.check_server(args.base_url, model, setup_hint=f"--setup --model {args.base_model}")

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="not-needed", timeout=args.timeout, max_retries=1)
    reward = GRPOReward()
    print("Choosing days where the answer matters (train seeds, random scenario, level-2 news)...", flush=True)
    prompts = pick_prompts(reward, args.prompts, TRAIN_SEEDS[:40])
    calls = len(prompts) * args.samples * len(temperatures)
    print(f"Model {model}: {len(prompts)} days x {args.samples} answers x {len(temperatures)} temperature(s) "
          f"= {calls} model calls", flush=True)

    report: Dict[str, Any] = {"model": model, "samples": args.samples, "runs": []}
    for temperature in temperatures:
        print(f"\n===== temperature {temperature} =====", flush=True)
        results = []
        for prompt in prompts:
            started = time.time()
            answers = sample_answers(client, model, prompt, args.samples, temperature)
            result = analyse(reward, prompt, answers)
            results.append(result)
            print_prompt(result, args.show)
            print(f"  ({time.time() - started:.0f}s)", flush=True)
        summary = verdict(results)
        print(f"\n{summary}")
        report["runs"].append({"temperature": temperature, "verdict": summary, "results": results})

    args.results_dir.mkdir(parents=True, exist_ok=True)
    safe_model = "".join(c if c.isalnum() or c == "." else "-" for c in model)
    out = args.results_dir / f"diversity_{safe_model}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
