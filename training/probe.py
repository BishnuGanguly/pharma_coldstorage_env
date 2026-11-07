"""
The probe exam: a fixed set of news days the model answers greedily, scored with the
GRPO reward. Run it before, during and after training to see whether the model gets
better at reading the news.

    items = build_probe(seeds=range(9000, 9100), count=100)       # held out from training
    texts = my_generate([i.messages for i in items])              # one answer per item
    results = score_probe(items, texts)
    summarize_probe(items, results)   # {"reward": ..., "pct_of_best": ..., "right_product": ...}

No model is needed here: the caller generates the answers. Everything else is the
simulator.
"""

from __future__ import annotations

import json
import math
import statistics as st
from dataclasses import dataclass
from random import Random
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tasks import TASK_REGISTRY
from training.grpo_reward import (PENALTY_NO_JSON, PENALTY_OUT_OF_RANGE, DayPrompt, GRPOReward, build_state,
                                  news_day_prompts, prompt_messages)

PROBE_SEEDS = range(9000, 9100)    # train-split seeds kept out of training, for the exam during training
REFERENCE_DAYS = (0, 2, 4, 7, 10)  # answers tried on a prompt's news SKUs: the best of them is "best achievable"


def news_skus(prompt: DayPrompt) -> List[str]:
    """The SKUs named by the news visible on that day."""
    env, _, _ = build_state(prompt)
    return sorted({e.sku_id for e in env._episode_config.news if e.announce_day <= prompt.day <= e.end_day})


def reference_rewards(reward: GRPOReward, prompt: DayPrompt, skus: Optional[List[str]] = None) -> Dict[int, float]:
    """The reward of answering 0, +2, +4, +7 and +10 days for the SKUs in the news."""
    skus = news_skus(prompt) if skus is None else skus
    return {v: reward.score_answer(prompt, json.dumps({k: v for k in skus}))["reward"] for v in REFERENCE_DAYS}


@dataclass
class ProbeItem:
    prompt: DayPrompt
    messages: List[Dict[str, str]]
    skus: List[str]                 # SKUs in the news that day
    reference: Dict[int, float]     # reward of +0/+2/+4/+7/+10 on those SKUs

    @property
    def best(self) -> float:
        """Best achievable reward (at least 0, since answering 0 scores 0)."""
        return max(self.reference.values())

    @property
    def zero_is_best(self) -> bool:
        """Every reference answer above 0 loses: the right move is to hold back."""
        return all(v <= 0 for d, v in self.reference.items() if d)


def build_probe(seeds: Iterable[int] = PROBE_SEEDS, count: int = 100, reward: Optional[GRPOReward] = None,
                tasks: Optional[Sequence[str]] = None, salt: str = "probe") -> List[ProbeItem]:
    """`count` news days drawn evenly from the tasks of these episodes (deterministic)."""
    reward = reward or GRPOReward()
    tasks = sorted(tasks or TASK_REGISTRY)
    seeds = list(seeds)
    per_task = [news_day_prompts([t], seeds) for t in tasks]
    picked: List[DayPrompt] = []
    for i, prompts in enumerate(per_task):
        want = count // len(tasks) + (i < count % len(tasks))
        picked += sorted(Random(f"{salt}|{tasks[i]}").sample(prompts, min(want, len(prompts))),
                         key=lambda p: (p.seed, p.day))
    items = []
    for p in picked:
        skus = news_skus(p)
        items.append(ProbeItem(p, prompt_messages(p), skus, reference_rewards(reward, p, skus)))
    return items


def score_probe(items: Sequence[ProbeItem], texts: Sequence[str],
                reward: Optional[GRPOReward] = None) -> List[Dict[str, Any]]:
    """The reward and parts of each answer, plus the behaviour checks."""
    if len(items) != len(texts):
        raise ValueError(f"{len(items)} probe items but {len(texts)} answers")
    reward = reward or GRPOReward()
    results = []
    for item, text in zip(items, texts):
        s = reward.score_answer(item.prompt, text)
        extra = {k: v for k, v in s["extra_days"].items() if v > 0}
        top = max(extra, key=extra.get) if extra else None
        results.append({
            **s,
            "text": text,
            "task": item.prompt.task, "seed": item.prompt.seed, "day": item.prompt.day,
            "best": item.best,
            "no_json": s["penalty"] == PENALTY_NO_JSON,
            "out_of_range": s["penalty"] == PENALTY_OUT_OF_RANGE,
            "all_zero": s["penalty"] != PENALTY_NO_JSON and not extra,
            "right_product": None if top is None else top in item.skus,
            "insulin_habit": None if "insulin" in item.skus else "insulin" in extra,
            "held_back": (s["penalty"] != PENALTY_NO_JSON and not extra) if item.zero_is_best else None,
        })
    return results


def _share(values: Iterable[Optional[bool]]) -> float:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else float("nan")


def summarize_probe(items: Sequence[ProbeItem], results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """
    reward: mean reward per answer (points; answering 0 everywhere scores 0).
    pct_of_best: total reward / total best achievable reward, in %.
    right_product: answers adding days whose largest extra is on a SKU in the news.
    insulin_habit: days without insulin news where insulin got extra days.
    held_back: days where every +N loses, answered with 0.
    """
    best = sum(i.best for i in items)
    total = sum(r["reward"] for r in results)
    return {
        "n": len(results),
        "reward": total / len(results) if results else float("nan"),
        "pct_of_best": 100.0 * total / best if best > 0 else float("nan"),
        "no_json": _share(r["no_json"] for r in results),
        "out_of_range": _share(r["out_of_range"] for r in results),
        "all_zero": _share(r["all_zero"] for r in results),
        "right_product": _share(r["right_product"] for r in results),
        "insulin_habit": _share(r["insulin_habit"] for r in results),
        "held_back": _share(r["held_back"] for r in results),
    }


def compare_probes(candidate: Sequence[Dict[str, Any]], reference: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Paired difference in reward per probe item (candidate - reference), with a 95% interval."""
    if len(candidate) != len(reference):
        raise ValueError("both runs must answer the same probe items")
    diffs = [c["reward"] - r["reward"] for c, r in zip(candidate, reference)]
    n = len(diffs)
    sd = st.stdev(diffs) if n > 1 else 0.0
    return {"n": n, "mean": st.fmean(diffs) if n else float("nan"), "ci95": 1.96 * sd / math.sqrt(n) if n else float("nan"),
            "better": sum(d > 1e-9 for d in diffs), "worse": sum(d < -1e-9 for d in diffs)}
