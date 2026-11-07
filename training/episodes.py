"""
Full episodes with a local model, many at once: on each day every unfinished episode
asks the model for its answer, and all those prompts go to the model as one batch.
This plays exactly like eval.run_episode with the `llm` agent in the `adjust` format
(the same prompt, the same conversion of the answer into an order), but makes 60
batched calls instead of 60 calls per episode.

    records = run_llm_episodes(tasks, seeds, generate, news=2, scenario="random")
    # generate(list_of_chat_messages) -> list of answer strings, one per prompt

The records have the fields eval.paired_comparison and eval.summarize read.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Sequence, Tuple

import inference
from agents import Agent, baseline_policy
from eval import JSON_OBJECT
from models import InventoryState, PharmaAction
from server.my_env_environment import PharmaEnvironment
from tasks import compute_final_score

Generate = Callable[[List[List[Dict[str, str]]]], List[str]]
History = List[Tuple[int, str, str]]


def adjust_messages(obs: InventoryState, history: History) -> List[Dict[str, str]]:
    """The chat messages agents.LLMAgent sends in the `adjust` format."""
    report = inference.observation_to_dict(obs, include_cover=False)
    user = inference.build_user_prompt(report, history, step=obs.current_date + 1, action="adjust")
    return [{"role": "system", "content": inference.system_prompt("adjust")}, {"role": "user", "content": user}]


def answer_to_order(obs: InventoryState, text: str) -> str:
    """The order LLMAgent sends for a model answer in the `adjust` format."""
    extra = inference.adjust_days(obs, inference.parse_json_object(text))
    return json.dumps(baseline_policy(obs, extra_days=extra, min_extra=inference.ADJUST_DAYS_RANGE[0]))


class FlatExtraAgent(Agent):
    """The baseline with the same extra days on every SKU, every day ("+3 everywhere")."""

    def __init__(self, days: float = 3.0) -> None:
        self.days = days

    def act(self, obs: InventoryState, history: History):
        return json.dumps(baseline_policy(obs, extra_days={k: self.days for k in obs.skus})), None


def run_llm_episodes(
    tasks: Sequence[str],
    seeds: Sequence[int],
    generate: Generate,
    news: int = 2,
    scenario: str = "random",
    agent: str = "llm",
    model: str = "",
    split: str = "test",
    save_replies: bool = False,
) -> List[Dict[str, Any]]:
    """Play every (task, seed) episode to the end, one batched model call per day."""
    runs = []
    for task in tasks:
        for seed in seeds:
            env = PharmaEnvironment()
            obs = env.reset(task_name=task, seed=seed, news=news, scenario=scenario)
            runs.append({"task": task, "seed": seed, "env": env, "obs": obs, "history": [], "fill": 0.0,
                         "stockout": 0, "overflow": 0, "wasted": 0.0, "parse_failures": 0, "replies": []})
    while True:
        active = [r for r in runs if not r["obs"].done]
        if not active:
            break
        texts = generate([adjust_messages(r["obs"], r["history"]) for r in active])
        if len(texts) != len(active):
            raise ValueError(f"generate returned {len(texts)} answers for {len(active)} prompts")
        for r, text in zip(active, texts):
            text = (text or "").strip()
            obs, day = r["obs"], r["obs"].current_date
            reply = answer_to_order(obs, text)
            r["parse_failures"] += not JSON_OBJECT.search(text)
            obs = r["env"].step(PharmaAction(message=reply))
            fills = [s.demand_fulfilled_today for s in obs.skus.values()]
            r["fill"] += sum(fills) / len(fills)
            r["stockout"] += sum(f < 0.999 for f in fills)
            r["overflow"] += obs.inventory_excess_today > 0
            r["wasted"] += obs.inventory_excess_today
            r["history"].append((day, reply, inference.build_feedback(obs)))
            if save_replies:
                r["replies"].append({"day": day, "news": list(r["obs"].news), "reply": text})
            r["obs"] = obs
    records = []
    for r in runs:
        days = len(r["history"])
        record = {
            "agent": agent, "model": model or None, "action_format": "adjust",
            "task": r["task"], "seed": r["seed"], "split": split, "news": news, "scenario": scenario,
            "score": round(compute_final_score(r["obs"], r["env"]._episode_config), 6),
            "avg_fill": round(r["fill"] / days, 6) if days else 0.0,
            "stockout_sku_days": r["stockout"], "overflow_days": r["overflow"],
            "units_wasted": round(r["wasted"], 2), "parse_failures": r["parse_failures"],
            "llm_errors": 0, "days": days,
        }
        if save_replies:
            record["replies"] = r["replies"]
        records.append(record)
    return records


def run_rule_episodes(agent: Agent, name: str, tasks: Sequence[str], seeds: Sequence[int], news: int = 2,
                      scenario: str = "random", split: str = "test") -> List[Dict[str, Any]]:
    """Score records for a rule-based agent object (one not in eval.make_agent, e.g. FlatExtraAgent)."""
    records = []
    for task in tasks:
        for seed in seeds:
            env = PharmaEnvironment()
            obs = env.reset(task_name=task, seed=seed, news=news, scenario=scenario)
            agent.start_episode(env)
            history: History = []
            while not obs.done:
                day = obs.current_date
                reply, _ = agent.act(obs, history)
                obs = env.step(PharmaAction(message=reply))
                history.append((day, reply, inference.build_feedback(obs)))
            records.append({"agent": name, "task": task, "seed": seed, "split": split, "news": news,
                            "scenario": scenario, "score": round(compute_final_score(obs, env._episode_config), 6)})
    return records
