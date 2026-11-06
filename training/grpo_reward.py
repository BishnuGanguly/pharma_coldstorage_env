"""
GRPO reward for the `adjust` answer format. The rules, and why each was chosen, are in
training/REWARD.md.

    from training.grpo_reward import GRPOReward, news_day_prompts

    prompts = news_day_prompts(tasks, seeds=range(0, 300))   # train seeds, news days only
    reward = GRPOReward()
    reward(completions, task=[...], seed=[...], day=[...])   # one float per completion

GRPOReward follows the reward-function signature of TRL's GRPOTrainer: it receives the
completions plus the dataset's columns as keyword arguments, and returns one reward per
completion. Everything is computed with the simulator: no model calls.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import inference  # noqa: E402
from agents import BaselineAgent, baseline_policy  # noqa: E402
from models import InventoryState, PharmaAction  # noqa: E402
from server.my_env_environment import PharmaEnvironment  # noqa: E402

HORIZON = 15             # days played forward per answer
FUTURES = 4              # possible futures per prompt, shared by every answer to it
POINTS = 100.0 / HORIZON  # reward units: hundredths of a day's reward, per day
PENALTY_NO_JSON = -1.0
PENALTY_OUT_OF_RANGE = -0.25
SCENARIO = "random"
NEWS_LEVEL = 2
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)  # the same check as eval.py's parse failures

History = List[Tuple[int, str, str]]


@dataclass(frozen=True)
class DayPrompt:
    """One training prompt: one day of one episode."""

    task: str
    seed: int
    day: int
    scenario: str = SCENARIO
    news: int = NEWS_LEVEL


# ---------------------------------------------------------------------------
# Rebuilding a day's state and prompt
# ---------------------------------------------------------------------------

def build_state(prompt: DayPrompt) -> Tuple[PharmaEnvironment, InventoryState, History]:
    """
    The environment on `prompt.day`, reached by letting the baseline play every earlier
    day, and the action history an LLM agent would have (as in eval.run_episode).
    Deterministic: the same prompt always gives the same state.
    """
    env = PharmaEnvironment()
    obs = env.reset(task_name=prompt.task, seed=prompt.seed, news=prompt.news, scenario=prompt.scenario)
    agent, history = BaselineAgent(), []
    while obs.current_date < prompt.day:
        if obs.done:
            raise ValueError(f"day {prompt.day} is past the end of the episode")
        day = obs.current_date
        reply, _ = agent.act(obs, history)
        obs = env.step(PharmaAction(message=reply))
        history.append((day, reply, inference.build_feedback(obs)))
    return env, obs, history


def prompt_messages(prompt: DayPrompt) -> List[Dict[str, str]]:
    """The chat messages agents.LLMAgent sends on that day in the `adjust` format."""
    _, obs, history = build_state(prompt)
    report = inference.observation_to_dict(obs, include_cover=False)
    user = inference.build_user_prompt(report, history, step=obs.current_date + 1, action="adjust")
    return [{"role": "system", "content": inference.system_prompt("adjust")},
            {"role": "user", "content": user}]


def news_day_prompts(
    tasks: Iterable[str],
    seeds: Iterable[int],
    scenario: str = SCENARIO,
    news: int = NEWS_LEVEL,
) -> List[DayPrompt]:
    """Every day with visible news in the given episodes (days without news teach nothing)."""
    prompts = []
    for task in tasks:
        for seed in seeds:
            env = PharmaEnvironment()
            obs = env.reset(task_name=task, seed=seed, news=news, scenario=scenario)
            agent = BaselineAgent()
            while not obs.done:
                if obs.news:
                    prompts.append(DayPrompt(task, seed, obs.current_date, scenario, news))
                obs = env.step(PharmaAction(message=agent.act(obs, [])[0]))
    return prompts


# ---------------------------------------------------------------------------
# Parsing an answer
# ---------------------------------------------------------------------------

def parse_answer(obs: InventoryState, text: str) -> Tuple[Dict[str, float], float]:
    """
    Extra days per SKU from a completion, and its format penalty.
    No JSON object: answer 0 and PENALTY_NO_JSON. A value outside 0..10: clipped and
    PENALTY_OUT_OF_RANGE (once). Missing, unknown or non-numeric entries: 0, no penalty.
    """
    if not JSON_OBJECT.search(text or ""):
        return {}, PENALTY_NO_JSON
    raw = inference.parse_extra_days(obs, inference.parse_json_object(text))
    low, high = inference.ADJUST_DAYS_RANGE
    penalty = PENALTY_OUT_OF_RANGE if any(not low <= v <= high for v in raw.values()) else 0.0
    return inference.adjust_days(obs, raw), penalty


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def future_rng(prompt: DayPrompt, index: int) -> Random:
    """Future `index` of a prompt: the same for every answer to that prompt."""
    return Random(f"future|{prompt.task}|{prompt.seed}|{prompt.day}|{prompt.scenario}|{prompt.news}|{index}")


def play_forward(env: PharmaEnvironment, obs: InventoryState, extra: Dict[str, float], horizon: int = HORIZON) -> float:
    """Sum of the daily rewards: today's order with `extra`, then the baseline. Changes `env`."""
    total = 0.0
    for i in range(horizon):
        if obs.done:
            break
        orders = baseline_policy(obs, extra_days=extra if i == 0 else None)
        obs = env.step(PharmaAction(message=json.dumps(orders)))
        total += obs.reward or 0.0
    return total


def score(env: PharmaEnvironment, obs: InventoryState, prompt: DayPrompt, extra: Dict[str, float],
          futures: int = FUTURES, horizon: int = HORIZON) -> float:
    """S(extra): the mean of play_forward over the prompt's futures. Leaves `env` unchanged."""
    totals = []
    for index in range(futures):
        clone = copy.deepcopy(env)
        clone.resample_future_noise(future_rng(prompt, index))
        totals.append(play_forward(clone, clone._inventory_state, extra, horizon))
    return sum(totals) / len(totals)


class GRPOReward:
    """
    reward = (S(answer) - S(answer 0)) x 100 / HORIZON + format penalty   (see REWARD.md)

    Keeps the rebuilt state and S(answer 0) of the prompts it has seen, so the 8 answers
    to one prompt rebuild the state once.
    """

    __name__ = "pharma_adjust_reward"   # TRL logs reward functions by name

    def __init__(self, futures: int = FUTURES, horizon: int = HORIZON, cache_size: int = 256) -> None:
        self.futures, self.horizon, self.cache_size = futures, horizon, cache_size
        self._cache: Dict[DayPrompt, Tuple[PharmaEnvironment, InventoryState, float]] = {}

    def _state(self, prompt: DayPrompt) -> Tuple[PharmaEnvironment, InventoryState, float]:
        if prompt not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            env, obs, _ = build_state(prompt)
            self._cache[prompt] = (env, obs, score(env, obs, prompt, {}, self.futures, self.horizon))
        return self._cache[prompt]

    def score_answer(self, prompt: DayPrompt, text: str) -> Dict[str, Any]:
        """The reward of one answer, with its parts (for logging and tests)."""
        env, obs, zero = self._state(prompt)
        extra, penalty = parse_answer(obs, text)
        gain = 0.0 if not any(extra.values()) else (score(env, obs, prompt, extra, self.futures, self.horizon) - zero)
        return {"reward": gain * 100.0 / self.horizon + penalty, "gain_points": gain * 100.0 / self.horizon,
                "penalty": penalty, "extra_days": extra}

    def __call__(self, completions: Sequence[Any], task: Sequence[str], seed: Sequence[int],
                 day: Sequence[int], scenario: Optional[Sequence[str]] = None,
                 news: Optional[Sequence[int]] = None, **_: Any) -> List[float]:
        rewards = []
        for i, completion in enumerate(completions):
            prompt = DayPrompt(task[i], int(seed[i]), int(day[i]),
                               scenario[i] if scenario is not None else SCENARIO,
                               int(news[i]) if news is not None else NEWS_LEVEL)
            rewards.append(self.score_answer(prompt, completion_text(completion))["reward"])
        return rewards


def completion_text(completion: Any) -> str:
    """A completion as TRL passes it: a string, or a list of chat messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    return str(completion)


def dataset_rows(prompts: Iterable[DayPrompt]) -> List[Dict[str, Any]]:
    """Rows for a TRL dataset: the chat prompt plus the columns GRPOReward reads."""
    return [{"prompt": prompt_messages(p), **asdict(p)} for p in prompts]

