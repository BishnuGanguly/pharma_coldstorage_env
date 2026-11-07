"""
Training-health numbers for GRPO, collected while the reward is computed.

MonitoredReward is the GRPO reward (it returns exactly the same rewards) that also
remembers, for the completions it scored since the last pop_stats():

    no_json, out_of_range, all_zero  share of answers
    right_product                    answers adding days whose largest extra is on a SKU in the news
    zero_std_groups                  groups whose answers all got the same reward (nothing to learn)
    group_best_minus_mean            how far the best answer of a group is above the group mean

TRL already logs the mean reward, KL and completion length; a trainer callback adds these.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from training.grpo_reward import (NEWS_LEVEL, PENALTY_NO_JSON, PENALTY_OUT_OF_RANGE, SCENARIO, DayPrompt,
                                  GRPOReward, completion_text)


class MonitoredReward(GRPOReward):
    __name__ = "pharma_adjust_reward"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rows: List[Dict[str, Any]] = []

    def _news_skus(self, prompt: DayPrompt) -> set:
        env = self._state(prompt)[0]
        return {e.sku_id for e in env._episode_config.news if e.announce_day <= prompt.day <= e.end_day}

    def __call__(self, completions: Sequence[Any], task: Sequence[str], seed: Sequence[int],
                 day: Sequence[int], scenario: Optional[Sequence[str]] = None,
                 news: Optional[Sequence[int]] = None, **_: Any) -> List[float]:
        rewards = []
        for i, completion in enumerate(completions):
            prompt = DayPrompt(task[i], int(seed[i]), int(day[i]),
                               scenario[i] if scenario is not None else SCENARIO,
                               int(news[i]) if news is not None else NEWS_LEVEL)
            s = self.score_answer(prompt, completion_text(completion))
            extra = {k: v for k, v in s["extra_days"].items() if v > 0}
            top = max(extra, key=extra.get) if extra else None
            self._rows.append({"prompt": prompt, "reward": s["reward"],
                               "no_json": s["penalty"] == PENALTY_NO_JSON,
                               "out_of_range": s["penalty"] == PENALTY_OUT_OF_RANGE,
                               "all_zero": s["penalty"] != PENALTY_NO_JSON and not extra,
                               "right_product": None if top is None else top in self._news_skus(prompt)})
            rewards.append(s["reward"])
        return rewards

    def pop_stats(self) -> Dict[str, float]:
        """Numbers over the completions scored since the last call (empty if none)."""
        rows, self._rows = self._rows, []
        if not rows:
            return {}
        groups: Dict[DayPrompt, List[float]] = {}
        for r in rows:
            groups.setdefault(r["prompt"], []).append(r["reward"])
        right = [r["right_product"] for r in rows if r["right_product"] is not None]
        share = lambda key: sum(r[key] for r in rows) / len(rows)  # noqa: E731
        return {
            "health/no_json": share("no_json"),
            "health/out_of_range": share("out_of_range"),
            "health/all_zero": share("all_zero"),
            "health/right_product": sum(right) / len(right) if right else float("nan"),
            "health/zero_std_groups": sum(max(g) - min(g) < 1e-9 for g in groups.values()) / len(groups),
            "health/group_best_minus_mean": sum(max(g) - sum(g) / len(g) for g in groups.values()) / len(groups),
        }
