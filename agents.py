"""
Agents that play the Pharma environment, shared by the dashboard (server/dashboard.py)
and the evaluation script (eval.py).

Every agent has the same interface:

    agent.start_episode(env)            # called once, right after env.reset()
    reply, error = agent.act(obs, history)

`reply` is the text sent as PharmaAction(message=reply); `error` is None or a
description of what went wrong (e.g. a failed LLM call). `history` is the list of
(day, reply, feedback) tuples from earlier days, as built in inference.py.

Reference agents for evaluation:
- NothingAgent:  never orders - the floor.
- BaselineAgent: a simple order-up-to rule using only the observation.
- OracleAgent:   perfect foresight - reads the episode's hidden demand and lead
                 times from the environment and plans just-in-time deliveries.
                 It cheats by design: it only marks the best achievable score.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import inference
from models import InventoryState

History = List[Tuple[int, str, str]]


class Agent:
    """Base class: override act(); override start_episode() if the agent needs setup."""

    def start_episode(self, env: Any) -> None:
        pass

    def act(self, obs: InventoryState, history: History) -> Tuple[str, Optional[str]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Floor and baseline
# ---------------------------------------------------------------------------

class NothingAgent(Agent):
    def act(self, obs: InventoryState, history: History) -> Tuple[str, Optional[str]]:
        return "{}", None


# Extra safety days (from news) are capped so that lead time + safety + extra stays within
# this many days of demand: each storage pool holds about 21 days of its SKUs' base demand.
MAX_COVER_DAYS = 18.0
MAX_EXTRA_DAYS = 15.0


def baseline_lead_time(sku: Any) -> float:
    """The baseline's lead-time estimate: the most pessimistic of the average and the last 3."""
    return max([sku.avg_lead_time, *sku.lead_time_last3_orders])


def baseline_policy(
    obs: InventoryState,
    safety_days: float = 3.0,
    extra_days: Optional[Dict[str, float]] = None,
    min_extra: float = 0.0,
) -> Dict[str, float]:
    """
    Order-up-to policy that uses only what the agent can observe: keep enough
    stock plus inbound to cover the (most pessimistic) lead time plus a few
    safety days of recent demand.

    extra_days adds safety days per SKU (e.g. chosen from the news), clipped to
    [min_extra, MAX_EXTRA_DAYS] and so that the total cover stays within MAX_COVER_DAYS.
    min_extra below 0 lets an agent keep less than the default (never below the lead time).
    """
    extra_days = extra_days or {}
    inbound: Dict[str, float] = {}
    for sku_id, qty, _ in obs.expected_inbound_orders:
        inbound[sku_id] = inbound.get(sku_id, 0.0) + qty

    orders: Dict[str, float] = {}
    for sku_id, sku in obs.skus.items():
        demand = max(sku.avg_demand_last_5_days, sku.avg_demand_per_day)
        if demand <= 0:
            continue
        lead_time = baseline_lead_time(sku)
        extra = min(max(float(extra_days.get(sku_id, 0.0)), min_extra, -safety_days), MAX_EXTRA_DAYS,
                    max(MAX_COVER_DAYS - lead_time - safety_days, 0.0))
        shortfall = demand * (lead_time + safety_days + extra) - sku.inventory_on_hand - inbound.get(sku_id, 0.0)
        if shortfall > 0:
            orders[sku_id] = round(shortfall)
    return orders


class BaselineAgent(Agent):
    def act(self, obs: InventoryState, history: History) -> Tuple[str, Optional[str]]:
        return json.dumps(baseline_policy(obs)), None


# ---------------------------------------------------------------------------
# News: the baseline plus extra safety days for announced disruptions
# ---------------------------------------------------------------------------

def _surge_profile(event: Any, day: int) -> float:
    """
    Demand multiple implied by a demand_surge event's fields alone: rising from 1x five
    days before start to 1.5x at start, to the peak multiplier at peak_day, back to
    1.5x at end_day and to 1x five days later.
    """
    points = [(event.start_day - 5, 1.0), (event.start_day, 1.5), (event.peak_day, event.multiplier),
              (event.end_day, 1.5), (event.end_day + 5, 1.0)]
    if day <= points[0][0] or day >= points[-1][0]:
        return 1.0
    for (d0, v0), (d1, v1) in zip(points, points[1:]):
        if d0 <= day <= d1:
            return v0 if d1 == d0 else v0 + (v1 - v0) * (day - d0) / (d1 - d0)
    return 1.0


def news_extra_days(obs: InventoryState, events: List[Any], safety_days: float = 3.0) -> Dict[str, float]:
    """
    Extra safety days per SKU implied by the news events published so far (announce_day
    <= today <= end_day), worked out from the events' exact fields:

    supplier_delay (lead times x m): cover the slower lead time, m x avg_lead_time,
        less what the baseline already assumes.
    demand_surge: the baseline covers its horizon (lead time + safety days) at recent
        demand, which lags a rising wave; add the days that make up for the demand
        expected over that horizon compared with the last 5 days.
    """
    today = obs.current_date
    extra: Dict[str, float] = {}
    for e in events:
        sku = obs.skus.get(e.sku_id)
        if sku is None or not e.announce_day <= today <= e.end_day:
            continue
        lead_time = baseline_lead_time(sku)
        if e.kind == "supplier_delay":
            days = e.multiplier * sku.avg_lead_time - lead_time
        else:
            horizon = lead_time + safety_days
            ahead = [_surge_profile(e, today + k) for k in range(1, math.ceil(horizon) + 1)]
            recent = [_surge_profile(e, today - k) for k in range(5)]
            days = horizon * (sum(ahead) / len(ahead) / (sum(recent) / len(recent)) - 1.0)
        extra[e.sku_id] = extra.get(e.sku_id, 0.0) + max(days, 0.0)
    return extra


class NewsRuleAgent(Agent):
    """
    The baseline plus news_extra_days() from the exact event behind each published news
    item: what perfect understanding of the news is worth to the baseline. It reads the
    structured events (not the text), so it is a reference, like the oracle, and the
    teacher an LLM reading the text can learn from.
    """

    def __init__(self) -> None:
        self._events: List[Any] = []

    def start_episode(self, env: Any) -> None:
        self._events = list(env._episode_config.news)

    def act(self, obs: InventoryState, history: History) -> Tuple[str, Optional[str]]:
        return json.dumps(baseline_policy(obs, extra_days=news_extra_days(obs, self._events))), None


class NewsLLMAgent(Agent):
    """
    The hybrid: the baseline does the arithmetic, and an LLM reads the news text and
    answers only with extra safety days per SKU (inference.NEWS_SYSTEM_PROMPT). On days
    with no news the model is not called and the agent plays exactly like the baseline,
    so a confused model can only hurt on news days, and only within MAX_EXTRA_DAYS.
    """

    NO_CALL = "{}  (no news today: baseline only, model not called)"

    def __init__(self, model: str, api_key: str, base_url: str, client: Any = None,
                 timeout: float = 60.0) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)
        self.client = client
        self.model = model
        self.action = "news_extra_days"
        self.last_model_reply = ""

    def act(self, obs: InventoryState, history: History) -> Tuple[str, Optional[str]]:
        if not obs.news:
            self.last_model_reply = self.NO_CALL
            return json.dumps(baseline_policy(obs)), None
        self.last_model_reply = ""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": inference.NEWS_SYSTEM_PROMPT},
                    {"role": "user", "content": inference.build_news_prompt(obs)},
                ],
                temperature=inference.TEMPERATURE,
                max_tokens=inference.MAX_TOKENS,
            )
            text = (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            # The baseline still orders: a failed call costs only the news adjustment.
            return json.dumps(baseline_policy(obs)), f"{type(exc).__name__}: {exc}"
        self.last_model_reply = text
        extra = inference.parse_extra_days(obs, inference.parse_json_object(text))
        return json.dumps(baseline_policy(obs, extra_days=extra)), None


# ---------------------------------------------------------------------------
# Ceiling: perfect foresight
# ---------------------------------------------------------------------------

class OracleAgent(Agent):
    """
    Knows the episode's true demand for every day and the true lead time of an
    order placed on any day (both fixed by the seed since the noise is drawn at
    reset), and plans orders so each day's demand arrives as late as possible:

    1. Starting stock covers the earliest days.
    2. Each remaining day t is assigned to the order day whose delivery arrives
       latest while still arriving on or before t.
    3. Each day, place the orders planned for that day.

    Days before any delivery can arrive are lost for every agent, so the oracle
    marks the best score reachable, not 1.0.
    """

    def __init__(self) -> None:
        self._plan: Dict[int, Dict[str, float]] = {}

    def start_episode(self, env: Any) -> None:
        cfg = env._episode_config
        days = cfg.no_of_days
        all_skus = {sku_id: 1.0 for sku_id in cfg.skus}
        # Pure lookups into the pre-drawn noise: calling them changes nothing.
        demand = [env._get_true_demands(d) for d in range(days)]
        arrival = [{k: d + lt for k, lt in env._get_true_lead_times(all_skus, d).items()} for d in range(days)]

        self._plan = {d: {} for d in range(days)}
        for sku_id in cfg.skus:
            stock = cfg.initial_inventory.get(sku_id, 0.0)
            for t in range(days):
                need = demand[t][sku_id]
                used = min(stock, need)
                stock -= used
                need -= used
                if need <= 0:
                    continue
                feasible = [d for d in range(t + 1) if arrival[d][sku_id] <= t]
                if not feasible:
                    continue  # no order can arrive in time: unavoidable shortfall
                order_day = max(feasible, key=lambda d: (arrival[d][sku_id], d))
                self._plan[order_day][sku_id] = self._plan[order_day].get(sku_id, 0.0) + need

    def act(self, obs: InventoryState, history: History) -> Tuple[str, Optional[str]]:
        orders = {k: q for k, q in self._plan.get(obs.current_date, {}).items() if q > 0}
        return json.dumps(orders), None


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

class LLMAgent(Agent):
    """
    One chat call per day, with the prompt and history format from inference.py.

    action="adjust" (default): the baseline rule orders a default (lead time + 3 safety
    days) and the model answers only with adjustment days per SKU (-3 to +10), so a
    copied number, a 0 or a missing SKU plays like the baseline instead of emptying the
    shelf. days_of_cover is left out of the report, since small models copied it.
    action="days": the model answers with days of stock wanted per SKU, and
    inference.days_to_units() converts that into units before the order is sent.
    action="units": the model writes units directly (the benchmark's native format).

    `last_model_reply` keeps the model's own text for logging; act() returns the order
    actually sent to the environment.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        client: Any = None,
        timeout: float = 60.0,
        action: str = "adjust",
    ) -> None:
        """timeout: seconds per call. Raise it for large models on a CPU, where one call can take minutes."""
        if action not in inference.ACTION_FORMATS:
            raise ValueError(f"action must be one of {inference.ACTION_FORMATS}, not {action!r}")
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)
        self.client = client
        self.model = model
        self.action = action
        self.last_model_reply = ""

    def act(self, obs: InventoryState, history: History) -> Tuple[str, Optional[str]]:
        obs_dict = inference.observation_to_dict(obs, include_cover=self.action != "adjust")
        user_prompt = inference.build_user_prompt(obs_dict, history, step=obs.current_date + 1, action=self.action)
        self.last_model_reply = ""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": inference.system_prompt(self.action)},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=inference.TEMPERATURE,
                max_tokens=inference.MAX_TOKENS,
            )
            text = (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            # In "adjust" mode a failed call still orders the default; otherwise nothing.
            fallback = json.dumps(baseline_policy(obs)) if self.action == "adjust" else "{}"
            return fallback, f"{type(exc).__name__}: {exc}"
        self.last_model_reply = text
        if self.action == "adjust":
            extra = inference.adjust_days(obs, inference.parse_json_object(text))
            return json.dumps(baseline_policy(obs, extra_days=extra, min_extra=inference.ADJUST_DAYS_RANGE[0])), None
        if self.action == "days":
            return json.dumps(inference.days_to_units(obs, inference.parse_json_object(text))), None
        return text or "{}", None
