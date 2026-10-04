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


def baseline_policy(obs: InventoryState, safety_days: float = 3.0) -> Dict[str, float]:
    """
    Order-up-to policy that uses only what the agent can observe: keep enough
    stock plus inbound to cover the (most pessimistic) lead time plus a few
    safety days of recent demand.
    """
    inbound: Dict[str, float] = {}
    for sku_id, qty, _ in obs.expected_inbound_orders:
        inbound[sku_id] = inbound.get(sku_id, 0.0) + qty

    orders: Dict[str, float] = {}
    for sku_id, sku in obs.skus.items():
        demand = max(sku.avg_demand_last_5_days, sku.avg_demand_per_day)
        if demand <= 0:
            continue
        lead_time = max([sku.avg_lead_time, *sku.lead_time_last3_orders])
        shortfall = demand * (lead_time + safety_days) - sku.inventory_on_hand - inbound.get(sku_id, 0.0)
        if shortfall > 0:
            orders[sku_id] = round(shortfall)
    return orders


class BaselineAgent(Agent):
    def act(self, obs: InventoryState, history: History) -> Tuple[str, Optional[str]]:
        return json.dumps(baseline_policy(obs)), None


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
    """Same prompt and history format as inference.py, one chat call per day."""

    def __init__(self, model: str, api_key: str, base_url: str, client: Any = None, timeout: float = 60.0) -> None:
        """timeout: seconds per call. Raise it for large models on a CPU, where one call can take minutes."""
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)
        self.client = client
        self.model = model

    def act(self, obs: InventoryState, history: History) -> Tuple[str, Optional[str]]:
        obs_dict = inference.observation_to_dict(obs)
        user_prompt = inference.build_user_prompt(obs_dict, history, step=obs.current_date + 1)
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": inference.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=inference.TEMPERATURE,
                max_tokens=inference.MAX_TOKENS,
            )
            text = (completion.choices[0].message.content or "").strip()
            return text or "{}", None
        except Exception as exc:
            return "{}", f"{type(exc).__name__}: {exc}"
