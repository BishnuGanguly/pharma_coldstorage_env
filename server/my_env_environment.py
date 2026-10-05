from __future__ import annotations

import json
import math
import re
import uuid
from random import Random
from typing import Any, Dict, List, Optional, Tuple, Union

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    EpisodeConfig,
    InventoryState,
    PharmaAction,
    SKUEpisodeConfig,
    SKUState,
)
from news import visible_news
from tasks import compute_step_reward, get_task_config


# ---------------------------------------------------------------------------
# PharmaEnvironment
# ---------------------------------------------------------------------------

class PharmaEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._episode_config: Optional[EpisodeConfig] = None
        self._inventory_state: Optional[InventoryState] = None
        self._episode_id: str = ""
        self._step_count: int = 0
        self._rng: Optional[Random] = None
        # The episode's randomness, drawn once at reset as standard-normal values per
        # SKU and day and scaled by the SKU's std when used. Steps only read these
        # tables, so the agent's actions can never change the world it faces.
        self._demand_z: Dict[str, List[float]] = {}
        self._lead_time_z: Dict[str, List[float]] = {}
        # Ground-truth open orders: (sku_id, qty, order_day, true_arrival_day).
        # order_day is kept so the lead time can be revealed on delivery.
        self._open_orders: List[Tuple[str, float, int, int]] = []

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        episode_config: Optional[Union[EpisodeConfig, Dict[str, Any]]] = None,
        task_name: Optional[str] = None,
        news: int = 0,
        **kwargs: Any,
    ) -> InventoryState:
        """
        Start a new episode.

        The episode is chosen, in order of precedence, from:
          - episode_config: an EpisodeConfig or its dict form (as sent over HTTP/WS)
          - task_name:      a key of tasks.TASK_REGISTRY, built with `seed` and the
                            news level `news` (0 none, 1 exact, 2 varied; see news.py)
          - the built-in default config
        """
        self._rng = Random(seed)
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._open_orders = []
        if episode_config is not None:
            self._episode_config = EpisodeConfig.model_validate(episode_config)
        elif task_name is not None:
            self._episode_config = get_task_config(task_name, seed, news=news)
        else:
            self._episode_config = self._default_episode_config()
        self._draw_world_noise()
        self._inventory_state = self._build_initial_state()
        self._update_insights()
        self._update_news()
        return self._inventory_state

    def _draw_world_noise(self) -> None:
        """
        Deal all of the episode's noise up front from the seeded generator: one
        demand value per SKU per day, and one lead-time value per SKU per order day.
        The same seed therefore gives the same demand on every day, and the same
        lead time for an order placed on a given day, whatever the agent does.
        """
        days = self._episode_config.no_of_days
        skus = list(self._episode_config.skus)
        self._demand_z = {sku_id: [self._rng.gauss(0.0, 1.0) for _ in range(days)] for sku_id in skus}
        self._lead_time_z = {sku_id: [self._rng.gauss(0.0, 1.0) for _ in range(days)] for sku_id in skus}

    def _build_initial_state(self) -> InventoryState:
        cfg = self._episode_config
        skus = {
            sku_id: SKUState(
                sku_id=sku_id,
                cold_storage_required=sku_cfg.cold_storage_required,
                stockout_penalty=sku_cfg.stockout_penalty,
                waste_penalty=sku_cfg.waste_penalty,
                inventory_on_hand= cfg.initial_inventory.get(sku_id,0.0),
            )
            for sku_id, sku_cfg in cfg.skus.items()
        }

        cold_used = sum(s.inventory_on_hand for s in skus.values() if s.cold_storage_required)
        ambient_used = sum(s.inventory_on_hand for s in skus.values() if not s.cold_storage_required)

        return InventoryState(
            current_date=0,
            cold_storage_total_capacity=cfg.cold_storage_total_capacity,
            cold_storage_current_capacity=cold_used,
            ambient_storage_total_capacity=cfg.ambient_storage_total_capacity,
            ambient_storage_current_capacity=ambient_used,
            skus=skus,
            demand_history={sku_id: [] for sku_id in cfg.skus.keys()},
            lead_time_history={sku_id: [] for sku_id in cfg.skus.keys()},
        )

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def step(
        self,
        action: PharmaAction,
        **kwargs: Any,
    ) -> InventoryState:
        ts = self._inventory_state
        if ts is None:
            raise RuntimeError("Call reset() before step().")
        if ts.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._step_count += 1
        orders = self._sanitize_orders(self._parse_action(action))
        self._handle_action(orders)
        self._inventory_state.reward = compute_step_reward(ts)
        return self._inventory_state

    # -----------------------------------------------------------------------
    # Parse action
    # -----------------------------------------------------------------------

    def _parse_action(self, action: PharmaAction) -> Dict[str, Any]:
        if action.orders:
            return dict(action.orders)
        try:
            raw = re.search(r"\{.*\}", action.message or "", re.DOTALL)
            if raw:
                parsed = json.loads(raw.group())
                if isinstance(parsed, dict):
                    # Tolerate a wrapped form such as {"orders": {...}}.
                    if isinstance(parsed.get("orders"), dict):
                        parsed = parsed["orders"]
                    return parsed
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return {}

    def _sanitize_orders(self, orders: Dict[str, Any]) -> Dict[str, float]:
        """
        Keep only positive, finite quantities for SKUs that exist in this episode.
        SKU names are matched case-insensitively; numeric strings are accepted.
        Anything else is dropped rather than crashing the step.
        """
        known = {sku_id.lower(): sku_id for sku_id in self._inventory_state.skus}
        clean: Dict[str, float] = {}
        for name, qty in orders.items():
            sku_id = known.get(str(name).strip().lower())
            if sku_id is None or isinstance(qty, bool):
                continue
            try:
                qty = float(qty)
            except (TypeError, ValueError):
                continue
            if math.isfinite(qty) and qty > 0:
                clean[sku_id] = clean.get(sku_id, 0.0) + qty
        return clean

    # -----------------------------------------------------------------------
    # Handle action  (main step logic)
    # -----------------------------------------------------------------------

    def _handle_action(self, orders: Dict[str, float]) -> None:
        ts = self._inventory_state
        cfg = self._episode_config
        day = ts.current_date

        # 1. True demands for today
        true_demands = self._get_true_demands(day)

        # 2. Process inbound arrivals + overflow.
        #    Several orders for the same SKU can land on the same day, so
        #    quantities are summed per SKU before allocation.
        arriving = [o for o in self._open_orders if o[3] <= day]
        self._open_orders = [o for o in self._open_orders if o[3] > day]
        arriving_qty: Dict[str, float] = {}
        for sku_id, qty, _, _ in arriving:
            arriving_qty[sku_id] = arriving_qty.get(sku_id, 0.0) + qty

        cold_remaining = ts.cold_storage_total_capacity - ts.cold_storage_current_capacity
        ambient_remaining = ts.ambient_storage_total_capacity - ts.ambient_storage_current_capacity
        avg_demands = {sku_id: ts.skus[sku_id].avg_demand_per_day for sku_id in ts.skus}

        accepted, wasted = self._handle_inbound_overflow(
            list(arriving_qty.items()),
            cold_remaining,
            ambient_remaining,
            avg_demands,
        )

        for sku_id, qty in accepted.items():
            ts.skus[sku_id].inventory_on_hand += qty

        # 3. Update inventory excess with today's waste
        self._update_inventory_excess(arriving_qty, wasted)

        # 4 & 5. Fulfill demands
        self._fulfill_demands(true_demands)

        # 6. True lead times for agent orders + clamp to >= 1
        true_lead_times = self._get_true_lead_times(orders, day)

        # 7. Update actual inbound with true lead times
        self._update_actual_inbound(orders, true_lead_times, day)

        # 8. Update expected inbound with avg lead times
        self._update_expected_inbound(orders, day)

        # 9. Update history buffers. A lead time only becomes observable once
        #    the order has been delivered, never at order placement.
        for sku_id, demand in true_demands.items():
            ts.demand_history[sku_id].append(demand)

        for sku_id, _, order_day, arrival_day in sorted(arriving, key=lambda o: o[2]):
            ts.lead_time_history[sku_id].append(float(arrival_day - order_day))

        # 10. Recalculate insights
        self._update_insights()

        # 11. Recompute storage usage and advance date
        self._recompute_storage()
        ts.current_date += 1
        if ts.current_date >= cfg.no_of_days:
            ts.done = True
        self._update_news()

    # -----------------------------------------------------------------------
    # True demand
    # -----------------------------------------------------------------------

    def _get_true_demands(self, day: int) -> Dict[str, float]:
        demands: Dict[str, float] = {}
        for sku_id, sku_cfg in self._episode_config.skus.items():
            curve_val = sku_cfg.demand_curve[day] if sku_cfg.demand_curve else 1.0
            mean = sku_cfg.base_demand * curve_val
            noise = sku_cfg.demand_std * self._demand_z[sku_id][day]
            demands[sku_id] = max(0.0, mean + noise)
        return demands

    # -----------------------------------------------------------------------
    # True lead times
    # -----------------------------------------------------------------------

    def _get_true_lead_times(self, orders: Dict[str, float], day: int) -> Dict[str, int]:
        lead_times: Dict[str, int] = {}
        for sku_id in orders:
            sku_cfg = self._episode_config.skus.get(sku_id)
            if sku_cfg is None:
                continue
            curve_val = sku_cfg.lead_time_curve[day] if sku_cfg.lead_time_curve else 1.0
            mean = sku_cfg.base_lead_time * curve_val
            noise = sku_cfg.lead_time_std * self._lead_time_z[sku_id][day]
            lead_times[sku_id] = max(1, round(mean + noise))
        return lead_times

    # -----------------------------------------------------------------------
    # Inbound overflow
    # -----------------------------------------------------------------------

    def _handle_inbound_overflow(
        self,
        arriving_orders: List[Tuple[str, float]],
        cold_remaining: float,
        ambient_remaining: float,
        avg_demand_per_day: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Allocates arriving inbound to inventory using demand-ratio proportional allocation.
        Returns (accepted_qty_per_sku, wasted_qty_per_sku).
        Handles cold and ambient pools independently.
        """
        ts = self._inventory_state

        cold_orders = [
            (sku_id, qty) for sku_id, qty in arriving_orders
            if ts.skus[sku_id].cold_storage_required
        ]
        ambient_orders = [
            (sku_id, qty) for sku_id, qty in arriving_orders
            if not ts.skus[sku_id].cold_storage_required
        ]

        accepted: Dict[str, float] = {}
        wasted: Dict[str, float] = {}

        for pool_orders, capacity in [(cold_orders, cold_remaining), (ambient_orders, ambient_remaining)]:
            pool_accepted, pool_wasted = self._allocate_pool(pool_orders, capacity, avg_demand_per_day)
            accepted.update(pool_accepted)
            wasted.update(pool_wasted)

        return accepted, wasted

    def _allocate_pool(
        self,
        orders: List[Tuple[str, float]],
        capacity: float,
        avg_demand_per_day: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Allocates free capacity across SKUs proportional to avg daily demand.
        SKUs whose inbound fits within their share are accepted in full and the
        unused part of their share is redistributed; once no remaining SKU fits,
        each gets its share of what is left. Whatever does not fit is wasted.
        """
        if not orders:
            return {}, {}

        total_inbound = sum(qty for _, qty in orders)
        if total_inbound <= capacity:
            return dict(orders), {sku_id: 0.0 for sku_id, _ in orders}

        def weight(sku_id: str) -> float:
            return max(avg_demand_per_day.get(sku_id, 1.0), 1e-9)

        accepted: Dict[str, float] = {}
        pending: List[Tuple[str, float]] = list(orders)
        remaining_capacity = max(capacity, 0.0)

        while pending and remaining_capacity > 0:
            total_weight = sum(weight(sku_id) for sku_id, _ in pending)
            share = {sku_id: weight(sku_id) / total_weight * remaining_capacity for sku_id, _ in pending}
            fits = [(sku_id, qty) for sku_id, qty in pending if qty <= share[sku_id]]
            if not fits:
                # Nobody fits within their share: everyone gets exactly their share.
                for sku_id, _ in pending:
                    accepted[sku_id] = share[sku_id]
                break
            for sku_id, qty in fits:
                accepted[sku_id] = qty
                remaining_capacity -= qty
            pending = [(sku_id, qty) for sku_id, qty in pending if qty > share[sku_id]]

        wasted = {
            sku_id: qty - accepted.get(sku_id, 0.0)
            for sku_id, qty in orders
        }
        return accepted, wasted

    # -----------------------------------------------------------------------
    # Waste tracking
    # -----------------------------------------------------------------------

    def _update_inventory_excess(self, delivered: Dict[str, float], wasted: Dict[str, float]) -> None:
        """
        waste_fraction_today = sum(w_i * wasted_i / delivered_i) / sum(w_i)
        over the SKUs with a delivery today, where w_i is the SKU's waste_penalty.
        It is 0 when nothing arrived or every weight is 0.
        """
        ts = self._inventory_state
        # Units rejected today (reported to the agent; 0 = no waste).
        ts.inventory_excess_today = sum(wasted.values())

        weighted_fraction = 0.0
        total_weight = 0.0
        for sku_id, qty in delivered.items():
            if qty <= 0:
                continue
            weight = ts.skus[sku_id].waste_penalty
            weighted_fraction += weight * min(wasted.get(sku_id, 0.0) / qty, 1.0)
            total_weight += weight
        ts.waste_fraction_today = weighted_fraction / total_weight if total_weight > 0 else 0.0

        # Per-day waste-free score (feeds the episode score; +1 per zero-waste day).
        ts.inventory_excess_cumulative += 1.0 - ts.waste_fraction_today

    # -----------------------------------------------------------------------
    # Fulfill demands
    # -----------------------------------------------------------------------

    def _fulfill_demands(self, true_demands: Dict[str, float]) -> None:
        ts = self._inventory_state
        for sku_id, demand in true_demands.items():
            sku = ts.skus.get(sku_id)
            if sku is None:
                continue
            if demand == 0.0:
                sku.demand_fulfilled_today = 1.0
                sku.demand_fulfilled_cumulative += 1.0
                continue
            fulfilled = min(sku.inventory_on_hand, demand)
            sku.inventory_on_hand -= fulfilled
            sku.demand_fulfilled_today = fulfilled / demand
            sku.demand_fulfilled_cumulative += sku.demand_fulfilled_today

    # -----------------------------------------------------------------------
    # Actual inbound update
    # -----------------------------------------------------------------------

    def _update_actual_inbound(
        self,
        orders: Dict[str, float],
        true_lead_times: Dict[str, int],
        day: int,
    ) -> None:
        for sku_id, qty in orders.items():
            lead_time = true_lead_times.get(sku_id, 1)
            self._open_orders.append((sku_id, qty, day, day + lead_time))
        self._inventory_state.actual_inbound_orders = [
            (sku_id, qty, arrival_day) for sku_id, qty, _, arrival_day in self._open_orders
        ]

    # -----------------------------------------------------------------------
    # Expected inbound update
    # -----------------------------------------------------------------------

    def _update_expected_inbound(self, orders: Dict[str, float], day: int) -> None:
        ts = self._inventory_state
        ts.expected_inbound_orders = [
            (sku_id, qty, exp_day)
            for sku_id, qty, exp_day in ts.expected_inbound_orders
            if exp_day > day
        ]
        for sku_id, qty in orders.items():
            avg_lt = ts.skus[sku_id].avg_lead_time or self._episode_config.skus[sku_id].base_lead_time
            ts.expected_inbound_orders.append((sku_id, qty, day + round(avg_lt)))

    # -----------------------------------------------------------------------
    # Insights
    # -----------------------------------------------------------------------

    def _update_insights(self) -> None:
        ts = self._inventory_state
        for sku_id, sku in ts.skus.items():
            demand_hist = ts.demand_history.get(sku_id, [])
            lead_hist = ts.lead_time_history.get(sku_id, [])

            sku.avg_demand_per_day = (sum(demand_hist) / len(demand_hist)) if demand_hist else 0.0
            sku.avg_demand_last_5_days = (sum(demand_hist[-5:]) / min(5, len(demand_hist))) if demand_hist else 0.0

            sku.avg_lead_time = (sum(lead_hist) / len(lead_hist)) if lead_hist else float(self._episode_config.skus[sku_id].base_lead_time)
            sku.lead_time_last3_orders = lead_hist[-3:]

            sku.stockout_days_if_no_reorder = (
                sku.inventory_on_hand / sku.avg_demand_per_day
                if sku.avg_demand_per_day > 0 else 999.0
            )

    def _update_news(self) -> None:
        """Show the news published so far whose event is not over yet."""
        ts = self._inventory_state
        ts.news = visible_news(self._episode_config, ts.current_date)

    # -----------------------------------------------------------------------
    # Storage recompute
    # -----------------------------------------------------------------------

    def _recompute_storage(self) -> None:
        ts = self._inventory_state
        ts.cold_storage_current_capacity = sum(
            s.inventory_on_hand for s in ts.skus.values() if s.cold_storage_required
        )
        ts.ambient_storage_current_capacity = sum(
            s.inventory_on_hand for s in ts.skus.values() if not s.cold_storage_required
        )

    # -----------------------------------------------------------------------
    # State property
    # -----------------------------------------------------------------------

    @property
    def state(self) -> State:
        return State(episode_id=self._episode_id, step_count=self._step_count)

    # -----------------------------------------------------------------------
    # LLM prompt
    # -----------------------------------------------------------------------

    def to_llm_prompt(self) -> str:
        """
        The prompt an LLM agent sees for today: the system prompt plus today's report,
        exactly as built by inference.py (the single source of the prompt).
        """
        import inference  # imported lazily: only needed when a prompt is requested

        if self._inventory_state is None:
            return inference.SYSTEM_PROMPT + "\n\nNo state available. Call reset() first."
        report = inference.observation_to_dict(self._inventory_state)
        user_prompt = inference.build_user_prompt(report, [], step=self._inventory_state.current_date + 1)
        return inference.SYSTEM_PROMPT + "\n\n" + user_prompt

    # -----------------------------------------------------------------------
    # Default episode config
    # -----------------------------------------------------------------------

    def _default_episode_config(self) -> EpisodeConfig:
        D = 30
        return EpisodeConfig(
            task_name="default",
            no_of_days=D,
            cold_storage_total_capacity=500.0,
            ambient_storage_total_capacity=10000.0,
            initial_inventory={
                "insulin": 70.0,
                "paracetamol": 1000.0,
            },
            skus={
                "insulin": SKUEpisodeConfig(
                    sku_id="insulin",
                    no_of_days=D,
                    base_demand=10,
                    demand_std=1.5,
                    base_lead_time=3,
                    lead_time_std=0.5,
                    cold_storage_required=False,
                    stockout_penalty=100.0,
                    waste_penalty=5.0,
                ),
                "paracetamol": SKUEpisodeConfig(
                    sku_id="paracetamol",
                    no_of_days=D,
                    base_demand=200,
                    demand_std=20.0,
                    base_lead_time=5,
                    lead_time_std=1.0,
                    cold_storage_required=False,
                    stockout_penalty=20.0,
                    waste_penalty=1.0,
                ),
            },
        )
