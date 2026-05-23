from __future__ import annotations

import json
import re
import uuid
from random import Random
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    EpisodeConfig,
    InventoryState,
    PharmaAction,
    SKUEpisodeConfig,
    SKUState,
)
from  tasks import compute_step_reward, compute_final_score


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a pharmaceutical warehouse procurement agent managing a cold-chain inventory over a multi-day episode.

OBJECTIVE
---------
Every day you receive an inventory report and decide what to order.
Maximise prescription fill rate while avoiding stockouts, waste, and capacity overflow.

WAREHOUSE
---------
Two storage pools:
  cold_storage  — refrigerated, for cold-chain SKUs only.
  ambient       — standard storage, for all other SKUs.
Do not order quantities that would exceed the available capacity of either pool.

SKUS
----
Each SKU exposes:
  inventory_on_hand            — units on shelf right now.
  stockout_days_if_no_reorder  — days until stockout at current demand rate. Act before this hits 0.
  avg_demand_per_day           — rolling daily demand average. Use for reorder sizing.
  avg_demand_last_5_days       — recent daily demand average. Reacts faster to spikes.
  avg_lead_time                — average days from order to arrival. Use to time orders.
  lead_time_last3_orders       — recent lead times. Rising values = supply stress.
  expected_inbound_orders      — your open orders with estimated arrival dates.
  stockout_penalty             — criticality of this SKU. Higher = order first.

DECISION RULES
--------------
1. Order before stockout_days_if_no_reorder drops below avg_lead_time.
2. Higher stockout_penalty SKUs take priority when capacity or budget is tight.
3. expected_inbound_orders may be inaccurate if true lead times deviate from the average.
4. Capacity overflow is penalised — do not over-order.

ACTION FORMAT
-------------
Respond with a JSON object mapping SKU names to order quantities:
  {"insulin": 100, "paracetamol": 500}
To order nothing today respond with: {}
"""


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
        # self._demand_history: Dict[str, List[float]] = {}
        # self._lead_time_history: Dict[str, List[float]] = {}

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(
        self,
        episode_config: Optional[EpisodeConfig] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> InventoryState:
        self._rng = Random(seed)
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._episode_config = episode_config or self._default_episode_config()
        # self._demand_history = {sku_id: [] for sku_id in self._episode_config.skus}#issue
        # self._lead_time_history = {sku_id: [] for sku_id in self._episode_config.skus}#issue
        self._inventory_state = self._build_initial_state()
        return self._inventory_state

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
        orders = self._parse_action(action)
        self._handle_action(orders)
        self._inventory_state.reward = compute_step_reward(ts)
        return self._inventory_state

    # -----------------------------------------------------------------------
    # Parse action
    # -----------------------------------------------------------------------

    def _parse_action(self, action: PharmaAction) -> Dict[str, float]:
        if action.orders:
            return {k: v for k, v in action.orders.items() if v > 0}
        try:
            raw = re.search(r"\{.*\}", action.message, re.DOTALL)
            if raw:
                parsed = json.loads(raw.group())
                return {k: float(v) for k, v in parsed.items() if isinstance(v, (int, float)) and v > 0}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return {}

    # -----------------------------------------------------------------------
    # Handle action  (main step logic)
    # -----------------------------------------------------------------------

    def _handle_action(self, orders: Dict[str, float]) -> None:
        ts = self._inventory_state
        cfg = self._episode_config
        day = ts.current_date

        # 1. True demands for today
        true_demands = self._get_true_demands(day)

        # 2. Process inbound arrivals + overflow
        arriving = [
            (sku_id, qty, arr_day)
            for (sku_id, qty, arr_day) in ts.actual_inbound_orders
            if arr_day <= day
        ]
        ts.actual_inbound_orders = [
            o for o in ts.actual_inbound_orders if o[2] > day
        ]

        cold_remaining = ts.cold_storage_total_capacity - ts.cold_storage_current_capacity
        ambient_remaining = ts.ambient_storage_total_capacity - ts.ambient_storage_current_capacity
        avg_demands = {sku_id: ts.skus[sku_id].avg_demand_per_day for sku_id in ts.skus}

        accepted, wasted = self._handle_inbound_overflow(
            [(sku_id, qty) for sku_id, qty, _ in arriving],
            cold_remaining,
            ambient_remaining,
            avg_demands,
        )

        for sku_id, qty in accepted.items():
            ts.skus[sku_id].inventory_on_hand += qty

        # 3. Update inventory excess with today's waste
        self._update_inventory_excess(wasted)

        # 4 & 5. Fulfill demands
        self._fulfill_demands(true_demands)

        # 6. True lead times for agent orders + clamp to >= 1
        true_lead_times = self._get_true_lead_times(orders, day)

        # 7. Update actual inbound with true lead times
        self._update_actual_inbound(orders, true_lead_times, day)

        # 8. Update expected inbound with avg lead times
        self._update_expected_inbound(orders, day)

        # 9. Update history buffers
        for sku_id, demand in true_demands.items():
            ts = self._inventory_state
            ts.demand_history[sku_id].append(demand)

        for sku_id, lead_time in true_lead_times.items():
            ts = self._inventory_state
            ts.lead_time_history[sku_id].append(float(lead_time))

        # 10. Recalculate insights
        self._update_insights()

        # # 11. Recalculate global reward fields
        # self._update_global_reward_fields(wasted)

        # 12. Recompute storage usage and advance date
        self._recompute_storage()
        ts.current_date += 1
        if ts.current_date >= cfg.no_of_days:
            ts.done = True

    # -----------------------------------------------------------------------
    # True demand
    # -----------------------------------------------------------------------

    def _get_true_demands(self, day: int) -> Dict[str, float]:
        demands: Dict[str, float] = {}
        for sku_id, sku_cfg in self._episode_config.skus.items():
            curve_val = sku_cfg.demand_curve[day] if sku_cfg.demand_curve else 1.0
            mean = sku_cfg.base_demand * curve_val
            noise = self._rng.gauss(0, sku_cfg.demand_std) if sku_cfg.demand_std > 0 else 0.0
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
            noise = self._rng.gauss(0, sku_cfg.lead_time_std) if sku_cfg.lead_time_std > 0 else 0.0
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
        Recursively allocates capacity across SKUs proportional to avg daily demand.
        SKUs whose inbound is below their allocation donate freed capacity back to the pool.
        """
        if not orders:
            return {}, {}

        total_inbound = sum(qty for _, qty in orders)
        if total_inbound <= capacity:
            return dict(orders), {sku_id: 0.0 for sku_id, _ in orders}

        accepted: Dict[str, float] = {}
        pending: List[Tuple[str, float]] = list(orders)
        remaining_capacity = capacity

        while pending and remaining_capacity > 0:
            total_demand = sum(max(avg_demand_per_day.get(sku_id, 1.0), 1e-9) for sku_id, _ in pending)
            next_pending: List[Tuple[str, float]] = []
            freed = 0.0

            for sku_id, qty in pending:
                demand = max(avg_demand_per_day.get(sku_id, 1.0), 1e-9)
                ratio = demand / total_demand
                allocation = ratio * remaining_capacity
                if qty <= allocation:
                    accepted[sku_id] = qty
                    freed += allocation - qty
                else:
                    next_pending.append((sku_id, qty))

            remaining_capacity = freed
            if not next_pending or freed == 0.0:
                # Allocate remaining capacity to pending SKUs by ratio
                total_demand = sum(max(avg_demand_per_day.get(sku_id, 1.0), 1e-9) for sku_id, _ in next_pending)
                for sku_id, qty in next_pending:
                    demand = max(avg_demand_per_day.get(sku_id, 1.0), 1e-9)
                    ratio = demand / total_demand
                    accepted[sku_id] = ratio * remaining_capacity
                remaining_capacity = 0.0
                next_pending = []

            pending = next_pending

        wasted = {
            sku_id: qty - accepted.get(sku_id, 0.0)
            for sku_id, qty in orders
        }
        return accepted, wasted

    # -----------------------------------------------------------------------
    # Waste tracking
    # -----------------------------------------------------------------------

    def _update_inventory_excess(self, wasted: Dict[str, float]) -> None:
        #inventory_excess_cumulative = cumulative(1/1+invenetory_excess_today)
        ts = self._inventory_state
        #inventory excess today is calculated for step reward , the lower it is the better, 0 means no waste at all.
        ts.inventory_excess_today = 0.0
        for sku_id, waste_qty in wasted.items():
            ts.inventory_excess_today += waste_qty

        #inventory excess cumulative is calculated for episode score , the higher it is the better ,  5 means 5 days of zero waste.
            
        ts.inventory_excess_cumulative += (1/(1+ts.inventory_excess_today))

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
        ts = self._inventory_state
        for sku_id, qty in orders.items():
            lead_time = true_lead_times.get(sku_id, 1)
            ts.actual_inbound_orders.append((sku_id, qty, day + lead_time))

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

            avg_3d = (sum(demand_hist[-3:]) / min(3, len(demand_hist))) if len(demand_hist) >= 1 else 0.0
            avg_7d = (sum(demand_hist[-7:]) / min(7, len(demand_hist))) if len(demand_hist) >= 1 else 0.0
            #sku.demand_trend = avg_3d - avg_7d

            sku.avg_lead_time = (sum(lead_hist) / len(lead_hist)) if lead_hist else float(self._episode_config.skus[sku_id].base_lead_time)
            sku.lead_time_last3_orders = lead_hist[-3:]

            sku.stockout_days_if_no_reorder = (
                sku.inventory_on_hand / sku.avg_demand_per_day
                if sku.avg_demand_per_day > 0 else 999.0
            )

    # -----------------------------------------------------------------------
    # Global reward fields
    # -----------------------------------------------------------------------

    # def _update_global_reward_fields(self) -> None:
    #     ts = self._inventory_state
    #     total_capacity = ts.cold_storage_total_capacity + ts.ambient_storage_total_capacity
    #     total_inventory = sum(s.inventory_on_hand for s in ts.skus.values())
    #     ts.inventory_excess_today = max(1.0, total_inventory / total_capacity)
    #     ts.inventory_excess_cumulative += max(0.0, ts.inventory_excess_today - 1.0)

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
        if self._inventory_state is None:
            return _SYSTEM_PROMPT + "\n\nNo state available. Call reset() first."
        ts = self._inventory_state
        report = {
            "day": ts.current_date,
            "storage": {
                "cold_total": ts.cold_storage_total_capacity,
                "cold_used": round(ts.cold_storage_current_capacity, 1),
                "cold_ratio": round(ts.cold_storage_ratio, 3),
                "ambient_total": ts.ambient_storage_total_capacity,
                "ambient_used": round(ts.ambient_storage_current_capacity, 1),
                "ambient_ratio": round(ts.ambient_storage_ratio, 3),
            },
            "expected_inbound_orders": ts.expected_inbound_orders,
            "inventory": {
                sku_id: {
                    "inventory_on_hand": round(sku.inventory_on_hand, 1),
                    "avg_demand_per_day": round(sku.avg_demand_per_day, 2),
                    "avg_demand_last_5_days": round(sku.avg_demand_last_5_days, 2),
                    "avg_lead_time": round(sku.avg_lead_time, 1),
                    "lead_time_last3_orders": [round(x, 1) for x in sku.lead_time_last3_orders],
                    "stockout_days_if_no_reorder": round(sku.stockout_days_if_no_reorder, 1),
                    #"demand_trend": round(sku.demand_trend, 3),
                    "cold_storage_required": sku.cold_storage_required,
                    "stockout_penalty": sku.stockout_penalty,
                }
                for sku_id, sku in ts.skus.items()
            },
        }
        return (
            _SYSTEM_PROMPT
            + "\n\n--- TODAY'S INVENTORY REPORT ---\n"
            + json.dumps(report, indent=2)
            + "\n\n--- YOUR PROCUREMENT DECISION ---\n"
            + 'Respond with JSON: {"sku_name": quantity, ...}\n'
            + "Order nothing today with: {}\n"
        )

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