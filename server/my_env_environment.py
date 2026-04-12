from __future__ import annotations

import json
import re
import uuid
from math import exp
from random import Random
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import InventoryState, PharmaAction, SKUState, SupplierState
except ImportError:
    from models import InventoryState, PharmaAction, SKUState, SupplierState


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an intelligent pharmaceutical warehouse procurement agent.

OBJECTIVE
---------
Manage a cold-chain pharmaceutical warehouse over a 60-day episode.
Every day you receive an inventory report and decide what to order.
Your goal is to maximise prescription fill rate while minimising waste,
overstock, and unnecessary spend.

WAREHOUSE
---------
Two storage pools:
  cold_storage  — refrigerated, scarce, for insulin only.
  ambient       — standard storage, for all other drugs.

You cannot place an order if the relevant storage pool is at capacity.
You cannot place an order if procurement_budget_ratio is near 0.

SKUS
----
Each SKU has:
  inventory_on_hand        — units physically on shelf right now.
  inbound_expected_3d      — units you ordered, expected within 3 days (estimate).
  inbound_expected_7d      — units you ordered, expected within 7 days (estimate).
  demand_last_3d           — actual demand observed over last 3 days.
  demand_last_7d           — actual demand observed over last 7 days.
  demand_trend             — (demand_last_3d/3) - (demand_last_7d/7). Positive = accelerating.
  stockout_days_if_no_reorder — days until stockout if you order nothing today.
  coverage_gap_7d          — demand_last_7d - inventory_on_hand. Positive = shortage incoming.
  stockout_penalty         — cost of failing this SKU. Higher = more critical.
  substitute_coverage_ratio — fraction of demand coverable by substitute. 0.0 = no substitute.
  cold_storage_required    — True means this SKU needs the cold storage pool.

SUPPLIERS
---------
Each supplier has:
  last_observed_lead_time  — days from order to arrival (from last delivery).
  on_time_rate_14d         — fraction of recent orders delivered on time.
  disruption_active        — True if supplier is currently disrupted.
  sku_served               — list of SKUs this supplier can fulfill.
  cold_chain_certified     — must be True to supply insulin.
  unit_cost                — relative cost. Higher = more expensive.
  expedite_allowed         — True if emergency fast delivery is possible.

DECISION RULES
--------------
1. Insulin can ONLY be ordered from cold_chain_certified suppliers.
2. A supplier cannot fulfill SKUs not in its sku_served list.
3. If orders_overdue_count > 0, treat inbound_expected figures as unreliable.
4. If cold_chain_integrity_flag = False, insulin is destroyed — order immediately.
5. If epidemic_alert_flag = True, demand spike is imminent — pre-stock critical SKUs.

PRIORITY ORDER (when budget is tight)
--------------------------------------
insulin (no substitute, life-critical)
> bp_medication (chronic patients, serious if missed)
> paracetamol (substitute available)
> vitamins (fully substitutable, lowest penalty)

ACTION FORMAT (FOLLOW EXACT SYNTAX)
-------------------------------------
Respond with a JSON object mapping SKU names to [quantity, supplier_name]:

{
  "insulin":      [quantity, "FastPharma"],
  "paracetamol":  [quantity, "GlobalMed"]
}

Rules:
- Only include SKUs you want to order today. Omit SKUs you do not want to order.
- quantity must be a positive number.
- Use exact SKU names and supplier names as shown in the inventory report.
- If you want to order nothing today, respond with: {}
"""


# ---------------------------------------------------------------------------
# PharmaEnvironment
# ---------------------------------------------------------------------------

class PharmaEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(self, inventory_state: InventoryState | None = None) -> None:
        super().__init__()
        self._inventory_state: Optional[InventoryState] = inventory_state
        self._step_count: int = 0
        self._episode_id: str = str(uuid.uuid4()) if inventory_state is not None else ""

        # Hidden simulation state — never exposed to agent
        self._hidden: Dict[str, Any] = {}

        # Demand history per SKU for rolling window computations
        # {sku_id: [demand_day_0, demand_day_1, ...]}
        self._demand_history: Dict[str, List[float]] = {}

        # Open purchase orders
        # Each entry: {sku_id, supplier_id, qty, expected_arrival_day, placed_day}
        self._open_orders: List[Dict[str, Any]] = []

        # Delivery history per supplier for on_time_rate computation
        # {supplier_id: [(was_on_time: bool), ...]}
        self._delivery_history: Dict[str, List[bool]] = {}

        # Random number generator
        self._rng: Optional[Random] = None

    # -----------------------------------------------------------------------
    # Public OpenEnv API
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> InventoryState:
        """
        Start a fresh episode.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        episode_id : str, optional
            Identifier for this episode run.
        kwargs : dict
            task_config : dict
                Full task configuration produced by a task constructor.
                If not provided, a default baseline config is used.

        Returns
        -------
        InventoryState
            Initial state — the first daily inventory report.
        """
        self._rng = Random(seed)
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._open_orders = []
        self._demand_history = {}
        self._delivery_history = {}

        task_config = kwargs.get("task_config", self._default_task_config())

        # -- Build hidden simulation state ----------------------------------
        self._hidden = {
            "duration_days":      task_config["duration_days"],
            "base_demands":       task_config["base_demands"],
            "demand_curves":      task_config["demand_curves"],
            "supplier_stress":    task_config["supplier_stress"],
            "total_budget":       task_config["total_budget"],
            "remaining_budget":   task_config["total_budget"],

            # True supplier lead time parameters (never shown to agent)
            "supplier_params":    task_config["supplier_params"],

            # Cold chain breach probability per day
            "cold_chain_breach_prob": task_config.get("cold_chain_breach_prob", 0.005),

            # Disruption Markov chain state per supplier
            "disruption_state": {
                s: False for s in task_config["supplier_params"]
            },
            "disruption_params": task_config.get("disruption_params", {
                "FastPharma": {"p_onset": 0.02, "p_persist": 0.30},
                "GlobalMed":  {"p_onset": 0.08, "p_persist": 0.70},
            }),
        }

        # -- Initialise demand history with zeros ---------------------------
        for sku_id in task_config["base_demands"]:
            self._demand_history[sku_id] = [0.0] * 7

        for supplier_id in task_config["supplier_params"]:
            self._delivery_history[supplier_id] = []

        # -- Build InventoryState -------------------------------------------
        self._inventory_state = self._build_initial_state(task_config)

        return self._inventory_state

    # -----------------------------------------------------------------------

    def step(
        self,
        action: PharmaAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> InventoryState:
        """
        Execute one day's procurement action and advance the simulation by one day.

        Order of operations per day:
        1. Parse and validate the agent's orders.
        2. Place valid orders (added to open order pipeline).
        3. Advance simulation by one day:
            a. Sample true demand from hidden process.
            b. Consume inventory (reduce on-hand, create backorders if needed).
            c. Process inbound arrivals (orders whose lead time has elapsed).
            d. Update disruption Markov chains.
            e. Possibly trigger cold chain breach.
        4. Update all observable Layer 2 and Layer 3 signals.
        5. Compute reward.
        6. Check terminal condition.
        7. Return updated InventoryState.

        Parameters
        ----------
        action : PharmaAction
            Agent's procurement decision for today.

        Returns
        -------
        InventoryState
            Updated daily inventory report for the next decision.
        """
        ts = self._inventory_state
        if ts is None:
            raise RuntimeError("reset() must be called before step().")
        if ts.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._step_count += 1
        day = ts.current_date
        feedback_parts: List[str] = []

        # -- Step 1+2: Parse and place orders --------------------------------
        orders = self._parse_action(action)
        order_feedback = self._place_orders(orders, day)
        feedback_parts.extend(order_feedback)

        # -- Step 3a: Sample true demand ------------------------------------
        true_demands = self._sample_true_demand(day)

        # -- Step 3b: Consume inventory -------------------------------------
        consumption_feedback = self._consume_inventory(true_demands)
        feedback_parts.extend(consumption_feedback)

        # -- Step 3c: Process inbound arrivals ------------------------------
        arrival_feedback = self._process_arrivals(day)
        feedback_parts.extend(arrival_feedback)

        # -- Step 3d: Update disruption states ------------------------------
        self._update_disruptions(day)

        # -- Step 3e: Cold chain breach check --------------------------------
        breach_feedback = self._check_cold_chain_breach()
        if breach_feedback:
            feedback_parts.append(breach_feedback)

        # -- Step 4: Update observable signals ------------------------------
        self._update_demand_history(true_demands)
        self._update_layer2_signals(day)
        self._update_layer3_signals()
        self._update_overdue_orders(day)

        # -- Advance date ---------------------------------------------------
        ts.current_date = day + 1
        ts.season_phase = ts.current_date / self._hidden["duration_days"]

        # -- Step 5: Compute reward (placeholder — reward fn injected later) -
        ts.reward = self._compute_reward(true_demands)

        # -- Step 6: Terminal check -----------------------------------------
        if ts.current_date >= self._hidden["duration_days"]:
            ts.done = True
            feedback_parts.append(
                f"Episode complete. Final fill rate (30d): "
                f"{ts.prescription_fill_rate_30d:.2%}."
            )

        ts.last_action_feedback = " | ".join(feedback_parts) if feedback_parts else "Day processed."

        return ts

    # -----------------------------------------------------------------------

    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
        )

    # -----------------------------------------------------------------------
    # Reset helper — build initial InventoryState
    # -----------------------------------------------------------------------

    def _build_initial_state(self, task_config: Dict[str, Any]) -> InventoryState:
        """Construct the full InventoryState at episode start."""

        # -- Build per-SKU states -------------------------------------------
        skus: Dict[str, SKUState] = {}
        for sku_id, sku_cfg in task_config["sku_configs"].items():
            skus[sku_id] = SKUState(
                sku_id                   = sku_id,
                name                     = sku_cfg["name"],
                cold_storage_required    = sku_cfg["cold_storage_required"],
                stockout_penalty         = sku_cfg["stockout_penalty"],
                substitute_coverage_ratio= sku_cfg.get("substitute_coverage_ratio", 0.0),
                inventory_on_hand        = task_config["starting_inventory"][sku_id],
                # Layer 2 starts at zero — no history yet
                demand_last_3d           = 0.0,
                demand_last_7d           = 0.0,
                demand_trend             = 0.0,
                stockout_days_if_no_reorder = 999.0,
                coverage_gap_7d          = 0.0,
                inbound_expected_3d      = 0.0,
                inbound_expected_7d      = 0.0,
                backorders               = 0.0,
            )

        # -- Build per-supplier states --------------------------------------
        suppliers: Dict[str, SupplierState] = {}
        for sup_id, sup_cfg in task_config["supplier_configs"].items():
            suppliers[sup_id] = SupplierState(
                supplier_id              = sup_id,
                cold_chain_certified     = sup_cfg["cold_chain_certified"],
                sku_served               = sup_cfg["sku_served"],
                unit_cost                = sup_cfg["unit_cost"],
                expedite_allowed         = sup_cfg.get("expedite_allowed", False),
                expedite_cost_multiplier = sup_cfg.get("expedite_cost_multiplier", 1.0),
                last_observed_lead_time  = sup_cfg["base_lead_time_mean"],
                on_time_rate_14d         = 1.0,
                disruption_active        = False,
            )

        # -- Compute initial capacity values --------------------------------
        cold_used = sum(
            skus[s].inventory_on_hand
            for s in skus if skus[s].cold_storage_required
        )
        ambient_used = sum(
            skus[s].inventory_on_hand
            for s in skus if not skus[s].cold_storage_required
        )

        state = InventoryState(
            current_date                    = 0,
            season_phase                    = 0.0,
            prescription_fill_rate_30d      = 1.0,
            prescription_fill_rate_7d       = 1.0,
            epidemic_alert_flag             = False,
            days_since_epidemic_alert_fired = 0,
            supply_disruption_days_last_30d = 0,
            procurement_budget_ratio        = 1.0,
            cold_storage_total_capacity     = task_config["cold_storage_total_capacity"],
            cold_storage_current_capacity   = cold_used,
            cold_storage_capacity_ratio     = cold_used / task_config["cold_storage_total_capacity"],
            ambient_storage_total_capacity  = task_config["ambient_storage_total_capacity"],
            ambient_storage_current_capacity= ambient_used,
            ambient_capacity_ratio          = ambient_used / task_config["ambient_storage_total_capacity"],
            cold_chain_integrity_flag       = True,
            orders_overdue_count            = 0,
            overdue_qty_total               = 0.0,
            skus                            = skus,
            suppliers                       = suppliers,
            done                            = False,
            reward                          = 0.0,
        )

        return state

    # -----------------------------------------------------------------------
    # Action handling
    # -----------------------------------------------------------------------

    def _parse_action(self, action: PharmaAction) -> Dict[str, Tuple[float, str]]:
        """
        Extract orders from PharmaAction.
        Tries action.orders first, then falls back to parsing action.message.
        Returns {sku_id: (quantity, supplier_id)}.
        """
        if action.orders:
            return action.orders

        # Fallback: parse JSON from message string
        try:
            raw = re.search(r"\{.*\}", action.message, re.DOTALL)
            if raw:
                parsed = json.loads(raw.group())
                orders: Dict[str, Tuple[float, str]] = {}
                for sku_id, val in parsed.items():
                    if isinstance(val, (list, tuple)) and len(val) == 2:
                        orders[sku_id] = (float(val[0]), str(val[1]))
                return orders
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return {}

    # -----------------------------------------------------------------------

    def _place_orders(
        self,
        orders: Dict[str, Tuple[float, str]],
        day: int,
    ) -> List[str]:
        """
        Validate and place purchase orders.
        Returns feedback strings for each order attempt.
        """
        ts = self._inventory_state
        feedback: List[str] = []

        for sku_id, (qty, supplier_id) in orders.items():

            # -- Validate SKU exists ----------------------------------------
            sku = ts.get_sku(sku_id)
            if sku is None:
                feedback.append(f"Order rejected: unknown SKU '{sku_id}'.")
                continue

            # -- Validate supplier exists -----------------------------------
            supplier = ts.get_supplier(supplier_id)
            if supplier is None:
                feedback.append(f"Order rejected: unknown supplier '{supplier_id}'.")
                continue

            # -- Validate supplier serves this SKU --------------------------
            if sku_id not in supplier.sku_served:
                feedback.append(
                    f"Order rejected: '{supplier_id}' does not serve '{sku_id}'."
                )
                continue

            # -- Validate cold chain constraint for insulin -----------------
            if sku.cold_storage_required and not supplier.cold_chain_certified:
                feedback.append(
                    f"Order rejected: '{sku_id}' requires cold chain — "
                    f"'{supplier_id}' is not certified."
                )
                continue

            # -- Validate disruption ----------------------------------------
            if supplier.disruption_active:
                feedback.append(
                    f"Order rejected: '{supplier_id}' is currently disrupted."
                )
                continue

            # -- Validate quantity ------------------------------------------
            if qty <= 0:
                feedback.append(f"Order rejected: quantity must be positive for '{sku_id}'.")
                continue

            # -- Validate budget --------------------------------------------
            order_cost = qty * supplier.unit_cost
            total_budget = self._hidden["total_budget"]
            remaining_budget = self._hidden["remaining_budget"]
            if order_cost > remaining_budget:
                feedback.append(
                    f"Order rejected: '{sku_id}' order costs {order_cost:.1f} "
                    f"but only {remaining_budget:.1f} budget remaining."
                )
                continue

            # -- Validate capacity ------------------------------------------
            if sku.cold_storage_required:
                available_cold = (
                    ts.cold_storage_total_capacity - ts.cold_storage_current_capacity
                )
                if qty > available_cold:
                    feedback.append(
                        f"Order rejected: '{sku_id}' needs {qty} cold storage units, "
                        f"only {available_cold:.1f} available."
                    )
                    continue
            else:
                available_ambient = (
                    ts.ambient_storage_total_capacity - ts.ambient_storage_current_capacity
                )
                if qty > available_ambient:
                    feedback.append(
                        f"Order rejected: '{sku_id}' needs {qty} ambient units, "
                        f"only {available_ambient:.1f} available."
                    )
                    continue

            # -- Place order -------------------------------------------------
            # Sample true lead time from hidden parameters (agent does not see this)
            params = self._hidden["supplier_params"][supplier_id]
            day_idx = min(day, len(self._hidden["demand_curves"].get(sku_id, [1.0])) - 1)
            stress_mult = self._hidden["supplier_stress"].get(supplier_id, [1.0] * 100)
            stress = stress_mult[min(day, len(stress_mult) - 1)]

            true_lead_time = max(1, self._rng.gauss(
                params["lead_time_mean"] * stress,
                params["lead_time_std"] * stress,
            ))
            true_lead_time = round(true_lead_time)

            expected_arrival_day = day + round(supplier.last_observed_lead_time * stress)

            self._open_orders.append({
                "sku_id":               sku_id,
                "supplier_id":          supplier_id,
                "qty":                  qty,
                "true_arrival_day":     day + true_lead_time,   # hidden
                "expected_arrival_day": expected_arrival_day,    # observable
                "placed_day":           day,
                "expedited":            False,
            })

            # Deduct budget
            self._hidden["remaining_budget"] -= order_cost
            ts.procurement_budget_ratio = (
                self._hidden["remaining_budget"] / self._hidden["total_budget"]
            )

            feedback.append(
                f"Order placed: {qty:.0f} units of '{sku_id}' "
                f"from '{supplier_id}' (expected in ~{round(supplier.last_observed_lead_time * stress)} days)."
            )

        return feedback

    # -----------------------------------------------------------------------
    # Hidden simulation — demand
    # -----------------------------------------------------------------------

    def _sample_true_demand(self, day: int) -> Dict[str, float]:
        """
        Sample true demand for each SKU today.
        Uses hidden demand_curves and base_demands.
        Result is NOT shown to agent — only its inventory consequences are visible.
        """
        true_demands: Dict[str, float] = {}
        curves = self._hidden["demand_curves"]
        base   = self._hidden["base_demands"]

        for sku_id, base_demand in base.items():
            curve  = curves.get(sku_id, [1.0] * 100)
            idx    = min(day, len(curve) - 1)
            mean   = base_demand * curve[idx]
            std    = mean * 0.15  # 15% coefficient of variation
            demand = max(0.0, self._rng.gauss(mean, std))
            true_demands[sku_id] = demand

        return true_demands

    # -----------------------------------------------------------------------

    def _consume_inventory(self, true_demands: Dict[str, float]) -> List[str]:
        """
        Reduce inventory by true demand. Create backorders if insufficient stock.
        Returns feedback strings for stockout events.
        """
        ts = self._inventory_state
        feedback: List[str] = []

        for sku_id, demand in true_demands.items():
            sku = ts.get_sku(sku_id)
            if sku is None:
                continue

            if sku.inventory_on_hand >= demand:
                sku.inventory_on_hand -= demand
            else:
                # Partial fill — remaining demand becomes backorder
                unmet = demand - sku.inventory_on_hand
                sku.inventory_on_hand = 0.0
                sku.backorders += unmet
                feedback.append(
                    f"Stockout: '{sku_id}' — {unmet:.1f} units unmet, "
                    f"added to backorders (total: {sku.backorders:.1f})."
                )

        # Update capacity after consumption
        self._recompute_storage_usage()

        return feedback

    # -----------------------------------------------------------------------
    # Hidden simulation — arrivals
    # -----------------------------------------------------------------------

    def _process_arrivals(self, day: int) -> List[str]:
        """
        Check open orders for arrivals today (true_arrival_day <= day).
        Add arrived stock to inventory. Update supplier delivery history.
        """
        ts = self._inventory_state
        feedback: List[str] = []
        still_open: List[Dict] = []

        for order in self._open_orders:
            if order["true_arrival_day"] <= day:
                # Order arrives today
                sku = ts.get_sku(order["sku_id"])
                if sku is None:
                    continue

                sku.inventory_on_hand += order["qty"]

                # First try to clear backorders with new stock
                if sku.backorders > 0:
                    cleared = min(sku.backorders, order["qty"])
                    sku.backorders = max(0.0, sku.backorders - cleared)

                # Update supplier delivery history
                was_on_time = order["true_arrival_day"] <= order["expected_arrival_day"]
                sup_id = order["supplier_id"]
                if sup_id not in self._delivery_history:
                    self._delivery_history[sup_id] = []
                self._delivery_history[sup_id].append(was_on_time)

                # Update last_observed_lead_time for this supplier
                supplier = ts.get_supplier(sup_id)
                if supplier:
                    actual_lead = order["true_arrival_day"] - order["placed_day"]
                    supplier.last_observed_lead_time = float(actual_lead)

                feedback.append(
                    f"Arrived: {order['qty']:.0f} units of '{order['sku_id']}' "
                    f"from '{sup_id}'."
                )
            else:
                still_open.append(order)

        self._open_orders = still_open
        self._recompute_storage_usage()

        return feedback

    # -----------------------------------------------------------------------
    # Hidden simulation — disruptions
    # -----------------------------------------------------------------------

    def _update_disruptions(self, day: int) -> None:
        """
        Advance the Markov disruption chain for each supplier.
        Updates supplier.disruption_active based on hidden state transitions.
        """
        ts = self._inventory_state
        params = self._hidden["disruption_params"]
        state  = self._hidden["disruption_state"]

        total_disrupted_days = 0

        for sup_id in state:
            p = params.get(sup_id, {"p_onset": 0.05, "p_persist": 0.50})
            currently_disrupted = state[sup_id]

            if currently_disrupted:
                # Stay disrupted with probability p_persist
                if self._rng.random() < p["p_persist"]:
                    state[sup_id] = True
                    total_disrupted_days += 1
                else:
                    state[sup_id] = False
                    total_disrupted_days += 1
            else:
                # Enter disruption with probability p_onset
                if self._rng.random() < p["p_onset"]:
                    state[sup_id] = True
                    total_disrupted_days += 1
                else:
                    state[sup_id] = False

            # Update observable disruption_active (agent sees this AFTER it fires)
            supplier = ts.get_supplier(sup_id)
            if supplier:
                supplier.disruption_active = state[sup_id]

        # Update rolling disruption history
        ts.supply_disruption_days_last_30d = min(
            30, ts.supply_disruption_days_last_30d + total_disrupted_days
        )

    # -----------------------------------------------------------------------
    # Hidden simulation — cold chain breach
    # -----------------------------------------------------------------------

    def _check_cold_chain_breach(self) -> Optional[str]:
        """
        Possibly trigger a cold chain breach.
        If breach fires, destroy all insulin inventory.
        """
        ts = self._inventory_state
        p_breach = self._hidden["cold_chain_breach_prob"]

        if self._rng.random() < p_breach:
            insulin = ts.get_sku("insulin")
            if insulin and insulin.inventory_on_hand > 0:
                destroyed = insulin.inventory_on_hand
                insulin.inventory_on_hand = 0.0
                ts.cold_chain_integrity_flag = False
                self._recompute_storage_usage()
                return (
                    f"COLD CHAIN BREACH: {destroyed:.0f} units of insulin destroyed. "
                    f"Emergency reorder required immediately."
                )
        else:
            ts.cold_chain_integrity_flag = True

        return None

    # -----------------------------------------------------------------------
    # Observable signal updates
    # -----------------------------------------------------------------------

    def _update_demand_history(self, true_demands: Dict[str, float]) -> None:
        """Append today's true demand to rolling history."""
        for sku_id, demand in true_demands.items():
            if sku_id not in self._demand_history:
                self._demand_history[sku_id] = []
            self._demand_history[sku_id].append(demand)

    # -----------------------------------------------------------------------

    def _update_layer2_signals(self, day: int) -> None:
        """
        Recompute all Layer 2 insight signals from observable history.
        Called after demand history is updated.
        """
        ts = self._inventory_state

        for sku_id, sku in ts.skus.items():
            history = self._demand_history.get(sku_id, [])

            # Rolling demand windows from observable history
            sku.demand_last_3d = float(sum(history[-3:])) if len(history) >= 1 else 0.0
            sku.demand_last_7d = float(sum(history[-7:])) if len(history) >= 1 else 0.0

            # Recompute all derived insight signals
            ts.update_sku_insights(sku_id)

        # Update supplier on_time_rate from delivery history
        for sup_id, supplier in ts.suppliers.items():
            deliveries = self._delivery_history.get(sup_id, [])
            recent = deliveries[-14:]  # last 14 deliveries
            if recent:
                supplier.on_time_rate_14d = sum(recent) / len(recent)

        # Update prescription fill rates
        self._update_fill_rates(day)

        # Update epidemic alert
        self._update_epidemic_alert()

    # -----------------------------------------------------------------------

    def _update_layer3_signals(self) -> None:
        """
        Recompute Layer 3 real-state signals.
        Called after arrivals and consumption are processed.
        """
        ts = self._inventory_state

        # Recompute inbound expected per SKU
        for sku_id, sku in ts.skus.items():
            inbound_3d = sum(
                o["qty"] for o in self._open_orders
                if o["sku_id"] == sku_id
                and o["expected_arrival_day"] <= ts.current_date + 3
            )
            inbound_7d = sum(
                o["qty"] for o in self._open_orders
                if o["sku_id"] == sku_id
                and o["expected_arrival_day"] <= ts.current_date + 7
            )
            sku.inbound_expected_3d = inbound_3d
            sku.inbound_expected_7d = inbound_7d

        # Recompute capacity ratios
        ts.update_capacity_ratios()

    # -----------------------------------------------------------------------

    def _update_overdue_orders(self, day: int) -> None:
        """
        Count orders past their expected arrival date.
        Updates orders_overdue_count and overdue_qty_total.
        """
        ts = self._inventory_state
        overdue = [
            o for o in self._open_orders
            if o["expected_arrival_day"] < day
        ]
        ts.orders_overdue_count = len(overdue)
        ts.overdue_qty_total    = float(sum(o["qty"] for o in overdue))

    # -----------------------------------------------------------------------

    def _update_fill_rates(self, day: int) -> None:
        """
        Recompute prescription fill rates from backorder and demand history.
        Uses a rolling window approach.
        """
        ts = self._inventory_state

        if day == 0:
            return

        # Approximate fill rate: 1 - (backorders / total_demand_in_window)
        total_demand_7d  = 0.0
        total_demand_30d = 0.0
        total_backorders = 0.0

        for sku_id, sku in ts.skus.items():
            history = self._demand_history.get(sku_id, [])
            total_demand_7d  += float(sum(history[-7:]))
            total_demand_30d += float(sum(history[-30:]))
            total_backorders += sku.backorders

        if total_demand_7d > 0:
            ts.prescription_fill_rate_7d = max(
                0.0, 1.0 - (total_backorders / total_demand_7d)
            )
        if total_demand_30d > 0:
            ts.prescription_fill_rate_30d = max(
                0.0, 1.0 - (total_backorders / total_demand_30d)
            )

    # -----------------------------------------------------------------------

    def _update_epidemic_alert(self) -> None:
        """
        Fire epidemic_alert_flag when observed demand trend crosses threshold.
        Based on observable history only — not hidden epidemic state.
        Threshold: any SKU's 3-day average exceeds 14-day average by 40%.
        """
        ts = self._inventory_state
        alert_triggered = False

        for sku_id, sku in ts.skus.items():
            history = self._demand_history.get(sku_id, [])
            if len(history) < 7:
                continue

            avg_3d  = sum(history[-3:]) / 3.0
            avg_14d = sum(history[-14:]) / min(14, len(history))

            if avg_14d > 0 and avg_3d > avg_14d * 1.40:
                alert_triggered = True
                break

        if alert_triggered:
            if not ts.epidemic_alert_flag:
                ts.epidemic_alert_flag             = True
                ts.days_since_epidemic_alert_fired = 0
            else:
                ts.days_since_epidemic_alert_fired += 1
        else:
            ts.epidemic_alert_flag             = False
            ts.days_since_epidemic_alert_fired = 0

    # -----------------------------------------------------------------------
    # Storage helpers
    # -----------------------------------------------------------------------

    def _recompute_storage_usage(self) -> None:
        """Recompute cold and ambient storage current usage from SKU inventories."""
        ts = self._inventory_state

        cold_used = sum(
            sku.inventory_on_hand
            for sku in ts.skus.values()
            if sku.cold_storage_required
        )
        ambient_used = sum(
            sku.inventory_on_hand
            for sku in ts.skus.values()
            if not sku.cold_storage_required
        )

        ts.cold_storage_current_capacity    = cold_used
        ts.ambient_storage_current_capacity = ambient_used
        ts.update_capacity_ratios()

    # -----------------------------------------------------------------------
    # Reward (placeholder — real reward injected from tasks module)
    # -----------------------------------------------------------------------

    def _compute_reward(self, true_demands: Dict[str, float]) -> float:
        """
        Placeholder reward. Returns 0.0.
        Real reward function injected from tasks module.
        """
        return 0.0

    # -----------------------------------------------------------------------
    # LLM interface
    # -----------------------------------------------------------------------

    def to_llm_prompt(self) -> str:
        """
        Build the full prompt sent to the LLM for today's procurement decision.

        Returns system prompt + current daily inventory report as JSON.
        Agent sees only Layer 2 and Layer 3 signals.
        """
        if self._inventory_state is None:
            return _SYSTEM_PROMPT + "\n\nNo state available. Call reset() first."

        ts = self._inventory_state

        # -- Build agent-visible state dict ---------------------------------
        state_dict = {
            "day":          ts.current_date,
            "season_phase": round(ts.season_phase, 3),

            # Global Layer 2
            "service": {
                "prescription_fill_rate_30d":       round(ts.prescription_fill_rate_30d, 3),
                "prescription_fill_rate_7d":        round(ts.prescription_fill_rate_7d, 3),
                "epidemic_alert_flag":              ts.epidemic_alert_flag,
                "days_since_epidemic_alert_fired":  ts.days_since_epidemic_alert_fired,
                "supply_disruption_days_last_30d":  ts.supply_disruption_days_last_30d,
            },

            # Global Layer 3
            "warehouse": {
                "procurement_budget_ratio":         round(ts.procurement_budget_ratio, 3),
                "cold_storage_capacity_ratio":      round(ts.cold_storage_capacity_ratio, 3),
                "cold_storage_total_capacity":      ts.cold_storage_total_capacity,
                "cold_storage_current_capacity":    round(ts.cold_storage_current_capacity, 1),
                "ambient_capacity_ratio":           round(ts.ambient_capacity_ratio, 3),
                "ambient_storage_total_capacity":   ts.ambient_storage_total_capacity,
                "ambient_storage_current_capacity": round(ts.ambient_storage_current_capacity, 1),
                "cold_chain_integrity_flag":        ts.cold_chain_integrity_flag,
                "orders_overdue_count":             ts.orders_overdue_count,
                "overdue_qty_total":                round(ts.overdue_qty_total, 1),
            },

            # Per-SKU (Layer 2 + Layer 3)
            "inventory": {
                sku_id: {
                    "name":                         sku.name,

                    # Layer 3 — real state
                    "inventory_on_hand":            round(sku.inventory_on_hand, 1),
                    "backorders":                   round(sku.backorders, 1),
                    "inbound_expected_3d":          round(sku.inbound_expected_3d, 1),
                    "inbound_expected_7d":          round(sku.inbound_expected_7d, 1),
                    "cold_storage_required":        sku.cold_storage_required,

                    # Layer 2 — insights
                    "demand_last_3d":               round(sku.demand_last_3d, 1),
                    "demand_last_7d":               round(sku.demand_last_7d, 1),
                    "demand_trend":                 round(sku.demand_trend, 3),
                    "stockout_days_if_no_reorder":  round(sku.stockout_days_if_no_reorder, 1),
                    "coverage_gap_7d":              round(sku.coverage_gap_7d, 1),

                    # Fixed domain params
                    "stockout_penalty":             sku.stockout_penalty,
                    "substitute_coverage_ratio":    sku.substitute_coverage_ratio,
                }
                for sku_id, sku in ts.skus.items()
            },

            # Per-supplier (Layer 2 + Layer 3 + fixed params)
            "suppliers": {
                sup_id: {
                    "last_observed_lead_time":  round(sup.last_observed_lead_time, 1),
                    "on_time_rate_14d":         round(sup.on_time_rate_14d, 3),
                    "disruption_active":        sup.disruption_active,
                    "sku_served":               sup.sku_served,
                    "cold_chain_certified":     sup.cold_chain_certified,
                    "unit_cost":                sup.unit_cost,
                    "expedite_allowed":         sup.expedite_allowed,
                }
                for sup_id, sup in ts.suppliers.items()
            },

            "last_action_feedback": ts.last_action_feedback,
        }

        state_json = json.dumps(state_dict, indent=2)

        return (
            _SYSTEM_PROMPT
            + "\n\n--- TODAY'S INVENTORY REPORT ---\n"
            + state_json
            + "\n\n--- YOUR PROCUREMENT DECISION ---\n"
            + "Respond with a JSON object. "
            + "Format: {\"sku_name\": [quantity, \"supplier_name\"], ...}\n"
            + "Order nothing today with: {}\n"
        )

    # -----------------------------------------------------------------------
    # Default task config (baseline — no events)
    # -----------------------------------------------------------------------

    def _default_task_config(self) -> Dict[str, Any]:
        """
        Returns a minimal baseline task config with no demand events.
        Used when reset() is called without a task_config kwarg.
        """
        D = 60
        return {
            "duration_days": D,

            "sku_configs": {
                "insulin": {
                    "name":                    "Insulin",
                    "cold_storage_required":   True,
                    "stockout_penalty":        100.0,
                    "substitute_coverage_ratio": 0.0,
                },
                "paracetamol": {
                    "name":                    "Paracetamol",
                    "cold_storage_required":   False,
                    "stockout_penalty":        20.0,
                    "substitute_coverage_ratio": 0.6,
                },
                "bp_medication": {
                    "name":                    "BP Medication",
                    "cold_storage_required":   False,
                    "stockout_penalty":        60.0,
                    "substitute_coverage_ratio": 0.4,
                },
                "vitamins": {
                    "name":                    "Vitamins",
                    "cold_storage_required":   False,
                    "stockout_penalty":        5.0,
                    "substitute_coverage_ratio": 1.0,
                },
            },

            "supplier_configs": {
                "FastPharma": {
                    "cold_chain_certified":    True,
                    "sku_served":              ["insulin", "paracetamol", "bp_medication", "vitamins"],
                    "unit_cost":               1.4,
                    "expedite_allowed":        True,
                    "expedite_cost_multiplier": 2.0,
                    "base_lead_time_mean":     2.0,
                },
                "GlobalMed": {
                    "cold_chain_certified":    False,
                    "sku_served":              ["paracetamol", "bp_medication", "vitamins"],
                    "unit_cost":               1.0,
                    "expedite_allowed":        False,
                    "expedite_cost_multiplier": 1.0,
                    "base_lead_time_mean":     7.0,
                },
            },

            "supplier_params": {
                "FastPharma": {"lead_time_mean": 2.0, "lead_time_std": 0.5},
                "GlobalMed":  {"lead_time_mean": 7.0, "lead_time_std": 2.5},
            },

            "disruption_params": {
                "FastPharma": {"p_onset": 0.02, "p_persist": 0.30},
                "GlobalMed":  {"p_onset": 0.08, "p_persist": 0.70},
            },

            "base_demands": {
                "insulin":      10.0,
                "paracetamol":  200.0,
                "bp_medication": 50.0,
                "vitamins":      75.0,
            },

            # Flat curves — no events in baseline
            "demand_curves": {
                sku: [1.0] * D
                for sku in ["insulin", "paracetamol", "bp_medication", "vitamins"]
            },

            "supplier_stress": {
                "FastPharma": [1.0] * D,
                "GlobalMed":  [1.0] * D,
            },

            "starting_inventory": {
                "insulin":      70.0,
                "paracetamol":  1400.0,
                "bp_medication": 350.0,
                "vitamins":      525.0,
            },

            "cold_storage_total_capacity":  500.0,
            "ambient_storage_total_capacity": 20000.0,
            "total_budget":                 100000.0,
            "cold_chain_breach_prob":       0.005,
        }