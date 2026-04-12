from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# SupplierState
# ---------------------------------------------------------------------------

class SupplierState(BaseModel):
    """
    Represents one supplier's observable state.

    Layer 1 (hidden — never stored here):
        true lead_time_mean, lead_time_std, inbound_risk_probability
        These live only in the environment's internal simulation state.

    Layer 2 (insights — agent sees):
        last_observed_lead_time, on_time_rate_14d, sku_served

    Layer 3 (current real — agent sees):
        disruption_active
    """

    # -- Identity (fixed domain parameter) ----------------------------------
    supplier_id: str = Field(
        ...,
        description="Unique supplier identifier. Example: 'FastPharma'.",
    )

    cold_chain_certified: bool = Field(
        ...,
        description=(
            "True if this supplier can handle cold-chain SKUs (e.g. insulin). "
            "FastPharma=True, GlobalMed=False. Never changes during episode."
        ),
    )

    # -- Fixed domain parameters (visible, never change) --------------------
    sku_served: List[str] = Field(
        default_factory=list,
        description=(
            "List of SKU IDs this supplier can fulfill. "
            "FastPharma serves all SKUs. GlobalMed serves non-cold-chain only."
        ),
    )

    unit_cost: float = Field(
        ...,
        ge=0.0,
        description=(
            "Normalised cost per unit from this supplier. "
            "FastPharma=1.4 (premium), GlobalMed=1.0 (baseline)."
        ),
    )

    expedite_allowed: bool = Field(
        default=False,
        description=(
            "True if emergency expedited orders are possible. "
            "FastPharma=True, GlobalMed=False."
        ),
    )

    expedite_cost_multiplier: float = Field(
        default=1.0,
        ge=1.0,
        description=(
            "Cost multiplier when expediting an order. "
            "Only relevant if expedite_allowed=True."
        ),
    )

    # -- Layer 2: Historical insights (agent observes) ----------------------
    last_observed_lead_time: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Lead time (days) of the most recently completed delivery. "
            "Agent's best estimate of current lead time. "
            "Updates each time an order from this supplier arrives. "
            "Starts at 0 — no deliveries yet at episode start."
        ),
    )

    on_time_rate_14d: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of orders delivered on time over the last 14 days. "
            "Agent infers reliability from observed history, not ground truth. "
            "1.0 = perfectly reliable, 0.0 = all recent orders were late."
        ),
    )

    # -- Layer 3: Current real state (agent observes) -----------------------
    disruption_active: bool = Field(
        default=False,
        description=(
            "True if this supplier is currently disrupted. "
            "Agent discovers this AFTER disruption starts — reactive, not predictive. "
            "Set by the environment when the hidden Markov disruption chain fires."
        ),
    )


# ---------------------------------------------------------------------------
# SKUState
# ---------------------------------------------------------------------------

class SKUState(BaseModel):
    """
    Represents one SKU's full state across all three layers.

    Layer 1 (hidden — never stored here):
        demand_today, true_demand_mean_t, true_demand_std_t
        inbound qty and true arrival date (subject to lead time randomness)
        These live only in the environment's internal simulation state.

    Layer 2 (insights — agent observes):
        demand_last_3d, demand_last_7d, demand_trend,
        stockout_days_if_no_reorder, coverage_gap_7d

    Layer 3 (current real — agent observes):
        inventory_on_hand, inbound_expected_3d, inbound_expected_7d,
        stockout_penalty, cold_storage_required
    """

    # -- Identity (fixed) ---------------------------------------------------
    sku_id: str = Field(
        ...,
        description="Unique SKU identifier. Example: 'insulin', 'paracetamol'.",
    )

    name: str = Field(
        ...,
        description="Human-readable drug name for the LLM prompt.",
    )

    # -- Fixed domain parameters (visible, never change) --------------------
    cold_storage_required: bool = Field(
        ...,
        description=(
            "True if this SKU requires refrigerated cold storage. "
            "Insulin=True. All others=False. "
            "Determines which capacity pool is used."
        ),
    )

    stockout_penalty: float = Field(
        ...,
        ge=0.0,
        description=(
            "Cost incurred per unit of unmet demand. "
            "Encodes medical priority: insulin >> bp_med >> paracetamol >> vitamins. "
            "Used by the reward function. Visible to agent for triage decisions."
        ),
    )

    substitute_coverage_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of demand coverable by a substitute SKU if this one stockouts. "
            "insulin=0.0 (no substitute), vitamins=1.0 (fully substitutable). "
            "Informs the agent how critical a stockout is."
        ),
    )

    # -- Layer 2: Historical insights (agent observes) ----------------------
    demand_last_3d: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Total actual demand observed over the last 3 days. "
            "Agent sees consequences of hidden true demand. "
            "A spike shows up here 1-2 days after it starts."
        ),
    )

    demand_last_7d: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Total actual demand observed over the last 7 days. "
            "Provides a stable baseline window for trend detection."
        ),
    )

    demand_trend: float = Field(
        default=0.0,
        description=(
            "Demand acceleration signal. "
            "Computed as: (demand_last_3d / 3) - (demand_last_7d / 7). "
            "Positive = demand accelerating. Negative = demand decelerating. "
            "Derived from observable history only — no hidden process leakage."
        ),
    )

    stockout_days_if_no_reorder: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Days until stockout if the agent places no new orders today. "
            "Computed as: inventory_on_hand / (demand_last_3d / 3). "
            "Does NOT include inbound_expected — honest, conservative estimate. "
            "Most important urgency signal per SKU."
        ),
    )

    coverage_gap_7d: float = Field(
        default=0.0,
        description=(
            "Demand shortfall over the next 7 days. "
            "Computed as: demand_last_7d - inventory_on_hand. "
            "Negative = surplus (safe). Positive = shortage incoming. "
            "Uses observable quantities only — no oracle leakage."
        ),
    )

    # -- Layer 3: Current real state (agent observes) -----------------------
    inventory_on_hand: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Units physically on shelf right now. "
            "Reduced by true demand each day (agent sees result, not cause). "
            "Replenished when inbound orders arrive."
        ),
    )

    backorders: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Unfilled prescriptions for this SKU. "
            "Accumulates when inventory_on_hand reaches zero. "
            "Consequence of past stockouts — directly penalised in reward."
        ),
    )

    inbound_expected_3d: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Units the agent has ordered expected to arrive within 3 days. "
            "Based on last_observed_lead_time — estimate, not guarantee. "
            "Becomes wrong if supplier disrupts or true lead time extends. "
            "Agent discovers inaccuracy via orders_overdue signals."
        ),
    )

    inbound_expected_7d: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Units the agent has ordered expected to arrive within 7 days. "
            "Same caveats as inbound_expected_3d — estimate based on history."
        ),
    )


# ---------------------------------------------------------------------------
# InventoryState  (global warehouse state — the TaskState equivalent)
# ---------------------------------------------------------------------------

class InventoryState(Observation):
    """
    Full observable state of the pharmaceutical cold-chain warehouse.

    Sent to the agent as the daily inventory report.

    Structured into three layers per the environment design:
        Layer 1 (hidden): lives only in PharmaEnvironment._hidden
        Layer 2 (insights): demand history, service history, alerts
        Layer 3 (real state): current inventory, budget, capacity

    The agent sees ONLY layers 2 and 3 in the LLM prompt.
    """

    # -----------------------------------------------------------------------
    # GLOBAL — Layer 1 identifiers (observable, not random)
    # -----------------------------------------------------------------------

    current_date: int = Field(
        default=0,
        ge=0,
        description=(
            "Current day within the episode. 0-indexed. "
            "Day 0 = first day, day 59 = last day of 60-day episode."
        ),
    )

    season_phase: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Position within the seasonal cycle. 0.0 = start, 1.0 = end. "
            "Drives demand patterns: winter phase elevates paracetamol and vitamins. "
            "Observable — agent uses it to anticipate seasonal demand shifts."
        ),
    )

    # -----------------------------------------------------------------------
    # GLOBAL — Layer 2: Historical insights (agent observes)
    # -----------------------------------------------------------------------

    prescription_fill_rate_30d: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of prescriptions filled on time over the last 30 days. "
            "Core service-level KPI. Declining = agent is failing. "
            "Already low = damage done, recovery needed."
        ),
    )

    prescription_fill_rate_7d: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of prescriptions filled on time over the last 7 days. "
            "Shorter window catches deterioration faster than 30d metric. "
            "7d < 30d → things are getting worse right now."
        ),
    )

    epidemic_alert_flag: bool = Field(
        default=False,
        description=(
            "True when a public health authority alert is active. "
            "Fires when observed demand trend crosses a threshold — "
            "based on observable history, not hidden epidemic state. "
            "Gives agent a 3-5 day warning before demand peak. "
            "Clears after wave subsides. Re-fires for second wave."
        ),
    )

    days_since_epidemic_alert_fired: int = Field(
        default=0,
        ge=0,
        description=(
            "Days the epidemic alert has been continuously active. "
            "0 = alert just fired today (act immediately). "
            "5+ = alert has been active a while (wave probably near peak). "
            "Resets to 0 when alert clears between waves."
        ),
    )

    supply_disruption_days_last_30d: int = Field(
        default=0,
        ge=0,
        le=30,
        description=(
            "Total days any supplier was disrupted over the last 30 days. "
            "Agent infers whether disruption environment is benign or hostile. "
            "0 = clean month. 10+ = frequent disruptions, build safety stock."
        ),
    )

    # -----------------------------------------------------------------------
    # GLOBAL — Layer 3: Current real state (agent observes)
    # -----------------------------------------------------------------------

    procurement_budget_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Remaining procurement budget / total quarterly budget. "
            "Hard constraint on every reorder decision. "
            "Reaches 0 → agent cannot place any orders."
        ),
    )

    cold_storage_capacity_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Cold storage currently used / total cold storage capacity. "
            "Applies only to insulin. Separate scarce resource pool. "
            "Near 1.0 → cannot receive new insulin orders."
        ),
    )

    cold_storage_total_capacity: float = Field(
        ...,
        gt=0.0,
        description="Total cold storage capacity in units. Fixed for the episode.",
    )

    cold_storage_current_capacity: float = Field(
        default=0.0,
        ge=0.0,
        description="Cold storage units currently occupied by insulin inventory.",
    )

    ambient_capacity_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Ambient storage currently used / total ambient storage capacity. "
            "Applies to all non-cold-chain SKUs. "
            "Near 1.0 → must deplete stock before placing new orders."
        ),
    )

    ambient_storage_total_capacity: float = Field(
        ...,
        gt=0.0,
        description="Total ambient storage capacity in units. Fixed for the episode.",
    )

    ambient_storage_current_capacity: float = Field(
        default=0.0,
        ge=0.0,
        description="Ambient storage units currently occupied.",
    )

    cold_chain_integrity_flag: bool = Field(
        default=True,
        description=(
            "True = cold storage functioning normally. "
            "False = cold chain breach occurred — all insulin inventory condemned. "
            "Agent discovers this AFTER breach fires. "
            "Triggers emergency reorder requirement."
        ),
    )

    orders_overdue_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of purchase orders past their expected arrival date. "
            "Rises when true lead time exceeds historical estimate. "
            "Signals that inbound_expected figures are unreliable."
        ),
    )

    overdue_qty_total: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Total units stuck in overdue orders. "
            "Magnitude companion to orders_overdue_count. "
            "Large overdue_qty = significant inbound stock at risk."
        ),
    )

    # -----------------------------------------------------------------------
    # Per-SKU states
    # -----------------------------------------------------------------------

    skus: Dict[str, SKUState] = Field(
        default_factory=dict,
        description=(
            "Per-SKU state indexed by sku_id. "
            "Each SKUState contains inventory position, demand history, "
            "urgency signals, and fixed domain parameters."
        ),
    )

    # -----------------------------------------------------------------------
    # Per-supplier states
    # -----------------------------------------------------------------------

    suppliers: Dict[str, SupplierState] = Field(
        default_factory=dict,
        description=(
            "Per-supplier state indexed by supplier_id. "
            "Each SupplierState contains delivery history and disruption status."
        ),
    )

    # -----------------------------------------------------------------------
    # Episode metadata
    # -----------------------------------------------------------------------

    done: bool = Field(
        default=False,
        description="True when the episode has ended.",
    )

    reward: float = Field(
        default=0.0,
        description="Reward returned by the last step.",
    )

    last_action_feedback: str = Field(
        default="Episode started. Review inventory and make your first procurement decision.",
        description=(
            "Plain-English result of the last action. "
            "Tells you what was ordered, what arrived, what was fulfilled, "
            "and any warnings about overdue orders or disruptions."
        ),
    )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def get_sku(self, sku_id: str) -> Optional[SKUState]:
        return self.skus.get(sku_id)

    def get_supplier(self, supplier_id: str) -> Optional[SupplierState]:
        return self.suppliers.get(supplier_id)

    def update_capacity_ratios(self) -> None:
        """Recompute capacity ratios from current capacity values."""
        if self.cold_storage_total_capacity > 0:
            self.cold_storage_capacity_ratio = (
                self.cold_storage_current_capacity / self.cold_storage_total_capacity
            )
        if self.ambient_storage_total_capacity > 0:
            self.ambient_capacity_ratio = (
                self.ambient_storage_current_capacity / self.ambient_storage_total_capacity
            )

    def update_sku_insights(self, sku_id: str) -> None:
        """
        Recompute Layer 2 derived signals for one SKU.
        Called after inventory or demand history updates.
        """
        sku = self.skus.get(sku_id)
        if sku is None:
            return

        avg_3d = sku.demand_last_3d / 3.0 if sku.demand_last_3d > 0 else 0.0
        avg_7d = sku.demand_last_7d / 7.0 if sku.demand_last_7d > 0 else 0.0

        # Demand trend: acceleration signal
        sku.demand_trend = avg_3d - avg_7d

        # Stockout days: how long current stock lasts at recent demand rate
        if avg_3d > 0:
            sku.stockout_days_if_no_reorder = sku.inventory_on_hand / avg_3d
        else:
            sku.stockout_days_if_no_reorder = 999.0  # no demand → no stockout risk

        # Coverage gap: observable shortfall over 7 days
        sku.coverage_gap_7d = sku.demand_last_7d - sku.inventory_on_hand


# ---------------------------------------------------------------------------
# PharmaAction
# ---------------------------------------------------------------------------

class PharmaAction(Action):
    """
    Agent's daily procurement decision.

    Format: {sku_id: (order_quantity, supplier_id)}

    Example:
        {
            "insulin":      (100, "FastPharma"),
            "paracetamol":  (500, "GlobalMed"),
        }

    Rules:
    - Only SKUs in the order dict are ordered. Omitted SKUs = no order today.
    - order_quantity must be > 0.
    - supplier_id must be in the supplier's sku_served list.
    - insulin can only be ordered from cold_chain_certified suppliers.
    - Order is rejected if budget or capacity constraints are violated.
    """

    orders: Dict[str, Tuple[float, str]] = Field(
        default_factory=dict,
        description=(
            "Procurement orders as {sku_id: (order_quantity, supplier_id)}. "
            "Only include SKUs you want to order today. "
            "Example: {'insulin': (100, 'FastPharma'), 'paracetamol': (500, 'GlobalMed')}."
        ),
    )

    message: str = Field(
        default="",
        description=(
            "Raw LLM message string. Parsed by the environment "
            "to extract the orders dict."
        ),
    )