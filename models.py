from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
from openenv.core.env_server.types import Action, Observation


# ---------------------------------------------------------------------------
# SKUState
# ---------------------------------------------------------------------------

class SKUState(BaseModel):

    sku_id: str = Field(
        default="sku_123",
        description="Unique identifier for the SKU. Used as the key in all order and inventory dicts."
    )

    # -- Insights (derived from observable history — agent sees these) -------

    avg_demand_per_day: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Rolling average daily demand computed over the full history available so far. "
            "Primary signal for long-run reorder sizing."
        ),
    )

    avg_demand_last_5_days: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Average daily demand over the last 5 days. "
            "Reacts faster than avg_demand_per_day — use to detect short-term demand shifts."
        ),
    )

    avg_lead_time: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Rolling average lead time in days computed from all observed deliveries so far. "
            "Agent uses this to estimate expected_inbound_orders arrival dates."
        ),
    )

    lead_time_last3_orders: List[float] = Field(
        default_factory=list,
        description=(
            "Actual lead times of the 3 most recent deliveries, oldest first. "
            "Short window makes supply chain stress visible quickly. "
            "Capped at 3 entries — older values are dropped."
        ),
    )

    # demand_trend: float = Field(
    #     default=0.0,
    #     description=(
    #         "Demand acceleration signal: (avg_demand_last_3d / 3) - (avg_demand_last_7d / 7). "
    #         "Positive = demand accelerating. Negative = demand decelerating. "
    #         "Gives early warning of spikes before they fully show up in avg_demand_per_day."
    #     ),
    # )

    stockout_days_if_no_reorder: float = Field(
        default=999.0,
        ge=0.0,
        description=(
            "Days until stockout if no new orders are placed: inventory_on_hand / avg_demand_per_day. "
            "999.0 when demand is zero. Most urgent reorder signal per SKU."
        ),
    )

    # -- Current real state (depends on true hidden demand) ------------------

    cold_storage_required: bool = Field(
        default=False,
        description=(
            "True if this SKU requires refrigerated cold storage. "
            "Determines which capacity pool (cold vs ambient) is consumed."
        ),
    )

    inventory_on_hand: float = Field(
        default=0.0,
        ge=0.0,
        description="Units physically on shelf right now after today's demand has been consumed.",
    )

    stockout_penalty: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Per-unit penalty applied to unmet demand for this SKU. "
            "Encodes medical criticality — insulin >> antiflu >> vitamins."
        ),
    )

    waste_penalty: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight of this SKU's overflow waste: wasting a high-penalty SKU's delivery costs more. "
            "Reflects the cost of spoilage and wasted procurement spend."
        ),
    )

    # -- Reward tracking (environment writes, agent reads) -------------------

    demand_fulfilled_today: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of today's demand that was fulfilled: units_served / true_demand. "
            "1.0 = fully served. 0.5 = half the demand was met. 0.0 = complete stockout."
        ),
    )

    demand_fulfilled_cumulative: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Cumulative sum of demand_fulfilled_today across all days so far. "
            "Divide by current_date to get the episode fill rate for this SKU."
        ),
    )

    @field_validator("lead_time_last3_orders")
    @classmethod
    def cap_at_three(cls, v: List[float]) -> List[float]:
        return v[-3:] if len(v) > 3 else v


# ---------------------------------------------------------------------------
# InventoryState
# ---------------------------------------------------------------------------

class InventoryState(Observation):

    current_date: int = Field(
        default=0,
        ge=0,
        description="Current day index within the episode. 0-indexed.",
    )

    # -- Storage pools -------------------------------------------------------

    cold_storage_total_capacity: float = Field(
        gt=0.0,
        description="Maximum units of cold storage available. Fixed for the episode.",
    )

    cold_storage_current_capacity: float = Field(
        default=0.0,
        ge=0.0,
        description="Cold storage units currently occupied across all cold-chain SKUs.",
    )

    ambient_storage_total_capacity: float = Field(
        gt=0.0,
        description="Maximum units of ambient storage available. Fixed for the episode.",
    )

    ambient_storage_current_capacity: float = Field(
        default=0.0,
        ge=0.0,
        description="Ambient storage units currently occupied across all non-cold-chain SKUs.",
    )

    # -- Inbound orders ------------------------------------------------------

    actual_inbound_orders: List[Tuple[str, float, int]] = Field(
        default_factory=list,
        description=(
            "Ground-truth open orders tracked by the environment: (sku_id, quantity, true_arrival_day). "
            "true_arrival_day is sampled from the hidden lead time distribution at order placement. "
            "Never exposed directly to the agent — agent only sees expected_inbound_orders."
        ),
    )

    expected_inbound_orders: List[Tuple[str, float, int]] = Field(
        default_factory=list,
        description=(
            "Agent-visible inbound orders: (sku_id, quantity, expected_arrival_day). "
            "expected_arrival_day = order_date + avg_lead_time at time of ordering. "
            "Will diverge from actual_inbound_orders when true lead times deviate from the average."
        ),
    )

    # -- Per-SKU states ------------------------------------------------------

    skus: Dict[str, SKUState] = Field(
        default_factory=dict,
        description="Per-SKU runtime state keyed by sku_id.",
    )
    demand_history:Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Historical demand data for each SKU keyed by sku_id."
    )
    lead_time_history:Dict[str,List[float]] = Field(
        default_factory=dict,
        description="Historical lead time data for each SKU keyed by sku_id."
    )

    # -- Reward tracking (global) --------------------------------------------

    inventory_excess_today: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Units of today's deliveries rejected because their storage pool was full. "
            "0 means nothing was wasted."
        ),
    )

    waste_fraction_today: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Share of today's deliveries rejected for lack of storage, averaged over the SKUs "
            "that received a delivery and weighted by their waste_penalty. "
            "0 = everything fit (or nothing arrived), 1 = every delivery was rejected. "
            "Subtracted from the step reward."
        ),
    )

    inventory_excess_cumulative: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Running sum of (1 - waste_fraction_today): +1 for each day with no waste. "
            "Days played minus this value is the total waste fraction subtracted from the score."
        ),
    )

    # -- News ----------------------------------------------------------------

    news: List[str] = Field(
        default_factory=list,
        description=(
            "Announcements the agent can see today: every news item already published whose "
            "event has not ended yet, each starting with the day it was published. "
            "Empty when the episode has no news (news level 0)."
        ),
    )

    # -- Episode metadata ----------------------------------------------------

    done: bool = Field(
        default=False,
        description="True when current_date has reached no_of_days.",
    )

    reward: float = Field(
        default=0.0,
        description="Reward returned after the last step.",
    )

    # -- Helpers -------------------------------------------------------------

    def get_sku(self, sku_id: str) -> Optional[SKUState]:
        return self.skus.get(sku_id)

    @property
    def cold_storage_ratio(self) -> float:
        return self.cold_storage_current_capacity / self.cold_storage_total_capacity

    @property
    def ambient_storage_ratio(self) -> float:
        return self.ambient_storage_current_capacity / self.ambient_storage_total_capacity


# ---------------------------------------------------------------------------
# SKUEpisodeConfig
# ---------------------------------------------------------------------------

class SKUEpisodeConfig(BaseModel):

    sku_id: str = Field(
        default="sku_123",
        description="Must match the sku_id used in SKUState and all order tuples."
    )

    no_of_days: int = Field(
        default=60,
        gt=0,
        description="Episode length in days. All curves must have exactly this many entries.",
    )

    base_demand: float = Field(
        gt=0,
        default=10.0,
        description=(
            "Daily demand baseline in units, randomly sampled from a configured range at episode start. "
            "True daily demand = base_demand * demand_curve[day] + gauss(0, demand_std)."
        ),
    )

    demand_curve: List[float] = Field(
        default_factory=list,
        description=(
            "Per-day demand multiplier of length no_of_days. "
            "1.0 = baseline. >1.0 = spike. <1.0 = trough. "
            "Used to model seasonal patterns, epidemic waves, and flu seasons."
        ),
    )

    demand_std: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Standard deviation of Gaussian noise added to daily demand after curve scaling. "
            "Controls how noisy demand is around the seasonal mean."
        ),
    )

    base_lead_time: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Baseline supplier lead time in days, randomly sampled from a configured range at episode start. "
            "True lead time = base_lead_time * lead_time_curve[day] + gauss(0, lead_time_std)."
        ),
    )

    lead_time_curve: List[float] = Field(
        default_factory=list,
        description=(
            "Per-day lead time multiplier of length no_of_days. "
            "1.0 = baseline. >1.0 = supplier under stress. "
            "Used to model supply chain disruptions and peak-season delays."
        ),
    )

    lead_time_std: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Standard deviation of Gaussian noise added to lead time after curve scaling. "
            "Controls delivery time variability around the seasonal mean."
        ),
    )

    cold_storage_required: bool = Field(
        default=False,
        description=(
            "True if this SKU requires refrigerated cold storage. "
            "Copied into SKUState at episode initialisation."
        ),
    )
 
    stockout_penalty: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Per-unit penalty for unmet demand. Copied into SKUState at episode initialisation. "
            "Encodes medical criticality — insulin >> antiflu >> vitamins."
        ),
    )
 
    waste_penalty: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight of this SKU's overflow waste in waste_fraction_today. "
            "Copied into SKUState at episode initialisation."
        ),
    )

    inbound_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Probability that any given inbound order is lost in transit and never arrives. "
            "0.0 = all orders arrive. 0.1 = 10% chance of total loss per order. "
            "Agent only discovers this after the expected arrival date passes with no delivery."
        ),
    )

    @model_validator(mode="after")
    def validate_curve_lengths(self) -> SKUEpisodeConfig:
        if self.demand_curve and len(self.demand_curve) != self.no_of_days:
            raise ValueError(
                f"demand_curve length {len(self.demand_curve)} != no_of_days {self.no_of_days}"
            )
        if self.lead_time_curve and len(self.lead_time_curve) != self.no_of_days:
            raise ValueError(
                f"lead_time_curve length {len(self.lead_time_curve)} != no_of_days {self.no_of_days}"
            )
        return self


# ---------------------------------------------------------------------------
# NewsEvent
# ---------------------------------------------------------------------------

class NewsEvent(BaseModel):
    """
    A disruption of the episode, announced to the agent before it starts.

    The structured fields describe the event exactly (they come from the task's own
    demand and lead-time curves) and stay hidden; the agent only sees `message`,
    from `announce_day` until `end_day`.
    """

    kind: str = Field(
        description="'supplier_delay' (lead times multiplied) or 'demand_surge' (demand multiplied).",
    )
    sku_id: str = Field(description="The SKU the event affects.")
    start_day: int = Field(ge=0, description="First day of the event.")
    end_day: int = Field(ge=0, description="Last day of the event.")
    peak_day: int = Field(ge=0, description="Day of the strongest effect (start_day for a supplier delay).")
    multiplier: float = Field(
        gt=0.0,
        description="Peak multiplier of the SKU's lead time (supplier_delay) or demand (demand_surge).",
    )
    announce_day: int = Field(ge=0, description="Day the news is published.")
    message: str = Field(description="The news text the agent reads.")

    @model_validator(mode="after")
    def validate_days(self) -> NewsEvent:
        if not self.start_day <= self.peak_day <= self.end_day:
            raise ValueError("need start_day <= peak_day <= end_day")
        if self.announce_day > self.end_day:
            raise ValueError("announce_day must not be after end_day")
        return self


# ---------------------------------------------------------------------------
# EpisodeConfig  (top-level task definition)
# ---------------------------------------------------------------------------

class EpisodeConfig(BaseModel):

    task_name: str = Field(
        description="Human-readable task identifier. e.g. 'task1_supply_chain_broken'."
    )

    no_of_days: int = Field(
        gt=0,
        description="Episode length shared across all SKUs.",
    )

    cold_storage_total_capacity: float = Field(
        gt=0.0,
        description="Total cold storage capacity for the episode.",
    )

    ambient_storage_total_capacity: float = Field(
        gt=0.0,
        description="Total ambient storage capacity for the episode.",
    )

    skus: Dict[str, SKUEpisodeConfig] = Field(
        description=(
            "Per-SKU episode configs keyed by sku_id. "
            "Each entry fully specifies the hidden demand and lead time process for that SKU."
        ),
    )
    initial_inventory: Dict[str, float] = Field(
        default_factory=dict,
        description="Initial inventory levels for each SKU keyed by sku_id."
    )
    news: List[NewsEvent] = Field(
        default_factory=list,
        description=(
            "Disruptions announced to the agent ahead of time (see NewsEvent). "
            "Empty for news level 0, the original environment."
        ),
    )

    @model_validator(mode="after")
    def validate_sku_days_match(self) -> EpisodeConfig:
        for sku_id, cfg in self.skus.items():
            if cfg.no_of_days != self.no_of_days:
                raise ValueError(
                    f"SKU '{sku_id}' no_of_days={cfg.no_of_days} != episode no_of_days={self.no_of_days}"
                )
        return self


# ---------------------------------------------------------------------------
# PharmaAction
# ---------------------------------------------------------------------------

class PharmaAction(Action):

    orders: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Procurement orders for today: {sku_id: order_quantity}. "
            "Only include SKUs you want to order. Omitted SKUs = no order placed today. "
            "Quantities must be positive. Orders violating capacity at arrival will be rejected."
        ),
    )

    message: str = Field(
        default="",
        description=(
            "Raw LLM output string. Parsed by the environment to extract the orders dict "
            "when orders is not set directly."
        ),
    )
    @field_validator("orders", mode="before")
    @classmethod
    def parse_orders_json(cls, v: Any) -> Any:
        """Accept orders as a JSON string too, as sent by the /web Playground's text box."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return {}
            try:
                return json.loads(v)
            except json.JSONDecodeError as exc:
                raise ValueError(f'orders must be a JSON object like {{"insulin": 40}}: {exc}') from None
        return v
