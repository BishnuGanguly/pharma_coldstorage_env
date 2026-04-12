"""Pharmaceutical Warehouse Procurement Environment Client."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from models import InventoryState, PharmaAction, SKUState, SupplierState


class PharmaEnvClient(EnvClient[PharmaAction, InventoryState, State]):
    """
    Client for the PharmaEnvironment.

    Sends a text action message (or structured orders dict) to the server
    and receives a full InventoryState as the observation.

    Example:
        >>> with PharmaEnvClient(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.prescription_fill_rate_30d)
        ...
        ...     action = PharmaAction(orders={"insulin": (100.0, "FastPharma")})
        ...     result = client.step(action)
        ...     print(result.observation.last_action_feedback)
    """

    def _step_payload(self, action: PharmaAction) -> Dict[str, Any]:
        """Convert PharmaAction into a JSON-serializable payload."""
        return {
            "message": action.message,
            "orders": {
                sku_id: list(order)
                for sku_id, order in action.orders.items()
            },
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[InventoryState]:
        """Parse the server response into StepResult[InventoryState]."""
        obs_data = payload.get("observation", payload)
        observation = self._parse_inventory_state(obs_data, payload)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse the lightweight server-side metadata state."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _parse_inventory_state(
        self,
        obs_data: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> InventoryState:
        """Reconstruct a full InventoryState from the server payload."""

        skus: Dict[str, SKUState] = {}
        for sku_id, sku_data in obs_data.get("skus", {}).items():
            skus[sku_id] = SKUState(
                sku_id=sku_id,
                name=sku_data.get("name", sku_id),
                cold_storage_required=sku_data.get("cold_storage_required", False),
                stockout_penalty=sku_data.get("stockout_penalty", 0.0),
                substitute_coverage_ratio=sku_data.get("substitute_coverage_ratio", 0.0),
                inventory_on_hand=sku_data.get("inventory_on_hand", 0.0),
                backorders=sku_data.get("backorders", 0.0),
                inbound_expected_3d=sku_data.get("inbound_expected_3d", 0.0),
                inbound_expected_7d=sku_data.get("inbound_expected_7d", 0.0),
                demand_last_3d=sku_data.get("demand_last_3d", 0.0),
                demand_last_7d=sku_data.get("demand_last_7d", 0.0),
                demand_trend=sku_data.get("demand_trend", 0.0),
                stockout_days_if_no_reorder=sku_data.get("stockout_days_if_no_reorder", 999.0),
                coverage_gap_7d=sku_data.get("coverage_gap_7d", 0.0),
            )

        suppliers: Dict[str, SupplierState] = {}
        for sup_id, sup_data in obs_data.get("suppliers", {}).items():
            suppliers[sup_id] = SupplierState(
                supplier_id=sup_id,
                cold_chain_certified=sup_data.get("cold_chain_certified", False),
                sku_served=list(sup_data.get("sku_served", [])),
                unit_cost=sup_data.get("unit_cost", 1.0),
                expedite_allowed=sup_data.get("expedite_allowed", False),
                expedite_cost_multiplier=sup_data.get("expedite_cost_multiplier", 1.0),
                last_observed_lead_time=sup_data.get("last_observed_lead_time", 0.0),
                on_time_rate_14d=sup_data.get("on_time_rate_14d", 1.0),
                disruption_active=sup_data.get("disruption_active", False),
            )

        return InventoryState(
            current_date=obs_data.get("current_date", 0),
            season_phase=obs_data.get("season_phase", 0.0),
            prescription_fill_rate_30d=obs_data.get("prescription_fill_rate_30d", 1.0),
            prescription_fill_rate_7d=obs_data.get("prescription_fill_rate_7d", 1.0),
            epidemic_alert_flag=obs_data.get("epidemic_alert_flag", False),
            days_since_epidemic_alert_fired=obs_data.get("days_since_epidemic_alert_fired", 0),
            supply_disruption_days_last_30d=obs_data.get("supply_disruption_days_last_30d", 0),
            procurement_budget_ratio=obs_data.get("procurement_budget_ratio", 1.0),
            cold_storage_total_capacity=obs_data.get("cold_storage_total_capacity", 500.0),
            cold_storage_current_capacity=obs_data.get("cold_storage_current_capacity", 0.0),
            cold_storage_capacity_ratio=obs_data.get("cold_storage_capacity_ratio", 0.0),
            ambient_storage_total_capacity=obs_data.get("ambient_storage_total_capacity", 20000.0),
            ambient_storage_current_capacity=obs_data.get("ambient_storage_current_capacity", 0.0),
            ambient_capacity_ratio=obs_data.get("ambient_capacity_ratio", 0.0),
            cold_chain_integrity_flag=obs_data.get("cold_chain_integrity_flag", True),
            orders_overdue_count=obs_data.get("orders_overdue_count", 0),
            overdue_qty_total=obs_data.get("overdue_qty_total", 0.0),
            skus=skus,
            suppliers=suppliers,
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            last_action_feedback=obs_data.get(
                "last_action_feedback",
                "Episode started. Review inventory and make your first procurement decision.",
            ),
        )