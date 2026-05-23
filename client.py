"""Pharmaceutical Warehouse Procurement Environment Client."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from models import InventoryState, PharmaAction, SKUState


class PharmaEnvClient(EnvClient[PharmaAction, InventoryState, State]):
    """
    Client for the PharmaEnvironment.

    Sends a text action message (or structured orders dict) to the server
    and receives a full InventoryState as the observation.

    Example:
        >>> with PharmaEnvClient(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.current_date)
        ...
        ...     action = PharmaAction(orders={"insulin": 100.0})
        ...     result = client.step(action)
        ...     print(result.observation.skus["insulin"].inventory_on_hand)
    """

    def _step_payload(self, action: PharmaAction) -> Dict[str, Any]:
        """Convert PharmaAction into a JSON-serializable payload."""
        return {
            "message": action.message,
            "orders": {
                sku_id: order
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
                avg_demand_per_day=sku_data.get("avg_demand_per_day", 0.0),
                avg_demand_last_5_days=sku_data.get("avg_demand_last_5_days", 0.0),
                avg_lead_time=sku_data.get("avg_lead_time", 0.0),
                lead_time_last3_orders=sku_data.get("lead_time_last3_orders", []),
                stockout_days_if_no_reorder=sku_data.get("stockout_days_if_no_reorder", 999.0),
                cold_storage_required=sku_data.get("cold_storage_required", False),
                inventory_on_hand=sku_data.get("inventory_on_hand", 0.0),
                stockout_penalty=sku_data.get("stockout_penalty", 1.0),
                waste_penalty=sku_data.get("waste_penalty", 0.5),
                demand_fulfilled_today=sku_data.get("demand_fulfilled_today", 0.0),
                demand_fulfilled_cumulative=sku_data.get("demand_fulfilled_cumulative", 0.0),  
            )



        return InventoryState(
            current_date=obs_data.get("current_date", 0),
            cold_storage_total_capacity=obs_data.get("cold_storage_total_capacity", 500.0),
            cold_storage_current_capacity=obs_data.get("cold_storage_current_capacity", 0.0),
            ambient_storage_total_capacity=obs_data.get("ambient_storage_total_capacity", 20000.0),
            ambient_storage_current_capacity=obs_data.get("ambient_storage_current_capacity", 0.0),
            actual_inbound_orders=obs_data.get("actual_inbound_orders", []),
            expected_inbound_orders=obs_data.get("expected_inbound_orders", []),
            skus=skus,
            demand_history=obs_data.get("demand_history", {}),
            lead_time_history=obs_data.get("lead_time_history", {}),
            inventory_excess_today=obs_data.get("inventory_excess_today", 0.0),
            inventory_excess_cumulative=obs_data.get("inventory_excess_cumulative", 0.0),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
        )