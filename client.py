"""Pharmaceutical Warehouse Procurement Environment Client."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from models import InventoryState, PharmaAction


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
        obs_data = dict(payload.get("observation", payload))
        # reward/done live at the top level of the payload, not inside the observation.
        obs_data["done"] = payload.get("done", obs_data.get("done", False))
        reward = payload.get("reward", obs_data.get("reward"))
        obs_data["reward"] = reward if reward is not None else 0.0
        observation = InventoryState.model_validate(obs_data)

        return StepResult(
            observation=observation,
            reward=reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse the lightweight server-side metadata state."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
