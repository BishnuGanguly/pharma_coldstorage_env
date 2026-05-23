"""Smoke test for the Pharmaceutical Warehouse Procurement Environment."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from server.my_env_environment import PharmaEnvironment
from models import InventoryState, PharmaAction


def pretty(obj: Any) -> str:
    """Pretty-print Pydantic models or plain dicts."""
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(mode="json"), indent=2)
    if hasattr(obj, "dict"):
        return json.dumps(obj.dict(), indent=2, default=str)
    return json.dumps(obj, indent=2, default=str)


def print_state(label: str, state: Any) -> None:
    print(f"\n{'=' * 80}\n{label}\n{'=' * 80}")
    print(pretty(state))


def assert_basic_invariants(state: InventoryState) -> None:
    """Sanity checks for the InventoryState."""
    assert state.current_date >= 0
    assert 0.0 <= state.cold_storage_ratio <= 1.0
    assert 0.0 <= state.ambient_storage_ratio <= 1.0
    assert state.cold_storage_current_capacity <= state.cold_storage_total_capacity
    assert state.ambient_storage_current_capacity <= state.ambient_storage_total_capacity
    assert state.inventory_excess_today >= 0.0
    assert state.inventory_excess_cumulative >= 0.0
    assert len(state.skus) > 0
    for sku_id, sku in state.skus.items():
        assert sku.inventory_on_hand >= 0.0, f"{sku_id}: negative inventory_on_hand"
        assert 0.0 <= sku.demand_fulfilled_today <= 1.0, (
            f"{sku_id}: demand_fulfilled_today={sku.demand_fulfilled_today} out of [0, 1]"
        )
        assert sku.avg_demand_per_day >= 0.0, f"{sku_id}: negative avg_demand_per_day"
        assert sku.stockout_days_if_no_reorder >= 0.0, f"{sku_id}: negative stockout_days"
    print("  [invariants OK]")


def run_direct_environment_smoke_test() -> PharmaEnvironment:
    """
    Test the environment directly without HTTP.
    Fastest way to verify reset/step/state transitions.
    """
    env = PharmaEnvironment()

    # 1) Reset
    state = env.reset(seed=42)
    print_state("INITIAL STATE AFTER RESET", state)
    assert_basic_invariants(state)

    # 2) Show LLM prompt
    print("\n" + "=" * 80)
    print("LLM PROMPT")
    print("=" * 80)
    print(env.to_llm_prompt())

    # 3) Order nothing — empty action
    no_order_action = PharmaAction(orders={})
    state = env.step(no_order_action)
    print_state("STATE AFTER EMPTY ORDER ({})", state)
    assert_basic_invariants(state)

    # 4) Order insulin (cold-chain SKU)
    insulin_action = PharmaAction(orders={"insulin": 50.0})
    state = env.step(insulin_action)
    print_state("STATE AFTER ORDERING 50 INSULIN", state)
    assert_basic_invariants(state)

    # 5) Order paracetamol (ambient SKU)
    para_action = PharmaAction(orders={"paracetamol": 500.0})
    state = env.step(para_action)
    print_state("STATE AFTER ORDERING 500 PARACETAMOL", state)
    assert_basic_invariants(state)

    # 6) Multi-SKU order
    multi_action = PharmaAction(orders={
        "insulin":     100.0,
        "paracetamol": 300.0,
    })
    state = env.step(multi_action)
    print_state("STATE AFTER MULTI-SKU ORDER (insulin + paracetamol)", state)
    assert_basic_invariants(state)

    # 7) Parse action from raw LLM message string
    #    _parse_action extracts numeric values only — must be plain numbers, not tuples
    llm_message = '{"insulin": 40, "paracetamol": 300}'
    llm_action = PharmaAction(message=llm_message)
    state = env.step(llm_action)
    print_state("STATE AFTER LLM MESSAGE ACTION (parsed from JSON string)", state)
    assert_basic_invariants(state)

    # 8) Malformed message — should fallback gracefully (no orders placed)
    bad_action = PharmaAction(message="I would like to order some drugs please")
    state = env.step(bad_action)
    print_state("STATE AFTER MALFORMED LLM MESSAGE (should process with no orders)", state)
    assert_basic_invariants(state)

    # 9) Verify reward fields are populated after steps
    assert isinstance(state.reward, float), "reward should be a float"
    print(f"\n  reward={state.reward:.4f}  "
          f"excess_today={state.inventory_excess_today:.4f}  "
          f"excess_cumulative={state.inventory_excess_cumulative:.4f}")

    # 10) Check OpenEnv state metadata
    env_state = env.state
    print(f"\nOpenEnv State: episode_id={env_state.episode_id}, step_count={env_state.step_count}")
    assert env_state.step_count > 0

    print("\nDirect environment smoke test passed.")
    return env


async def run_http_client_smoke_test() -> None:
    """
    Optional client-side test.
    Requires the FastAPI server to be running locally first.

    Terminal:
        uvicorn app:app --host 127.0.0.1 --port 8000 --reload
    """
    try:
        from client import PharmaEnvClient
    except Exception as exc:
        print(f"\nSkipping client test: could not import client.py -> {exc}")
        return

    client = PharmaEnvClient(base_url="http://127.0.0.1:8000")
    try:
        result = await client.reset()
        print_state("CLIENT RESET OBSERVATION", result.observation)
        assert_basic_invariants(result.observation)

        # Order insulin via client
        action = PharmaAction(orders={"insulin": 50.0})
        result = await client.step(action)
        print_state("CLIENT STEP AFTER ORDERING INSULIN", result.observation)
        assert_basic_invariants(result.observation)

        # Multi-SKU order via client
        action = PharmaAction(orders={
            "paracetamol": 400.0,
            "insulin":     150.0,
        })
        result = await client.step(action)
        print_state("CLIENT STEP AFTER MULTI-SKU ORDER", result.observation)
        assert_basic_invariants(result.observation)

        # ordering insulin again to test cold storage capacity limits
        action = PharmaAction(orders={"insulin":50.0})
        result = await client.step(action)
        print("the result returned from the client is of type ", type(result))
        print("the raw result returned from the client is ", result)
        print_state("CLIENT STEP AFTER ORDERING INSULIN AGAIN", result.observation)
        assert_basic_invariants(result.observation)
        #three empty to deplete the cold storage.
        action = PharmaAction(orders={"insulin": 0.0})
        result = await client.step(action)
        print_state("CLIENT STEP AFTER 1ST EMPTY ORDER (depleting cold storage)", result.observation)
        assert_basic_invariants(result.observation)

        action = PharmaAction(orders={"insulin": 0.0})
        result = await client.step(action)
        print_state("CLIENT STEP AFTER 2ND EMPTY ORDER (depleting cold storage)", result.observation)
        assert_basic_invariants(result.observation)
        
        action = PharmaAction(orders={"insulin": 0.0})
        result = await client.step(action)
        print_state("CLIENT STEP AFTER 3 EMPTY ORDERS (depleting cold storage)", result.observation)
        assert_basic_invariants(result.observation)
        print("\nHTTP client smoke test passed.")
    finally:
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            await close_fn()


if __name__ == "__main__":
    run_direct_environment_smoke_test()

    # Uncomment only after your server is running locally:
    asyncio.run(run_http_client_smoke_test())