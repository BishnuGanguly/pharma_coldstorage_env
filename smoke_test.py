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
    assert 0 <= state.current_date
    assert 0.0 <= state.prescription_fill_rate_30d <= 1.0
    assert 0.0 <= state.prescription_fill_rate_7d <= 1.0
    assert 0.0 <= state.procurement_budget_ratio <= 1.0
    assert 0.0 <= state.cold_storage_capacity_ratio <= 1.0
    assert 0.0 <= state.ambient_capacity_ratio <= 1.0
    assert state.cold_storage_current_capacity <= state.cold_storage_total_capacity
    assert state.ambient_storage_current_capacity <= state.ambient_storage_total_capacity
    assert len(state.skus) > 0
    assert len(state.suppliers) > 0
    for sku in state.skus.values():
        assert sku.inventory_on_hand >= 0.0
        assert sku.backorders >= 0.0
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

    # 4) Order insulin from FastPharma (cold-chain certified)
    insulin_action = PharmaAction(orders={"insulin": (50.0, "FastPharma")})
    state = env.step(insulin_action)
    print_state("STATE AFTER ORDERING 50 INSULIN FROM FastPharma", state)
    assert_basic_invariants(state)

    # 5) Order paracetamol from GlobalMed (cheaper, non-cold-chain)
    para_action = PharmaAction(orders={"paracetamol": (500.0, "GlobalMed")})
    state = env.step(para_action)
    print_state("STATE AFTER ORDERING 500 PARACETAMOL FROM GlobalMed", state)
    assert_basic_invariants(state)

    # 6) Multi-SKU order
    multi_action = PharmaAction(orders={
        "bp_medication": (200.0, "FastPharma"),
        "vitamins":      (300.0, "GlobalMed"),
    })
    state = env.step(multi_action)
    print_state("STATE AFTER MULTI-SKU ORDER (bp_medication + vitamins)", state)
    assert_basic_invariants(state)

    # 7) Invalid: try to order insulin from GlobalMed (not cold-chain certified)
    invalid_action = PharmaAction(orders={"insulin": (30.0, "GlobalMed")})
    state = env.step(invalid_action)
    print_state("STATE AFTER INVALID ORDER (insulin from non-certified GlobalMed)", state)
    assert_basic_invariants(state)
    assert "rejected" in state.last_action_feedback.lower(), (
        f"Expected rejection feedback, got: {state.last_action_feedback}"
    )

    # 8) Parse action from raw LLM message string
    llm_message = '{"insulin": [40, "FastPharma"], "paracetamol": [300, "GlobalMed"]}'
    llm_action = PharmaAction(message=llm_message)
    state = env.step(llm_action)
    print_state("STATE AFTER LLM MESSAGE ACTION (parsed from JSON string)", state)
    assert_basic_invariants(state)

    # 9) Malformed message — should fallback gracefully
    bad_action = PharmaAction(message="I would like to order some drugs please")
    state = env.step(bad_action)
    print_state("STATE AFTER MALFORMED LLM MESSAGE (should process with no orders)", state)
    assert_basic_invariants(state)

    # 10) Check OpenEnv state metadata
    env_state = env.state
    print(f"\nOpenEnv State: episode_id={env_state.episode_id}, step_count={env_state.step_count}")
    assert env_state.step_count == 9

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
        action = PharmaAction(orders={"insulin": (50.0, "FastPharma")})
        result = await client.step(action)
        print_state("CLIENT STEP AFTER ORDERING INSULIN", result.observation)
        assert_basic_invariants(result.observation)

        # Multi-SKU order via client
        action = PharmaAction(orders={
            "paracetamol":  (400.0, "GlobalMed"),
            "bp_medication": (150.0, "FastPharma"),
        })
        result = await client.step(action)
        print_state("CLIENT STEP AFTER MULTI-SKU ORDER", result.observation)
        assert_basic_invariants(result.observation)

        print("\nHTTP client smoke test passed.")
    finally:
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            await close_fn()


if __name__ == "__main__":
    run_direct_environment_smoke_test()

    # Uncomment only after your server is running locally:
    # asyncio.run(run_http_client_smoke_test())