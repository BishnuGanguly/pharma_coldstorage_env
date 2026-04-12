# smoke_test_pipeline_env.py
from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Make local imports work when running this file directly.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from server.my_env_environment import PipelineEnvironment
from models import PipelineAction
from tasks import create_tasks_class


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


def assert_basic_invariants(state: Any) -> None:
    """Sanity checks for your environment state."""
    assert state.current_time >= 0
    assert state.time_budget > 0
    assert state.step_size > 0
    assert state.resources.workers_free <= state.resources.workers_total
    assert state.resources.cpu_free <= state.resources.cpu_total
    assert len(state.ready_jobs) >= 0
    assert len(state.running_jobs) >= 0
    assert len(state.completed_jobs) >= 0


def run_direct_environment_smoke_test() -> PipelineEnvironment:
    """
    Test the environment directly without HTTP.
    This is the fastest way to verify reset/step/state transitions.
    """
    env = PipelineEnvironment()
    #job_blueprints=create_tasks_class.JOB_BLUEPRINT_LIST[0]
    # 1) reset
    state = env.reset()
    print_state("INITIAL STATE AFTER RESET", state)
    assert_basic_invariants(state)

    # 2) show prompt that will be sent to the LLM
    print("\n" + "=" * 80)
    print("LLM PROMPT")
    print("=" * 80)
    print(env.to_llm_prompt())

    # 3) send a WAIT action
    wait_action = PipelineAction(message="[wait]")
    state = env.step(wait_action)
    print_state("STATE AFTER [wait]", state)
    assert_basic_invariants(state)

    # 4) if any ready job exists, schedule the first one
    if state.ready_jobs:
        job_id = state.ready_jobs[0]
        schedule_action = PipelineAction(message=f"[schedule: {{{job_id}}}]")
        state = env.step(schedule_action)
        print_state(f"STATE AFTER [schedule: {{{job_id}}}]", state)
        assert_basic_invariants(state)
    else:
        print("\nNo ready jobs found after WAIT, so skipping schedule test.")

    # 5) try a malformed action to ensure parser fallback works
    bad_action = PipelineAction(message="this is not a valid command")
    state = env.step(bad_action)
    print_state("STATE AFTER INVALID ACTION (should fallback / error feedback)", state)
    assert_basic_invariants(state)

    print("\nDirect environment smoke test passed.")
    return env


async def run_http_client_smoke_test() -> None:
    """
    Optional client-side test.
    Requires your FastAPI server to be running locally first.

    Example terminal:
        uvicorn app:app --host 127.0.0.1 --port 8000 --reload
    """
    try:
        from client import PipelineEnvClient
    except Exception as exc:
        print(f"\nSkipping client test: could not import client.py -> {exc}")
        return

    client = PipelineEnvClient(base_url="http://127.0.0.1:8000")
    try:
        
        result = await client.reset(job_blueprints=create_tasks_class.JOB_BLUEPRINT_LIST[0])
        print_state("CLIENT RESET OBSERVATION", result.observation)
        assert_basic_invariants(result.observation)

        if result.observation.ready_jobs:
            job_id = result.observation.ready_jobs[0]
            action = PipelineAction(message=f"[schedule: {{{job_id}}}]")
            result = await client.step(action)
            print_state(f"CLIENT STEP AFTER SCHEDULING {job_id}", result.observation)
            assert_basic_invariants(result.observation)
        else:
            result = await client.step(PipelineAction(message="[wait]"))
            print_state("CLIENT STEP AFTER [wait]", result.observation)
            assert_basic_invariants(result.observation)

        print("\nHTTP client smoke test passed.")
    finally:
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            await close_fn()


if __name__ == "__main__":
    run_direct_environment_smoke_test()

    # Uncomment this only after your server is running locally.
    #asyncio.run(run_http_client_smoke_test())