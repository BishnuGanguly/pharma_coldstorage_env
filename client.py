"""
FastAPI application for the Pharmaceutical Warehouse Procurement Environment.

Endpoints:
    POST /reset   — Reset the environment, returns initial InventoryState
    POST /step    — Execute a PharmaAction, returns updated InventoryState
    GET  /state   — Get current OpenEnv State (episode_id, step_count)
    GET  /schema  — Get action/observation JSON schemas
    WS   /ws      — WebSocket endpoint for persistent sessions

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with:\n    uv sync\n"
    ) from e

try:
    from models import InventoryState, PharmaAction
    from my_env_environment import PharmaEnvironment
except ModuleNotFoundError:
    from models import InventoryState, PharmaAction
    from server.my_env_environment import PharmaEnvironment


app = create_app(
    PharmaEnvironment,
    PharmaAction,
    InventoryState,
    env_name="pharma_env",
    max_concurrent_envs=1,
)


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()