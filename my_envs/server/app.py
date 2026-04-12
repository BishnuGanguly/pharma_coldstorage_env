# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Data Pipeline Orchestration Environment.

Endpoints:
    POST /reset   — Reset the environment, returns initial TaskState
    POST /step    — Execute a PipelineAction, returns updated TaskState
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
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with:\n    uv sync\n"
    ) from e

try:
    from models import PipelineAction, TaskState
    from my_env_environment import PipelineEnvironment
except ModuleNotFoundError:
    from models import PipelineAction, TaskState
    from server.my_env_environment import PipelineEnvironment


app = create_app(
    PipelineEnvironment,
    PipelineAction,
    TaskState,
    env_name="pipeline_env",
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


if __name__ == '__main__':
    main()

# def main(host: str = "0.0.0.0", port: int = 8000):
#     """Entry point for direct execution."""
#     import uvicorn
#     uvicorn.run(app, host=host, port=port)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--port", type=int, default=8000)
#     args = parser.parse_args()
#     main(port=args.port)