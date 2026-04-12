# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """My Env Environment Client."""

# from typing import Dict

# from openenv.core import EnvClient
# from openenv.core.client_types import StepResult
# from openenv.core.env_server.types import State

# from .models import MyAction, MyObservation


# class MyEnv(
#     EnvClient[MyAction, MyObservation, State]
# ):
#     """
#     Client for the My Env Environment.

#     This client maintains a persistent WebSocket connection to the environment server,
#     enabling efficient multi-step interactions with lower latency.
#     Each client instance has its own dedicated environment session on the server.

#     Example:
#         >>> # Connect to a running server
#         >>> with MyEnv(base_url="http://localhost:8000") as client:
#         ...     result = client.reset()
#         ...     print(result.observation.echoed_message)
#         ...
#         ...     result = client.step(MyAction(message="Hello!"))
#         ...     print(result.observation.echoed_message)

#     Example with Docker:
#         >>> # Automatically start container and connect
#         >>> client = MyEnv.from_docker_image("my_env-env:latest")
#         >>> try:
#         ...     result = client.reset()
#         ...     result = client.step(MyAction(message="Test"))
#         ... finally:
#         ...     client.close()
#     """

#     def _step_payload(self, action: MyAction) -> Dict:
#         """
#         Convert MyAction to JSON payload for step message.

#         Args:
#             action: MyAction instance

#         Returns:
#             Dictionary representation suitable for JSON encoding
#         """
#         return {
#             "message": action.message,
#         }

#     def _parse_result(self, payload: Dict) -> StepResult[MyObservation]:
#         """
#         Parse server response into StepResult[MyObservation].

#         Args:
#             payload: JSON response data from server

#         Returns:
#             StepResult with MyObservation
#         """
#         obs_data = payload.get("observation", {})
#         observation = MyObservation(
#             echoed_message=obs_data.get("echoed_message", ""),
#             message_length=obs_data.get("message_length", 0),
#             done=payload.get("done", False),
#             reward=payload.get("reward"),
#             metadata=obs_data.get("metadata", {}),
#         )

#         return StepResult(
#             observation=observation,
#             reward=payload.get("reward"),
#             done=payload.get("done", False),
#         )

#     def _parse_state(self, payload: Dict) -> State:
#         """
#         Parse server response into State object.

#         Args:
#             payload: JSON response from state request

#         Returns:
#             State object with episode_id and step_count
#         """
#         return State(
#             episode_id=payload.get("episode_id"),
#             step_count=payload.get("step_count", 0),
#         )
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline Environment Client."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from models import JobNode, PipelineAction, ResourceState, TaskState


class PipelineEnvClient(EnvClient[PipelineAction, TaskState, State]):
    """
    Client for the PipelineEnvironment.

    This client sends a text action message to the server and receives a
    full TaskState as the observation.
    """

    def _step_payload(self, action: PipelineAction) -> Dict[str, Any]:
        """
        Convert PipelineAction into a JSON-serializable payload.

        The server-side environment expects only a text message.
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TaskState]:
        """
        Parse the server response into StepResult[TaskState].

        The server returns an observation payload that should match the
        TaskState structure: resources, jobs, DAG edges, derived lists, etc.
        """
        obs_data = payload.get("observation", payload)

        resources_data = obs_data.get("resources", {})
        resources = ResourceState(
            workers_total=resources_data.get("workers_total", 0),
            cpu_total=resources_data.get("cpu_total", 0),
            workers_free=resources_data.get("workers_free", 0),
            cpu_free=resources_data.get("cpu_free", 0),
        )

        jobs = [self._parse_job_node(job_data) for job_data in obs_data.get("jobs", [])]

        running_jobs_raw = obs_data.get("running_jobs", [])
        running_jobs: List[List[Any]] = []
        for item in running_jobs_raw:
            if isinstance(item, (list, tuple)):
                running_jobs.append(list(item))
            else:
                running_jobs.append([item])

        observation = TaskState(
            current_time=obs_data.get("current_time", 0),
            time_budget=obs_data.get("time_budget", 0),
            step_size=obs_data.get("step_size", 5),
            task_deadline=obs_data.get("task_deadline", 0),
            resources=resources,
            jobs=jobs,
            edges=[tuple(edge) for edge in obs_data.get("edges", [])],
            critical_path=[tuple(edge) for edge in obs_data.get("critical_path", [])],
            ready_jobs=list(obs_data.get("ready_jobs", [])),
            running_jobs=running_jobs,
            completed_jobs=list(obs_data.get("completed_jobs", [])),
            last_action_feedback=obs_data.get(
                "last_action_feedback",
                "Episode started. Make your first scheduling decision.",
            ),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse the lightweight server-side metadata state.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    def _parse_job_node(self, data: Dict[str, Any]) -> JobNode:
        """
        Parse one job entry from the server payload into JobNode.
        """
        return JobNode(
            id=data.get("id", ""),
            duration=data.get("duration", 1),
            cpu_required=data.get("cpu_required", 1),
            critical_path=data.get("critical_path", False),
            depends_on=list(data.get("depends_on", [])),
            status=data.get("status", "pending"),
            sla_deadline=data.get("sla_deadline"),
            worker_occupied=data.get("worker_occupied", False),
            cpu_occupied=data.get("cpu_occupied", 0),
        )