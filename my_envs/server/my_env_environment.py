from __future__ import annotations

import json
import re
import uuid
from typing import Any, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# from OpenEnv.envs.my_env.tasks import create_tasks_class

try:
    from models import JobNode, PipelineAction, ResourceState, TaskState
    from tasks import compute_step_reward
    from tasks import create_tasks_class
except ImportError:
    from models import JobNode, PipelineAction, ResourceState, TaskState
    from tasks import compute_step_reward
    from tasks import create_tasks_class


# ---------------------------------------------------------------------------
# DAG blueprints — plain dicts, permanent, never mutated
# reset() reads these and builds fresh JobNode objects every episode.
# ---------------------------------------------------------------------------

_WORKERS_TOTAL = 3
_CPU_TOTAL     = 8
_TIME_BUDGET   = 180   # total episode minutes
_STEP_SIZE     = 5     # minutes advanced by WAIT
_TASK_DEADLINE = 120   # final SLA (minutes from episode start)

# ---------------------------------------------------------------------------
# System prompt shown to the LLM before every state observation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an intelligent data pipeline scheduler.

OBJECTIVE
---------
Complete all ETL jobs in the pipeline before the task_deadline runs out.
Prioritise jobs on the critical_path — missing their deadlines fails the pipeline.

RESOURCES
---------
You have a fixed pool of workers and CPU slots.
A job can only start if workers_free >= 1 AND cpu_free >= job.cpu_required.

ACTION FORMAT
-------------
To do nothing and let time pass 5 minutes:
    [wait]

To start one or more READY jobs (check ready_jobs list):
    [schedule: {job_id}]
    [schedule: {job_id_1, job_id_2}]

Only jobs in ready_jobs can be scheduled.
Jobs in running_jobs are already executing — each entry is [id, minutes_remaining].
Jobs in completed_jobs are finished and cannot be rescheduled.

STRATEGY TIPS
-------------
- Always check ready_jobs and resources before deciding.
- Jobs on the critical_path should be scheduled as early as possible.
- Scheduling multiple jobs in one action uses fewer time steps.
- If no jobs are ready and jobs are running, use [wait] to let them finish.
"""


# ---------------------------------------------------------------------------
# PipelineEnvironment
# ---------------------------------------------------------------------------

class PipelineEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(self, task_state: TaskState|None = None) -> None:
        """
        Initialise the environment.

        Parameters
        ----------
        task_state : TaskState, optional
            If provided, the environment starts from this existing state
            instead of calling reset(). Useful for loading a saved episode
            or continuing from a checkpoint.
            If None, the environment is uninitialised until reset() is called.
        """
        super().__init__()

        self._task_state = task_state
        self._step_count: int = 0
        self._episode_id: str = str(uuid.uuid4()) if task_state is not None else ""

    # -----------------------------------------------------------------------
    # Public OpenEnv API
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TaskState:
        """
        Start a fresh episode from the blueprint DAG.

        Builds fresh JobNode / ResourceState / TaskState objects.
        All counters and timers reset to their initial values.

        SLA ratio metadata is initialised to zeros / None here so the
        reward function always finds the keys present from step 0.

        Returns the initial TaskState (first observation the agent sees).
        """
        if kwargs.get("job_blueprints") is None:
            self._task_state = self.build_taskstate(create_tasks_class.JOB_BLUEPRINT_LIST[0])
            self._task_state.sync_job_lists
        if kwargs.get("job_blueprints") is not None:
            self._task_state = self.build_taskstate(kwargs["job_blueprints"])
            self._task_state.sync_job_lists()
        # else:
        #     tasks_class = create_tasks_class()
        #     self._task_state = tasks_class.tasks[0]

        # Initialise SLA ratio counters to episode-start defaults.
        # At t=0 no sla_deadline has expired yet, so both ratios are None.
        self._task_state.init_sla_metadata()

        self._reset_rubric()
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        return self._task_state

    # -----------------------------------------------------------------------

    def step(
        self,
        action: PipelineAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TaskState:
        """
        Execute one agent action and return the updated TaskState.

        Parameters
        ----------
        action : PipelineAction
            Action with a `message` string produced by the LLM.
            Parse it with parse_llm_output() from raw LLM text first.

        Returns
        -------
        TaskState
            Updated state. The LLM reads this as its next observation.
        """
        if self._task_state is None:
            raise RuntimeError("reset() must be called before step().")
        if self._task_state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._step_count += 1

        # -- Parse the message string ---------------------------------------
        command, job_ids = self._parse_message(action.message)

        # -- Dispatch -------------------------------------------------------
        if command == "wait":
            feedback = self._handle_wait()

        elif command == "schedule":
            feedback = self._handle_schedule(job_ids)

        else:
            feedback = (
                f"Could not parse '{action.message}'. "
                f"Use '[wait]' or '[schedule: {{id1, id2}}]'."
            )

        self._task_state.last_action_feedback = feedback

        # -- Terminal check -------------------------------------------------
        ts = self._task_state
        all_done = all(j.status == "done" for j in ts.jobs)
        time_up  = ts.current_time >= ts.time_budget

        if all_done or time_up:
            ts.done = True

        ts.sync_job_lists()

        # -- Compute and store per-step reward ------------------------------
        # update_sla_ratios() was already called inside _handle_wait() or
        # _handle_schedule(), so metadata ratios are current at this point.
        ts.reward = compute_step_reward(ts)

        return ts

    # -----------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Return OpenEnv State metadata (episode_id, step_count)."""
        return State(
            episode_id = self._episode_id,
            step_count = self._step_count,
        )

    def build_taskstate(self, job_blueprints: List[dict]) -> TaskState:
        jobs: List[JobNode] = []
        for bp in job_blueprints[1:]:
            status = "ready" if not bp["depends_on"] else "pending"
            jobs.append(JobNode(
                id              = bp["id"],
                duration        = bp["duration"],
                cpu_required    = bp["cpu_required"],
                depends_on      = list(bp["depends_on"]),
                sla_deadline    = bp["sla_deadline"],
                critical_path   = bp["critical_path"],
                status          = status,
                worker_occupied = False,
                cpu_occupied    = 0,
            ))

        # -- Build edges from depends_on ------------------------------------
        edges: List[Tuple[str, str]] = [
            (parent, job.id)
            for job in jobs
            for parent in job.depends_on
        ]
        critical_edges: List[Tuple[str, str]] = [
            (p, c) for (p, c) in edges
            if any(j.id == c and j.critical_path for j in jobs)
        ]

        # -- Build ResourceState (all slots free) ---------------------------
        resources = ResourceState(
            workers_total = job_blueprints[0]["WORKERS_TOTAL"],
            cpu_total     = job_blueprints[0]["CPU_TOTAL"],
            workers_free  = job_blueprints[0]["WORKERS_TOTAL"],
            cpu_free      = job_blueprints[0]["CPU_TOTAL"],
        )

        # -- Assemble TaskState ---------------------------------------------
        task_state = TaskState(
            current_time         = 0,
            time_budget          = job_blueprints[0]["TIME_BUDGET"],
            step_size            = job_blueprints[0]["STEP_SIZE"],
            task_deadline        = job_blueprints[0]["TASK_DEADLINE"],
            resources            = resources,
            jobs                 = jobs,
            edges                = edges,
            critical_path        = critical_edges,
            last_action_feedback = "Episode started. Make your first scheduling decision.",
            done                 = False,
            reward               = 0.0,
        )
        task_state.sync_job_lists()
        return task_state

    # -----------------------------------------------------------------------
    # Message parsing (private)
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_message(message: str) -> Tuple[str, List[str]]:
        """
        Parse the LLM's message string into (command, job_ids).

        Recognised formats
        ------------------
        [wait]                         → ("wait", [])
        [schedule: {id1}]              → ("schedule", ["id1"])
        [schedule: {id1, id2}]         → ("schedule", ["id1", "id2"])

        Unrecognised input             → ("unknown", [])

        Tolerant: strips whitespace and is case-insensitive on keywords.
        """
        stripped = message.strip().lower()

        # -- WAIT -----------------------------------------------------------
        if re.search(r"\[wait\]", stripped):
            return "wait", []

        # -- SCHEDULE -------------------------------------------------------
        match = re.search(r"\[schedule\s*:\s*\{([^}]*)\}\]", stripped)
        if match:
            raw_ids = match.group(1)
            job_ids = [jid.strip() for jid in raw_ids.split(",") if jid.strip()]
            return "schedule", job_ids

        return "unknown", []

    # -----------------------------------------------------------------------
    # Action handlers (private)
    # -----------------------------------------------------------------------

    def _handle_wait(self) -> str:
        """
        Advance time by step_size (5 min) and tick all running jobs.

        Steps
        -----
        1. Advance current_time by step_size.
        2. Reduce task_deadline and per-job sla_deadlines by step_size.
        3. Reduce remaining_time of every running job by step_size.
        4. Complete jobs whose remaining_time reached zero.
        5. Unlock jobs whose dependencies are now all done.
        6. Recompute SLA ratio metadata (both overall and critical-path).

        Returns a human-readable feedback string.
        """
        ts = self._task_state
        T  = ts.step_size

        # 1 + 2: advance clock and all deadlines
        ts.current_time   += T
        ts.task_deadline  -= T
        self._reduce_sla_deadlines(T)

        # 3: tick running jobs
        for job in ts.jobs:
            if job.status == "running":
                current_remaining = job.metadata.get("time_remaining", job.duration)
                job.metadata["time_remaining"] = current_remaining - T

        # 4: complete finished jobs
        finished = self._complete_finished_jobs()

        # 5: unlock newly unblocked jobs
        unlocked = self._unlock_ready_jobs()

        # 6: recompute SLA ratios now that deadlines have been decremented
        #    and any newly finished jobs are marked done.
        ts.update_sla_ratios()

        parts = [f"Waited {T} min. Time is now {ts.current_time} min."]
        if finished:
            parts.append(f"Completed: {finished}.")
        if unlocked:
            parts.append(f"Now ready: {unlocked}.")

        # Append SLA ratio info to feedback when data exists
        sla_ratio = ts.metadata["sla_success_ratio"]
        if sla_ratio is not None:
            n_comp  = ts.metadata["sla_due_completed"]
            n_total = ts.metadata["sla_due_total"]
            parts.append(
                f"SLA success ratio: {sla_ratio:.2f} ({n_comp}/{n_total})."
            )

        critical_ratio = ts.metadata["critical_sla_success_ratio"]
        if critical_ratio is not None:
            nc_comp  = ts.metadata["critical_sla_due_completed"]
            nc_total = ts.metadata["critical_sla_due_total"]
            parts.append(
                f"Critical-path SLA success ratio: "
                f"{critical_ratio:.2f} ({nc_comp}/{nc_total})."
            )

        return " ".join(parts)

    # -----------------------------------------------------------------------

    def _handle_schedule(self, job_ids: List[str]) -> str:
        """
        Schedule one or more jobs with event-driven time advancement.

        Full logic when running jobs already exist
        ------------------------------------------
        1.  Validate each requested job_id.
        2.  Find the running job with the smallest remaining_time → T.
        3.  Advance current_time by T.
        4.  Reduce task_deadline and per-job sla_deadlines by T.
        5.  Reduce remaining_time of every other running job by T.
        6.  Complete the earliest-finishing running job (free its resources).
        7.  Unlock any jobs whose deps are now all done.
        8.  Add each validated new job to running with time_remaining = duration.
        9.  Deduct each new job's resources from ResourceState.
        10. Recompute SLA ratio metadata (both overall and critical-path).

        When no running jobs exist
        --------------------------
        Skip steps 2–6. Just start each job with time_remaining = duration.
        SLA ratios are still recomputed (step 10) in case a deadline expired
        at the moment the schedule action fires.

        Parameters
        ----------
        job_ids : list of str
            Job IDs parsed from the LLM's message.

        Returns a human-readable feedback string.
        """
        ts = self._task_state
        feedback_parts: List[str] = []
        valid_jobs: List[JobNode] = []

        # -- Validate each requested job ------------------------------------
        for job_id in job_ids:
            ok, reason = self._validate_schedule(job_id)
            if not ok:
                feedback_parts.append(f"Cannot schedule '{job_id}': {reason}")
            else:
                valid_jobs.append(ts.get_job(job_id))

        if not valid_jobs:
            return " | ".join(feedback_parts) or "No valid jobs to schedule."

        # -- Determine T (time until next completion event) -----------------
        running_nodes = [j for j in ts.jobs if j.status == "running"]

        if running_nodes:
            # Find the running job that will finish soonest
            earliest: JobNode = min(
                running_nodes,
                key=lambda j: j.metadata.get("time_remaining", j.duration),
            )
            T = earliest.metadata.get("time_remaining", earliest.duration)
            T = max(T, 0)

            # Advance clock and all deadlines by T
            ts.current_time  += T
            ts.task_deadline -= T
            self._reduce_sla_deadlines(T)

            # Reduce remaining time of all other running jobs by T
            for job in running_nodes:
                if job.id != earliest.id:
                    current = job.metadata.get("time_remaining", job.duration)
                    job.metadata["time_remaining"] = current - T

            # Complete the earliest-finishing job
            self._complete_one_job(earliest)
            feedback_parts.append(
                f"Time advanced {T} min. '{earliest.id}' completed."
            )

            # Unlock jobs freed by this completion
            unlocked = self._unlock_ready_jobs()
            if unlocked:
                feedback_parts.append(f"Now ready: {unlocked}.")

        else:
            # No running jobs — start new jobs without advancing time
            T = 0

        # -- Start each validated new job -----------------------------------
        scheduled: List[str] = []
        for job in valid_jobs:
            ok, reason = self._validate_schedule(job.id)  # re-check after time advance
            if not ok:
                feedback_parts.append(f"'{job.id}' no longer schedulable: {reason}")
                continue

            # Job starts now; T minutes have already elapsed this step
            time_remaining = max(job.duration - T, 1)

            job.status           = "running"
            job.worker_occupied  = True
            job.cpu_occupied     = job.cpu_required
            job.metadata["time_remaining"] = time_remaining

            # Deduct resources
            ts.resources.workers_free -= 1
            ts.resources.cpu_free     -= job.cpu_required

            scheduled.append(job.id)

        if scheduled:
            feedback_parts.append(f"Scheduled: {scheduled}.")

        # -- Recompute SLA ratios after all state changes -------------------
        # Deadlines were decremented by T above (via _reduce_sla_deadlines),
        # so any job whose deadline just hit 0 is captured here.
        ts.update_sla_ratios()

        sla_ratio = ts.metadata["sla_success_ratio"]
        if sla_ratio is not None:
            n_comp  = ts.metadata["sla_due_completed"]
            n_total = ts.metadata["sla_due_total"]
            feedback_parts.append(
                f"SLA success ratio: {sla_ratio:.2f} ({n_comp}/{n_total})."
            )

        critical_ratio = ts.metadata["critical_sla_success_ratio"]
        if critical_ratio is not None:
            nc_comp  = ts.metadata["critical_sla_due_completed"]
            nc_total = ts.metadata["critical_sla_due_total"]
            feedback_parts.append(
                f"Critical-path SLA success ratio: "
                f"{critical_ratio:.2f} ({nc_comp}/{nc_total})."
            )

        return " ".join(feedback_parts) if feedback_parts else "No action taken."

    # -----------------------------------------------------------------------
    # Simulation helpers (private)
    # -----------------------------------------------------------------------

    def _validate_schedule(self, job_id: str) -> Tuple[bool, str]:
        """
        Check whether job_id can be scheduled right now.

        Returns (True, success_msg) or (False, reason_msg).
        """
        ts  = self._task_state
        job = ts.get_job(job_id)

        if job is None:
            return False, f"not found. Known jobs: {[j.id for j in ts.jobs]}"

        if job.status == "pending":
            return False, f"still PENDING, waiting for {job.depends_on}"

        if job.status == "running":
            return False, "already RUNNING"

        if job.status == "done":
            return False, "already DONE"

        if ts.resources.workers_free < 1:
            return False, f"no free workers (running: {[r[0] for r in ts.running_jobs]})"

        if ts.resources.cpu_free < job.cpu_required:
            return (
                False,
                f"needs {job.cpu_required} CPU but only "
                f"{ts.resources.cpu_free} free",
            )

        return True, f"'{job_id}' scheduled."

    # -----------------------------------------------------------------------

    def _complete_one_job(self, job: JobNode) -> None:
        """
        Mark a single job as done and free its resources.
        """
        ts = self._task_state
        job.status          = "done"
        job.worker_occupied = False
        job.cpu_occupied    = 0
        job.metadata["time_remaining"] = 0

        ts.resources.workers_free += 1
        ts.resources.cpu_free     += job.cpu_required

    # -----------------------------------------------------------------------

    def _complete_finished_jobs(self) -> List[str]:
        """
        Complete all RUNNING jobs whose time_remaining <= 0.

        Returns list of completed job IDs (for feedback strings).
        """
        finished: List[str] = []
        for job in self._task_state.jobs:
            if job.status == "running":
                remaining = job.metadata.get("time_remaining", job.duration)
                if remaining <= 0:
                    self._complete_one_job(job)
                    finished.append(job.id)
        return finished

    # -----------------------------------------------------------------------

    def _unlock_ready_jobs(self) -> List[str]:
        """
        Move PENDING jobs to READY when all their dependencies are done.

        Returns list of newly unlocked job IDs (for feedback strings).
        """
        ts       = self._task_state
        done_ids = {j.id for j in ts.jobs if j.status == "done"}
        unlocked: List[str] = []

        for job in ts.jobs:
            if job.status == "pending":
                if all(dep in done_ids for dep in job.depends_on):
                    job.status = "ready"
                    unlocked.append(job.id)

        return unlocked

    # -----------------------------------------------------------------------

    def _reduce_sla_deadlines(self, minutes: int) -> None:
        """
        Subtract `minutes` from every job's sla_deadline (if it has one).
        """
        for job in self._task_state.jobs:
            if job.sla_deadline is not None:
                job.sla_deadline -= minutes

    # -----------------------------------------------------------------------
    # LLM interface helpers (public)
    # -----------------------------------------------------------------------

    @staticmethod
    def parse_llm_output(raw: str) -> PipelineAction:
        """
        Extract a valid action message from raw LLM text.

        Handles common LLM output patterns:
        - Clean format:   [wait]  or  [schedule: {id}]
        - With reasoning: "I will schedule ingest_sales.\n[schedule: {ingest_sales}]"
        - Markdown fence: ```\n[schedule: {id}]\n```

        Falls back to "[wait]" if nothing valid is found.
        """
        # Strip markdown fences
        cleaned = re.sub(r"```[a-z]*\s*", "", raw).strip()
        cleaned = re.sub(r"```", "", cleaned).strip()

        # Look for [wait] or [schedule: {...}]
        wait_match     = re.search(r"\[wait\]", cleaned, re.IGNORECASE)
        schedule_match = re.search(r"\[schedule\s*:\s*\{[^}]*\}\]", cleaned, re.IGNORECASE)

        if schedule_match:
            return PipelineAction(message=schedule_match.group(0))

        if wait_match:
            return PipelineAction(message="[wait]")

        # Nothing found — default to wait
        return PipelineAction(message="[wait]")

    # -----------------------------------------------------------------------

    def to_llm_prompt(self) -> str:
        """
        Build the full prompt sent to the LLM for the next decision.

        Returns a string containing:
        1. The system prompt (objective, rules, action format).
        2. The current TaskState serialised as JSON.

        The caller (baseline script) passes this as the user message
        to the OpenAI chat completion API.
        """
        if self._task_state is None:
            return _SYSTEM_PROMPT + "\n\nNo state available. Call reset() first."

        ts = self._task_state

        # Build a clean state dict for the LLM
        state_dict = {
            "current_time_minutes"  : ts.current_time,
            "time_budget_minutes"   : ts.time_budget,
            "task_deadline_minutes" : ts.task_deadline,
            "resources": {
                "workers_free"  : ts.resources.workers_free,
                "workers_total" : ts.resources.workers_total,
                "cpu_free"      : ts.resources.cpu_free,
                "cpu_total"     : ts.resources.cpu_total,
            },
            "ready_jobs"    : ts.ready_jobs,
            "running_jobs"  : ts.running_jobs,   # [[id, remaining_time], ...]
            "completed_jobs": ts.completed_jobs,
            "jobs": [
                {
                    "id"           : j.id,
                    "status"       : j.status,
                    "duration"     : j.duration,
                    "cpu_required" : j.cpu_required,
                    "depends_on"   : j.depends_on,
                    "sla_deadline" : j.sla_deadline,
                    "critical_path": j.critical_path,
                }
                for j in ts.jobs
            ],
            "edges"          : ts.edges,
            "critical_path"  : ts.critical_path,
            "last_feedback"  : ts.last_action_feedback,
            "episode_done"   : ts.done,
        }

        state_json = json.dumps(state_dict, indent=2)

        return (
            _SYSTEM_PROMPT
            + "\n\n--- CURRENT PIPELINE STATE ---\n"
            + state_json
            + "\n\n--- YOUR DECISION ---\n"
            + "Respond with exactly one action: [wait] or [schedule: {job_id}]\n"
        )