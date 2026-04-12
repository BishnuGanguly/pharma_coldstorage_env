from __future__ import annotations

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator
from openenv.core.env_server.types import Action, Observation, State


class JobNode(Observation):

    # -- Identity -----------------------------------------------------------
    id: str = Field(
        ...,
        description=(
            "Unique job identifier used in SCHEDULE actions. "
            "Example: 'ingest_sales'."
        ),
    )

    # -- Static config (set at reset, never change during episode) ----------
    duration: int = Field(
        ...,
        ge=1,
        description="How many minutes this job runs once started.",
    )

    cpu_required: int = Field(
        ...,
        ge=1,
        description=(
            "CPU slots this job consumes from the shared pool while running. "
            "Check ResourceState.cpu_free before scheduling."
        ),
    )

    critical_path: bool = Field(
        default=False,
        description=(
            "True if this job is on the critical path to the final deadline. "
            "Delaying it directly risks the overall SLA. Prioritise these."
        ),
    )

    depends_on: List[str] = Field(
        default_factory=list,
        description=(
            "IDs of jobs that must be DONE before this job becomes READY. "
            "Empty list means no dependencies — job starts READY at reset."
        ),
    )

    # -- Live state (mutated by env after every action) ---------------------
    status: str = Field(
        default="pending",
        description=(
            "Current lifecycle state. "
            "pending  — dependencies not met, cannot schedule. "
            "ready    — can be scheduled right now. "
            "running  — occupying a worker, timer counting down. "
            "done     — finished, resources released."
        ),
    )

    sla_deadline: Optional[int] = Field(
        default=None,
        description=(
            "Minutes remaining until this job's SLA deadline. "
            "Decremented by step_size on every WAIT action. "
            "None means no individual SLA for this job. "
            "When it reaches 0 and job is not DONE, a violation is recorded."
        ),
    )

    worker_occupied: bool = Field(
        default=False,
        description=(
            "True while this job is RUNNING and holding a worker slot. "
            "Set to True by _apply_schedule(). "
            "Set back to False when job reaches DONE."
        ),
    )

    cpu_occupied: int = Field(
        default=0,
        description=(
            "CPU slots currently held by this job. "
            "Equals cpu_required while status == running, 0 otherwise. "
            "Used by ResourceState to track pool usage."
        ),
    )


# ---------------------------------------------------------------------------
# ResourceState
# ---------------------------------------------------------------------------

class ResourceState(Observation):

    # -- Total capacity (set at reset, constant for the episode) ------------
    workers_total: int = Field(
        ...,
        ge=1,
        description="Total worker slots available in this task configuration.",
    )

    cpu_total: int = Field(
        ...,
        ge=1,
        description="Total CPU slots in the shared pool for this task.",
    )

    # -- Live availability (mutated by env) ---------------------------------
    workers_free: int = Field(
        ...,
        ge=0,
        description=(
            "Worker slots currently free. "
            "You can only schedule a job if workers_free >= 1."
        ),
    )

    cpu_free: int = Field(
        ...,
        ge=0,
        description=(
            "CPU slots currently free. "
            "You can only schedule a job if cpu_free >= job.cpu_required."
        ),
    )

    # -- Derived properties (computed from above, not stored) ---------------
    @property
    def workers_occupied(self) -> int:
        """Worker slots currently in use."""
        return self.workers_total - self.workers_free

    @property
    def cpu_occupied(self) -> int:
        """CPU slots currently in use."""
        return self.cpu_total - self.cpu_free

    def can_fit(self, job: JobNode) -> bool:
        """
        Return True if this job can be scheduled right now.

        Checks both constraints: at least one worker free AND
        enough CPU slots free. Does not check job status —
        the env validates status separately in _validate_action().
        """
        return self.workers_free >= 1 and self.cpu_free >= job.cpu_required

    # -- Invariant validator ------------------------------------------------
    @model_validator(mode="after")
    def free_cannot_exceed_total(self) -> "ResourceState":
        """free slots can never exceed total slots — hard invariant."""
        if self.workers_free > self.workers_total:
            raise ValueError(
                f"workers_free ({self.workers_free}) "
                f"cannot exceed workers_total ({self.workers_total})"
            )
        if self.cpu_free > self.cpu_total:
            raise ValueError(
                f"cpu_free ({self.cpu_free}) "
                f"cannot exceed cpu_total ({self.cpu_total})"
            )
        return self


# ---------------------------------------------------------------------------
# TaskState
# ---------------------------------------------------------------------------

class TaskState(Observation):

    # -- Clock --------------------------------------------------------------
    current_time: int = Field(
        default=0,
        ge=0,
        description="Minutes elapsed since episode start.",
    )

    time_budget: int = Field(
        ...,
        gt=0,
        description=(
            "Total minutes available in this episode. "
            "Episode ends when current_time >= time_budget."
        ),
    )

    step_size: int = Field(
        default=5,
        gt=0,
        description="Minutes advanced by each WAIT action.",
    )

    # -- Task deadline ------------------------------------------------------
    task_deadline: int = Field(
        ...,
        description=(
            "Minutes remaining until the final episode deadline. "
            "Decremented by step_size on every WAIT action. "
            "When this hits 0 and the terminal job is not DONE, "
            "the episode ends with a large SLA miss penalty."
        ),
    )

    # -- Resources ----------------------------------------------------------
    resources: ResourceState = Field(
        ...,
        description=(
            "Current worker and CPU pool state. "
            "Check resources.workers_free and resources.cpu_free "
            "before scheduling any job."
        ),
    )

    # -- DAG ----------------------------------------------------------------
    jobs: List[JobNode] = Field(
        default_factory=list,
        description=(
            "All jobs currently visible to the agent. "
            "Hidden jobs (partial DAG mode) are excluded until relevant. "
            "Each job shows its status, resource needs, and dependencies."
        ),
    )

    edges: List[Tuple[str, str]] = Field(
        default_factory=list,
        description=(
            "All directed edges in the DAG as (parent_id, child_id) pairs. "
            "An edge ('A', 'B') means job B cannot start until job A is DONE. "
            "Example: [('ingest_sales', 'clean_sales'), "
            "('clean_sales', 'compute_revenue')]. "
            "Use this to understand the full dependency structure "
            "and plan which jobs to prioritise."
        ),
    )

    critical_path: List[Tuple[str, str]] = Field(
        default_factory=list,
        description=(
            "Edges on the critical path to the final deadline. "
            "Same (parent, child) format as edges. "
            "Delaying any job on this path directly causes deadline miss. "
            "Example: [('clean_sales', 'compute_revenue'), "
            "('compute_revenue', 'daily_summary')]."
        ),
    )

    # -- Derived job lists (maintained by env, read by LLM) ----------------
    ready_jobs: List[str] = Field(
        default_factory=list,
        description=(
            "IDs of jobs whose status is READY right now. "
            "These are the only jobs you can SCHEDULE. "
            "Refreshed automatically after every action."
        ),
    )

    running_jobs: List[List] = Field(
        default_factory=list,
        description=(
            "Jobs currently RUNNING on workers. "
            "Each entry is [task_id, remaining_time_minutes]. "
            "Example: [['ingest_sales', 15], ['ingest_events', 8]]. "
            "remaining_time counts down to zero when the job finishes. "
            "Refreshed automatically after every action."
        ),
    )

    completed_jobs: List[str] = Field(
        default_factory=list,
        description=(
            "IDs of jobs that have reached DONE status. "
            "Their resources have been released. "
            "Refreshed automatically after every action."
        ),
    )

    # -- SLA ratio metrics (stored in metadata, updated after every action) -
    # These are NOT fields — they live in metadata dict (inherited from
    # Observation) so the reward function can read them without polluting
    # the Pydantic schema.  Access via:
    #   ts.metadata["sla_success_ratio"]          → float | None
    #   ts.metadata["critical_sla_success_ratio"] → float | None
    #
    # None  = no SLA-due jobs have been observed yet this episode.
    # float = completed_sla_due / total_sla_due  (both counts cumulative).
    #
    # The env also stores the raw counts for transparency:
    #   ts.metadata["sla_due_total"]              → int
    #   ts.metadata["sla_due_completed"]          → int
    #   ts.metadata["critical_sla_due_total"]     → int
    #   ts.metadata["critical_sla_due_completed"] → int

    # -- Agent feedback -----------------------------------------------------
    last_action_feedback: str = Field(
        default="Episode started. Make your first scheduling decision.",
        description=(
            "Plain-English result of the last action. "
            "Tells you if the action succeeded, failed and why, "
            "or what changed in the environment after the action."
        ),
    )

    # -- Helpers ------------------------------------------------------------
    def sync_job_lists(self) -> None:
        self.ready_jobs     = [j.id for j in self.jobs if j.status == "ready"]
        self.running_jobs = [
            [j.id, j.metadata.get("time_remaining", j.duration)]
            for j in self.jobs
            if j.status == "running"]
        self.completed_jobs = [j.id for j in self.jobs if j.status == "done"]

    def get_job(self, job_id: str) -> Optional[JobNode]:
        """Return the JobNode with the given id, or None if not found."""
        for job in self.jobs:
            if job.id == job_id:
                return job
        return None

    def minutes_remaining(self) -> int:
        """Minutes left before time budget runs out."""
        return max(0, self.time_budget - self.current_time)

    def init_sla_metadata(self) -> None:
        """
        Initialise SLA ratio metadata keys to their episode-start defaults.

        Call this once inside reset() after the TaskState is constructed.
        All counts start at 0; ratios start at None (no data yet).
        """
        self.metadata["sla_due_total"]              = 0
        self.metadata["sla_due_completed"]          = 0
        self.metadata["critical_sla_due_total"]     = 0
        self.metadata["critical_sla_due_completed"] = 0
        self.metadata["sla_success_ratio"]          = None
        self.metadata["critical_sla_success_ratio"] = None

    def update_sla_ratios(self) -> None:
        """
        Recompute SLA ratio metadata from the current job list.

        Definition
        ----------
        A job is "SLA-due" when its sla_deadline is not None AND <= 0,
        meaning the deadline has arrived or passed this step.

        sla_success_ratio          = completed SLA-due jobs / all SLA-due jobs
        critical_sla_success_ratio = completed critical SLA-due / critical SLA-due

        Counts are CUMULATIVE across the episode (jobs can only become
        SLA-due once), so we scan all jobs each time and overwrite the
        counts — this is safe because sla_deadline only decreases.

        Both ratios are None until at least one SLA-due job exists.
        """
        sla_due_jobs = [
            j for j in self.jobs
            if j.sla_deadline is not None and j.sla_deadline <= 0
        ]
        completed_ids = {j.id for j in self.jobs if j.status == "done"}

        # -- All SLA-due jobs -----------------------------------------------
        total     = len(sla_due_jobs)
        completed = sum(1 for j in sla_due_jobs if j.id in completed_ids)

        self.metadata["sla_due_total"]     = total
        self.metadata["sla_due_completed"] = completed
        self.metadata["sla_success_ratio"] = (
            completed / total if total > 0 else None
        )

        # -- Critical-path SLA-due jobs only --------------------------------
        critical_due      = [j for j in sla_due_jobs if j.critical_path]
        critical_total     = len(critical_due)
        critical_completed = sum(1 for j in critical_due if j.id in completed_ids)

        self.metadata["critical_sla_due_total"]     = critical_total
        self.metadata["critical_sla_due_completed"] = critical_completed
        self.metadata["critical_sla_success_ratio"] = (
            critical_completed / critical_total if critical_total > 0 else None
        )


class PipelineAction(Action):

    message: str = Field(
        ...,
        description=(
            "The scheduling decision as a plain string. "
            "Use '[wait]' to advance time by 5 minutes. "
            "Use '[schedule: {job_id}]' or '[schedule: {id1, id2}]' "
            "to start one or more READY jobs."
        ),
    )