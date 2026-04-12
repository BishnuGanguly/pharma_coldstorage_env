"""
inference.py
============
Baseline inference script for the Data Pipeline Orchestration Environment.

Runs all 3 competition tasks (easy → medium → difficult) in sequence.
Each task is a separate episode with its own DAG and resource config,
defined in tasks.py via create_tasks_class.

Environment variables
---------------------
API_BASE_URL       LLM API endpoint  (default: HuggingFace router)
MODEL_NAME         Model identifier  (default: Qwen2.5-72B-Instruct)
HF_TOKEN           API key
LOCAL_IMAGE_NAME   Docker image name (used by from_docker_image)
ENV_BASE_URL       Server URL when not using Docker (default: localhost:8000)

STDOUT format (mandatory)
--------------------------
[START] task=<task> env=<benchmark> model=<model>
[STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Score formula
-------------
    score = 0.5 * mean(per_step_rewards) + 0.5 * (jobs_completed / total_jobs)

where per_step_rewards = [compute_step_reward(ts) at each step]
                       = [0.7 * sla_ratio + 0.3 * critical_ratio] per step

Success = score >= SUCCESS_THRESHOLD
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

try:
    from client import PipelineEnvClient
    from models import PipelineAction, TaskState
    from tasks import create_tasks_class
except ImportError:
    from client import PipelineEnvClient
    from models import PipelineAction, TaskState
    from tasks import create_tasks_class

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE_NAME        = os.getenv("LOCAL_IMAGE_NAME")
API_KEY           = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL      = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME        = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK         = "pipeline_env"
MAX_STEPS         = 50
TEMPERATURE       = 0.2
MAX_TOKENS        = 80
SUCCESS_THRESHOLD = 0.8

# One name per task — must align with the order in create_tasks_class.tasks
TASK_NAMES = ["pipeline_easy", "pipeline_medium", "pipeline_difficult"]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an intelligent data pipeline scheduler.

    OBJECTIVE
    ---------
    Complete all ETL jobs before the task_deadline runs out.
    Prioritise jobs marked critical_path=true — missing their deadlines fails the pipeline.

    RESOURCES
    ---------
    You share a pool of workers and CPU slots across all running jobs.
    A job can only start if:  workers_free >= 1  AND  cpu_free >= job.cpu_required

    ACTION FORMAT  (respond with EXACTLY one of these two formats, nothing else)
    ------------
    To advance time by 5 minutes:
        [wait]

    To start one or more jobs from ready_jobs:
        [schedule: {job_id}]
        [schedule: {job_id_1, job_id_2}]

    RULES
    -----
    - Only jobs listed in ready_jobs can be scheduled.
    - running_jobs entries are [id, minutes_remaining].
    - If ready_jobs is empty and running_jobs is non-empty, you MUST use [wait].
    - Always check cpu_free and workers_free before scheduling.
    - Schedule multiple jobs in one action when resources allow — it is more efficient.
""").strip()

# ---------------------------------------------------------------------------
# Logging helpers (mandatory STDOUT format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def observation_to_dict(obs) -> dict:
    """Convert TaskState observation to a clean dict for the LLM prompt."""
    return {
        "current_time_minutes"  : obs.current_time,
        "task_deadline_minutes" : obs.task_deadline,
        "resources": {
            "workers_free"  : obs.resources.workers_free,
            "workers_total" : obs.resources.workers_total,
            "cpu_free"      : obs.resources.cpu_free,
            "cpu_total"     : obs.resources.cpu_total,
        },
        "ready_jobs"    : obs.ready_jobs,
        "running_jobs"  : obs.running_jobs,   # [[id, minutes_remaining], ...]
        "completed_jobs": obs.completed_jobs,
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
            for j in obs.jobs
        ],
        "edges"         : obs.edges,
        "critical_path" : obs.critical_path,
        "last_feedback" : obs.last_action_feedback,
    }


def build_user_prompt(obs_dict: dict, step: int) -> str:
    """Build the user-turn message from the current observation dict."""
    state_json = json.dumps(obs_dict, indent=2)
    return textwrap.dedent(f"""
        Step {step}. Current pipeline state:

        {state_json}

        Respond with exactly one action: [wait] or [schedule: {{job_id}}]
    """).strip()


def compute_jobs_completion_ratio(obs) -> float:
    """
    Jobs-completion ratio = jobs_done / total_jobs.

    This is the second component of the final task score.
    Range: [0.0, 1.0]
    """
    total = len(obs.jobs)
    if total == 0:
        return 0.0
    done_count = sum(1 for j in obs.jobs if j.status == "done")
    return done_count / total


def compute_final_score(rewards: List[float], obs) -> float:
    """
    Final task score combining reward quality and pipeline completion.

    Formula
    -------
        score = 0.5 * mean(per_step_rewards) + 0.5 * (jobs_done / total_jobs)

    Rationale for each term
    -----------------------
    mean(per_step_rewards):
        Captures HOW WELL the agent respected SLA deadlines throughout the
        episode. An agent that completes all jobs but misses every SLA scores
        poorly here. An agent that finishes fast with no violations scores ~1.0.
        We use mean (not sum) here because the per-step reward is already
        bounded in [0, 1] and we want this term to stay in [0, 1] regardless
        of episode length — making it directly combinable with the ratio term.

    jobs_done / total_jobs:
        Captures WHETHER the agent completed the pipeline at all. This anchors
        the score against agents that earn good SLA ratios by doing nothing
        (since no-deadline = no violation). Without this term, an agent that
        never schedules a single job could earn 1.0 on SLA ratios.

    The 50/50 weighting means both dimensions matter equally: a perfect SLA
    score with 50% completion = 0.75, same as perfect completion with 50% SLA.

    Range: [0.0, 1.0]
    """
    if not rewards:
        mean_reward = 0.0
    else:
        mean_reward = sum(rewards) / len(rewards)

    completion_ratio = compute_jobs_completion_ratio(obs)

    raw = 0.5 * mean_reward + 0.5 * completion_ratio
    # Clamp strictly to [0.01, 0.99] — same range as step rewards.
    return round(max(0.01, min(0.99, raw)), 4)

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_llm_action(client: OpenAI, obs_dict: dict, step: int) -> str:
    """Call the LLM and return the raw action string. Falls back to [wait]."""
    user_prompt = build_user_prompt(obs_dict, step)
    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            stream      = False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "[wait]"
    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        return "[wait]"

# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

async def run_episode(
    client: OpenAI,
    task_name: str,
    Job_blueprint: dict,
) -> None:
    """
    Run one full episode for a single task and emit mandatory STDOUT lines.

    Parameters
    ----------
    client : OpenAI
        Initialised OpenAI client.
    task_name : str
        Human-readable task name used in [START] log line.
    Job_blueprint : dict
        Job blueprint list from tasks.py (the full list[dict] for one task).
        Passed to env.reset(job_blueprints=Job_blueprint).
    """
    # -- Connect --------------------------------------------------------
    if IMAGE_NAME:
        env = await PipelineEnvClient.from_docker_image(IMAGE_NAME)
    else:
        base_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
        env = PipelineEnvClient(base_url=base_url)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_obs = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # -- Reset with the task-specific blueprint ----------------------
        result = await env.reset(job_blueprints=Job_blueprint)
        last_obs = result.observation

        # -- Step loop ---------------------------------------------------
        for step in range(1, MAX_STEPS + 1):

            if result.done:
                break

            obs_dict   = observation_to_dict(last_obs)
            raw_action = get_llm_action(client, obs_dict, step)

            action = PipelineAction(message=raw_action)
            result = await env.step(action)

            # reward is set by compute_step_reward() inside environment.step()
            # and stored on ts.reward before the TaskState is returned.
            reward   = result.reward if result.reward is not None else 0.0
            done     = result.done
            last_obs = result.observation

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=raw_action, reward=reward, done=done, error=None)

            if done:
                break

        # -- Final score -------------------------------------------------
        if last_obs is not None:
            score = compute_final_score(rewards, last_obs)
        success = score >= SUCCESS_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ---------------------------------------------------------------------------
# Main — run all 3 tasks in sequence
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Entry point.

    Runs all 3 task blueprints (easy → medium → difficult) as separate
    episodes, emitting START / STEP / END log lines for each.
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_tasks = create_tasks_class.JOB_BLUEPRINT_LIST

    for task_name, job_blueprint in zip(TASK_NAMES, all_tasks):
        await run_episode(
            client        = client,
            task_name     = task_name,
            Job_blueprint = job_blueprint,
        )


if __name__ == "__main__":
    asyncio.run(main())