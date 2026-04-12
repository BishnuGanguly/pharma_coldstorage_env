---
title: ETL Pipeline Scheduling Environment
emoji: 🔧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# ETL Pipeline Scheduler: Operational AI Benchmark

The ETL Pipeline Scheduling Environment is an OpenEnv-compliant benchmark designed to test whether LLM agents can reason about **time, resources, and dependencies simultaneously** — the core challenge of real-world data engineering operations.

An agent observes a live Directed Acyclic Graph (DAG) of interdependent ETL jobs. Each job has a duration, a CPU cost, upstream dependencies, and an optional SLA deadline. The agent must decide which jobs to start and when to wait, all under a shared pool of workers and CPU slots. Greedy agents fail. Agents that can reason about critical paths, resource contention windows, and multi-horizon deadlines succeed.

---

## Motivation

Production data pipelines fail in subtle ways that expose gaps in LLM operational reasoning:

1. **Dependency-aware scheduling**: A job can only start when all its upstream dependencies are complete. Agents that ignore the DAG structure block themselves immediately.
2. **Resource contention**: Multiple jobs compete for a shared CPU pool. Scheduling a low-priority job at the wrong moment can starve a critical-path job of the resources it needs, causing a cascade of SLA failures.
3. **Deadline horizon reasoning**: SLA deadlines count from episode start, not from when a job becomes schedulable. An agent must work backwards from deadlines to decide *now* what to schedule *next*.
4. **Resisting temptation**: Some jobs are ready to schedule but harmful to schedule — because they consume resources needed by higher-priority jobs arriving soon. The agent must learn to say no.

---

## Environment Specification

### Observation Space

Each step the agent receives a structured JSON state object containing:

| Field | Description |
| :--- | :--- |
| `current_time_minutes` | Simulated minutes elapsed since episode start |
| `task_deadline_minutes` | Minutes remaining until the final pipeline SLA |
| `resources` | `workers_free`, `workers_total`, `cpu_free`, `cpu_total` |
| `ready_jobs` | List of job IDs that can be scheduled right now |
| `running_jobs` | List of `[job_id, minutes_remaining]` pairs |
| `completed_jobs` | List of finished job IDs |
| `jobs` | Full metadata for every job: `duration`, `cpu_required`, `depends_on`, `sla_deadline`, `critical_path` |
| `edges` | All DAG dependency edges as `(parent_id, child_id)` pairs |
| `critical_path` | Subset of edges on the critical path to the final deadline |
| `last_feedback` | Plain-English result of the previous action |

### Action Space

The agent responds with exactly one of the following per step:

| Action | Effect |
| :--- | :--- |
| `[wait]` | Advance simulation time by 5 minutes; running jobs tick forward |
| `[schedule: {job_id}]` | Start a single READY job (if resources allow) |
| `[schedule: {job_id_1, job_id_2}]` | Start multiple READY jobs in one action |

A job can only be scheduled if `workers_free >= 1` **and** `cpu_free >= job.cpu_required`. Batching multiple jobs into one schedule action is more efficient than issuing separate actions.

---

## Tasks and Difficulty

| Task | Difficulty | Jobs | Workers | CPU | Time Budget | Key Challenge |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `pipeline_easy` | **Easy** | 3 | 2 | 4 | 60 min | Zero slack — act on step 1 or miss the SLA |
| `pipeline_medium` | **Medium** | 6 | 3 | 6 | 120 min | CPU trap job that starves the critical path |
| `pipeline_difficult` | **Hard** | 10 | 4 | 8 | 180 min | Two large jobs that cannot run simultaneously |

### `pipeline_easy` — Zero-Slack Linear Chain

A three-job sequential pipeline: `ingest_customers → clean_customers → build_report`. All jobs run on the same single critical path. `clean_customers` carries an SLA deadline at **t=40 min**. Since `ingest_customers` takes 10 minutes, the agent must schedule it on step 1 with no waiting. Any idle step at the start causes the SLA to expire. Tests immediate critical-path recognition and the understanding that SLA deadlines are absolute, not relative.

```
ingest_customers (10 min) → clean_customers (10 min, SLA=40) → build_report (10 min, SLA=60)
```

### `pipeline_medium` — The CPU Trap

Two parallel ingest streams converge at `compute_revenue`. A third job, `ingest_logs`, is ready at t=0 with no dependencies — but it consumes 3 of 6 available CPU slots. If the agent schedules it, `clean_sales` (also requiring 3 CPU) cannot start when `ingest_sales` finishes, causing `clean_sales` to breach its SLA at t=50, which cascades into `compute_revenue` (SLA=90) and `summary_report` (SLA=110). The agent must identify and defer the trap job while parallelising the two safe ingest streams.

```
ingest_sales (15 min, 2 CPU) → clean_sales (15 min, 2 CPU, SLA=50) ──┐
ingest_events (12 min, 1 CPU) → clean_events (10 min, 1 CPU)         ├→ compute_revenue (20 min, 3 CPU, SLA=90) → summary_report (10 min, SLA=110)
ingest_logs [TRAP] (20 min, 3 CPU) ─────────────────── off-path ─────┘
```

### `pipeline_difficult` — Double Diamond with a Time Bomb

Two overlapping diamond-shaped dependency chains share a single terminal bottleneck (`daily_summary`). The central challenge is that `compute_revenue` (3 CPU, SLA=100) and `anomaly_detection` (3 CPU, SLA=130) together require 6 CPU but only 8 are available — they cannot run simultaneously alongside other active jobs and must be sequenced in deadline order. An additional temptation job (`enrich_reference`, no dependencies, 2 CPU) must be scheduled in a precise window: too early it starves the first clean wave; too late it delays `anomaly_detection` past its SLA. Requires genuine multi-horizon deadline reasoning across 4 workers.

```
ingest_sales (20 min) → clean_sales (15 min, SLA=70) ────────────────────────────────────────────┐
ingest_events (15 min) → clean_events (10 min) ──┬──→ compute_revenue (20 min, 3 CPU, SLA=100) ──┤
ingest_inventory (12 min) → clean_inventory ─────┤                                               ├→ daily_summary (12 min, SLA=150)
enrich_reference [window job] (18 min, 2 CPU) ───┴──→ compute_metrics (18 min) → anomaly_detection (25 min, 3 CPU, SLA=130) ──┘
```

---

## Scoring

### Per-Step Reward

At every step, the environment computes:

```
step_reward = clamp(0.7 × sla_success_ratio + 0.3 × critical_sla_success_ratio, 0.01, 0.99)
```

| Term | Definition |
| :--- | :--- |
| `sla_success_ratio` | `(SLA-due jobs completed on time) / (all SLA-due jobs)` |
| `critical_sla_success_ratio` | `(critical-path SLA-due jobs completed) / (critical-path SLA-due jobs)` |

Both ratios default to **1.0** when no SLA deadlines have yet expired — the agent starts with full credit and loses it as deadlines pass with incomplete jobs.

### Final Task Score

```
task_score = clamp(0.5 × mean(step_rewards) + 0.5 × (jobs_completed / total_jobs), 0.01, 0.99)
```

The two terms measure complementary things: `mean(step_rewards)` captures *how well* SLAs were respected throughout the episode; `jobs_completed / total_jobs` captures *whether* the pipeline was actually finished. An agent that completes the pipeline but misses every SLA, and an agent that respects every SLA but never finishes the pipeline, both score around 0.5.

**Success threshold: 0.80**

---

## Baseline Benchmarks

Evaluated across all 3 tasks. Scores represent Final Task Score [0.01 – 0.99].

| Model | `pipeline_easy` | `pipeline_medium` | `pipeline_difficult` | Success Rate |
| :--- | :---: | :---: | :---: | :---: |
| *Results to be published post-evaluation* | — | — | — | — |

---

## Setup and Usage

### 1. Installation

```bash
uv sync
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Set HF_TOKEN, MODEL_NAME, API_BASE_URL as needed
```

### 3. Run the Server

```bash
uv run server
```

### 4. Run the Benchmark

```bash
python inference.py
```

### Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | — | HuggingFace API key |
| `LOCAL_IMAGE_NAME` | — | Docker image name (optional) |
| `ENV_BASE_URL` | `http://localhost:8000` | Server URL when not using Docker |

---

## Project Structure

```
pipeline_env/
├── environment.py   # Core simulation: DAG execution, resource tracking, time advancement
├── models.py        # Pydantic schemas: JobNode, TaskState, ResourceState, PipelineAction
├── tasks.py         # Task blueprints (easy/medium/difficult) + reward function
├── inference.py     # Benchmark execution: runs all 3 tasks, emits START/STEP/END logs
├── client.py        # OpenEnv async client
└── openenv.yaml     # Environment manifest
```

---

## Key Design Decisions

**Event-driven time advancement**: The `[schedule]` action does not cost a full time step. Instead, time advances exactly to the moment when the earliest running job finishes, making scheduling decisions effectively free in terms of simulated time. Only `[wait]` burns a fixed 5-minute block. This means the agent is rewarded for parallelism and penalised for unnecessary waiting.

**SLA deadlines are absolute**: All `sla_deadline` values in job definitions count from `t=0` (episode start), not from when the job becomes schedulable. This forces the agent to reason about the full DAG timeline upfront rather than reacting greedily step-by-step.

**Separate `time_budget` and `task_deadline`**: `time_budget` is the hard episode termination limit. `task_deadline` is the SLA for the final pipeline job. `task_deadline < time_budget` in medium and hard tasks, meaning the agent has buffer time but no SLA buffer — finishing late is not the same as finishing on time.