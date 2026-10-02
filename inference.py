"""
inference.py
============
Baseline inference script for the Pharma Inventory Management Environment.

Runs every task in tasks.TASK_REGISTRY (supply_chain_broken, flu_season,
epidemic_two_wave) in sequence. Each task is a separate 60-day episode.

Environment variables
---------------------
API_BASE_URL       LLM API endpoint  (default: HuggingFace router)
MODEL_NAME         Model identifier  (default: Qwen2.5-72B-Instruct)
HF_TOKEN           API key
LOCAL_IMAGE_NAME   Docker image name (used by from_docker_image)
ENV_BASE_URL       Server URL when not using Docker (default: localhost:8000)
TASK_SEED          Seed for task generation and environment noise (default: 42)

STDOUT format (mandatory)
--------------------------
[START] task=<task> env=<benchmark> model=<model>
[STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Score formula (see tasks.compute_final_score)
---------------------------------------------
    final_score = (0.6 * mean_sku(sum_days(demand_fulfilled_today))
                 + 0.4 * sum_days(1 - waste_fraction_today)) / no_of_days

    waste_fraction_today is the waste_penalty-weighted share of the day's
    deliveries rejected because storage was full.

Success = score >= SUCCESS_THRESHOLD (0.80)

Action history
--------------
Every LLM call receives the recent (day, action, feedback) history from the
current episode, where feedback summarises the fill rate per SKU and any
overflow waste caused by the previous action.
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from client import PharmaEnvClient
from models import EpisodeConfig, InventoryState, PharmaAction
from tasks import TASK_REGISTRY, compute_final_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME        = os.getenv("LOCAL_IMAGE_NAME")
API_KEY           = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL      = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME        = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_SEED         = int(os.getenv("TASK_SEED", "42"))

BENCHMARK         = "pharma_coldchain_env"
TEMPERATURE       = 0.2
MAX_TOKENS        = 256         # orders can be multi-SKU JSON
SUCCESS_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a pharmaceutical warehouse procurement agent managing a cold-chain inventory over a multi-day episode.

    OBJECTIVE
    ---------
    Every day you receive an inventory report and decide what to order.
    Maximise prescription fill rate while avoiding stockouts, waste, and capacity overflow.

    WAREHOUSE
    ---------
    Two storage pools:
      cold_storage  — refrigerated, for cold-chain SKUs only (cold_storage_required = True).
      ambient       — standard storage, for all other SKUs.
    Do not order quantities that would exceed available capacity.
    
    SKUS
    ----
    Each SKU exposes:
      inventory_on_hand            — units physically on shelf right now.
      avg_demand_per_day           — rolling daily demand average. Use for reorder sizing.
      avg_demand_last_5_days       — recent demand average. Reacts faster to spikes.
      avg_lead_time                — average days from order placement to arrival.
      lead_time_last3_orders       — actual lead times of last 3 deliveries. Rising = supply stress.
      expected_inbound_orders      — your open orders: (sku_id, quantity, expected_arrival_day).
      stockout_days_if_no_reorder  — days until stockout at the current demand rate.
      stockout_penalty             — criticality of this SKU. Higher = order first.
      waste_penalty                — cost of this SKU's deliveries overflowing storage. Higher = avoid over-ordering it.
      cold_storage_required        — True means this SKU uses the cold storage pool.

    DECISION RULES
    --------------
    1. Order before stockout_days_if_no_reorder drops below avg_lead_time.
    2. Higher stockout_penalty SKUs take priority when storage capacity is tight.
    3. expected_inbound_orders may be inaccurate — true lead times deviate from avg_lead_time.
    4. Deliveries that do not fit in storage are wasted. The penalty is the share of the
       delivery lost, weighted by SKU (insulin waste costs the most) — do not over-order.
    5. Use avg_demand_last_5_days to detect short-term demand spikes.

    PRIORITY ORDER
    --------------
    Higher stockout_penalty = higher priority.
    insulin (100) > bp_medication (60) > hydroxychloroquine (35) > paracetamol (20) > vitamins (5)

    ACTION FORMAT
    -------------
    Respond with a JSON object mapping SKU names to order quantities. No extra text.

    To place orders:
        {"insulin": 100, "paracetamol": 500}

    To order nothing today:
        {}

    Rules:
    - Only include SKUs you want to order today.
    - Quantities must be positive numbers.
    - Use exact SKU names as shown in the inventory report.
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
    error_val = " ".join(error.split()) if error else "null"
    # The log format is line-based: collapse any newlines/indentation from the LLM.
    action = " ".join(action.split()) or "{}"
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

def observation_to_dict(obs: InventoryState) -> Dict[str, Any]:
    """
    Convert InventoryState to a clean dict for the LLM prompt.
    Shows only Layer 2 (insights) and Layer 3 (real state).
    Layer 1 (hidden) is never included.
    """
    return {
        "day":  obs.current_date,
        "storage": {
            "cold_total":    obs.cold_storage_total_capacity,
            "cold_used":     round(obs.cold_storage_current_capacity, 1),
            "cold_ratio":    round(obs.cold_storage_ratio, 3),
            "ambient_total": obs.ambient_storage_total_capacity,
            "ambient_used":  round(obs.ambient_storage_current_capacity, 1),
            "ambient_ratio": round(obs.ambient_storage_ratio, 3),
        },
        "expected_inbound_orders": obs.expected_inbound_orders,
        "inventory": {
            sku_id: {
                "inventory_on_hand":            round(sku.inventory_on_hand, 1),
                "avg_demand_per_day":           round(sku.avg_demand_per_day, 2),
                "avg_demand_last_5_days":       round(sku.avg_demand_last_5_days, 2),
                "avg_lead_time":                round(sku.avg_lead_time, 1),
                "lead_time_last3_orders":       sku.lead_time_last3_orders,
                "stockout_days_if_no_reorder":  round(sku.stockout_days_if_no_reorder, 1),
                "cold_storage_required":        sku.cold_storage_required,
                "stockout_penalty":             sku.stockout_penalty,
                "waste_penalty":                sku.waste_penalty,
            }
            for sku_id, sku in obs.skus.items()
        },
    }


def build_feedback(obs: InventoryState) -> str:
    """Summarise the outcome of the last step for the agent's action history."""
    fill = {
        sku_id: round(sku.demand_fulfilled_today, 2)
        for sku_id, sku in obs.skus.items()
    }
    stockouts = [sku_id for sku_id, f in fill.items() if f < 1.0]
    return json.dumps({
        "fill_rate": fill,
        "short_skus": stockouts,
        "units_wasted_overflow": round(obs.inventory_excess_today, 1),
        "waste_fraction": round(obs.waste_fraction_today, 3),
        "reward": round(obs.reward or 0.0, 3),
    })


def build_action_history_str(
    action_history: List[Tuple[int, str, str]],
    max_entries: int = 10,
) -> str:
    """
    Build a compact JSON string of recent action history.

    Each entry is [day, action_taken, environment_feedback].
    Truncated to the most recent max_entries to keep context window manageable.
    The agent uses this to:
      - See what it ordered and when
      - See what arrived and what was rejected
      - Detect patterns (e.g. repeated stockouts → should have ordered more)
      - Avoid re-ordering something that is already in the inbound pipeline

    Parameters
    ----------
    action_history : list of (day, action_str, feedback_str)
    max_entries    : maximum number of past steps to include

    Returns a formatted string block, or empty string if no history yet.
    """
    if not action_history:
        return ""

    recent = action_history[-max_entries:]
    entries = [
        {"day": day, "action": action, "feedback": feedback}
        for day, action, feedback in recent
    ]
    return json.dumps(entries, indent=2)


def build_user_prompt(
    obs_dict: Dict[str, Any],
    action_history: List[Tuple[int, str, str]],
    step: int,
) -> str:
    """
    Build the user-turn message from:
      1. The current inventory report (today's state)
      2. The action history (what has happened so far this episode)

    The history gives the agent memory across steps without requiring
    the LLM to maintain its own internal state.
    """
    state_json   = json.dumps(obs_dict, indent=2)
    history_str  = build_action_history_str(action_history)

    history_block = ""
    if history_str:
        history_block = textwrap.dedent(f"""
            --- RECENT ACTION HISTORY (last {min(10, len(action_history))} days) ---
            {history_str}

        """).strip() + "\n\n"

    return textwrap.dedent(f"""
        Day {obs_dict['day']} — Step {step}

        {history_block}--- TODAY'S INVENTORY REPORT ---
        {state_json}

        --- YOUR PROCUREMENT DECISION ---
        Respond with a single JSON object on one line.
        To order: {{"sku_name": quantity, ...}}
        To skip:  {{}}
    """).strip()

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_llm_action(
    client: OpenAI,
    obs_dict: Dict[str, Any],
    action_history: List[Tuple[int, str, str]],
    step: int,
) -> str:
    """
    Call the LLM with the current inventory report and full action history.
    Returns the raw action string. Falls back to '{}' (no order) on error.
    """
    user_prompt = build_user_prompt(obs_dict, action_history, step)
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
        return text if text else "{}"
    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        return "{}"

# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

async def run_episode(
    client: OpenAI,
    task_name: str,
    episode_config: EpisodeConfig,
    seed: int,
) -> None:
    """
    Run one full episode for a single task config.

    Parameters
    ----------
    client         : OpenAI        — initialised LLM client
    task_name      : str           — task name used in the log lines
    episode_config : EpisodeConfig — the task definition sent to the server
    seed           : int           — seed for the environment's demand/lead-time noise

    Emits mandatory [START] / [STEP] / [END] log lines to STDOUT.

    Action history
    --------------
    Maintained as a list of (day, action_str, feedback_str) tuples.
    Passed to every LLM call so the agent has full memory of the episode.
    Capped at last 10 entries in the prompt to keep context manageable,
    but the full list is retained internally for scoring and debugging.
    """
    if IMAGE_NAME:
        env = await PharmaEnvClient.from_docker_image(IMAGE_NAME)
    else:
        base_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
        env = PharmaEnvClient(base_url=base_url)

    # Per-episode state
    rewards:        List[float]                  = []
    action_history: List[Tuple[int, str, str]]   = []
    steps_taken  = 0
    score        = 0.01
    success      = False
    last_obs     = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # -- Reset with this task's config ----------------------------------
        # The config is sent as plain JSON; the server re-validates it.
        result   = await env.reset(
            episode_config=episode_config.model_dump(mode="json"),
            seed=seed,
        )
        last_obs = result.observation

        # -- Step loop ------------------------------------------------------
        for step in range(1, episode_config.no_of_days + 1):

            if result.done:
                break

            # Build observation dict (Layer 2 + 3 only — no hidden state)
            obs_dict = observation_to_dict(last_obs)

            # Get LLM action — passes full action history for context
            raw_action = get_llm_action(
                client         = client,
                obs_dict       = obs_dict,
                action_history = action_history,
                step           = step,
            )

            # Step the environment
            action = PharmaAction(message=raw_action)
            result = await env.step(action)

            reward   = result.reward if result.reward is not None else 0.0
            done     = result.done
            last_obs = result.observation

            # Record this step in action history
            # Stores: (day_index, what_agent_decided, what_environment_reported)
            action_history.append((
                obs_dict["day"],
                raw_action,
                build_feedback(last_obs),
            ))

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=raw_action, reward=reward, done=done, error=None)

            if done:
                break

        # -- Final score ----------------------------------------------------
        if last_obs is not None:
            score = compute_final_score( last_obs,episode_config)

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error in {task_name}: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

        log_end(
            success = success,
            steps   = steps_taken,
            score   = score,
            rewards = rewards,
        )

# ---------------------------------------------------------------------------
# Main — run all tasks in sequence
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Entry point.

    Runs every task in TASK_REGISTRY in sequence:
        supply_chain_broken
        flu_season
        epidemic_two_wave

    Each task is a separate 60-day episode with its own demand curves,
    supplier stress schedule, and starting inventory. Tasks are built with
    TASK_SEED so runs are reproducible.
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_name, task_fn in TASK_REGISTRY.items():
        try:
            episode_config = task_fn(TASK_SEED)
            await run_episode(
                client=client,
                task_name=task_name,
                episode_config=episode_config,
                seed=TASK_SEED,
            )
        except Exception as exc:
            print(f"[DEBUG] Task {task_name} failed entirely: {exc}", flush=True)
            log_end(success=False, steps=0, score=0.01, rewards=[])


if __name__ == "__main__":
    asyncio.run(main())