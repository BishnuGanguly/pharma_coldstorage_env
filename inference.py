"""
inference.py
============
Baseline inference script for the Pharma Cold-Chain Inventory Management Environment.

Runs all 9 competition tasks (3 scenarios × 3 difficulties) in sequence.
Each task is a separate 60-day episode defined in tasks.py via TASK_CONFIGS.

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
    final_score = 0.5 * mean(step_rewards)
                + 0.3 * prescription_fill_rate_30d
                + 0.2 * (1 - total_backorder_ratio)

Success = score >= SUCCESS_THRESHOLD (0.80)

Action history
--------------
Every LLM call receives the full history of (day, action, feedback) tuples
from the current episode as a compact JSON array in the user prompt.
This gives the agent memory of what it ordered, what arrived, and what
feedback the environment returned — without bloating the context window
with full state snapshots at every prior step.
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from client import PipelineEnvClient
    from models import InventoryState, PharmaAction
    from tasks import TASK_CONFIGS, compute_final_score
except ImportError:
    from client import PipelineEnvClient
    from models import InventoryState, PharmaAction
    from tasks import TASK_CONFIGS, compute_final_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME        = os.getenv("LOCAL_IMAGE_NAME")
API_KEY           = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL      = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME        = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK         = "pharma_coldchain_env"
MAX_STEPS         = 60          # one step per episode day
TEMPERATURE       = 0.2
MAX_TOKENS        = 256         # orders can be multi-SKU JSON
SUCCESS_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an intelligent pharmaceutical warehouse procurement agent.

    OBJECTIVE
    ---------
    Manage a cold-chain warehouse stocking four drugs over a 60-day episode.
    Every day you receive an inventory report and decide what to order.
    Maximise prescription fill rate while minimising stockouts, waste, and budget overrun.

    WAREHOUSE
    ---------
    Two storage pools:
      cold_storage  — refrigerated, scarce, for insulin only.
      ambient       — standard storage, for all other drugs.

    You cannot place an order if the relevant storage pool is at capacity.
    You cannot place an order if procurement_budget_ratio is near 0.

    SKUS
    ----
    Each SKU has:
      inventory_on_hand        — units physically on shelf right now.
      inbound_expected_3d      — units you ordered, expected within 3 days (estimate only).
      inbound_expected_7d      — units you ordered, expected within 7 days (estimate only).
      demand_last_3d           — actual demand observed over last 3 days.
      demand_last_7d           — actual demand observed over last 7 days.
      demand_trend             — (demand_last_3d/3) - (demand_last_7d/7). Positive = accelerating.
      stockout_days_if_no_reorder — days until stockout if you order nothing today.
      coverage_gap_7d          — demand_last_7d - inventory_on_hand. Positive = shortage incoming.
      stockout_penalty         — cost of failing this SKU. Higher = more critical.
      substitute_coverage_ratio — fraction of demand coverable by substitute. 0.0 = no substitute.

    SUPPLIERS
    ---------
    Each supplier has:
      last_observed_lead_time  — days from order to arrival (from most recent delivery).
      on_time_rate_14d         — fraction of recent orders delivered on time.
      disruption_active        — True if supplier is currently disrupted.
      sku_served               — list of SKUs this supplier can fulfill.
      cold_chain_certified     — must be True to supply insulin.
      unit_cost                — relative cost. Higher = more expensive.
      expedite_allowed         — True if emergency fast delivery is possible.

    CRITICAL RULES
    --------------
    1. Insulin can ONLY be ordered from cold_chain_certified suppliers (FastPharma only).
    2. A disrupted supplier cannot receive orders — they will be rejected.
    3. If cold_chain_integrity_flag = False, insulin is destroyed — order immediately.
    4. If orders_overdue_count > 0, inbound_expected figures are unreliable.
    5. If epidemic_alert_flag = True, a demand spike is imminent — pre-stock now.

    PRIORITY ORDER (when budget is tight)
    --------------------------------------
    insulin (no substitute, life-critical)
    > bp_medication (chronic patients, serious if missed)
    > paracetamol (substitute partially available)
    > vitamins (fully substitutable, lowest penalty)

    ACTION FORMAT (FOLLOW EXACT SYNTAX)
    -------------------------------------
    Respond with a single JSON object on one line. No extra text.

    To place orders:
        {"insulin": [100, "FastPharma"], "paracetamol": [500, "GlobalMed"]}

    To order nothing today:
        {}

    Rules:
    - Only include SKUs you want to order today.
    - quantity must be a positive number.
    - Use exact SKU names and supplier names as shown in the inventory report.
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

def observation_to_dict(obs: InventoryState) -> Dict[str, Any]:
    """
    Convert InventoryState to a clean dict for the LLM prompt.
    Shows only Layer 2 (insights) and Layer 3 (real state).
    Layer 1 (hidden) is never included.
    """
    return {
        "day":          obs.current_date,
        "season_phase": round(obs.season_phase, 3),

        "service": {
            "prescription_fill_rate_30d":      round(obs.prescription_fill_rate_30d, 3),
            "prescription_fill_rate_7d":       round(obs.prescription_fill_rate_7d, 3),
            "epidemic_alert_flag":             obs.epidemic_alert_flag,
            "days_since_epidemic_alert_fired": obs.days_since_epidemic_alert_fired,
            "supply_disruption_days_last_30d": obs.supply_disruption_days_last_30d,
        },

        "warehouse": {
            "procurement_budget_ratio":         round(obs.procurement_budget_ratio, 3),
            "cold_storage_capacity_ratio":      round(obs.cold_storage_capacity_ratio, 3),
            "cold_storage_total_capacity":      obs.cold_storage_total_capacity,
            "cold_storage_current_capacity":    round(obs.cold_storage_current_capacity, 1),
            "ambient_capacity_ratio":           round(obs.ambient_capacity_ratio, 3),
            "ambient_storage_total_capacity":   obs.ambient_storage_total_capacity,
            "ambient_storage_current_capacity": round(obs.ambient_storage_current_capacity, 1),
            "cold_chain_integrity_flag":        obs.cold_chain_integrity_flag,
            "orders_overdue_count":             obs.orders_overdue_count,
            "overdue_qty_total":                round(obs.overdue_qty_total, 1),
        },

        "inventory": {
            sku_id: {
                "name":                        sku.name,
                "inventory_on_hand":           round(sku.inventory_on_hand, 1),
                "backorders":                  round(sku.backorders, 1),
                "inbound_expected_3d":         round(sku.inbound_expected_3d, 1),
                "inbound_expected_7d":         round(sku.inbound_expected_7d, 1),
                "cold_storage_required":       sku.cold_storage_required,
                "demand_last_3d":              round(sku.demand_last_3d, 1),
                "demand_last_7d":              round(sku.demand_last_7d, 1),
                "demand_trend":                round(sku.demand_trend, 3),
                "stockout_days_if_no_reorder": round(sku.stockout_days_if_no_reorder, 1),
                "coverage_gap_7d":             round(sku.coverage_gap_7d, 1),
                "stockout_penalty":            sku.stockout_penalty,
                "substitute_coverage_ratio":   sku.substitute_coverage_ratio,
            }
            for sku_id, sku in obs.skus.items()
        },

        "suppliers": {
            sup_id: {
                "last_observed_lead_time": round(sup.last_observed_lead_time, 1),
                "on_time_rate_14d":        round(sup.on_time_rate_14d, 3),
                "disruption_active":       sup.disruption_active,
                "sku_served":              sup.sku_served,
                "cold_chain_certified":    sup.cold_chain_certified,
                "unit_cost":               sup.unit_cost,
                "expedite_allowed":        sup.expedite_allowed,
            }
            for sup_id, sup in obs.suppliers.items()
        },

        "last_action_feedback": obs.last_action_feedback,
    }


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
        To order: {{"sku_name": [quantity, "supplier_name"], ...}}
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
    task_config: Dict[str, Any],
) -> None:
    """
    Run one full 60-day episode for a single task config.

    Parameters
    ----------
    client      : OpenAI — initialised LLM client
    task_config : dict   — task config from tasks.py (one of TASK_CONFIGS)

    Emits mandatory [START] / [STEP] / [END] log lines to STDOUT.

    Action history
    --------------
    Maintained as a list of (day, action_str, feedback_str) tuples.
    Passed to every LLM call so the agent has full memory of the episode.
    Capped at last 10 entries in the prompt to keep context manageable,
    but the full list is retained internally for scoring and debugging.
    """
    task_name = f"{task_config['task_id']}_{task_config['difficulty']}"

    if IMAGE_NAME:
        env = await PipelineEnvClient.from_docker_image(IMAGE_NAME)
    else:
        base_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
        env = PipelineEnvClient(base_url=base_url)

    # Per-episode state
    rewards:        List[float]                  = []
    action_history: List[Tuple[int, str, str]]   = []
    steps_taken  = 0
    score        = 0.0
    success      = False
    last_obs     = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # -- Reset with this task's config ----------------------------------
        result   = await env.reset(task_config=task_config)
        last_obs = result.observation

        # -- Step loop ------------------------------------------------------
        for step in range(1, MAX_STEPS + 1):

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
                last_obs.last_action_feedback,
            ))

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=raw_action, reward=reward, done=done, error=None)

            if done:
                break

        # -- Final score ----------------------------------------------------
        if last_obs is not None:
            score = compute_final_score(rewards, last_obs)

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
# Main — run all 9 tasks in sequence
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Entry point.

    Runs all 9 task configs in sequence:
        supply_chain_broken  (easy → medium → hard)
        flu_season           (easy → medium → hard)
        epidemic_two_wave    (easy → medium → hard)

    Each task is a separate 60-day episode with its own demand curves,
    supplier stress schedule, and starting inventory.
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_config in TASK_CONFIGS:
        await run_episode(client=client, task_config=task_config)


if __name__ == "__main__":
    asyncio.run(main())