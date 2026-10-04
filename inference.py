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
    final_score = clip(sum_days(step_reward) / no_of_days, 0, 1)
    step_reward = mean_sku(demand_fulfilled_today) - waste_fraction_today

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
    Each SKU in the report shows:
      inventory_on_hand            — units physically on shelf right now.
      inbound_units                — units you have already ordered that have not arrived yet.
      days_of_cover                — (inventory_on_hand + inbound_units) / recent daily demand:
                                     how many days your stock plus open orders will last.
                                     null until any demand has been observed.
      avg_demand_per_day           — daily demand averaged over the whole episode so far.
      avg_demand_last_5_days       — recent daily demand. Reacts faster to spikes.
      avg_lead_time                — average days from order placement to arrival.
      lead_time_last3_orders       — actual lead times of the last 3 deliveries. Rising = supply stress.
      stockout_days_if_no_reorder  — days until the shelf is empty, ignoring open orders.
      stockout_penalty             — how critical this SKU is. Higher = protect it first.
      waste_penalty                — cost of this SKU's deliveries overflowing storage.
      cold_storage_required        — true means this SKU uses the cold storage pool.

    DECISION RULES
    --------------
    1. Decide for EVERY SKU in the report, every day. Each one needs its own quantity.
    2. A SKU needs an order when its days_of_cover is below avg_lead_time plus a safety
       margin of about 3 days. If days_of_cover is comfortably above that, do not order it.
    3. Size an order to bring days_of_cover back up to about avg_lead_time + 3 days:
       roughly avg_demand_last_5_days x (avg_lead_time + 3) - inventory_on_hand - inbound_units.
    4. If lead_time_last3_orders is rising, suppliers are slowing down: use the larger
       lead time, and order earlier.
    5. Deliveries that do not fit in storage are wasted, and wasted cold-storage (insulin)
       deliveries cost the most. Never order far more than a SKU needs.
    6. When storage is tight, protect SKUs with a higher stockout_penalty first.

    ACTION FORMAT
    -------------
    Respond with one JSON object mapping SKU names to the number of units to order today.
    Use the exact SKU names from the report and compute each quantity from the report.
    Leave out SKUs that do not need an order. If nothing needs ordering, respond with {}.

    Format:
        {"<sku_name>": <units>, "<sku_name>": <units>}
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

def inbound_units_by_sku(obs: InventoryState) -> Dict[str, float]:
    """Units ordered but not yet arrived, per SKU (from the agent-visible expected orders)."""
    inbound = {sku_id: 0.0 for sku_id in obs.skus}
    for sku_id, qty, _ in obs.expected_inbound_orders:
        inbound[sku_id] = inbound.get(sku_id, 0.0) + qty
    return inbound


def days_of_cover(on_hand: float, inbound: float, recent_demand: float) -> Optional[float]:
    """How many days stock plus open orders last at the recent demand rate; None if no demand seen yet."""
    if recent_demand <= 0:
        return None
    return round((on_hand + inbound) / recent_demand, 1)


def observation_to_dict(obs: InventoryState) -> Dict[str, Any]:
    """
    Convert InventoryState to a clean dict for the LLM prompt.
    Shows only what the agent may observe, plus two values computed from it
    (inbound_units, days_of_cover) so a model does not have to do that
    arithmetic itself. Hidden ground truth is never included.
    """
    inbound = inbound_units_by_sku(obs)
    return {
        "day":  obs.current_date,
        "storage": {
            "cold_total":    round(obs.cold_storage_total_capacity, 1),
            "cold_used":     round(obs.cold_storage_current_capacity, 1),
            "cold_ratio":    round(obs.cold_storage_ratio, 3),
            "ambient_total": round(obs.ambient_storage_total_capacity, 1),
            "ambient_used":  round(obs.ambient_storage_current_capacity, 1),
            "ambient_ratio": round(obs.ambient_storage_ratio, 3),
        },
        "expected_inbound_orders": obs.expected_inbound_orders,
        "inventory": {
            sku_id: {
                "inventory_on_hand":            round(sku.inventory_on_hand, 1),
                "inbound_units":                round(inbound[sku_id], 1),
                "days_of_cover":                days_of_cover(
                    sku.inventory_on_hand, inbound[sku_id],
                    max(sku.avg_demand_last_5_days, sku.avg_demand_per_day),
                ),
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


def format_report(obs_dict: Dict[str, Any]) -> str:
    """
    Render the report as valid JSON with one line per section and per SKU, which
    reads well and uses far fewer tokens than fully indented JSON.
    """
    def compact(value: Any) -> str:
        return json.dumps(value, separators=(", ", ": "))

    lines = ["{"]
    keys = list(obs_dict)
    for i, key in enumerate(keys):
        comma = "," if i < len(keys) - 1 else ""
        value = obs_dict[key]
        if isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
            lines.append(f'  "{key}": {{')
            items = list(value.items())
            for j, (name, entry) in enumerate(items):
                lines.append(f'    "{name}": {compact(entry)}' + ("," if j < len(items) - 1 else ""))
            lines.append("  }" + comma)
        else:
            lines.append(f'  "{key}": {compact(value)}{comma}')
    lines.append("}")
    return "\n".join(lines)


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


HISTORY_DAYS = 3


def build_action_history_str(
    action_history: List[Tuple[int, str, str]],
    max_entries: int = HISTORY_DAYS,
) -> str:
    """
    Build a compact JSON string of the results of the last few days.

    action_history holds (day, action_str, feedback_str) tuples. Only the day and
    its results (fill rates, short SKUs, waste, reward) are shown, never the
    model's own earlier replies: small models tend to copy those and repeat the
    same order every day. What is already on order is visible in the report
    (inbound_units), so the raw replies are not needed.

    Returns a formatted string block, or an empty string if there is no history yet.
    """
    if not action_history:
        return ""

    entries = []
    for day, _action, feedback in action_history[-max_entries:]:
        try:
            result = json.loads(feedback)
        except (TypeError, ValueError):
            result = feedback
        entries.append({"day": day, "result": result})
    return json.dumps(entries)


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
    parts = [f"Day {obs_dict['day']} — Step {step}"]
    history_str = build_action_history_str(action_history)
    if history_str:
        n_days = min(HISTORY_DAYS, len(action_history))
        parts.append(f"--- RESULTS OF THE LAST {n_days} DAY{'S' if n_days != 1 else ''} ---\n{history_str}")
    parts.append(f"--- TODAY'S INVENTORY REPORT ---\n{format_report(obs_dict)}")
    parts.append(
        "--- YOUR PROCUREMENT DECISION ---\n"
        "Go through every SKU, then respond with a single JSON object on one line:\n"
        '{"<sku_name>": <units>, ...} for the SKUs that need an order, or {} if none do.'
    )
    return "\n\n".join(parts)

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