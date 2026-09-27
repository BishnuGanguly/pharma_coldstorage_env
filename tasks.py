from __future__ import annotations

from math import exp
from random import Random
from typing import Callable, Dict, List, Optional

from models import EpisodeConfig, InventoryState, SKUEpisodeConfig


# ---------------------------------------------------------------------------
# Curve builders
# ---------------------------------------------------------------------------

def gaussian_curve(peak_day: int, amplitude: float, width: float, duration: int) -> List[float]:
    return [
        1.0 + amplitude * exp(-0.5 * ((day - peak_day) / width) ** 2)
        for day in range(duration)
    ]


def flat_curve(duration: int, value: float = 1.0) -> List[float]:
    return [value] * duration


def step_curve(base_curve: List[float], start_day: int, end_day: int, amplitude: float, duration: int) -> List[float]:
    return [
        amplitude if start_day <= day <= end_day else base_curve[day]
        for day in range(duration)
    ]


def combine_curves(curve_a: List[float], curve_b: List[float]) -> List[float]:
    return [a * b for a, b in zip(curve_a, curve_b)]


# ---------------------------------------------------------------------------
# SKU catalogue
# ---------------------------------------------------------------------------
# Ranges are (low, high) and are sampled once per task build, so every
# episode built with the same seed is identical.
# All five SKUs currently use ambient storage; cold storage is reserved for
# future cold-chain SKUs.

SKU_CATALOGUE: Dict[str, Dict] = {
    "insulin": {
        "cold_storage_required": False,
        "stockout_penalty":      100.0,
        "base_demand":           (8, 12),
        "base_lead_time":        (4, 6),
        "starting_inventory":    (1, 5),
    },
    "paracetamol": {
        "cold_storage_required": False,
        "stockout_penalty":      20.0,
        "base_demand":           (180, 220),
        "base_lead_time":        (1, 3),
        "starting_inventory":    (1, 5),
    },
    "bp_medication": {
        "cold_storage_required": False,
        "stockout_penalty":      60.0,
        "base_demand":           (40, 60),
        "base_lead_time":        (2, 4),
        "starting_inventory":    (1, 5),
    },
    "vitamins": {
        "cold_storage_required": False,
        "stockout_penalty":      5.0,
        "base_demand":           (60, 90),
        "base_lead_time":        (1, 2),
        "starting_inventory":    (1, 5),
    },
    "hydroxychloroquine": {
        "cold_storage_required": False,
        "stockout_penalty":      35.0,
        "base_demand":           (20, 40),
        "base_lead_time":        (5, 8),
        "starting_inventory":    (1, 5),
    },
}

EPISODE_DAYS = 60
COLD_STORAGE_CAPACITY = 500.0
AMBIENT_STORAGE_CAPACITY = 25000.0


def _build_episode(
    task_name: str,
    rng: Random,
    demand_curves: Dict[str, List[float]],
    lead_time_curves: Dict[str, List[float]],
    duration: int = EPISODE_DAYS,
) -> EpisodeConfig:
    """Assemble an EpisodeConfig from per-SKU curves, sampling base values from the catalogue."""
    skus: Dict[str, SKUEpisodeConfig] = {}
    initial_inventory: Dict[str, float] = {}
    for sku_id, spec in SKU_CATALOGUE.items():
        skus[sku_id] = SKUEpisodeConfig(
            sku_id=sku_id,
            no_of_days=duration,
            base_demand=rng.uniform(*spec["base_demand"]),
            demand_curve=demand_curves.get(sku_id) or flat_curve(duration),
            base_lead_time=rng.uniform(*spec["base_lead_time"]),
            lead_time_curve=lead_time_curves.get(sku_id) or flat_curve(duration),
            cold_storage_required=spec["cold_storage_required"],
            stockout_penalty=spec["stockout_penalty"],
        )
        initial_inventory[sku_id] = rng.uniform(*spec["starting_inventory"])

    return EpisodeConfig(
        task_name=task_name,
        no_of_days=duration,
        skus=skus,
        initial_inventory=initial_inventory,
        cold_storage_total_capacity=COLD_STORAGE_CAPACITY,
        ambient_storage_total_capacity=AMBIENT_STORAGE_CAPACITY,
    )


# ---------------------------------------------------------------------------
# Task 1 — Supply Chain Broken (60 days)
# ---------------------------------------------------------------------------

def supply_chain_broken(seed: Optional[int] = None) -> EpisodeConfig:
    """
    Flat demand; lead times step up for insulin (days 15-25) and then for
    paracetamol (days 30-40). The agent must build safety stock before each
    disruption, even though shelves look fine at the time.
    """
    rng = Random(seed)
    D = EPISODE_DAYS
    lead_time_curves = {
        "insulin":     step_curve(flat_curve(D), 15, 25, rng.uniform(2.0, 5.0), D),
        "paracetamol": step_curve(flat_curve(D), 30, 40, rng.uniform(2.0, 5.0), D),
    }
    return _build_episode("supply_chain_broken", rng, {}, lead_time_curves)


# ---------------------------------------------------------------------------
# Task 2 — Flu Season (60 days)
# ---------------------------------------------------------------------------

def task_flu_season(seed: Optional[int] = None) -> EpisodeConfig:
    """
    Paracetamol demand follows a Gaussian flu wave (peak day 30, 2x amplitude)
    while its lead times step up during days 30-40. The agent must pre-stock
    before the peak.
    """
    rng = Random(seed)
    D = EPISODE_DAYS
    demand_curves = {
        "paracetamol": gaussian_curve(peak_day=30, amplitude=2, width=10, duration=D),
    }
    lead_time_curves = {
        "paracetamol": step_curve(flat_curve(D), 30, 40, rng.uniform(2.0, 5.0), D),
    }
    return _build_episode("flu_season", rng, demand_curves, lead_time_curves)


# ---------------------------------------------------------------------------
# Task 3 — Epidemic Two Waves (60 days)
# ---------------------------------------------------------------------------

def task_epidemic(seed: Optional[int] = None) -> EpisodeConfig:
    """
    Two-wave epidemic. Primary SKU: Hydroxychloroquine (HCQ).

    Wave 1 peaks at day 17. Wave 2 peaks at day 45 and is larger.
    Between waves there is a trough — the agent must NOT relax.
    Insulin sees a mild secondary elevation.
    """
    rng = Random(seed)
    D = EPISODE_DAYS
    demand_curves = {
        "hydroxychloroquine": combine_curves(
            gaussian_curve(17, 1.8, 6.0, D),
            gaussian_curve(45, 2.5, 6.0, D),
        ),
        "insulin": gaussian_curve(17 + 3, rng.uniform(0.1, 0.2), 10, D),
    }
    lead_time_curves = {
        "hydroxychloroquine": step_curve(flat_curve(D), 17, 27, rng.uniform(2.0, 5.0), D),
        "insulin":            step_curve(flat_curve(D), 20, 30, rng.uniform(1.0, 3.0), D),
    }
    return _build_episode("epidemic_two_wave", rng, demand_curves, lead_time_curves)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, Callable[[Optional[int]], EpisodeConfig]] = {
    "supply_chain_broken": supply_chain_broken,
    "flu_season":          task_flu_season,
    "epidemic_two_wave":   task_epidemic,
}


def get_task_config(task_name: str, seed: Optional[int] = None) -> EpisodeConfig:
    """Build the EpisodeConfig for a registered task."""
    try:
        task_fn = TASK_REGISTRY[task_name]
    except KeyError:
        raise ValueError(
            f"Unknown task '{task_name}'. Available tasks: {sorted(TASK_REGISTRY)}"
        ) from None
    return task_fn(seed)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_step_reward(inventory_state: InventoryState) -> float:
    """
    step_reward = mean(demand_fulfilled_today over SKUs)
                - inventory_excess_today / (1 + inventory_excess_today)

    Range is (-1, 1]: the waste term is in [0, 1).
    """
    ts = inventory_state
    if not ts.skus:
        return 0.0
    demand_fulfilled = sum(s.demand_fulfilled_today for s in ts.skus.values()) / len(ts.skus)
    excess = ts.inventory_excess_today
    return demand_fulfilled - excess / (1.0 + excess)


def compute_final_score(
    inventory_state: InventoryState,
    episode_config: EpisodeConfig,
) -> float:
    """
    final_score = (0.6 * mean(demand_fulfilled_cumulative over SKUs)
                 + 0.4 * inventory_excess_cumulative) / no_of_days

    inventory_excess_cumulative accumulates 1 / (1 + inventory_excess_today)
    each day inside the environment, so both terms are per-day sums in [0, 1]
    and the final score lies in [0, 1].
    """
    ts = inventory_state
    if not ts.skus:
        return 0.0
    demand_fulfilled = sum(s.demand_fulfilled_cumulative for s in ts.skus.values()) / len(ts.skus)
    return (demand_fulfilled * 0.6 + ts.inventory_excess_cumulative * 0.4) / episode_config.no_of_days
