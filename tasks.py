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
# waste_penalty weights each SKU's share of overflow waste in the reward
# (see PharmaEnvironment._update_inventory_excess); only ratios matter.
# Insulin is the cold-chain SKU (refrigerated pool); the other four use ambient storage.

SKU_CATALOGUE: Dict[str, Dict] = {
    "insulin": {
        "cold_storage_required": True,
        "stockout_penalty":      100.0,
        "waste_penalty":         10.0,
        "base_demand":           (8, 12),
        "base_lead_time":        (4, 6),
        "starting_inventory":    (1, 5),
    },
    "paracetamol": {
        "cold_storage_required": False,
        "stockout_penalty":      20.0,
        "waste_penalty":         1.0,
        "base_demand":           (180, 220),
        "base_lead_time":        (1, 3),
        "starting_inventory":    (1, 5),
    },
    "bp_medication": {
        "cold_storage_required": False,
        "stockout_penalty":      60.0,
        "waste_penalty":         3.0,
        "base_demand":           (40, 60),
        "base_lead_time":        (2, 4),
        "starting_inventory":    (1, 5),
    },
    "vitamins": {
        "cold_storage_required": False,
        "stockout_penalty":      5.0,
        "waste_penalty":         0.5,
        "base_demand":           (60, 90),
        "base_lead_time":        (1, 2),
        "starting_inventory":    (1, 5),
    },
    "hydroxychloroquine": {
        "cold_storage_required": False,
        "stockout_penalty":      35.0,
        "waste_penalty":         4.0,
        "base_demand":           (20, 40),
        "base_lead_time":        (5, 8),
        "starting_inventory":    (1, 5),
    },
}

EPISODE_DAYS = 60
# Each storage pool holds this many days of its SKUs' combined base demand.
# 21 days leaves ~25% headroom over the worst case a perfect-foresight plan
# needs on any task (about 17 days in flu_season, 12 days for insulin), so
# good play always fits while hoarding overflows.
AMBIENT_STORAGE_DAYS = 21.0
COLD_STORAGE_DAYS = 21.0


def _build_episode(
    task_name: str,
    rng: Random,
    demand_curves: Dict[str, List[float]],
    lead_time_curves: Dict[str, List[float]],
    duration: int = EPISODE_DAYS,
    ambient_storage_days: float = AMBIENT_STORAGE_DAYS,
    cold_storage_days: float = COLD_STORAGE_DAYS,
) -> EpisodeConfig:
    """
    Assemble an EpisodeConfig from per-SKU curves, sampling base values from the catalogue.

    Each storage pool's capacity is `*_storage_days` times the summed base demand of
    the SKUs stored in it, so capacity scales with the demand drawn for the episode.
    """
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
            waste_penalty=spec["waste_penalty"],
        )
        initial_inventory[sku_id] = rng.uniform(*spec["starting_inventory"])

    # Computed from demand already drawn above: no extra random draws, so the rest
    # of each seeded episode is unchanged. A pool with no SKUs keeps a token 1 unit
    # (EpisodeConfig requires positive capacities).
    cold_demand = sum(c.base_demand for c in skus.values() if c.cold_storage_required)
    ambient_demand = sum(c.base_demand for c in skus.values() if not c.cold_storage_required)

    return EpisodeConfig(
        task_name=task_name,
        no_of_days=duration,
        skus=skus,
        initial_inventory=initial_inventory,
        cold_storage_total_capacity=max(cold_storage_days * cold_demand, 1.0),
        ambient_storage_total_capacity=max(ambient_storage_days * ambient_demand, 1.0),
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
    step_reward = mean(demand_fulfilled_today over SKUs) - waste_fraction_today

    waste_fraction_today is the waste_penalty-weighted share of today's
    deliveries rejected for lack of storage, in [0, 1]. Range is [-1, 1].
    """
    ts = inventory_state
    if not ts.skus:
        return 0.0
    demand_fulfilled = sum(s.demand_fulfilled_today for s in ts.skus.values()) / len(ts.skus)
    return demand_fulfilled - ts.waste_fraction_today


def average_step_reward(inventory_state: InventoryState, days: int) -> float:
    """
    Sum of the step rewards earned so far divided by `days`, clipped to [0, 1].

    Rebuilt from the environment's running totals, no extra state needed:
      sum of mean fill over days  = mean over SKUs of demand_fulfilled_cumulative
      sum of waste_fraction_today = days played - inventory_excess_cumulative
                                    (that total adds 1 - waste_fraction_today each day)
    """
    ts = inventory_state
    if days <= 0 or not ts.skus:
        return 0.0
    fulfilled = sum(s.demand_fulfilled_cumulative for s in ts.skus.values()) / len(ts.skus)
    wasted = ts.current_date - ts.inventory_excess_cumulative
    return min(1.0, max(0.0, (fulfilled - wasted) / days))


def compute_final_score(
    inventory_state: InventoryState,
    episode_config: EpisodeConfig,
) -> float:
    """
    final_score = clip(sum of step rewards / no_of_days, 0, 1)
                = clip(mean over days of (mean fill - waste_fraction_today), 0, 1)

    The episode score is the average daily reward, so an agent is evaluated on
    exactly what it is rewarded for each step. Ordering nothing scores ~0.
    Days not played (an episode cut short) count as 0.
    """
    return average_step_reward(inventory_state, episode_config.no_of_days)
