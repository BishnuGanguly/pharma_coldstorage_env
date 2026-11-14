from __future__ import annotations

from math import ceil, exp, log, sqrt
from random import Random
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from models import EpisodeConfig, InventoryState, SKUEpisodeConfig
from news import add_news
from world import World, WorldError, build_episode, default_world, load_world


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
# SKU catalogue: read from the built-in pharma world (worlds/pharma.yaml)
# ---------------------------------------------------------------------------
# The world file is the single source of the catalogue. The names below keep the
# shape older code expects; ranges are (low, high), sampled once per task build, so
# every episode built with the same seed is identical. waste_penalty weights each
# SKU's share of overflow waste in the reward; only ratios matter.

PHARMA_WORLD: World = default_world()


def _as_range(amount: Any) -> Tuple[float, float]:
    return amount if isinstance(amount, tuple) else (amount, amount)


def _catalogue(world: World) -> Dict[str, Dict]:
    return {
        sku_id: {
            "cold_storage_required": world.storage[p.storage].cold_chain,
            "stockout_penalty":      p.stockout_penalty,
            "waste_penalty":         p.waste_penalty,
            "base_demand":           _as_range(p.base_demand),
            "base_lead_time":        _as_range(p.base_lead_time),
            "starting_inventory":    _as_range(p.starting_inventory),
        }
        for sku_id, p in world.products.items()
    }


SKU_CATALOGUE: Dict[str, Dict] = _catalogue(PHARMA_WORLD)

EPISODE_DAYS = PHARMA_WORLD.days
# Each storage pool holds this many days of its SKUs' combined base demand.
_POOL_DAYS = {pool.cold_chain: pool.capacity_days for pool in PHARMA_WORLD.storage.values()}
AMBIENT_STORAGE_DAYS = _POOL_DAYS[False]
COLD_STORAGE_DAYS = _POOL_DAYS[True]

# Day-to-day randomness. Daily demand varies with a standard deviation of
# DEMAND_NOISE_CV x base demand, and each order's lead time with LEAD_TIME_NOISE_STD
# days. The environment draws this noise once per episode from the seed, so it
# never depends on what the agent does.
DEMAND_NOISE_CV = PHARMA_WORLD.noise.demand_cv
LEAD_TIME_NOISE_STD = PHARMA_WORLD.noise.lead_time_std


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
        base_demand = rng.uniform(*spec["base_demand"])
        skus[sku_id] = SKUEpisodeConfig(
            sku_id=sku_id,
            no_of_days=duration,
            base_demand=base_demand,
            demand_curve=demand_curves.get(sku_id) or flat_curve(duration),
            demand_std=DEMAND_NOISE_CV * base_demand,
            base_lead_time=rng.uniform(*spec["base_lead_time"]),
            lead_time_curve=lead_time_curves.get(sku_id) or flat_curve(duration),
            lead_time_std=LEAD_TIME_NOISE_STD,
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
# Scenarios: fixed or random
# ---------------------------------------------------------------------------
# "fixed": the original tasks. Every seed disrupts the same SKUs on the same days;
#   only the size of each disruption changes.
# "random": per seed, the disrupted SKUs and the disruption days are drawn, and
#   storage holds RANDOM_STORAGE_DAYS of demand instead of 21. With fixed SKUs and
#   21 days of storage, stockpiling a task's usual SKUs every day, without reading
#   the news, earned most of what perfect news reading earns; drawing the SKUs and
#   making stockpiling overflow sooner leaves most of that gain to reading.
# The scenario has its own generator, so a seed's other random values (base demand,
# lead times, starting stock, daily noise, the size of each disruption) are the
# same in both scenarios.
SCENARIOS = ("fixed", "random")
RANDOM_STORAGE_DAYS = 14.0
SKU_IDS = tuple(SKU_CATALOGUE)


def _scenario_rng(task_name: str, seed: Optional[int], scenario: str) -> Optional[Random]:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, not {scenario!r}")
    return Random(f"scenario|{task_name}|{seed}") if scenario == "random" else None


def _storage_days(srng: Optional[Random]) -> Dict[str, float]:
    days = RANDOM_STORAGE_DAYS if srng else AMBIENT_STORAGE_DAYS
    return {"ambient_storage_days": days, "cold_storage_days": days if srng else COLD_STORAGE_DAYS}


def surge_half_width(amplitude: float, width: float) -> float:
    """Days from the peak of a gaussian_curve to where it falls below 1.5x normal."""
    return width * sqrt(2 * log(2 * amplitude))


# ---------------------------------------------------------------------------
# Task 1 — Supply Chain Broken (60 days)
# ---------------------------------------------------------------------------

def supply_chain_broken(seed: Optional[int] = None, scenario: str = "fixed") -> EpisodeConfig:
    """
    Flat demand; lead times step up for insulin (days 15-25) and then for
    paracetamol (days 30-40). The agent must build safety stock before each
    disruption, even though shelves look fine at the time.

    Random scenario: two different SKUs are delayed, drawn from all five. The first
    delay starts on day 10-20 and lasts 9-13 days; the second starts 3-10 days after
    it ends and lasts 9-13 days.
    """
    rng = Random(seed)
    srng = _scenario_rng("supply_chain_broken", seed, scenario)
    D = EPISODE_DAYS
    first, second = "insulin", "paracetamol"
    first_start, first_end, second_start, second_end = 15, 25, 30, 40
    if srng:
        first, second = srng.sample(SKU_IDS, 2)
        first_start = srng.randint(10, 20)
        first_end = first_start + srng.randint(8, 12)
        second_start = min(first_end + srng.randint(3, 10), 45)
        second_end = second_start + srng.randint(8, 12)
    lead_time_curves = {
        first:  step_curve(flat_curve(D), first_start, first_end, rng.uniform(2.0, 5.0), D),
        second: step_curve(flat_curve(D), second_start, second_end, rng.uniform(2.0, 5.0), D),
    }
    return _build_episode("supply_chain_broken", rng, {}, lead_time_curves, **_storage_days(srng))


# ---------------------------------------------------------------------------
# Task 2 — Flu Season (60 days)
# ---------------------------------------------------------------------------

def task_flu_season(seed: Optional[int] = None, scenario: str = "fixed") -> EpisodeConfig:
    """
    Paracetamol demand follows a Gaussian flu wave (peak day 30, 2x amplitude)
    while its lead times step up during days 30-40. The agent must pre-stock
    before the peak.

    Random scenario: the wave hits one SKU drawn from all five. It peaks on day 24-38
    at 2.5-3.5x normal (width 8-12), never reaching 1.5x before day 8; the same SKU's
    delay starts within 3 days of the peak and lasts 11 days.
    """
    rng = Random(seed)
    srng = _scenario_rng("flu_season", seed, scenario)
    D = EPISODE_DAYS
    sku = "paracetamol"
    peak, amplitude, width, delay_start = 30, 2.0, 10.0, 30
    if srng:
        sku = srng.choice(SKU_IDS)
        amplitude = srng.uniform(1.5, 2.5)
        width = srng.uniform(8.0, 12.0)
        earliest_peak = max(24, ceil(8 + surge_half_width(amplitude, width)))
        peak = srng.randint(earliest_peak, max(earliest_peak, 38))
        delay_start = peak + srng.randint(-3, 3)
    demand_curves = {
        sku: gaussian_curve(peak_day=peak, amplitude=amplitude, width=width, duration=D),
    }
    lead_time_curves = {
        sku: step_curve(flat_curve(D), delay_start, delay_start + 10, rng.uniform(2.0, 5.0), D),
    }
    return _build_episode("flu_season", rng, demand_curves, lead_time_curves, **_storage_days(srng))


# ---------------------------------------------------------------------------
# Task 3 — Epidemic Two Waves (60 days)
# ---------------------------------------------------------------------------

def task_epidemic(seed: Optional[int] = None, scenario: str = "fixed") -> EpisodeConfig:
    """
    Two-wave epidemic. Primary SKU: Hydroxychloroquine (HCQ).

    Wave 1 peaks at day 17. Wave 2 peaks at day 45 and is larger.
    Between waves there is a trough — the agent must NOT relax.
    Insulin sees a mild secondary elevation.

    Random scenario: the primary and secondary SKUs (HCQ and insulin above) are two
    different SKUs drawn from all five. Wave 1 peaks on day 13-23 and wave 2 26-32
    days later (by day 52); the two delays keep their place relative to wave 1, each
    shifted by up to 2 days.
    """
    rng = Random(seed)
    srng = _scenario_rng("epidemic_two_wave", seed, scenario)
    D = EPISODE_DAYS
    primary, secondary = "hydroxychloroquine", "insulin"
    wave1, wave2, primary_delay, secondary_delay = 17, 45, 17, 20
    if srng:
        primary, secondary = srng.sample(SKU_IDS, 2)
        wave1 = srng.randint(13, 23)
        # 26+ days apart: closer waves merge into one (the trough stays above 1.5x).
        wave2 = min(wave1 + srng.randint(26, 32), 52)
        primary_delay = wave1 + srng.randint(-2, 2)
        secondary_delay = wave1 + 3 + srng.randint(-2, 2)
    demand_curves = {
        primary: combine_curves(
            gaussian_curve(wave1, 1.8, 6.0, D),
            gaussian_curve(wave2, 2.5, 6.0, D),
        ),
        secondary: gaussian_curve(wave1 + 3, rng.uniform(0.1, 0.2), 10, D),
    }
    lead_time_curves = {
        primary:   step_curve(flat_curve(D), primary_delay, primary_delay + 10, rng.uniform(2.0, 5.0), D),
        secondary: step_curve(flat_curve(D), secondary_delay, secondary_delay + 10, rng.uniform(1.0, 3.0), D),
    }
    return _build_episode("epidemic_two_wave", rng, demand_curves, lead_time_curves, **_storage_days(srng))


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, Callable[..., EpisodeConfig]] = {
    "supply_chain_broken": supply_chain_broken,
    "flu_season":          task_flu_season,
    "epidemic_two_wave":   task_epidemic,
}


# Seed split for evaluation and training. Train (and tune) only on TRAIN_SEEDS;
# report results on TEST_SEEDS, which a trained model must never have seen.
# eval.py always takes the first N seeds of a split, so runs are comparable.
TRAIN_SEEDS = range(0, 10_000)
TEST_SEEDS = range(10_000, 20_000)
SEED_SPLITS = {"train": TRAIN_SEEDS, "test": TEST_SEEDS}


def get_task_config(
    task_name: str, seed: Optional[int] = None, news: int = 0, scenario: str = "fixed",
    world: Union[World, str, Mapping[str, Any], None] = None,
) -> EpisodeConfig:
    """
    Build the EpisodeConfig for a task of a world.

    world: a World, a dict, or a built-in world's name (see world.load_world);
    default the built-in pharma world, whose tasks are TASK_REGISTRY's.
    news: 0 (none, the original tasks), 1 or 2 announces the task's disruptions
    ahead of time (see news.py). News never changes the episode's world.
    scenario: "fixed" (the tasks as the world defines them) or "random" (disrupted
    SKUs and days drawn per seed, less storage; see SCENARIOS). "random" is only
    available for the pharma world for now.
    """
    world = PHARMA_WORLD if world is None else load_world(world)
    if task_name not in world.tasks:
        raise ValueError(
            f"Unknown task '{task_name}' in world '{world.name}'. Available tasks: {sorted(world.tasks)}"
        )
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, not {scenario!r}")
    if scenario == "fixed":
        config = build_episode(world, task_name, seed)
    elif world != PHARMA_WORLD or task_name not in TASK_REGISTRY:
        raise WorldError(f"scenario='random' is only available for the built-in pharma world, not world "
                         f"'{world.name}'. In a world file, a task draws its own products and days per seed: "
                         "give it `draw` names and day ranges (see world.py), and use scenario='fixed'")
    else:
        config = TASK_REGISTRY[task_name](seed, scenario=scenario)
    return add_news(config, news, seed)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def weighted_fill(skus: Dict[str, Any], attribute: str = "demand_fulfilled_today") -> float:
    """
    Mean of a fill value over SKUs, weighted by each SKU's stockout_penalty:

        Σ stockout_penalty_i × fill_i / Σ stockout_penalty_i

    so running out of insulin (penalty 100) costs 20x more than running out of
    vitamins (penalty 5). Falls back to the plain mean if no SKU has a penalty.
    """
    if not skus:
        return 0.0
    weights = {k: max(float(s.stockout_penalty), 0.0) for k, s in skus.items()}
    total = sum(weights.values())
    if total <= 0:
        return sum(getattr(s, attribute) for s in skus.values()) / len(skus)
    return sum(weights[k] * getattr(s, attribute) for k, s in skus.items()) / total


def compute_step_reward(inventory_state: InventoryState) -> float:
    """
    step_reward = weighted_fill(demand_fulfilled_today) - waste_fraction_today

    The fill rate is averaged over SKUs with stockout_penalty as the weight, just as
    waste_fraction_today is the waste_penalty-weighted share of today's deliveries
    rejected for lack of storage, in [0, 1]. Range is [-1, 1].
    """
    ts = inventory_state
    if not ts.skus:
        return 0.0
    return weighted_fill(ts.skus) - ts.waste_fraction_today


def average_step_reward(inventory_state: InventoryState, days: int) -> float:
    """
    Sum of the step rewards earned so far divided by `days`, clipped to [0, 1].

    Rebuilt from the environment's running totals, no extra state needed:
      sum of weighted fill over days = weighted mean over SKUs of demand_fulfilled_cumulative
                                       (the weights are fixed for the episode)
      sum of waste_fraction_today = days played - inventory_excess_cumulative
                                    (that total adds 1 - waste_fraction_today each day)
    """
    ts = inventory_state
    if days <= 0 or not ts.skus:
        return 0.0
    fulfilled = weighted_fill(ts.skus, "demand_fulfilled_cumulative")
    wasted = ts.current_date - ts.inventory_excess_cumulative
    return min(1.0, max(0.0, (fulfilled - wasted) / days))


def compute_final_score(
    inventory_state: InventoryState,
    episode_config: EpisodeConfig,
) -> float:
    """
    final_score = clip(sum of step rewards / no_of_days, 0, 1)
                = clip(mean over days of (weighted fill - waste_fraction_today), 0, 1)

    The episode score is the average daily reward, so an agent is evaluated on
    exactly what it is rewarded for each step. Ordering nothing scores ~0.
    Days not played (an episode cut short) count as 0.
    """
    return average_step_reward(inventory_state, episode_config.no_of_days)
