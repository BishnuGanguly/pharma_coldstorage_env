from __future__ import annotations

from math import exp
from random import Random
from typing import Any, Dict, List, Optional

try:
    from models import InventoryState, SKUState, SupplierState
except ImportError:
    from models import InventoryState, SKUState, SupplierState


# ---------------------------------------------------------------------------
# Curve builders
# ---------------------------------------------------------------------------

def gaussian_curve(peak_day: int, amplitude: float, width: float, duration: int) -> List[float]:
    """
    Returns a list of daily demand multipliers for the full episode.
    1.0 = normal demand. 2.8 = demand is 2.8x baseline at peak.

    Parameters
    ----------
    peak_day  : day within episode where peak occurs (0-indexed)
    amplitude : peak elevation above baseline (1.5 → demand reaches 2.5x at peak)
    width     : standard deviation in days — larger = slower, wider wave
    duration  : total episode length in days
    """
    return [
        1.0 + amplitude * exp(-0.5 * ((day - peak_day) / width) ** 2)
        for day in range(duration)
    ]


def flat_curve(duration: int, value: float = 1.0) -> List[float]:
    """Constant multiplier — no demand event on this SKU."""
    return [value] * duration


def step_curve(start_day: int, amplitude: float, duration: int) -> List[float]:
    """
    Sustained step elevation starting at start_day.
    Models supply chain stress where demand rises and stays elevated.
    """
    return [
        1.0 + amplitude if day >= start_day else 1.0
        for day in range(duration)
    ]


def combine_curves(curve_a: List[float], curve_b: List[float]) -> List[float]:
    """
    Stack two demand events on the same SKU multiplicatively.
    Used for two-wave epidemic scenarios.
    """
    return [a * b for a, b in zip(curve_a, curve_b)]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sample_base_demands(rng: Random) -> Dict[str, float]:
    return {
        "insulin":      rng.uniform(8,   12),
        "paracetamol":  rng.uniform(180, 220),
        "bp_medication":rng.uniform(40,   60),
        "vitamins":     rng.uniform(60,   90),
    }


def _sample_hcq_base_demands(rng: Random) -> Dict[str, float]:
    base = _sample_base_demands(rng)
    base["hydroxychloroquine"] = rng.uniform(20, 40)
    return base


def _starting_inventory(base: Dict[str, float], cover_days: float, rng: Random) -> Dict[str, float]:
    """
    Starting inventory = cover_days × daily base demand × small noise.
    cover_days:
        easy   → 14  (comfortable)
        medium → 7   (tight)
        hard   → 4   (already stressed)
    """
    return {
        sku: rng.uniform(0.9, 1.1) * cover_days * demand
        for sku, demand in base.items()
    }


def _base_supplier_configs() -> Dict[str, Any]:
    return {
        "FastPharma": {
            "cold_chain_certified":     True,
            "sku_served":               ["insulin", "paracetamol", "bp_medication", "vitamins"],
            "unit_cost":                1.4,
            "expedite_allowed":         True,
            "expedite_cost_multiplier": 2.0,
            "base_lead_time_mean":      2.0,
        },
        "GlobalMed": {
            "cold_chain_certified":     False,
            "sku_served":               ["paracetamol", "bp_medication", "vitamins"],
            "unit_cost":                1.0,
            "expedite_allowed":         False,
            "expedite_cost_multiplier": 1.0,
            "base_lead_time_mean":      7.0,
        },
    }


def _hcq_supplier_configs() -> Dict[str, Any]:
    """Supplier configs extended to serve hydroxychloroquine."""
    configs = _base_supplier_configs()
    configs["FastPharma"]["sku_served"].append("hydroxychloroquine")
    configs["GlobalMed"]["sku_served"].append("hydroxychloroquine")
    return configs


def _base_supplier_params() -> Dict[str, Any]:
    return {
        "FastPharma": {"lead_time_mean": 2.0, "lead_time_std": 0.5},
        "GlobalMed":  {"lead_time_mean": 7.0, "lead_time_std": 2.5},
    }


def _base_disruption_params() -> Dict[str, Any]:
    return {
        "FastPharma": {"p_onset": 0.02, "p_persist": 0.30},
        "GlobalMed":  {"p_onset": 0.08, "p_persist": 0.70},
    }


def _base_sku_configs(include_hcq: bool = False) -> Dict[str, Any]:
    configs = {
        "insulin": {
            "name":                      "Insulin",
            "cold_storage_required":     True,
            "stockout_penalty":          100.0,
            "substitute_coverage_ratio": 0.0,
        },
        "paracetamol": {
            "name":                      "Paracetamol",
            "cold_storage_required":     False,
            "stockout_penalty":          20.0,
            "substitute_coverage_ratio": 0.6,
        },
        "bp_medication": {
            "name":                      "BP Medication (Amlodipine)",
            "cold_storage_required":     False,
            "stockout_penalty":          60.0,
            "substitute_coverage_ratio": 0.4,
        },
        "vitamins": {
            "name":                      "Vitamins",
            "cold_storage_required":     False,
            "stockout_penalty":          5.0,
            "substitute_coverage_ratio": 1.0,
        },
    }
    if include_hcq:
        configs["hydroxychloroquine"] = {
            "name":                      "Hydroxychloroquine",
            "cold_storage_required":     False,
            "stockout_penalty":          35.0,
            "substitute_coverage_ratio": 0.2,
        }
    return configs


# ---------------------------------------------------------------------------
# Task 1 — Supply Chain Broken (60 days)
# ---------------------------------------------------------------------------

def task_supply_chain_broken(difficulty: str, seed: int, control: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Supply chain stress scenario.

    Both suppliers experience extended lead times from a fixed day.
    GlobalMed goes fully offline for a window in the middle of the episode.
    Demand barely changes — the challenge is entirely supply-side.

    What the agent must learn
    -------------------------
    Lead times have doubled but shelves look fine today.
    Order more, earlier, before the stress window hits.
    Reactive ordering arrives too late.

    Difficulty controls
    -------------------
    easy   : stress starts late (day 25), mild multiplier, GlobalMed offline only 5 days
    medium : stress starts day 15, 2.5x multiplier, offline 10 days
    hard   : stress starts day 8, 4x multiplier, offline 15 days starting before agent
             has had time to build safety stock
    """
    rng = Random(seed)
    control = control or {}
    D = 60

    stress_start     = control.get("stress_start_day",      {"easy": 25, "medium": 15, "hard": 8}[difficulty])
    gm_lead_mult     = control.get("gm_lead_multiplier",    {"easy": 1.5, "medium": 2.5, "hard": 4.0}[difficulty])
    fp_lead_mult     = control.get("fp_lead_multiplier",    {"easy": 1.0, "medium": 1.5, "hard": 2.0}[difficulty])
    offline_start    = control.get("hard_disruption_start", {"easy": 40, "medium": 35, "hard": 25}[difficulty])
    offline_duration = control.get("hard_disruption_duration", {"easy": 5, "medium": 10, "hard": 15}[difficulty])
    cover_days       = {"easy": 14.0, "medium": 7.0, "hard": 4.0}[difficulty]

    base = _sample_base_demands(rng)

    # Slight step elevation in demand when supply stress rumours spread
    demand_curves = {
        sku: step_curve(stress_start, amplitude=rng.uniform(0.05, 0.15), duration=D)
        for sku in base
    }

    # GlobalMed: elevated lead time then offline window
    gm_stress = []
    for day in range(D):
        if offline_start <= day < offline_start + offline_duration:
            gm_stress.append(999.0)   # offline — orders rejected in env
        elif day >= stress_start:
            gm_stress.append(gm_lead_mult)
        else:
            gm_stress.append(1.0)

    # FastPharma: partial stress, hits a few days later than GlobalMed
    fp_stress = [
        fp_lead_mult if day >= stress_start + 4 else 1.0
        for day in range(D)
    ]

    return {
        "task_id":          "supply_chain_broken",
        "difficulty":       difficulty,
        "duration_days":    D,
        "sku_configs":      _base_sku_configs(),
        "supplier_configs": _base_supplier_configs(),
        "supplier_params":  _base_supplier_params(),
        "disruption_params":_base_disruption_params(),
        "base_demands":     base,
        "demand_curves":    demand_curves,
        "supplier_stress":  {"FastPharma": fp_stress, "GlobalMed": gm_stress},
        "starting_inventory": _starting_inventory(base, cover_days, rng),
        "cold_storage_total_capacity":   500.0,
        "ambient_storage_total_capacity":25000.0,
        "total_budget":     100000.0,
        "cold_chain_breach_prob": 0.003,
        "seed":             seed,
    }


# ---------------------------------------------------------------------------
# Task 2 — Flu Season (60 days)
# ---------------------------------------------------------------------------

def task_flu_season(difficulty: str, seed: int, control: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Flu season demand surge.

    Paracetamol demand rises sharply (wide gaussian — sustained flu season).
    Insulin demand spikes suddenly when flu hits diabetics hard.
    Vitamins see a shorter panic-buying bump.
    BP medication barely moves.
    GlobalMed lead times increase during the flu peak (logistics workers sick).

    What the agent must learn
    -------------------------
    Paracetamol and insulin surge simultaneously.
    Budget is tight — insulin has no substitute, protect it first.
    Epidemic alert fires a few days before paracetamol peaks —
    agent must pre-stock before the surge, not react to it.

    Difficulty controls
    -------------------
    easy   : flu onset day 25, mild amplitudes, no supply stress
    medium : flu onset day 15, significant amplitudes, GlobalMed slows
    hard   : flu onset day 8, large amplitudes, GlobalMed slows early
              overlapping insulin and paracetamol peaks strain budget
    """
    rng = Random(seed)
    control = control or {}
    D = 60

    onset        = control.get("flu_onset_day",     {"easy": 25, "medium": 15, "hard": 8}[difficulty])
    para_amp     = control.get("para_amplitude",    {"easy": 0.8, "medium": 1.5, "hard": 2.2}[difficulty])
    insulin_amp  = control.get("insulin_amplitude", {"easy": 0.2, "medium": 0.5, "hard": 0.8}[difficulty])
    lead_stress  = control.get("lead_stress_day",   {"easy": 999, "medium": onset + 8, "hard": onset + 3}[difficulty])
    cover_days   = {"easy": 14.0, "medium": 7.0, "hard": 4.0}[difficulty]

    base = _sample_base_demands(rng)

    demand_curves = {
        # Paracetamol: wide sustained surge
        "paracetamol":  gaussian_curve(onset + 10, para_amp,    width=12, duration=D),
        # Insulin: sharper spike (diabetics hit by flu)
        "insulin":      gaussian_curve(onset + 15, insulin_amp, width=6,  duration=D),
        # Vitamins: short panic-buying bump
        "vitamins":     gaussian_curve(onset + 5,  rng.uniform(0.3, 0.5), width=8, duration=D),
        # BP med: flat
        "bp_medication": flat_curve(D, 1.0),
    }

    gm_stress = [
        1.8 if lead_stress <= day <= lead_stress + 20 else 1.0
        for day in range(D)
    ]

    return {
        "task_id":          "flu_season",
        "difficulty":       difficulty,
        "duration_days":    D,
        "sku_configs":      _base_sku_configs(),
        "supplier_configs": _base_supplier_configs(),
        "supplier_params":  _base_supplier_params(),
        "disruption_params":_base_disruption_params(),
        "base_demands":     base,
        "demand_curves":    demand_curves,
        "supplier_stress":  {"FastPharma": flat_curve(D, 1.0), "GlobalMed": gm_stress},
        "starting_inventory": _starting_inventory(base, cover_days, rng),
        "cold_storage_total_capacity":   500.0,
        "ambient_storage_total_capacity":25000.0,
        "total_budget":     100000.0,
        "cold_chain_breach_prob": 0.004,
        "seed":             seed,
    }


# ---------------------------------------------------------------------------
# Task 3 — Epidemic Two Waves (60 days)
# ---------------------------------------------------------------------------

def task_epidemic(difficulty: str, seed: int, control: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Two-wave epidemic scenario. Primary SKU: Hydroxychloroquine (HCQ).

    Wave 1 peaks at day 17. Wave 2 peaks at day 45 and is larger than wave 1.
    Between waves there is a trough — the agent must NOT relax, wave 2 is coming.
    Lead times for GlobalMed worsen at each wave peak (logistics under pressure).
    FastPharma gets slight stress during wave 2 only.
    Insulin sees a mild secondary elevation (patients under epidemic stress).

    What the agent must learn
    -------------------------
    Pre-stock HCQ before day 17 — ordering at the peak arrives too late.
    Trough is a trap: second wave is larger, must keep ordering.
    Lead times increase exactly when demand peaks — orders placed late arrive late.
    Budget must stay protected for insulin throughout.

    Difficulty controls
    -------------------
    easy   : wave1 amp 1.2, wave2 smaller (1.0), wide waves, stress after peak
    medium : wave1 amp 1.8, wave2 amp 2.5 (larger), stress concurrent with peak
    hard   : wave1 amp 2.5, wave2 amp 3.5, narrow waves, stress before peak
    """
    rng = Random(seed)
    control = control or {}
    D = 60

    # Fixed peak positions — your explicit control requirement
    wave1_peak   = control.get("wave1_peak_day",   17)
    wave2_peak   = control.get("wave2_peak_day",   45)

    wave1_amp    = control.get("wave1_amplitude",  {"easy": 1.2, "medium": 1.8, "hard": 2.5}[difficulty])
    wave2_amp    = control.get("wave2_amplitude",  {"easy": 1.0, "medium": 2.5, "hard": 3.5}[difficulty])
    wave_width   = control.get("wave_width",       {"easy": 9.0, "medium": 6.0, "hard": 4.0}[difficulty])
    cover_days   = {"easy": 14.0, "medium": 7.0, "hard": 4.0}[difficulty]

    # Lead stress timing relative to wave peaks
    ls1 = control.get("lead_stress_wave1_day",
                       {"easy": wave1_peak + 3, "medium": wave1_peak - 2, "hard": wave1_peak - 5}[difficulty])
    ls2 = control.get("lead_stress_wave2_day",
                       {"easy": wave2_peak + 3, "medium": wave2_peak - 3, "hard": wave2_peak - 5}[difficulty])

    base = _sample_hcq_base_demands(rng)

    # HCQ: two gaussian waves stacked
    hcq_curve = combine_curves(
        gaussian_curve(wave1_peak, wave1_amp, wave_width, D),
        gaussian_curve(wave2_peak, wave2_amp, wave_width, D),
    )

    demand_curves = {
        "hydroxychloroquine": hcq_curve,
        "insulin":            gaussian_curve(wave1_peak + 3, rng.uniform(0.1, 0.2), 10, D),
        "paracetamol":        flat_curve(D, 1.0),
        "bp_medication":      flat_curve(D, 1.0),
        "vitamins":           flat_curve(D, 1.0),
    }

    # GlobalMed: stress windows around both peaks (worse at wave 2)
    gm_lm1 = rng.uniform(1.8, 2.5)
    gm_lm2 = rng.uniform(2.5, 3.5)
    gm_stress = [
        gm_lm1 if ls1 <= day <= wave1_peak + 8
        else gm_lm2 if ls2 <= day <= wave2_peak + 10
        else 1.0
        for day in range(D)
    ]

    # FastPharma: slight stress during wave 2 only
    fp_stress = [
        rng.uniform(1.2, 1.5) if wave2_peak - 5 <= day <= wave2_peak + 8
        else 1.0
        for day in range(D)
    ]

    return {
        "task_id":          "epidemic_two_wave",
        "difficulty":       difficulty,
        "duration_days":    D,
        "sku_configs":      _base_sku_configs(include_hcq=True),
        "supplier_configs": _hcq_supplier_configs(),
        "supplier_params":  _base_supplier_params(),
        "disruption_params":_base_disruption_params(),
        "base_demands":     base,
        "demand_curves":    demand_curves,
        "supplier_stress":  {"FastPharma": fp_stress, "GlobalMed": gm_stress},
        "starting_inventory": _starting_inventory(base, cover_days, rng),
        "cold_storage_total_capacity":   500.0,
        "ambient_storage_total_capacity":25000.0,
        "total_budget":     100000.0,
        "cold_chain_breach_prob": 0.005,
        "seed":             seed,
    }


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "supply_chain_broken": task_supply_chain_broken,
    "flu_season":          task_flu_season,
    "epidemic_two_wave":   task_epidemic,
}

# Convenience: all 9 canonical configs (3 tasks × 3 difficulties)
TASK_CONFIGS: List[Dict[str, Any]] = [
    task_supply_chain_broken("easy",   seed=1),
    task_supply_chain_broken("medium", seed=2),
    task_supply_chain_broken("hard",   seed=3),
    task_flu_season("easy",            seed=4),
    task_flu_season("medium",          seed=5),
    task_flu_season("hard",            seed=6),
    task_epidemic("easy",              seed=7),
    task_epidemic("medium",            seed=8),
    task_epidemic("hard",              seed=9),
]


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_step_reward(inventory_state: InventoryState) -> float:
    """
    Per-step reward for the pharma cold-chain procurement agent.

    Formula
    -------
        reward = w_fill  * fill_rate_7d
               + w_crit  * critical_sku_fill
               + w_waste * (1 - waste_ratio)
               + w_budget* budget_ratio

    Components
    ----------
    fill_rate_7d
        prescription_fill_rate_7d from InventoryState.
        Core service metric. Captures whether demand is being met recently.
        Weight 0.50 — most important signal.

    critical_sku_fill
        Weighted fill rate for critical SKUs only (insulin, bp_medication).
        Computed as 1 - (backorders / demand_last_7d) for each critical SKU.
        Heavily penalises insulin and BP med stockouts specifically.
        Weight 0.30 — second most important.

    waste_ratio
        Expiry pressure signal: mean expiry_pressure_7d across all SKUs.
        1.0 - waste_ratio rewards the agent for NOT over-stocking perishables.
        Weight 0.10.

    budget_ratio
        Remaining procurement budget / total.
        Rewards fiscal discipline — don't drain the budget in early days.
        Weight 0.10.

    None handling
    -------------
    At episode start, backorders are 0 and demand history is 0.
    Both fill components default to 1.0 (clean slate — no failures yet).
    Reward naturally starts at ~1.0 and drops as failures accumulate.

    Range: [0.01, 0.99]
    """
    ts = inventory_state

    # -- Component 1: overall fill rate (7-day rolling) ---------------------
    fill_7d = ts.prescription_fill_rate_7d  # already in [0, 1]

    # -- Component 2: critical SKU fill rate --------------------------------
    CRITICAL_SKUS = {"insulin", "bp_medication"}
    critical_scores = []
    for sku_id in CRITICAL_SKUS:
        sku = ts.get_sku(sku_id)
        if sku is None:
            continue
        demand_7d = sku.demand_last_7d
        if demand_7d > 0:
            fill = max(0.0, 1.0 - (sku.backorders / demand_7d))
        else:
            fill = 1.0  # no demand yet — no failure
        critical_scores.append(fill)

    critical_fill = sum(critical_scores) / len(critical_scores) if critical_scores else 1.0

    # -- Component 3: waste ratio (lower expiry pressure = better) ----------
    expiry_pressures = [
        sku.metadata.get("expiry_pressure_7d", 0.0) if hasattr(sku, "metadata")
        else getattr(sku, "expiry_pressure_7d", 0.0)  # SKUState field if added
        for sku in ts.skus.values()
    ]
    mean_expiry = sum(expiry_pressures) / len(expiry_pressures) if expiry_pressures else 0.0
    waste_component = 1.0 - min(1.0, mean_expiry)

    # -- Component 4: budget discipline ------------------------------------
    budget_component = ts.procurement_budget_ratio

    # -- Weighted sum -------------------------------------------------------
    raw = (
        0.50 * fill_7d
        + 0.30 * critical_fill
        + 0.10 * waste_component
        + 0.10 * budget_component
    )

    # Clamp to (0.01, 0.99) — same convention as ETL env
    return round(max(0.01, min(0.99, raw)), 4)


def compute_final_score(
    step_rewards: List[float],
    inventory_state: InventoryState,
) -> float:
    """
    Final episode score.

    Formula
    -------
        final_score = 0.5 * mean(step_rewards)
                    + 0.3 * prescription_fill_rate_30d
                    + 0.2 * (1 - total_backorder_ratio)

    The three terms measure:
        mean(step_rewards)         — quality of decisions throughout the episode
        fill_rate_30d              — end-of-episode service level
        1 - backorder_ratio        — whether accumulated debt was resolved

    Range: [0.01, 0.99]
    """
    ts = inventory_state

    mean_step = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0

    fill_30d = ts.prescription_fill_rate_30d

    # Total backorder ratio: backorders / total demand last 30 days
    total_backorders = sum(sku.backorders for sku in ts.skus.values())
    total_demand_30d = sum(sku.demand_last_7d * (30 / 7) for sku in ts.skus.values())
    backorder_ratio = min(1.0, total_backorders / total_demand_30d) if total_demand_30d > 0 else 0.0

    raw = (
        0.5  * mean_step
        + 0.3  * fill_30d
        + 0.2  * (1.0 - backorder_ratio)
    )

    return round(max(0.01, min(0.99, raw)), 4)