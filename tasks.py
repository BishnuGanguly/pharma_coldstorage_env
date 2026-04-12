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
    return [
        1.0 + amplitude * exp(-0.5 * ((day - peak_day) / width) ** 2)
        for day in range(duration)
    ]
 
 
def flat_curve(duration: int, value: float = 1.0) -> List[float]:
    return [value] * duration
 
 
def step_curve(start_day: int, amplitude: float, duration: int) -> List[float]:
    return [
        1.0 + amplitude if day >= start_day else 1.0
        for day in range(duration)
    ]
 
 
def combine_curves(curve_a: List[float], curve_b: List[float]) -> List[float]:
    return [a * b for a, b in zip(curve_a, curve_b)]
 
 
# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
 
def _sample_base_demands(rng: Random) -> Dict[str, float]:
    return {
        "insulin":       rng.uniform(8,   12),
        "paracetamol":   rng.uniform(180, 220),
        "bp_medication": rng.uniform(40,   60),
        "vitamins":      rng.uniform(60,   90),
    }
 
 
def _sample_hcq_base_demands(rng: Random) -> Dict[str, float]:
    base = _sample_base_demands(rng)
    base["hydroxychloroquine"] = rng.uniform(20, 40)
    return base
 
 
def _starting_inventory(base: Dict[str, float], cover_days: float, rng: Random) -> Dict[str, float]:
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
    GlobalMed goes fully offline (stress=999) for a window — orders are
    rejected by the environment when stress >= 100.
    Demand barely changes — the challenge is entirely supply-side.
 
    Difficulty controls
    -------------------
    easy   : stress starts late (day 25), mild multiplier, GlobalMed offline 5 days
    medium : stress starts day 15, 2.5x multiplier, offline 10 days
    hard   : stress starts day 8, 4x multiplier, offline 15 days
    """
    rng = Random(seed)
    control = control or {}
    D = 60
 
    stress_start     = control.get("stress_start_day",         {"easy": 25, "medium": 15, "hard": 8}[difficulty])
    gm_lead_mult     = control.get("gm_lead_multiplier",       {"easy": 1.5, "medium": 2.5, "hard": 4.0}[difficulty])
    fp_lead_mult     = control.get("fp_lead_multiplier",       {"easy": 1.0, "medium": 1.5, "hard": 2.0}[difficulty])
    offline_start    = control.get("hard_disruption_start",    {"easy": 40, "medium": 35, "hard": 25}[difficulty])
    offline_duration = control.get("hard_disruption_duration", {"easy": 5, "medium": 10, "hard": 15}[difficulty])
    cover_days       = {"easy": 14.0, "medium": 7.0, "hard": 4.0}[difficulty]
 
    base = _sample_base_demands(rng)
 
    demand_curves = {
        sku: step_curve(stress_start, amplitude=rng.uniform(0.05, 0.15), duration=D)
        for sku in base
    }
 
    # GlobalMed: stress=999 during offline window → env rejects all orders.
    # Outside offline window: elevated lead-time multiplier from stress_start.
    gm_stress = []
    for day in range(D):
        if offline_start <= day < offline_start + offline_duration:
            gm_stress.append(999.0)   # offline — _place_orders rejects at stress >= 100
        elif day >= stress_start:
            gm_stress.append(gm_lead_mult)
        else:
            gm_stress.append(1.0)
 
    fp_stress = [
        fp_lead_mult if day >= stress_start + 4 else 1.0
        for day in range(D)
    ]
 
    return {
        "task_id":           "supply_chain_broken",
        "difficulty":        difficulty,
        "duration_days":     D,
        "sku_configs":       _base_sku_configs(),
        "supplier_configs":  _base_supplier_configs(),
        "supplier_params":   _base_supplier_params(),
        "disruption_params": _base_disruption_params(),
        "base_demands":      base,
        "demand_curves":     demand_curves,
        "supplier_stress":   {"FastPharma": fp_stress, "GlobalMed": gm_stress},
        "starting_inventory":_starting_inventory(base, cover_days, rng),
        "cold_storage_total_capacity":   500.0,
        "ambient_storage_total_capacity":25000.0,
        "total_budget":      100000.0,
        "cold_chain_breach_prob": 0.003,
        "seed":              seed,
    }
 
 
# ---------------------------------------------------------------------------
# Task 2 — Flu Season (60 days)
# ---------------------------------------------------------------------------
 
def task_flu_season(difficulty: str, seed: int, control: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Flu season demand surge.
 
    Paracetamol demand rises sharply (wide gaussian — sustained flu season).
    Insulin demand spikes when flu hits diabetics hard.
    Vitamins see a short panic-buying bump.
    BP medication barely moves.
    GlobalMed lead times increase during the flu peak.
 
    Difficulty controls
    -------------------
    easy   : flu onset day 25, mild amplitudes, no supply stress
    medium : flu onset day 15, significant amplitudes, GlobalMed slows
    hard   : flu onset day 8, large amplitudes, GlobalMed slows early
    """
    rng = Random(seed)
    control = control or {}
    D = 60
 
    onset       = control.get("flu_onset_day",     {"easy": 25, "medium": 15, "hard": 8}[difficulty])
    para_amp    = control.get("para_amplitude",    {"easy": 0.8, "medium": 1.5, "hard": 2.2}[difficulty])
    insulin_amp = control.get("insulin_amplitude", {"easy": 0.2, "medium": 0.5, "hard": 0.8}[difficulty])
    lead_stress = control.get("lead_stress_day",   {"easy": 999, "medium": onset + 8, "hard": onset + 3}[difficulty])
    cover_days  = {"easy": 14.0, "medium": 7.0, "hard": 4.0}[difficulty]
 
    base = _sample_base_demands(rng)
 
    demand_curves = {
        "paracetamol":   gaussian_curve(onset + 10, para_amp,    width=12, duration=D),
        "insulin":       gaussian_curve(onset + 15, insulin_amp, width=6,  duration=D),
        "vitamins":      gaussian_curve(onset + 5,  rng.uniform(0.3, 0.5), width=8, duration=D),
        "bp_medication": flat_curve(D, 1.0),
    }
 
    gm_stress = [
        1.8 if lead_stress <= day <= lead_stress + 20 else 1.0
        for day in range(D)
    ]
 
    return {
        "task_id":           "flu_season",
        "difficulty":        difficulty,
        "duration_days":     D,
        "sku_configs":       _base_sku_configs(),
        "supplier_configs":  _base_supplier_configs(),
        "supplier_params":   _base_supplier_params(),
        "disruption_params": _base_disruption_params(),
        "base_demands":      base,
        "demand_curves":     demand_curves,
        "supplier_stress":   {"FastPharma": flat_curve(D, 1.0), "GlobalMed": gm_stress},
        "starting_inventory":_starting_inventory(base, cover_days, rng),
        "cold_storage_total_capacity":   500.0,
        "ambient_storage_total_capacity":25000.0,
        "total_budget":      100000.0,
        "cold_chain_breach_prob": 0.004,
        "seed":              seed,
    }
 
 
# ---------------------------------------------------------------------------
# Task 3 — Epidemic Two Waves (60 days)
# ---------------------------------------------------------------------------
 
def task_epidemic(difficulty: str, seed: int, control: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Two-wave epidemic. Primary SKU: Hydroxychloroquine (HCQ).
 
    Wave 1 peaks at day 17. Wave 2 peaks at day 45 and is larger.
    Between waves there is a trough — the agent must NOT relax.
    Lead times for GlobalMed worsen at each wave peak.
    FastPharma gets slight stress during wave 2 only.
    Insulin sees a mild secondary elevation.
 
    Difficulty controls
    -------------------
    easy   : wave1 amp 1.2, wave2 amp 1.0 (smaller), wide waves
    medium : wave1 amp 1.8, wave2 amp 2.5 (larger), stress concurrent with peak
    hard   : wave1 amp 2.5, wave2 amp 3.5, narrow waves, stress before peak
    """
    rng = Random(seed)
    control = control or {}
    D = 60
 
    wave1_peak  = control.get("wave1_peak_day",  17)
    wave2_peak  = control.get("wave2_peak_day",  45)
    wave1_amp   = control.get("wave1_amplitude", {"easy": 1.2, "medium": 1.8, "hard": 2.5}[difficulty])
    wave2_amp   = control.get("wave2_amplitude", {"easy": 1.0, "medium": 2.5, "hard": 3.5}[difficulty])
    wave_width  = control.get("wave_width",      {"easy": 9.0, "medium": 6.0, "hard": 4.0}[difficulty])
    cover_days  = {"easy": 14.0, "medium": 7.0, "hard": 4.0}[difficulty]
 
    ls1 = control.get("lead_stress_wave1_day",
                      {"easy": wave1_peak + 3, "medium": wave1_peak - 2, "hard": wave1_peak - 5}[difficulty])
    ls2 = control.get("lead_stress_wave2_day",
                      {"easy": wave2_peak + 3, "medium": wave2_peak - 3, "hard": wave2_peak - 5}[difficulty])
 
    base = _sample_hcq_base_demands(rng)
 
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
 
    gm_lm1 = rng.uniform(1.8, 2.5)
    gm_lm2 = rng.uniform(2.5, 3.5)
    gm_stress = [
        gm_lm1 if ls1 <= day <= wave1_peak + 8
        else gm_lm2 if ls2 <= day <= wave2_peak + 10
        else 1.0
        for day in range(D)
    ]
 
    fp_stress = [
        rng.uniform(1.2, 1.5) if wave2_peak - 5 <= day <= wave2_peak + 8
        else 1.0
        for day in range(D)
    ]
 
    return {
        "task_id":           "epidemic_two_wave",
        "difficulty":        difficulty,
        "duration_days":     D,
        "sku_configs":       _base_sku_configs(include_hcq=True),
        "supplier_configs":  _hcq_supplier_configs(),
        "supplier_params":   _base_supplier_params(),
        "disruption_params": _base_disruption_params(),
        "base_demands":      base,
        "demand_curves":     demand_curves,
        "supplier_stress":   {"FastPharma": fp_stress, "GlobalMed": gm_stress},
        "starting_inventory":_starting_inventory(base, cover_days, rng),
        "cold_storage_total_capacity":   500.0,
        "ambient_storage_total_capacity":25000.0,
        "total_budget":      100000.0,
        "cold_chain_breach_prob": 0.005,
        "seed":              seed,
    }
 
 
# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
 
TASK_REGISTRY = {
    "supply_chain_broken": task_supply_chain_broken,
    "flu_season":          task_flu_season,
    "epidemic_two_wave":   task_epidemic,
}
 
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
        reward = 0.50 * fill_rate_7d
               + 0.30 * critical_sku_fill
               + 0.10 * (1 - overstock_pressure)
               + 0.10 * budget_ratio
 
    Components
    ----------
    fill_rate_7d
        prescription_fill_rate_7d — whether demand is being met right now.
        Weight 0.50 — most important signal.
 
    critical_sku_fill
        Weighted fill rate for insulin and bp_medication only.
        1 - (backorders / demand_last_7d) per critical SKU.
        Weight 0.30 — heavily penalises life-critical stockouts.
 
    1 - overstock_pressure   [FIX 4]
        Originally referenced sku.metadata["expiry_pressure_7d"] which does
        not exist on SKUState (inherits BaseModel, not Observation, so has no
        .metadata dict). The hasattr guard silently made this component always
        return 0.0, making waste_component permanently 1.0 and non-functional.
 
        Replaced with a computable overstock pressure metric:
            days_cover = inventory_on_hand / (demand_last_7d / 7)
            overstock_pressure = clamp((days_cover - 30) / 30, 0, 1)
        A SKU with > 30 days cover has expiry risk. The component rewards the
        agent for NOT over-stocking perishables beyond a 30-day buffer.
        When demand_last_7d = 0 (no history yet), pressure is 0 (no penalty).
        Weight 0.10.
 
    budget_ratio
        Remaining procurement budget / total.
        Rewards fiscal discipline.
        Weight 0.10.
 
    Range: [0.01, 0.99]
    """
    ts = inventory_state
 
    # Component 1: overall 7-day fill rate
    fill_7d = ts.prescription_fill_rate_7d
 
    # Component 2: critical SKU fill rate
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
            fill = 1.0
        critical_scores.append(fill)
    critical_fill = sum(critical_scores) / len(critical_scores) if critical_scores else 1.0
 
    # Component 3: overstock pressure (replaces broken expiry_pressure_7d).
    # Penalises holding > 30 days of cover — expiry / storage waste risk.
    overstock_pressures = []
    for sku in ts.skus.values():
        avg_daily = sku.demand_last_7d / 7.0 if sku.demand_last_7d > 0 else 0.0
        if avg_daily > 0:
            days_cover = sku.inventory_on_hand / avg_daily
            pressure = min(1.0, max(0.0, (days_cover - 30.0) / 30.0))
        else:
            pressure = 0.0   # no demand history yet — no penalty
        overstock_pressures.append(pressure)
    mean_overstock = sum(overstock_pressures) / len(overstock_pressures) if overstock_pressures else 0.0
    waste_component = 1.0 - mean_overstock
 
    # Component 4: budget discipline
    budget_component = ts.procurement_budget_ratio
 
    raw = (
          0.50 * fill_7d
        + 0.30 * critical_fill
        + 0.10 * waste_component
        + 0.10 * budget_component
    )
 
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
 
    Range: [0.01, 0.99]
    """
    ts = inventory_state
 
    mean_step = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
    fill_30d  = ts.prescription_fill_rate_30d
 
    total_backorders  = sum(sku.backorders for sku in ts.skus.values())
    total_demand_30d  = sum(sku.demand_last_7d * (30 / 7) for sku in ts.skus.values())
    backorder_ratio   = min(1.0, total_backorders / total_demand_30d) if total_demand_30d > 0 else 0.0
 
    raw = (
          0.5 * mean_step
        + 0.3 * fill_30d
        + 0.2 * (1.0 - backorder_ratio)
    )
 
    return round(max(0.01, min(0.99, raw)), 4)
 