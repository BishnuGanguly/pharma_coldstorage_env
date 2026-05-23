from __future__ import annotations
 
from math import exp
from random import Random
from typing import Any, Dict, List, Optional
 
try:
    from models import InventoryState,SKUState, EpisodeConfig,SKUEpisodeConfig
except ImportError:
    from models import InventoryState, SKUState
 
 
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
        for day in range(duration)]
 
 
def combine_curves(curve_a: List[float], curve_b: List[float]) -> List[float]:
    return [a * b for a, b in zip(curve_a, curve_b)]
 
 
rng = Random(42)
base_sku_configs = {
        "insulin": {
            "name":                      "insulin",
            "cold_storage_required":     False,
            "stockout_penalty":          100.0,
            "base_demand":              rng.uniform(8,   12),
            "base_lead_time":           rng.uniform(4,   6),
            "starting_inventory":       rng.uniform(1,5)
        },
        "paracetamol": {
            "name":                      "paracetamol",
            "cold_storage_required":     False,
            "stockout_penalty":          20.0,
            "base_demand":              rng.uniform(180, 220),
            "base_lead_time":           rng.uniform(1,   3),
            "starting_inventory":       rng.uniform(1,5)
        },
        "bp_medication": {
            "name":                      "bp_medication",
            "cold_storage_required":     False,
            "stockout_penalty":          60.0,
            "base_demand":              rng.uniform(40,   60),
            "base_lead_time":           rng.uniform(2,   4),
            "starting_inventory":       rng.uniform(1,5)
        },
        "vitamins": {
            "name":                      "vitamins",
            "cold_storage_required":     False,
            "stockout_penalty":          5.0,
            "base_demand":              rng.uniform(60,   90),
            "base_lead_time":           rng.uniform(1,   2),
            "starting_inventory":       rng.uniform(1,5)
        },
        "hydroxychloroquine": {
            "name":                      "hydroxychloroquine",
            "cold_storage_required":     False,
            "stockout_penalty":          35.0,
            "base_demand":              rng.uniform(20, 40),
            "base_lead_time":           rng.uniform(5,   8),
            "starting_inventory":       rng.uniform(1,5)
        }
    }
    

 

#items that are neede to be modelled per task are as follows:
# Task Name
# No of days for the task.

# sku for each task .
# cold storage total capacity.
#(initial  innventory for both storage type) not sure .
# ambient storage total capacity.
# initial inventory for each SKU .


# - base_demand curves (per SKU)
# - base lead time curves (per SKU)
# - demand_multiplier curves (per SKU)
# - lead_time_multiplier curves (per SKU).
# all three task will have 5 squ which are insulin, paracetamol, bp_medication, vitamins and hydroxychloroquine (hcq) .
# intially for each task there will be no need for an cold storage in the first version.
# ---------------------------------------------------------------------------
# Task 1 — Supply Chain Broken (60 days)
# ---------------------------------------------------------------------------


def supply_chain_broken() -> EpisodeConfig:
    """
    supply chain broken , 60 days
    lead time increses for some suppliers , making the agent to order large quantities before even if the demand is not very high.
    """
    #for each squ we have to model lead time and demand curve.
    #in task 1 we will base demand only. but will have step  curve first for insulin and after it ends then for paracetamol .


    
    D = 60
 
    
    skuconfigdict ={}
    demand_curves = {
        sku: flat_curve(duration=D)
        for sku in base_sku_configs.keys()
    }
    lead_time_curves = {sku: flat_curve(duration=D) for sku in base_sku_configs.keys()}
 
    lead_time_curves["insulin"] = step_curve(lead_time_curves["insulin"], 15, 25, amplitude=rng.uniform(2.0, 5.0), duration=D)
    lead_time_curves["paracetamol"] = step_curve(lead_time_curves["paracetamol"], 30, 40, amplitude=rng.uniform(2.0, 5.0), duration=D)
    for key,value in base_sku_configs.items():
        sku = SKUEpisodeConfig()
        sku.sku_id = value["name"]
        sku.no_of_days = D
        sku.base_demand = value["base_demand"]
        sku.demand_curve = demand_curves[key]
        sku.base_lead_time = value["base_lead_time"]
        sku.lead_time_curve = lead_time_curves[key]
        sku.cold_storage_required = value["cold_storage_required"]
        sku.stockout_penalty = value["stockout_penalty"]
        skuconfigdict[key] = sku

    return EpisodeConfig(
        task_name="supply_chain_broken",
        no_of_days=D,
        skus=skuconfigdict,
        initial_inventory={sku: value["starting_inventory"] for sku, value in base_sku_configs.items()},
        cold_storage_total_capacity=500.0,
        ambient_storage_total_capacity=25000.0,
    )
 
 
# ---------------------------------------------------------------------------
# Task 2 — Flu Season (60 days)
# ---------------------------------------------------------------------------
 
def task_flu_season() -> EpisodeConfig:
    """
    there is constant high in demand for a medicine such as Antiflu ,and there are sudden spikes for critical medicine like 
    insulin. and lead time also increses for high demand medicines.
    Difficulty controls
    """

    D = 60
 
    
    skuconfigdict ={}
    demand_curves = {
        sku: flat_curve(duration=D)
        for sku in base_sku_configs.keys()
    }
  
    demand_curves["paracetamol"] = gaussian_curve(peak_day= 30 , amplitude= 2 , width=10 , duration=D)

    lead_time_curves = {sku: flat_curve(duration=D) for sku in base_sku_configs.keys()}
    lead_time_curves["paracetamol"] = step_curve(lead_time_curves["paracetamol"], 30, 40, amplitude=rng.uniform(2.0, 5.0), duration=D)
    
    
    for key,value in base_sku_configs.items():
        sku = SKUEpisodeConfig()
        sku.sku_id = value["name"]
        sku.no_of_days = D
        sku.base_demand = value["base_demand"]
        sku.demand_curve = demand_curves[key]
        sku.base_lead_time = value["base_lead_time"]
        sku.lead_time_curve = lead_time_curves[key]
        sku.cold_storage_required = value["cold_storage_required"]
        sku.stockout_penalty = value["stockout_penalty"]
        skuconfigdict[key] = sku

    return EpisodeConfig(
        task_name="flu_season",
        no_of_days=D,
        skus=skuconfigdict,
        initial_inventory={sku: value["starting_inventory"] for sku, value in base_sku_configs.items()},
        cold_storage_total_capacity=500.0,
        ambient_storage_total_capacity=25000.0,
    )
 
# ---------------------------------------------------------------------------
# Task 3 — Epidemic Two Waves (60 days)
# ---------------------------------------------------------------------------
 
def task_epidemic() -> EpisodeConfig:
    """
    Two-wave epidemic. Primary SKU: Hydroxychloroquine (HCQ).
 
    Wave 1 peaks at day 17. Wave 2 peaks at day 45 and is larger.
    Between waves there is a trough — the agent must NOT relax.
    Insulin sees a mild secondary elevation.
    
    """
   
    D = 60
 
    hcq_curve = combine_curves(
        gaussian_curve(17, 1.8, 6.0, D),
        gaussian_curve(45, 2.5, 6.0, D),
    )
 
    demand_curves = {
        "hydroxychloroquine": hcq_curve,
        "insulin":            gaussian_curve(17 + 3, rng.uniform(0.1, 0.2), 10, D),
        "paracetamol":        flat_curve(D, 1.0),
        "bp_medication":      flat_curve(D, 1.0),
        "vitamins":           flat_curve(D, 1.0),
    }
 
    lead_time_curves = {
        "hydroxychloroquine": step_curve(flat_curve(D), 17, 27, rng.uniform(2.0, 5.0), D),
        "insulin":            step_curve(flat_curve(D), 20, 30, rng.uniform(1.0, 3.0), D),
        "paracetamol":        flat_curve(D),
        "bp_medication":      flat_curve(D),
        "vitamins":           flat_curve(D),
    }
    skuconfigdict ={}
    for key,value in base_sku_configs.items():
        sku = SKUEpisodeConfig()
        sku.sku_id = value["name"]
        sku.no_of_days = D
        sku.base_demand = value["base_demand"]
        sku.demand_curve = demand_curves[key]
        sku.base_lead_time = value["base_lead_time"]
        sku.lead_time_curve = lead_time_curves[key]
        sku.cold_storage_required = value["cold_storage_required"]
        sku.stockout_penalty = value["stockout_penalty"]
        skuconfigdict[key] = sku

 
    return EpisodeConfig(
       task_name="task_epidemic",
        no_of_days=D,
        skus=skuconfigdict,
        initial_inventory={sku: value["starting_inventory"] for sku, value in base_sku_configs.items()},
        cold_storage_total_capacity=500.0,
        ambient_storage_total_capacity=25000.0,
    )
 
 
# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
 
TASK_REGISTRY = {
    "supply_chain_broken": supply_chain_broken,
    "flu_season":          task_flu_season,
    "epidemic_two_wave":   task_epidemic,
}
 
 
 
# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------
 
def compute_step_reward(inventory_state: InventoryState) -> float:
    """
   deamand fulfilled today -(inventory_excess_today /1+inventory_excess_today)

    """
    ts = inventory_state
    demand_fulfilled = 0.0
    inventory_excess = 0.0 
 
    skus = ts.skus
    for key, values in skus.items():
        demand_fulfilled = (demand_fulfilled+values.demand_fulfilled_today)
    demand_fulfilled = demand_fulfilled/len(skus)
    inventory_excess = (inventory_excess+ts.inventory_excess_today)

 
    return demand_fulfilled - (inventory_excess / (1.0 + inventory_excess))
 
 
def compute_final_score(
    inventory_state: InventoryState,
    episode_config: EpisodeConfig,
) -> float:
    """
    Final episode score.
    inventory_excess_cumulative = cumulative(1/1+invenetory_excess_today) this is handled in the inside the environement.
    (deamand_fullfilled_cumulaive*0.6 + invenetory_excess_cumulative*0.4)/ no_of_days_in_episoede
 

    """
    ts = inventory_state
    demand_fulfilled_cumulative = 0.0
    inventory_excess_cumulative = 0.0
    suks = ts.skus
    for key, values in ts.skus.items():
        demand_fulfilled_cumulative = (demand_fulfilled_cumulative + values.demand_fulfilled_cumulative)
    demand_fulfilled_cumulative = demand_fulfilled_cumulative/len(suks)
    inventory_excess_cumulative = (inventory_excess_cumulative + ts.inventory_excess_cumulative)

    return (demand_fulfilled_cumulative*0.6 + inventory_excess_cumulative*0.4)/episode_config.no_of_days
 