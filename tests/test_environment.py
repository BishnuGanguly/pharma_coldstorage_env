"""Unit tests for the simulation core (no server required)."""

from __future__ import annotations

import pytest

from models import EpisodeConfig, PharmaAction, SKUEpisodeConfig
from server.my_env_environment import PharmaEnvironment
from tasks import TASK_REGISTRY, compute_final_score, get_task_config


def _deterministic_config(lead_time: float = 3.0, days: int = 10) -> EpisodeConfig:
    """Single-SKU episode with no noise so arrivals and demand are exact."""
    return EpisodeConfig(
        task_name="deterministic",
        no_of_days=days,
        cold_storage_total_capacity=100.0,
        ambient_storage_total_capacity=1000.0,
        initial_inventory={"insulin": 50.0},
        skus={
            "insulin": SKUEpisodeConfig(
                sku_id="insulin",
                no_of_days=days,
                base_demand=10.0,
                base_lead_time=lead_time,
                stockout_penalty=100.0,
                waste_penalty=1.0,
            ),
        },
    )


def _two_pool_config(insulin_waste_penalty: float = 10.0, vitamins_waste_penalty: float = 0.5) -> EpisodeConfig:
    """Insulin in cold storage (100 units), vitamins in ambient (1000), both starting empty."""
    days = 5
    def sku(sku_id: str, cold: bool, waste_penalty: float) -> SKUEpisodeConfig:
        return SKUEpisodeConfig(
            sku_id=sku_id, no_of_days=days, base_demand=1.0, base_lead_time=1.0,
            cold_storage_required=cold, waste_penalty=waste_penalty,
        )
    return EpisodeConfig(
        task_name="two_pools",
        no_of_days=days,
        cold_storage_total_capacity=100.0,
        ambient_storage_total_capacity=1000.0,
        skus={
            "insulin": sku("insulin", True, insulin_waste_penalty),
            "vitamins": sku("vitamins", False, vitamins_waste_penalty),
        },
    )


def _deliver(cfg: EpisodeConfig, orders: dict):
    """Order on day 0; with a 1-day lead time everything arrives on day 1."""
    env = PharmaEnvironment()
    env.reset(episode_config=cfg)
    env.step(PharmaAction(orders=orders))
    return env.step(PharmaAction())


def test_reset_accepts_config_dict():
    """Over HTTP/WS the config arrives as plain JSON, not as a model instance."""
    env = PharmaEnvironment()
    cfg = get_task_config("flu_season", seed=1)
    state = env.reset(episode_config=cfg.model_dump(mode="json"), seed=1)
    assert set(state.skus) == set(cfg.skus)


def test_reset_by_task_name_is_reproducible():
    a = PharmaEnvironment().reset(task_name="epidemic_two_wave", seed=7)
    b = PharmaEnvironment().reset(task_name="epidemic_two_wave", seed=7)
    assert a.model_dump() == b.model_dump()


def test_reset_unknown_task_name():
    with pytest.raises(ValueError, match="Unknown task"):
        PharmaEnvironment().reset(task_name="nope")


@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY))
def test_tasks_are_seeded(task_name):
    assert get_task_config(task_name, seed=3) == get_task_config(task_name, seed=3)
    assert get_task_config(task_name, seed=3) != get_task_config(task_name, seed=4)


@pytest.mark.parametrize(
    "message",
    [
        '{"Insulin": 50}',                     # wrong case
        '{"unobtainium": 50}',                 # unknown SKU
        '{"insulin": -5, "paracetamol": "x"}', # invalid quantities
        '{"insulin": NaN}',                    # non-finite
        'no json here',
    ],
)
def test_bad_actions_do_not_crash(message):
    env = PharmaEnvironment()
    env.reset(seed=0)
    state = env.step(PharmaAction(message=message))
    assert state.current_date == 1


def test_sku_names_are_case_insensitive_and_strings_accepted():
    env = PharmaEnvironment()
    env.reset(episode_config=_deterministic_config())
    state = env.step(PharmaAction(message='Here you go: {"INSULIN": "40"}'))
    assert [(s, q) for s, q, _ in state.expected_inbound_orders] == [("insulin", 40.0)]


def test_same_day_arrivals_are_not_lost():
    env = PharmaEnvironment()
    cfg = _deterministic_config(lead_time=3.0)
    env.reset(episode_config=cfg)
    env.step(PharmaAction(orders={"insulin": 100.0}))   # day 0, arrives day 3
    cfg.skus["insulin"].base_lead_time = 2.0
    env._episode_config = cfg
    env.step(PharmaAction(orders={"insulin": 100.0}))   # day 1, arrives day 3
    env.step(PharmaAction())
    state = env.step(PharmaAction())                    # day 3: both arrive
    # 50 start + 200 delivered - 4 days * 10 demand
    assert state.skus["insulin"].inventory_on_hand == pytest.approx(210.0)
    assert state.inventory_excess_today == 0.0


def test_lead_time_revealed_only_on_delivery():
    env = PharmaEnvironment()
    env.reset(episode_config=_deterministic_config(lead_time=3.0))
    state = env.step(PharmaAction(orders={"insulin": 10.0}))
    assert state.skus["insulin"].lead_time_last3_orders == []
    env.step(PharmaAction())
    env.step(PharmaAction())
    state = env.step(PharmaAction())                    # delivered on day 3
    assert state.skus["insulin"].lead_time_last3_orders == [3.0]
    assert state.actual_inbound_orders == []


def test_overflow_is_recorded_as_waste():
    env = PharmaEnvironment()
    env.reset(episode_config=_deterministic_config(lead_time=1.0))
    env.step(PharmaAction(orders={"insulin": 5000.0}))  # capacity is 1000
    state = env.step(PharmaAction())
    assert state.inventory_excess_today > 0
    assert state.skus["insulin"].inventory_on_hand <= 1000.0
    assert state.reward < 1.0


def test_full_episode_scores_in_unit_range():
    env = PharmaEnvironment()
    cfg = get_task_config("supply_chain_broken", seed=0)
    state = env.reset(episode_config=cfg, seed=0)
    steps = 0
    while not state.done:
        orders = {
            sku_id: sku.avg_demand_per_day * 6
            for sku_id, sku in state.skus.items()
            if sku.stockout_days_if_no_reorder < 6
        }
        state = env.step(PharmaAction(orders=orders))
        steps += 1
        assert -1.0 < state.reward <= 1.0
    assert steps == cfg.no_of_days
    assert 0.0 <= compute_final_score(state, cfg) <= 1.0
    with pytest.raises(RuntimeError):
        env.step(PharmaAction())


def test_orders_accept_json_string():
    """The /web Playground sends the Orders box as a string."""
    assert PharmaAction(orders='{"insulin": 25}').orders == {"insulin": 25.0}
    assert PharmaAction(orders="  ").orders == {}
    with pytest.raises(ValueError, match="JSON object"):
        PharmaAction(orders="not json")


def test_waste_fraction_is_share_of_delivery():
    cfg = _deterministic_config(lead_time=1.0)
    cfg.initial_inventory = {"insulin": 0.0}
    state = _deliver(cfg, {"insulin": 2000.0})   # 1000 fit, 1000 rejected
    assert state.inventory_excess_today == pytest.approx(1000.0)
    assert state.waste_fraction_today == pytest.approx(0.5)
    assert state.reward == pytest.approx(1.0 - 0.5)


def test_tiny_overflow_costs_almost_nothing():
    cfg = _deterministic_config(lead_time=1.0)
    cfg.initial_inventory = {"insulin": 0.0}
    state = _deliver(cfg, {"insulin": 1001.0})   # 1 unit of 1001 rejected
    assert state.inventory_excess_today == pytest.approx(1.0)
    assert state.waste_fraction_today == pytest.approx(1 / 1001)


def test_waste_is_weighted_by_waste_penalty():
    # Half of the insulin delivery overflows cold storage; vitamins fit.
    insulin_wasted = _deliver(_two_pool_config(), {"insulin": 200.0, "vitamins": 500.0})
    assert insulin_wasted.waste_fraction_today == pytest.approx(10.0 * 0.5 / 10.5)
    # Half of the vitamins delivery overflows ambient storage; insulin fits.
    vitamins_wasted = _deliver(_two_pool_config(), {"insulin": 50.0, "vitamins": 2000.0})
    assert vitamins_wasted.waste_fraction_today == pytest.approx(0.5 * 0.5 / 10.5)
    assert insulin_wasted.reward < vitamins_wasted.reward


def test_no_delivery_or_zero_weights_mean_no_waste_penalty():
    state = _deliver(_two_pool_config(), {})
    assert state.waste_fraction_today == 0.0
    unweighted = _deliver(_two_pool_config(0.0, 0.0), {"insulin": 200.0})
    assert unweighted.inventory_excess_today == pytest.approx(100.0)   # still reported
    assert unweighted.waste_fraction_today == 0.0


def test_cumulative_waste_score_adds_one_minus_fraction():
    cfg = _deterministic_config(lead_time=1.0)
    cfg.initial_inventory = {"insulin": 0.0}
    state = _deliver(cfg, {"insulin": 2000.0})   # day 0: no waste, day 1: half
    assert state.inventory_excess_cumulative == pytest.approx(1.0 + 0.5)


@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY))
def test_tasks_give_every_sku_a_waste_penalty(task_name):
    cfg = get_task_config(task_name, seed=0)
    assert all(sku.waste_penalty > 0 for sku in cfg.skus.values())


@pytest.mark.parametrize(
    "orders, demand, expected_stored",
    [
        ([("a", 2000.0)], {"a": 10.0}, {"a": 1000.0}),
        ([("a", 900.0), ("b", 900.0)], {"a": 10.0, "b": 10.0}, {"a": 500.0, "b": 500.0}),
        ([("a", 100.0), ("b", 2000.0)], {"a": 10.0, "b": 10.0}, {"a": 100.0, "b": 900.0}),
        ([("a", 100.0), ("b", 2000.0), ("c", 2000.0)], {"a": 10.0, "b": 10.0, "c": 30.0},
         {"a": 100.0, "b": 225.0, "c": 675.0}),
    ],
)
def test_overflowing_deliveries_fill_the_free_space(orders, demand, expected_stored):
    """Regression: oversized deliveries used to be rejected entirely instead of filling free space."""
    accepted, wasted = PharmaEnvironment()._allocate_pool(orders, 1000.0, demand)
    assert accepted == pytest.approx(expected_stored)
    assert sum(accepted.values()) == pytest.approx(1000.0)
    for sku_id, qty in orders:
        assert accepted[sku_id] + wasted[sku_id] == pytest.approx(qty)


# ---------------------------------------------------------------------------
# Storage capacity sized from demand
# ---------------------------------------------------------------------------

from tasks import AMBIENT_STORAGE_DAYS, COLD_STORAGE_DAYS  # noqa: E402


def _perfect_foresight_peak(cfg: EpisodeConfig, sku_ids) -> float:
    """
    Peak units held by a plan that knows the future and has each day's demand
    delivered as late as possible (true lead times, no noise in the built-in tasks).
    This is the least storage any plan that avoids avoidable stockouts can use.
    """
    days = cfg.no_of_days
    held = [0.0] * days
    for sku_id in sku_ids:
        sku = cfg.skus[sku_id]
        demand = [sku.base_demand * m for m in sku.demand_curve]
        arrivals = [d + max(1, round(sku.base_lead_time * sku.lead_time_curve[d])) for d in range(days)]
        for t in range(days):
            feasible = [a for a in arrivals if a <= t]
            if feasible:
                for day in range(max(feasible), t + 1):
                    held[day] += demand[t]
    return max(held)


@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY))
def test_storage_is_sized_from_demand(task_name):
    cfg = get_task_config(task_name, seed=5)
    cold = [s for s in cfg.skus.values() if s.cold_storage_required]
    ambient = [s for s in cfg.skus.values() if not s.cold_storage_required]
    assert [s.sku_id for s in cold] == ["insulin"]
    assert cfg.cold_storage_total_capacity == pytest.approx(COLD_STORAGE_DAYS * sum(s.base_demand for s in cold))
    assert cfg.ambient_storage_total_capacity == pytest.approx(AMBIENT_STORAGE_DAYS * sum(s.base_demand for s in ambient))


@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY))
def test_perfect_foresight_plan_fits_in_storage(task_name):
    """Capacity must never make good play impossible."""
    for seed in range(50):
        cfg = get_task_config(task_name, seed)
        cold = [k for k, s in cfg.skus.items() if s.cold_storage_required]
        ambient = [k for k, s in cfg.skus.items() if not s.cold_storage_required]
        assert _perfect_foresight_peak(cfg, cold) <= cfg.cold_storage_total_capacity, seed
        assert _perfect_foresight_peak(cfg, ambient) <= cfg.ambient_storage_total_capacity, seed


def test_hoarding_insulin_overflows_cold_storage():
    env = PharmaEnvironment()
    state = env.reset(task_name="flu_season", seed=0)
    month_of_insulin = 30 * env._episode_config.skus["insulin"].base_demand
    for _ in range(10):   # lead time is at most ~6 days, so the order lands within 10 days
        state = env.step(PharmaAction(orders={"insulin": month_of_insulin} if state.current_date == 0 else {}))
        if state.inventory_excess_today > 0:
            break
    assert state.inventory_excess_today > 0
    assert state.waste_fraction_today > 0
    assert state.cold_storage_current_capacity <= state.cold_storage_total_capacity + 1e-6


# ---------------------------------------------------------------------------
# Final score = average daily step reward
# ---------------------------------------------------------------------------

def _play(task_name: str, seed: int, policy):
    env = PharmaEnvironment()
    state = env.reset(task_name=task_name, seed=seed)
    rewards = []
    while not state.done:
        state = env.step(PharmaAction(orders=policy(state)))
        rewards.append(state.reward)
    return state, env._episode_config, rewards


def _order_up_to(state, cover_days: float = 6.0):
    return {
        sku_id: sku.avg_demand_per_day * cover_days
        for sku_id, sku in state.skus.items()
        if sku.stockout_days_if_no_reorder < cover_days
    }


@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY))
def test_final_score_is_mean_step_reward(task_name):
    state, cfg, rewards = _play(task_name, 3, _order_up_to)
    expected = min(1.0, max(0.0, sum(rewards) / cfg.no_of_days))
    assert compute_final_score(state, cfg) == pytest.approx(expected)


@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY))
def test_ordering_nothing_scores_about_zero(task_name):
    """It used to score ~0.40 because waste-free days were credited even with no deliveries."""
    state, cfg, _ = _play(task_name, 0, lambda s: {})
    assert compute_final_score(state, cfg) < 0.02


def test_reckless_over_ordering_scores_below_sensible_ordering():
    reckless = lambda s: {k: 10 * v for k, v in _order_up_to(s).items()}
    sensible = [compute_final_score(*_play("supply_chain_broken", seed, _order_up_to)[:2]) for seed in range(5)]
    careless = [compute_final_score(*_play("supply_chain_broken", seed, reckless)[:2]) for seed in range(5)]
    assert sum(careless) < sum(sensible)


def test_partial_episode_counts_unplayed_days_as_zero():
    env = PharmaEnvironment()
    state = env.reset(task_name="flu_season", seed=1)
    cfg = env._episode_config
    rewards = []
    for _ in range(10):
        state = env.step(PharmaAction(orders=_order_up_to(state)))
        rewards.append(state.reward)
    assert compute_final_score(state, cfg) == pytest.approx(max(0.0, sum(rewards)) / cfg.no_of_days)
