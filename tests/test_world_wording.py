"""Tests for prompts and news written from the world file (world.py wording, news.py, inference.py)."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import inference
from agents import LLMAgent, NewsLLMAgent
from models import InventoryState, PharmaAction
from server.my_env_environment import PharmaEnvironment
from tasks import PHARMA_WORLD, TASK_REGISTRY, get_task_config
from world import WorldError, build_episode, load_world

ROOT = Path(__file__).resolve().parent.parent
GROCERY = load_world(ROOT / "examples" / "worlds" / "grocery.yaml")
PHARMA_WORDS = ("insulin", "pharma", "prescription", "paracetamol", "hydroxychloroquine")
ACTIONS = ("units", "days", "adjust")


def grocery_data():
    return yaml.safe_load((ROOT / "examples" / "worlds" / "grocery.yaml").read_text())


def pharma_copy(**changes):
    """The pharma world file with a change, which makes it a world of its own."""
    data = yaml.safe_load((ROOT / "worlds" / "pharma.yaml").read_text())
    data.update(changes or {"description": "A copy."})
    return load_world(data)


def reset(world, task, news=2, seed=1):
    env = PharmaEnvironment()
    return env, env.reset(world=world, task_name=task, seed=seed, news=news)


def two_pool_grocery():
    data = grocery_data()
    del data["storage"]["freezer"]
    data["products"]["ice_cream"]["storage"] = "fridge"
    return load_world(data)


# ---------------------------------------------------------------------------
# What an episode carries
# ---------------------------------------------------------------------------

def test_only_the_built_in_pharma_world_keeps_the_original_wording():
    for task in TASK_REGISTRY:
        config = build_episode(PHARMA_WORLD, task, 3)
        assert config.context is None and config.news_wording is None
    copied = build_episode(pharma_copy(), "flu_season", 3)
    assert copied.context.business == "a pharmaceutical warehouse"
    assert copied.context.products["bp_medication"].name == "blood-pressure medication"
    assert copied.news_wording.surge_causes["paracetamol"] == "an early flu season"


def test_context_comes_from_the_world_file_with_sensible_defaults():
    config = build_episode(GROCERY, "holiday_rush", 1)
    products = config.context.products
    assert config.context.business == "a neighbourhood grocery store"
    assert (products["milk"].name, products["milk"].aliases) == ("fresh milk", ["whole milk"])
    assert products["ice_cream"].name == "ice cream"                  # default: the id with spaces
    assert config.news_wording.surge_causes["ice_cream"] == "a heat wave"
    bare = grocery_data()
    del bare["business"], bare["news"]
    plain = build_episode(load_world(bare), "holiday_rush", 1)
    assert plain.context.business == "a warehouse" and plain.news_wording.default_surge_cause == "a rise in demand"
    assert plain.news_wording.delay_causes                             # the general defaults


def test_observation_carries_the_context_and_survives_a_round_trip():
    _, obs = reset(GROCERY, "holiday_rush")
    assert obs.context.business == "a neighbourhood grocery store"
    assert InventoryState.model_validate(obs.model_dump(mode="json")).context == obs.context
    assert PharmaEnvironment().reset(task_name="flu_season", seed=1).context is None


@pytest.mark.parametrize("change, message", [
    ({"news": {"surge_causes": {"bread": "a bake sale"}}}, "unknown product(s) ['bread']"),
    ({"news": {"delay_causes": []}}, "delay_causes"),
    ({"business": ""}, "business"),
])
def test_bad_wording_is_explained(change, message):
    with pytest.raises(WorldError) as info:
        load_world({**grocery_data(), **change})
    assert message in str(info.value)


def test_names_and_aliases_must_point_to_one_product():
    data = grocery_data()
    data["products"]["yogurt"]["aliases"] = ["Fresh Milk"]            # milk's name
    with pytest.raises(WorldError, match="names both product 'milk' and product 'yogurt'"):
        load_world(data)


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def test_news_uses_the_worlds_product_names_and_causes():
    messages = [e.message for task in ("holiday_rush", "delivery_strike") for seed in range(12)
                for e in get_task_config(task, seed, news=2, world=GROCERY).news]
    text = " ".join(messages)
    assert "fresh milk" in text and "ice cream" in text and "ice_cream" not in text
    assert "a holiday promotion" in text and "a heat wave" in text
    assert "a breakdown of the refrigerated delivery lorry" in text    # the cold-chain cause, for milk
    assert "a flooded motorway" in text or "a strike at the regional distribution centre" in text
    assert not any(word in text.lower() for word in ("outbreak", "hospital", "pharmac", "health authorities"))
    assert len({m.split(":")[0].split(" ")[0] for m in messages}) >= 4  # several templates in use


def test_cold_chain_causes_fall_back_to_the_general_ones_when_empty():
    data = grocery_data()
    data["news"]["cold_chain_delay_causes"] = []
    world = load_world(data)
    messages = [e.message for seed in range(8) for e in get_task_config("delivery_strike", seed, news=2, world=world).news]
    assert all(any(c in m for c in world.news.delay_causes) for m in messages)


def test_a_worlds_own_wording_draws_the_same_random_numbers():
    # A copy of pharma with the same words: the same days and, for delays (shared templates), the same text.
    copy_world = pharma_copy()
    for task in TASK_REGISTRY:
        for seed in range(6):
            original = get_task_config(task, seed, news=2).news
            copied = get_task_config(task, seed, news=2, world=copy_world).news
            assert [(e.kind, e.sku_id, e.announce_day) for e in original] == [(e.kind, e.sku_id, e.announce_day) for e in copied]
            for a, b in zip(original, copied):
                if a.kind == "supplier_delay":
                    assert a.message == b.message
                else:
                    assert a.message != b.message and "Health authorities" not in b.message
    level1 = get_task_config("flu_season", 2, news=1)
    assert [e.message for e in level1.news] == [e.message for e in get_task_config("flu_season", 2, news=1, world=copy_world).news]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def test_pharma_prompts_are_the_original_constants():
    obs = PharmaEnvironment().reset(task_name="flu_season", seed=1, news=2)
    assert inference.system_prompt("units", obs) == inference.system_prompt("units") == inference.SYSTEM_PROMPT
    assert inference.system_prompt("days", obs) == inference.SYSTEM_PROMPT_DAYS
    assert inference.system_prompt("adjust", obs) == inference.SYSTEM_PROMPT_ADJUST
    assert inference.news_system_prompt(obs) == inference.news_system_prompt() == inference.NEWS_SYSTEM_PROMPT
    assert inference.sku_aliases(obs) is inference.SKU_ALIASES


@pytest.mark.parametrize("action", ACTIONS)
def test_prompts_describe_the_world_and_its_pools(action):
    _, obs = reset(GROCERY, "holiday_rush")
    prompt = inference.system_prompt(action, obs)
    assert prompt.startswith("You are the procurement agent of a neighbourhood grocery store,")
    assert "milk       - fresh milk, sold daily, spoils fast" in prompt and "ice_cream  - ice cream" in prompt
    assert "lists every storage pool by name" in prompt and "storage_pool                 — " in prompt
    lowered = prompt.lower()
    assert not any(word in lowered for word in PHARMA_WORDS)
    assert not any(key in prompt for key in ("cold_days", "ambient_days", "cold_ratio", "ambient_ratio"))
    # Only the pharma-specific sentences change: the answer format is the original's.
    assert prompt.split("ACTION FORMAT")[1] == inference.system_prompt(action).split("ACTION FORMAT")[1]


@pytest.mark.parametrize("action", ACTIONS)
def test_a_cold_plus_ambient_world_keeps_the_cold_and_ambient_report_keys(action):
    _, obs = reset(two_pool_grocery(), "normal_weeks", news=0)
    prompt = inference.system_prompt(action, obs)
    assert "grocery store" in prompt and "Two storage pools:" in prompt and "cold_days and ambient_days" in prompt
    assert "storage_pool                 — " not in prompt
    assert not any(word in prompt.lower() for word in PHARMA_WORDS)


def test_every_swap_still_matches_the_prompt_texts():
    # If a prompt is reworded, a swap that no longer matches must be noticed here.
    rules = inference._UNITS_RULES + inference._DAYS_RULES + inference._ADJUST_RULES
    for old, _ in inference._GENERIC_SWAPS + inference._POOL_SWAPS:
        assert old in rules, old
    with pytest.raises(ValueError, match="prompt text not found"):
        inference._swap("abc", "x", "y")


def test_news_adviser_prompt_lists_the_worlds_products():
    _, obs = reset(GROCERY, "holiday_rush")
    prompt = inference.news_system_prompt(obs)
    assert "the procurement team of a neighbourhood grocery store" in prompt
    assert "milk       - fresh milk, sold daily, spoils fast, kept in cold storage" in prompt
    assert "biscuits   - biscuits\n\nGUIDELINES" in prompt
    assert not any(word in prompt.lower() for word in PHARMA_WORDS)
    assert prompt.split("GUIDELINES")[1] == inference.NEWS_SYSTEM_PROMPT.split("GUIDELINES")[1]


def test_answers_may_use_a_products_name_or_alias():
    _, obs = reset(GROCERY, "holiday_rush")
    reply = {"Fresh Milk": 3, "ice cream": 4, "RICE": 1, "bread": 9, "hcq": 2}
    assert inference.parse_extra_days(obs, reply) == {"milk": 3.0, "ice_cream": 4.0, "rice": 1.0}
    assert inference.parse_extra_days(obs, {"whole milk": 2}) == {"milk": 2.0}
    pharma = PharmaEnvironment().reset(task_name="flu_season", seed=1)
    assert inference.parse_extra_days(pharma, {"HCQ": 2, "amlodipine": 1}) == {"hydroxychloroquine": 2.0, "bp_medication": 1.0}


def test_agents_send_the_worlds_prompts():
    sent = []

    def create(**kwargs):
        sent.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"milk": 2}'))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    env, obs = reset(GROCERY, "holiday_rush")
    while not obs.news:
        obs = env.step(PharmaAction(message="{}"))
    LLMAgent("stub", "x", "x", client=client).act(obs, [])
    NewsLLMAgent("stub", "x", "x", client=client).act(obs, [])
    assert sent[0][0]["content"] == inference.system_prompt("adjust", obs)
    assert sent[1][0]["content"] == inference.news_system_prompt(obs)
    assert "grocery store" in sent[0][0]["content"] and "grocery store" in sent[1][0]["content"]
    assert "neighbourhood grocery store" in env.to_llm_prompt()
