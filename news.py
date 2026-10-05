"""
News: disruptions announced to the agent a few days before they start.

Every built-in task already contains its disruptions, as per-SKU demand and lead-time
curves (tasks.py). This module reads those curves, turns each disruption into a
NewsEvent, and writes the text the agent will read. The world itself is unchanged: news
only tells the agent, ahead of time, what the curves will do.

News levels
-----------
    0  no news: the original environment.
    1  one fixed template with exact days and multipliers (easy; a parser could read it).
    2  varied wording with approximate timing and severity, given relative to the day
       the news is published (needs reading comprehension).

Each event is published a random 5-10 days before it starts (never before day 0).
All of the news randomness comes from its own generator seeded from the task and seed,
so adding news never changes an episode's demand, lead times or noise.
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Tuple

from models import EpisodeConfig, NewsEvent

NEWS_LEVELS = (0, 1, 2)
# Days between publication and the start of the event, drawn per event.
ANNOUNCE_DAYS_AHEAD = (5, 10)
# A demand surge is the part of the demand curve at or above this multiple of normal.
SURGE_THRESHOLD = 1.5
# A lead-time curve above this counts as a supplier delay.
DELAY_THRESHOLD = 1.0001

# How each SKU is named in level-2 news: the drug, as a news story would put it.
PRODUCT_NAMES: Dict[str, str] = {
    "insulin": "insulin",
    "paracetamol": "paracetamol",
    "bp_medication": "amlodipine (blood-pressure medication)",
    "vitamins": "vitamin supplements",
    "hydroxychloroquine": "hydroxychloroquine (HCQ)",
}

DELAY_CAUSES: Tuple[str, ...] = (
    "a port strike",
    "a fire at a manufacturing plant",
    "a customs backlog",
    "a shortage of raw materials",
    "a truck drivers' strike",
)
COLD_CHAIN_CAUSES: Tuple[str, ...] = (
    "a strike at the cold-chain carrier",
    "a breakdown of refrigerated transport",
)
OUTBREAKS: Dict[str, str] = {
    "paracetamol": "an early flu season",
    "hydroxychloroquine": "a new epidemic wave",
}


# ---------------------------------------------------------------------------
# Finding the events in the curves
# ---------------------------------------------------------------------------

def _runs(values: List[float], above: float) -> List[Tuple[int, int]]:
    """(first_day, last_day) of each run of consecutive days with value >= above."""
    runs, start = [], None
    for day, value in enumerate(values):
        if value >= above and start is None:
            start = day
        elif value < above and start is not None:
            runs.append((start, day - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def find_events(config: EpisodeConfig) -> List[Dict]:
    """
    The disruptions in an episode's curves, as dicts of NewsEvent fields (no text yet),
    in the order they start: a supplier delay for every run of raised lead times, a
    demand surge for every run of demand at or above SURGE_THRESHOLD x normal.
    """
    events = []
    for sku_id, sku in config.skus.items():
        for kind, curve, above in (("supplier_delay", sku.lead_time_curve, DELAY_THRESHOLD),
                                   ("demand_surge", sku.demand_curve, SURGE_THRESHOLD)):
            for start, end in _runs(curve or [], above):
                window = curve[start:end + 1]
                peak = start + window.index(max(window))
                events.append({"kind": kind, "sku_id": sku_id, "start_day": start, "end_day": end,
                               "peak_day": start if kind == "supplier_delay" else peak,
                               "multiplier": round(max(window), 2)})
    return sorted(events, key=lambda e: (e["start_day"], e["sku_id"], e["kind"]))


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def exact_message(event: Dict) -> str:
    """Level 1: one fixed template with the exact days and multiplier."""
    sku, start, end, m = event["sku_id"], event["start_day"], event["end_day"], event["multiplier"]
    if event["kind"] == "supplier_delay":
        return (f"SUPPLIER DELAY - {sku}: orders placed from day {start} to day {end} "
                f"will take about {m:.1f}x the normal lead time.")
    return (f"DEMAND SURGE - {sku}: demand will be at least {SURGE_THRESHOLD:.1f}x normal from day {start} "
            f"to day {end}, peaking at about {m:.1f}x normal around day {event['peak_day']}.")


def _in_days(days: int) -> str:
    """When something starts, as a phrase that reads after "starting" or "expected ..."."""
    if days <= 1:
        return "tomorrow"
    if days in (6, 8):
        return "next week"
    return f"in about {days} days"


def _duration(days: int) -> str:
    """How long something lasts, as a phrase that reads after "for" or "last"."""
    if days >= 14:
        return f"roughly {round(days / 7)} weeks"
    return f"about {days} days"


def _delay_severity(m: float, rng: Random) -> str:
    """How much slower deliveries get, as a verb phrase: in a rounded number or in words."""
    if rng.random() < 0.5:
        return f"take around {round(m)} times longer than usual" if m >= 1.75 else "take somewhat longer than usual"
    if m < 2.5:
        return "be noticeably slower than usual"
    if m < 3.75:
        return "be severely delayed"
    return "be extremely delayed, several times slower than usual"


def _surge_size(m: float) -> str:
    if m < 1.75:
        return "rise by about half"
    if m < 2.5:
        return "roughly double"
    if m < 3.5:
        return "roughly triple"
    return "rise several-fold"


def varied_message(event: Dict, rng: Random) -> str:
    """
    Level 2: one of several wordings, with timing relative to the publication day and
    approximate (+-1 day on the start, +-2 on the length and peak) and severity in words
    or a rounded number.
    """
    product = PRODUCT_NAMES.get(event["sku_id"], event["sku_id"])
    lead = event["start_day"] - event["announce_day"]
    when = _in_days(max(1, lead + rng.randint(-1, 1)))
    length = _duration(max(2, event["end_day"] - event["start_day"] + 1 + rng.randint(-2, 2)))
    m = event["multiplier"]

    if event["kind"] == "supplier_delay":
        causes = COLD_CHAIN_CAUSES if event["sku_id"] == "insulin" else DELAY_CAUSES
        cause = rng.choice(causes)
        severity = _delay_severity(m, rng)
        templates = (
            f"Supply alert: {cause} is expected to hit {product} deliveries {when}. "
            f"New orders could {severity} for {length}.",
            f"{product[0].upper() + product[1:]} suppliers report {cause}. Starting {when}, new orders "
            f"may {severity}; the disruption could last {length}.",
            f"Logistics warning: because of {cause}, deliveries of {product} will {severity}, "
            f"starting {when}, for {length}.",
        )
        return rng.choice(templates)

    outbreak = OUTBREAKS.get(event["sku_id"], "an outbreak")
    peak_in = max(1, event["peak_day"] - event["announce_day"] + rng.randint(-2, 2))
    surge = _surge_size(m)
    templates = (
        f"Health authorities warn of {outbreak}. Demand for {product} is expected to {surge} "
        f"{when}, peaking in about {peak_in} days.",
        f"Hospitals are preparing for {outbreak}: {product} prescriptions could {surge} {when} "
        f"and stay high for {length}.",
        f"Pharmacy chains report early signs of {outbreak}. Expect {product} demand to {surge}, "
        f"starting {when}; the peak is likely in about {peak_in} days.",
    )
    return rng.choice(templates)


# ---------------------------------------------------------------------------
# Adding news to an episode
# ---------------------------------------------------------------------------

def add_news(config: EpisodeConfig, level: int, seed: Optional[int]) -> EpisodeConfig:
    """
    Return `config` with its disruptions announced as news at the given level.
    Level 0 returns it unchanged. The episode's world (curves, noise) is never altered.
    """
    if level not in NEWS_LEVELS:
        raise ValueError(f"news level must be one of {NEWS_LEVELS}, not {level!r}")
    if level == 0:
        return config
    rng = Random(f"news|{config.task_name}|{seed}")
    events = []
    for event in find_events(config):
        event["announce_day"] = max(0, event["start_day"] - rng.randint(*ANNOUNCE_DAYS_AHEAD))
        event["message"] = exact_message(event) if level == 1 else varied_message(event, rng)
        events.append(NewsEvent(**event))
    return config.model_copy(update={"news": events})


def visible_news(config: EpisodeConfig, day: int) -> List[str]:
    """The news the agent sees on `day`: published already, event not over yet; oldest first."""
    return [f"Day {e.announce_day}: {e.message}"
            for e in sorted(config.news, key=lambda e: e.announce_day)
            if e.announce_day <= day <= e.end_day]
