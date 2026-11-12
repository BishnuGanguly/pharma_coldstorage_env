"""
Worlds: an inventory setup described in a file instead of in code.

A world file (YAML or JSON) lists the products, the storage they live in, and the
tasks: each task is the same products under a different set of events (supplier
delays, demand waves, demand steps). `build_episode(world, task, seed)` turns it into
the EpisodeConfig the environment runs. The built-in pharma world (worlds/pharma.yaml)
is the original environment, so anything that works for pharma works for any world.

    from world import load_world, build_episode
    world = load_world("worlds/my_store.yaml")      # or load_world("pharma")
    config = build_episode(world, "holiday_rush", seed=0)

Numbers and ranges
------------------
Any amount can be a single number or a range [low, high]. A range is drawn uniformly
once per episode from the seed, so the same seed always builds the same episode. Days
(peak_day, start_day, end_day) are whole numbers counted from 0.

Draw order (what makes a seed reproducible): first every event of the task, in the
order listed, drawing its ranges in field order (supplier_delay: multiplier;
demand_wave: amplitude, width; demand_step: multiplier); then every product, in the
order listed, drawing base_demand, base_lead_time, starting_inventory.

Events
------
    supplier_delay  lead times are `multiplier` x normal for orders placed on days
                    start_day..end_day. Delays on the same product apply in order, so
                    a later one wins where two overlap.
    demand_wave     demand rises and falls as a bell curve around peak_day: at the peak
                    it is (1 + amplitude) x normal; `width` (days) sets how wide it is.
    demand_step     demand is `multiplier` x normal on days start_day..end_day
                    (a promotion above 1, a closure below 1).
Demand events on the same product multiply together, so two waves make two peaks.

Storage
-------
Any number of pools, with any names. Each holds `capacity_days` days of the combined
base demand of the products stored in it, so capacity follows the demand drawn for
the episode; a pool no product uses keeps a token capacity of 1 unit. Deliveries that
do not fit in their pool are wasted, pool by pool. `cold_chain: true` marks
refrigerated pools. With exactly one cold-chain pool and one other pool (like pharma)
the LLM report keeps the original cold/ambient layout; otherwise it lists every pool.
"""

from __future__ import annotations

import json
from math import exp
from pathlib import Path
from random import Random
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from models import EpisodeConfig, SKUEpisodeConfig, StoragePoolConfig

WORLDS_DIR = Path(__file__).resolve().parent / "worlds"
DEFAULT_WORLD = "pharma"


class WorldError(ValueError):
    """A world file that cannot be read or does not describe a valid world."""


# ---------------------------------------------------------------------------
# Numbers and ranges
# ---------------------------------------------------------------------------

Amount = Union[float, Tuple[float, float]]


def _check_amount(value: Any, *, positive: bool = False, non_negative: bool = False) -> Amount:
    """A number, or a [low, high] range with low <= high; optionally > 0 or >= 0."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("a range must be [low, high]")
        low, high = (float(v) for v in value)
        if low > high:
            raise ValueError(f"range [{low}, {high}] has low > high")
        values: Tuple[float, ...] = (low, high)
        result: Amount = (low, high)
    else:
        if isinstance(value, bool):
            raise ValueError("expected a number or a [low, high] range")
        values = (float(value),)
        result = float(value)
    if positive and min(values) <= 0:
        raise ValueError("must be greater than 0")
    if non_negative and min(values) < 0:
        raise ValueError("must be 0 or more")
    return result


def draw(rng: Random, amount: Amount) -> float:
    """A fixed number as is; a range drawn uniformly (one draw from rng)."""
    if isinstance(amount, tuple):
        return rng.uniform(*amount)
    return amount


class _Strict(BaseModel):
    # Unknown keys are errors: a typo in a world file must not be silently ignored.
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class Noise(_Strict):
    demand_cv: float = Field(
        default=0.12, ge=0.0,
        description="Daily demand varies with a standard deviation of demand_cv x base demand.",
    )
    lead_time_std: float = Field(
        default=0.75, ge=0.0, description="Each order's lead time varies with this standard deviation, in days.",
    )


class StoragePool(_Strict):
    capacity_days: float = Field(gt=0.0, description="Days of its products' combined base demand the pool holds.")
    cold_chain: bool = Field(default=False, description="True for refrigerated storage.")


class Product(_Strict):
    storage: str = Field(description="Name of the storage pool the product is kept in.")
    base_demand: Any = Field(description="Units per day: a number or a [low, high] range.")
    base_lead_time: Any = Field(description="Days from order to delivery: a number or a [low, high] range.")
    starting_inventory: Any = Field(default=0.0, description="Units on the shelf on day 0.")
    stockout_penalty: float = Field(ge=0.0, description="How much a shortage of this product costs (relative).")
    waste_penalty: float = Field(ge=0.0, description="How much a wasted delivery of this product costs (relative).")

    @field_validator("base_demand", "base_lead_time")
    @classmethod
    def _positive(cls, v: Any) -> Amount:
        return _check_amount(v, positive=True)

    @field_validator("starting_inventory")
    @classmethod
    def _non_negative(cls, v: Any) -> Amount:
        return _check_amount(v, non_negative=True)


class SupplierDelay(_Strict):
    type: Literal["supplier_delay"]
    product: str
    start_day: int = Field(ge=0)
    end_day: int = Field(ge=0)
    multiplier: Any = Field(description="Lead time x this, e.g. [2, 5].")

    @field_validator("multiplier")
    @classmethod
    def _positive(cls, v: Any) -> Amount:
        return _check_amount(v, positive=True)

    @model_validator(mode="after")
    def _days(self) -> SupplierDelay:
        if self.start_day > self.end_day:
            raise ValueError("start_day must not be after end_day")
        return self


class DemandWave(_Strict):
    type: Literal["demand_wave"]
    product: str
    peak_day: int = Field(ge=0)
    amplitude: Any = Field(description="Extra demand at the peak: 2 means 3x normal.")
    width: Any = Field(description="Days from the peak to where the wave has fallen by about 40%.")

    @field_validator("amplitude")
    @classmethod
    def _non_negative(cls, v: Any) -> Amount:
        return _check_amount(v, non_negative=True)

    @field_validator("width")
    @classmethod
    def _positive(cls, v: Any) -> Amount:
        return _check_amount(v, positive=True)


class DemandStep(_Strict):
    type: Literal["demand_step"]
    product: str
    start_day: int = Field(ge=0)
    end_day: int = Field(ge=0)
    multiplier: Any = Field(description="Demand x this on the days of the step.")

    @field_validator("multiplier")
    @classmethod
    def _non_negative(cls, v: Any) -> Amount:
        return _check_amount(v, non_negative=True)

    @model_validator(mode="after")
    def _days(self) -> DemandStep:
        if self.start_day > self.end_day:
            raise ValueError("start_day must not be after end_day")
        return self


Event = Annotated[Union[SupplierDelay, DemandWave, DemandStep], Field(discriminator="type")]


class Task(_Strict):
    description: str = ""
    events: List[Event] = Field(default_factory=list)


class World(_Strict):
    name: str = Field(min_length=1)
    description: str = ""
    days: int = Field(gt=0, description="Episode length in days.")
    noise: Noise = Field(default_factory=Noise)
    storage: Dict[str, StoragePool] = Field(min_length=1)
    products: Dict[str, Product] = Field(min_length=1)
    tasks: Dict[str, Task] = Field(min_length=1)

    @model_validator(mode="after")
    def _references(self) -> World:
        for name, product in self.products.items():
            if product.storage not in self.storage:
                raise ValueError(f"product '{name}' uses storage '{product.storage}', "
                                 f"which is not defined (storage pools: {sorted(self.storage)})")
        for task_name, task in self.tasks.items():
            for i, event in enumerate(task.events):
                where = f"task '{task_name}', event {i + 1} ({event.type})"
                if event.product not in self.products:
                    raise ValueError(f"{where}: unknown product '{event.product}' "
                                     f"(products: {sorted(self.products)})")
                last = getattr(event, "end_day", getattr(event, "peak_day", 0))
                if last >= self.days:
                    raise ValueError(f"{where}: day {last} is past the end of the episode "
                                     f"(days 0 to {self.days - 1})")
        return self

    @property
    def task_names(self) -> List[str]:
        return list(self.tasks)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects a key written twice in one mapping (YAML would keep the last)."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> Dict[Any, Any]:
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key '{key}'", key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def builtin_worlds() -> List[str]:
    """Names of the worlds shipped in worlds/ (e.g. 'pharma')."""
    return sorted(p.stem for p in WORLDS_DIR.glob("*.yaml"))


def _read(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorldError(f"cannot read world file {path}: {exc.strerror or exc}") from None
    try:
        if path.suffix.lower() == ".json":
            return json.loads(text)
        return yaml.load(text, Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise WorldError(f"{path} is not valid {'JSON' if path.suffix.lower() == '.json' else 'YAML'}: {exc}") from None


def _format_errors(exc: ValidationError, source: str) -> str:
    lines = [f"{source} is not a valid world:"]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        msg = err["msg"].removeprefix("Value error, ")
        lines.append(f"  - {loc}: {msg}" if loc else f"  - {msg}")
    return "\n".join(lines)


def load_world(source: Union[str, Path, Mapping[str, Any], World] = DEFAULT_WORLD,
               allow_files: bool = True) -> World:
    """
    A World from: a World (returned as is), a dict, the name of a built-in world
    ('pharma'), or the path of a .yaml/.yml/.json file. Raises WorldError with every
    problem found, each with where it is.

    allow_files=False accepts only Worlds, dicts and built-in names: use it for input
    from the network, which must not make the server read its own files.
    """
    if isinstance(source, World):
        return source
    if isinstance(source, Mapping):
        data, label = dict(source), "world"
    elif not allow_files and str(source) not in builtin_worlds():
        raise WorldError(f"world must be a built-in name {builtin_worlds()} or a world given as a mapping, "
                         f"not '{source}'")
    else:
        path = Path(source)
        if path.suffix == "" and not path.exists():
            if str(source) not in builtin_worlds():
                raise WorldError(f"no built-in world named '{source}' (built-in: {builtin_worlds()}); "
                                 "to load a file, give its path, e.g. worlds/my_store.yaml")
            path = WORLDS_DIR / f"{source}.yaml"
        data, label = _read(path), str(path)
    if not isinstance(data, dict):
        raise WorldError(f"{label} must contain a mapping (name, days, storage, products, tasks)")
    try:
        return World.model_validate(data)
    except ValidationError as exc:
        raise WorldError(_format_errors(exc, label)) from None


# ---------------------------------------------------------------------------
# Building an episode
# ---------------------------------------------------------------------------

def wave_curve(peak_day: int, amplitude: float, width: float, days: int) -> List[float]:
    """1 + amplitude x a bell curve centred on peak_day (the same as tasks.gaussian_curve)."""
    return [1.0 + amplitude * exp(-0.5 * ((day - peak_day) / width) ** 2) for day in range(days)]


def build_episode(world: World, task_name: str, seed: Optional[int] = None) -> EpisodeConfig:
    """
    The EpisodeConfig of one task of `world` for `seed` (see the module docstring for
    the draw order that makes it reproducible). Equal arguments give equal configs.
    """
    try:
        task = world.tasks[task_name]
    except KeyError:
        raise WorldError(f"world '{world.name}' has no task '{task_name}' (tasks: {world.task_names})") from None
    rng = Random(seed)
    days = world.days

    # 1. Events, in order. Curves start flat; demand events multiply, delays overwrite.
    demand: Dict[str, List[float]] = {}
    lead: Dict[str, List[float]] = {}
    for event in task.events:
        if isinstance(event, SupplierDelay):
            m = draw(rng, event.multiplier)
            curve = lead.setdefault(event.product, [1.0] * days)
            for day in range(event.start_day, event.end_day + 1):
                curve[day] = m
        elif isinstance(event, DemandWave):
            amplitude = draw(rng, event.amplitude)
            width = draw(rng, event.width)
            wave = wave_curve(event.peak_day, amplitude, width, days)
            previous = demand.get(event.product)
            demand[event.product] = wave if previous is None else [a * b for a, b in zip(previous, wave)]
        else:  # DemandStep
            m = draw(rng, event.multiplier)
            curve = demand.setdefault(event.product, [1.0] * days)
            for day in range(event.start_day, event.end_day + 1):
                curve[day] *= m

    # 2. Products, in order.
    skus: Dict[str, SKUEpisodeConfig] = {}
    initial_inventory: Dict[str, float] = {}
    for sku_id, product in world.products.items():
        base_demand = draw(rng, product.base_demand)
        skus[sku_id] = SKUEpisodeConfig(
            sku_id=sku_id,
            no_of_days=days,
            base_demand=base_demand,
            demand_curve=demand.get(sku_id) or [1.0] * days,
            demand_std=world.noise.demand_cv * base_demand,
            base_lead_time=draw(rng, product.base_lead_time),
            lead_time_curve=lead.get(sku_id) or [1.0] * days,
            lead_time_std=world.noise.lead_time_std,
            cold_storage_required=world.storage[product.storage].cold_chain,
            storage_pool=product.storage,
            stockout_penalty=product.stockout_penalty,
            waste_penalty=product.waste_penalty,
        )
        initial_inventory[sku_id] = draw(rng, product.starting_inventory)

    # 3. Storage: each pool holds capacity_days of its products' base demand; a pool
    #    with no products keeps a token 1 unit, as capacities must be positive.
    storage_pools: Dict[str, StoragePoolConfig] = {}
    for pool_name, pool in world.storage.items():
        pool_demand = sum(skus[s].base_demand for s, p in world.products.items() if p.storage == pool_name)
        storage_pools[pool_name] = StoragePoolConfig(capacity=max(pool.capacity_days * pool_demand, 1.0),
                                                     cold_chain=pool.cold_chain)

    return EpisodeConfig(
        task_name=task_name,
        no_of_days=days,
        skus=skus,
        initial_inventory=initial_inventory,
        storage_pools=storage_pools,
    )
