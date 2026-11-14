# What the agent sees and how it answers

The daily report an agent receives, and the three answer formats an LLM agent can use.

## What the agent observes

Each step the agent receives a full `InventoryState` rendered as a JSON report. Per-SKU signals:

| Field | Description |
|---|---|
| `inventory_on_hand` | Units on shelf right now |
| `stockout_days_if_no_reorder` | Days until stockout at current demand rate — most urgent signal |
| `avg_demand_per_day` | Rolling average demand over all history |
| `avg_demand_last_5_days` | Short-window average — reacts faster to spikes |
| `avg_lead_time` | Rolling average delivery time from all past orders |
| `lead_time_last3_orders` | Lead times of the last 3 delivered orders — rising values signal supply stress. A lead time only becomes visible once that order arrives |
| `expected_inbound_orders` | Open orders with estimated arrival day |
| `stockout_penalty` | Criticality weight for this SKU |

Global signals:

| Field | Description |
|---|---|
| `cold_storage_ratio` | Cold pool utilisation (0–1) |
| `ambient_storage_ratio` | Ambient pool utilisation (0–1) |
| `inventory_excess_today` | Units of today's deliveries rejected because storage was full |
| `waste_fraction_today` | Share of today's deliveries rejected, weighted by `waste_penalty` (0–1) |
| `inventory_excess_cumulative` | Sum of `1 - waste_fraction_today` so far (+1 per waste-free day) |


## Answer formats

Each day the agent responds with a JSON object mapping SKU names to order quantities:

```json
{"insulin": 100, "paracetamol": 500}
```

Only include SKUs you want to order. Send `{}` to place no orders today.

The LLM prompt (`inference.py`, also used by `agents.LLMAgent`, `eval.py`, the dashboard and `PharmaEnvironment.to_llm_prompt()`) shows the report one SKU per line and adds two values computed from the observation: `inbound_units` (already ordered, not yet arrived) and `days_of_cover` ((on hand + inbound) ÷ recent daily demand). It shows the results of the last 3 days but not the model's own earlier replies, and its instructions contain no example quantities: small models tend to copy either one instead of reading the report.

**Three action formats for LLM agents** (`action=` in `agents.LLMAgent`, `--action` in `eval.py` and `local_llm/run_local_llm.py`):

| Format | The model answers | Worst case |
|---|---|---|
| `adjust` (default) | extra days per SKU, 0 to +10, added to the baseline's default of lead time + 3 safety days | about the baseline: a copied number, a 0, a missing SKU or a reply without JSON keeps the default |
| `days` | the days of stock it wants per SKU | an empty shelf: an answer of 0 orders nothing |
| `units` | units to order per SKU | anything |

**`adjust`:** the baseline rule does the ordering and the model only decides where to keep extra stock on top of the default, e.g. `{"paracetamol": 4, "vitamins": 0, "insulin": 1}`. The report leaves out `days_of_cover`: Qwen 2.5 3B, answering in `days`, copied it until its stock ran out (scores 0.17-0.44). On 20 test seeds per task, answering 0 for everything scores 0.869 (the baseline), +3 for everything 0.886, +5 for everything 0.893 and +10 for everything (e.g. a copied large number, clipped) 0.882; the oracle reaches 0.932. Negative values are not allowed: keeping less than the default only added stockouts (-1 for everything: 0.853, -3: 0.728).

**`days`:** the model answers with the **days of stock** it wants per SKU, e.g. `{"paracetamol": 8, "insulin": 6}`, and `inference.days_to_units()` turns that into an order using only observed values:

```
units ordered = days x recent daily demand - inventory_on_hand - inbound_units
```

Recent daily demand is the larger of `avg_demand_last_5_days` and `avg_demand_per_day`. Days are capped at 21 (the storage size), and orders under one unit are dropped. The model decides *how much cover* it wants; the arithmetic, which small models get wrong, is done in code. With `--action units` the model writes units itself, using `SYSTEM_PROMPT` (the format `inference.py` and the benchmark use). For example, Gemma 2 2B scored 0.24 in units mode, mostly because it copied the report's small numbers as order sizes.

The environment also accepts a raw LLM message string — it extracts the first valid JSON object from the text automatically. SKU names are matched case-insensitively and numeric strings are accepted; unknown SKUs, non-positive or non-numeric quantities are ignored instead of failing the step.
