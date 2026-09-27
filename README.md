---
title: Pharma Cold-Chain Inventory Environment
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Pharma Inventory Management — OpenEnv Benchmark

An OpenEnv-compliant benchmark where an LLM agent manages a pharmaceutical warehouse over a 60-day episode. Every day the agent reads an inventory report and decides what to order and how much. Demand is stochastic, lead times are uncertain, and disruptions hit without warning. The agent must act under uncertainty with no visibility into the underlying processes driving the environment.

---

## The Problem

A warehouse stocks five drugs: insulin, BP medication, paracetamol, vitamins, and hydroxychloroquine. Every day the agent must decide:

- **What to order** — which SKUs need replenishment, and which can wait.
- **How much** — enough to cover demand without wasting capacity on overstock.

The hard parts are:

- You cannot see today's true demand — only recent observed history.
- Lead times are stochastic and can extend without warning during supply disruptions.
- Insulin has no substitute and a 100-point stockout penalty. Vitamins carry a 5-point penalty. The agent must learn this hierarchy.
- Over-ordering wastes storage capacity and is penalised — but under-ordering causes stockouts.

---

## Warehouse

### Storage Pools

| Pool | Capacity | Used for |
|---|:---:|---|
| Cold storage | 500 units | Cold-chain SKUs (reserved for future use) |
| Ambient storage | 25,000 units | All current SKUs |

### SKUs

All five SKUs in the current version use ambient storage.

| SKU | Drug | Stockout Penalty | Base Demand | Base Lead Time |
|---|---|:---:|---|---|
| `insulin` | Insulin | 100 | 8–12 units/day | 4–6 days |
| `bp_medication` | Amlodipine | 60 | 40–60 units/day | 2–4 days |
| `paracetamol` | Paracetamol | 20 | 180–220 units/day | 1–3 days |
| `vitamins` | Vitamins | 5 | 60–90 units/day | 1–2 days |
| `hydroxychloroquine` | Hydroxychloroquine (HCQ) | 35 | 20–40 units/day | 5–8 days |

Base demand, base lead time and starting inventory (1–5 units, so every episode starts nearly empty) are sampled when the task is built, from the ranges above. Tasks are seeded: the same `seed` always produces the same episode. True daily demand is `base_demand × demand_curve[day]`, plus Gaussian noise when a SKU's `demand_std` is set (the built-in tasks leave it at 0); lead times work the same way with `lead_time_curve` and `lead_time_std`.

---

## What the Agent Observes

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
| `inventory_excess_today` | Waste from overflow arrivals today |
| `inventory_excess_cumulative` | Cumulative waste across the episode |

---

## Action Format

Each day the agent responds with a JSON object mapping SKU names to order quantities:

```json
{"insulin": 100, "paracetamol": 500}
```

Only include SKUs you want to order. Send `{}` to place no orders today.

The environment also accepts a raw LLM message string — it extracts the first valid JSON object from the text automatically. SKU names are matched case-insensitively and numeric strings are accepted; unknown SKUs, non-positive or non-numeric quantities are ignored instead of failing the step.

---

## Tasks

Three scenario types. All episodes run 60 days. Base demand and lead times are randomised at episode start within the ranges in the SKU table.

### Task 1 — Supply Chain Broken (`supply_chain_broken`)

Demand is flat across all SKUs. The challenge is entirely supply-side: lead times extend sharply for two SKUs in sequence, forcing the agent to build safety stock before the disruption hits.

- Insulin lead times step up during days 15–25.
- Paracetamol lead times step up during days 30–40.
- Step amplitudes are randomised at episode start (2×–5× baseline).

The agent that waits for stockout signals will order too late — orders placed after the lead time extends arrive after the stockout.

![Supply Chain Broken — Demand & Lead Times](plots/task1_supply_chain.png)


### Task 2 — Flu Season (`flu_season`)

A flu outbreak drives a Gaussian demand surge for paracetamol, peaking at day 30 with amplitude 2× baseline. All other SKU demands remain flat.

- Paracetamol demand follows a bell curve (peak day 30, width 10).
- Paracetamol lead times also step up during days 30–40 (logistics stress during the peak).
- The agent must pre-stock before day 30 — ordering at peak demand while lead times are stretched leaves no time for delivery.

![Flu Season — Demand Curves](plots/task2_flu_season.png)


### Task 3 — Epidemic, Two Waves (`epidemic_two_wave`)

A two-wave epidemic drives hydroxychloroquine (HCQ) demand through two sequential spikes. Insulin sees a mild secondary elevation following the first wave.

- HCQ wave 1: Gaussian peak at day 17 (amplitude 1.8×).
- HCQ wave 2: Gaussian peak at day 45 (amplitude 2.5×) — larger than wave 1.
- Insulin: mild Gaussian elevation peaking ~day 20.
- HCQ lead times step up during days 17–27 (amplitude 2×–5×).
- Insulin lead times step up during days 20–30 (amplitude 1×–3×).

The trough between waves (days 27–40) is a trap: demand is low but the agent must keep ordering because wave 2 is larger and HCQ lead times are long (5–8 day baseline).

![Epidemic Two Wave — HCQ Demand](plots/task3_epidemic.png)

---

## Scoring

### Per-Step Reward

At each step the reward is:

```
step_reward = mean(demand_fulfilled_today across all SKUs)
            - inventory_excess_today / (1 + inventory_excess_today)
```

- `demand_fulfilled_today` per SKU is the fraction of true demand served (0.0 = full stockout, 1.0 = fully served).
- The excess term penalises waste from overflow arrivals. It is bounded to [0, 1), so the step reward lies in (-1, 1].

### Final Episode Score

```
final_score = (mean(demand_fulfilled_cumulative across all SKUs) × 0.6
            +  inventory_excess_cumulative × 0.4)
            / no_of_days
```

- `demand_fulfilled_cumulative` is the sum of `demand_fulfilled_today` across all days for each SKU.
- `inventory_excess_cumulative` is the sum of `1 / (1 + inventory_excess_today)` across all days — higher values mean less daily waste, so a higher score here is better.
- Dividing by `no_of_days` normalises the score to a per-day average.

---

## Baselines

| Model | Supply Chain | Flu Season | Epidemic Two Waves |
|---|:---:|:---:|:---:|
| *Results to be published post-evaluation* | — | — | — |

---

## Setup

```bash
# Install dependencies
uv sync

# Start the environment server
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Run the benchmark (separate terminal)
uv run python inference.py
```

### Web Playground

The Docker image sets `ENABLE_WEB_INTERFACE=true`, so the server also serves OpenEnv's Gradio Playground at `/web` (`/` redirects there). On the Hugging Face Space it is the page that opens.

- **Reset** starts a new episode, then **Step** advances one day.
- Type orders in either box: **Orders** takes a JSON object such as `{"insulin": 40}`, and **Message** takes free text containing one, as an LLM would reply.
- The raw JSON shows the full `InventoryState`, including ground-truth fields such as `actual_inbound_orders` that the agent's prompt in `inference.py` leaves out.
- The Reset button takes no parameters, so it loads the small 30-day, two-SKU default episode. To play one of the three tasks, reset through the API and keep using the page:

```bash
curl -X POST http://localhost:8000/web/reset \
     -H 'Content-Type: application/json' \
     -d '{"task_name": "flu_season", "seed": 42}'
```

To run it locally:

```bash
ENABLE_WEB_INTERFACE=true uv run uvicorn server.app:app --port 8000
# open http://localhost:8000/web
```

### Choosing a task on reset

`reset()` accepts either a registered task name or a full config:

```python
await env.reset(task_name="flu_season", seed=42)
await env.reset(episode_config=get_task_config("flu_season", seed=42).model_dump(mode="json"))
```

With neither, a small 30-day two-SKU default episode is used.

### Tests

```bash
uv pip install pytest
uv run python -m pytest tests      # unit tests for the simulation
uv run python smoke_test.py        # direct environment walkthrough
uv run python smoke_test.py --http # also exercises the client (server must be running)
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | — | HuggingFace API key |
| `ENV_BASE_URL` | `http://localhost:8000` | Server URL |
| `LOCAL_IMAGE_NAME` | — | Run the server from this Docker image instead of `ENV_BASE_URL` |
| `TASK_SEED` | `42` | Seed for task generation and environment noise |

---

## Project Structure

```
pharma_coldstorage_env/
├── models.py          # Pydantic schemas: SKUState, InventoryState,
│                      # PharmaAction, EpisodeConfig, SKUEpisodeConfig
├── tasks.py           # Task constructors (demand/lead-time curves)
│                      # + compute_step_reward, compute_final_score
├── inference.py       # Benchmark runner — loops over TASK_REGISTRY
├── client.py          # OpenEnv async HTTP client (PharmaEnvClient)
├── smoke_test.py      # Direct environment test (no server required)
├── tests/             # pytest unit tests
├── openenv.yaml       # Environment manifest
└── server/
    ├── app.py                 # FastAPI application
    └── my_env_environment.py  # Core simulation: demand sampling,
                               # arrivals, overflow, insights, LLM prompt
```