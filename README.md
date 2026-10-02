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
| Cold storage | 21 days of insulin demand (about 170–250 units) | Insulin |
| Ambient storage | 21 days of the other four SKUs' combined demand (about 6,300–8,600 units) | Paracetamol, BP medication, vitamins, HCQ |

Each pool's capacity is 21 times the summed base demand of the SKUs stored in it, so it scales with the demand drawn for each episode. That is about 25% more than the most a perfect-foresight plan ever needs on any task, so good play always fits, while over-ordering overflows and the excess is wasted (see Scoring). `AMBIENT_STORAGE_DAYS` and `COLD_STORAGE_DAYS` in `tasks.py` set the sizes.

### SKUs

Insulin is the cold-chain SKU; the other four use ambient storage.

| SKU | Drug | Stockout Penalty | Waste Penalty | Base Demand | Base Lead Time |
|---|---|:---:|:---:|---|---|
| `insulin` | Insulin | 100 | 10 | 8–12 units/day | 4–6 days |
| `bp_medication` | Amlodipine | 60 | 3 | 40–60 units/day | 2–4 days |
| `paracetamol` | Paracetamol | 20 | 1 | 180–220 units/day | 1–3 days |
| `vitamins` | Vitamins | 5 | 0.5 | 60–90 units/day | 1–2 days |
| `hydroxychloroquine` | Hydroxychloroquine (HCQ) | 35 | 4 | 20–40 units/day | 5–8 days |

Base demand, base lead time and starting inventory (1–5 units, so every episode starts nearly empty) are sampled when the task is built, from the ranges above. Tasks are seeded: the same `seed` always produces the same episode.

Every day is also a little random:

- **Demand:** true daily demand is `base_demand × demand_curve[day]` plus Gaussian noise with a standard deviation of 12% of base demand (`DEMAND_NOISE_CV` in `tasks.py`).
- **Lead times:** each order's lead time is `base_lead_time × lead_time_curve[day]` plus Gaussian noise with a standard deviation of 0.75 days (`LEAD_TIME_NOISE_STD`), rounded to whole days and at least 1.

The environment draws all of this noise once, at `reset`, from the seed. The same seed therefore gives the same demand every day, and the same lead time for an order placed on a given day, **whatever the agent does**. That makes comparisons between agents fair, and lets you replay one situation with different actions, e.g. to score several candidate answers for RL training.

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
| `inventory_excess_today` | Units of today's deliveries rejected because storage was full |
| `waste_fraction_today` | Share of today's deliveries rejected, weighted by `waste_penalty` (0–1) |
| `inventory_excess_cumulative` | Sum of `1 - waste_fraction_today` so far (+1 per waste-free day) |

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
            - waste_fraction_today
```

- `demand_fulfilled_today` per SKU is the fraction of true demand served (0.0 = full stockout, 1.0 = fully served).
- `waste_fraction_today` is the share of the day's deliveries rejected because their storage pool was full. It is computed per SKU that received a delivery, then averaged using each SKU's `waste_penalty` as the weight:

  ```
  waste_fraction_today = Σ waste_penalty_i × (wasted_i / delivered_i) / Σ waste_penalty_i
  ```

  It is 0 when nothing arrived, and 1 when every delivery was rejected, so the step reward lies in [-1, 1]. Wasting 1 unit of a 3,000-unit delivery costs almost nothing; wasting half of an insulin delivery costs far more than wasting half of a vitamins delivery.

### Final Episode Score

```
final_score = clip( sum of step rewards / no_of_days , 0, 1 )
            = clip( mean over days of (mean fill today − waste_fraction_today) , 0, 1 )
```

- The episode score is simply the **average daily step reward**, so an agent is evaluated on exactly what it is rewarded for each day (useful when training with RL).
- An agent that orders nothing serves almost no demand and scores about 0. Waste only ever subtracts, so a day without deliveries earns nothing extra.
- Days not played (an episode cut short) count as 0.
- `tasks.compute_final_score` rebuilds the sum from running totals in the state: `demand_fulfilled_cumulative` per SKU, and `inventory_excess_cumulative`, which adds `1 − waste_fraction_today` each day.

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

### Live dashboard (Custom tab)

Next to the Playground, `/web` has a **Custom** tab that plays a whole episode and redraws charts after every simulated day:

- **Pick** a task, a seed and an agent, then press **Run episode** (**Stop** cancels).
- **Agents:**
  - *Baseline*: a simple order-up-to rule. It needs no API key and finishes in seconds.
  - *LLM agent*: uses the same prompt and action history as `inference.py`, one model call per day, so a 60-day episode makes 60 calls and takes about 1–5 minutes.
- **Charts:** stock on hand vs. demand, true lead time vs. the agent's estimate, orders and deliveries, step reward, and unmet demand for every SKU and day. **Orange lines are hidden ground truth the agent never sees**, so you can judge its decisions.
- **Tables:** every agent decision (the model's raw reply and the orders accepted), plus a per-day data table for the selected SKU.

To use the LLM agent on the Hugging Face Space, add your token as a **Secret** named `HF_TOKEN` (Space → Settings → Variables and secrets), or paste a token under *LLM settings* for a single run. Calls go through [Inference Providers](https://huggingface.co/docs/inference-providers) and count against that account's credits. `MODEL_NAME` sets the default model and `API_BASE_URL` points it at any OpenAI-compatible endpoint. Each run uses its own environment, and runs are queued one at a time.

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
    ├── dashboard.py           # Live dashboard tab at /web (charts + agents)
    └── my_env_environment.py  # Core simulation: demand sampling,
                               # arrivals, overflow, insights, LLM prompt
```