# Pharma Cold-Chain Inventory Management Environment

An OpenEnv-compliant benchmark for intelligent pharmaceutical warehouse procurement under uncertainty.

An LLM agent manages a cold-chain warehouse stocking four drugs. Every day it receives an inventory report and decides what to order, how much, and from which supplier. Demand is stochastic. Lead times are uncertain. Suppliers can disrupt. The agent never sees the true underlying processes — it must infer risk from observable history and act under uncertainty.

---

## Motivation

Pharmaceutical procurement fails in ways that directly harm patients. The decisions are made daily, the stakes are high, and the uncertainty is irreducible. This benchmark tests whether LLM agents can reason about the class of problems where:

1. **You cannot see the future demand** — only recent history and weak signals. Acting on the signal before the spike arrives is the skill being tested.
2. **Lead times are uncertain and can extend** — ordering today does not guarantee stock arrives when you expect it. A supplier disruption can silently invalidate your inbound estimates.
3. **Resources are constrained and shared** — a fixed procurement budget must cover all four drugs. Spending heavily on vitamins today might mean you cannot afford insulin tomorrow.
4. **Priorities are asymmetric** — insulin has no substitute. A stockout is a patient harm event. Vitamins are fully substitutable. The agent must learn this hierarchy from penalty signals, not explicit rules.
5. **Inaction is costly, but over-action is also costly** — ordering too much wastes budget and risks expiry. The agent must find the right amount, at the right time, from the right supplier.

---

## Environment Design

### The Three-Layer State Architecture

Every state variable belongs to one of three layers:

| Layer | Visible to Agent | Description |
| :--- | :---: | :--- |
| **Layer 1 — Hidden** | ✗ | True demand today, true lead times, disruption Markov state, breach probability |
| **Layer 2 — Insights** | ✓ | Demand history, delivery history, epidemic alert, fill rate trends |
| **Layer 3 — Real State** | ✓ | Inventory on hand, backorders, inbound estimates, budget, capacity |

The agent sees only Layers 2 and 3. It must infer the hidden layer from observable consequences — exactly as a real procurement manager would.

### What Is Hidden (Layer 1)

```
true_demand_today           sampled from time-varying LogNormal process
true_demand_mean_t          seasonal + event-driven mean (never shown)
true_demand_std_t           true variance (never shown)
true_lead_time              actual days until order arrives (stochastic)
disruption_markov_state     current state of the disruption chain per supplier
p_onset / p_persist         true disruption transition probabilities
cold_chain_breach_prob      daily probability of refrigeration failure
```

### What the Agent Observes (Layers 2 + 3)

```
# Global — Layer 2
prescription_fill_rate_30d      rolling 30-day fill rate
prescription_fill_rate_7d       rolling 7-day fill rate (detects recent decline)
epidemic_alert_flag             fires when demand trend crosses threshold
days_since_epidemic_alert_fired alert duration (0 = just fired, 5+ = near peak)
supply_disruption_days_last_30d how hostile the disruption environment has been

# Global — Layer 3
procurement_budget_ratio        remaining / total quarterly budget
cold_storage_capacity_ratio     insulin storage used / max
ambient_capacity_ratio          B,C,D storage used / max
cold_chain_integrity_flag       False = breach fired, all insulin destroyed
orders_overdue_count            orders past expected arrival date
overdue_qty_total               units stuck in overdue orders

# Per SKU — Layer 2 (insights)
demand_last_3d                  actual demand over last 3 days
demand_last_7d                  actual demand over last 7 days
demand_trend                    (demand_last_3d/3) - (demand_last_7d/7)
stockout_days_if_no_reorder     days until stockout if agent does nothing today
coverage_gap_7d                 demand_last_7d - inventory_on_hand

# Per SKU — Layer 3 (real state)
inventory_on_hand               units physically on shelf
backorders                      unfilled prescriptions accumulated
inbound_expected_3d             units ordered, expected within 3 days (estimate)
inbound_expected_7d             units ordered, expected within 7 days (estimate)

# Per Supplier — Layer 2 (insights)
last_observed_lead_time         lead time from most recent delivery
on_time_rate_14d                fraction of recent orders delivered on time

# Per Supplier — Layer 3 (real state)
disruption_active               True if supplier confirmed disrupted right now
```

---

## The Warehouse

### SKUs

| SKU | Drug | Storage | Stockout Penalty | Substitute | Key Risk |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `insulin` | Insulin | Cold (2–8°C) | 100 | None (0.0) | Cold chain breach, single supplier |
| `bp_medication` | Amlodipine (BP Med) | Ambient | 60 | Partial (0.4) | Chronic patients, supply disruption |
| `paracetamol` | Paracetamol | Ambient | 20 | Partial (0.6) | Flu season demand spikes |
| `vitamins` | Vitamins | Ambient | 5 | Full (1.0) | Seasonal winter demand, low priority |

### Suppliers

| Supplier | Cold Certified | SKUs Served | Lead Time | Reliability | Cost | Expedite |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| `FastPharma` | ✓ | All four | ~2 days (σ=0.5) | High (0.95) | 1.4x | ✓ (2x cost) |
| `GlobalMed` | ✗ | Paracetamol, BP, Vitamins | ~7 days (σ=2.5) | Lower (0.70) | 1.0x | ✗ |

**Critical constraint**: Insulin can only be ordered from `FastPharma`. `GlobalMed` is not cold-chain certified. This is a hard constraint the agent must discover from the `cold_chain_certified` field — not from an explicit rule.

---

## Action Space

The agent responds each day with a JSON object:

```json
{
  "insulin":      [100, "FastPharma"],
  "paracetamol":  [500, "GlobalMed"]
}
```

- Include only SKUs you want to order today. Omit SKUs = no order.
- `quantity` must be positive.
- `supplier_name` must be in the supplier's `sku_served` list.
- Insulin orders to non-certified suppliers are rejected.
- Orders exceeding budget or storage capacity are rejected.
- Orders to disrupted suppliers are rejected.
- Send `{}` to place no orders today.

---

## Tasks and Difficulty

Three scenario types, three difficulty levels each. All episodes run 60 days.

### Task 1 — Supply Chain Broken

Supply chain stress. Both suppliers experience extended lead times. GlobalMed goes fully offline for a window mid-episode. Demand barely changes — the challenge is entirely supply-side.

**What the agent must learn**: Shelves look fine today but orders placed now will take 2–4x longer to arrive. Build safety stock before the stress window hits. Reactive ordering arrives too late.

| Difficulty | Stress Start | GlobalMed Lead Mult | Offline Window | Starting Cover |
| :--- | :---: | :---: | :---: | :---: |
| Easy | Day 25 | 1.5x | 5 days @ day 40 | 14 days |
| Medium | Day 15 | 2.5x | 10 days @ day 35 | 7 days |
| Hard | Day 8 | 4.0x | 15 days @ day 25 | 4 days |

```
Day 1–14:   Normal operations. Window to build stock.
Day 15+:    GlobalMed lead times extending.
Day 35–44:  GlobalMed fully offline. FastPharma only, at premium cost.
Day 45–60:  Recovery. Lead times normalise.
```

### Task 2 — Flu Season

Flu season demand surge. Paracetamol demand rises sharply. Insulin spikes when flu hits diabetics. GlobalMed slows as logistics workers fall sick.

**What the agent must learn**: Epidemic alert fires a few days before the peak. Pre-stock before the surge — not after. Insulin and paracetamol peak close together, forcing budget triage. Insulin must win.

| Difficulty | Flu Onset | Paracetamol Peak | Insulin Peak | Para Amplitude | Supply Stress |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Easy | Day 25 | Day 35 | Day 40 | 0.8x | None |
| Medium | Day 15 | Day 25 | Day 30 | 1.5x | GlobalMed 1.8x from day 23 |
| Hard | Day 8 | Day 18 | Day 23 | 2.2x | GlobalMed 1.8x from day 11 |

```
Day 1–onset:         Early season. Epidemic alert fires ~3 days before peak.
Day onset+10:        Paracetamol demand at peak.
Day onset+15:        Insulin demand spikes (flu hits diabetics).
Day onset+25:        Demand normalising.
```

### Task 3 — Epidemic, Two Waves (Hydroxychloroquine)

Two-wave epidemic. Primary drug: Hydroxychloroquine (HCQ). Wave 1 peaks at day 17. Wave 2 peaks at day 45 and is larger. Lead times worsen exactly when demand peaks.

**What the agent must learn**: Pre-stock HCQ before day 17. The trough between waves is a trap — wave 2 is larger, ordering must continue. Lead time stress hits before the peak at hard difficulty — orders placed at the peak arrive after it.

| Difficulty | Wave 1 Amplitude | Wave 2 Amplitude | Wave Width | Lead Stress Timing |
| :--- | :---: | :---: | :---: | :--- |
| Easy | 1.2x (2.2x total) | 1.0x (smaller) | Wide (9 days) | After peak |
| Medium | 1.8x (2.8x total) | 2.5x (larger) | Normal (6 days) | At peak |
| Hard | 2.5x (3.5x total) | 3.5x (much larger) | Narrow (4 days) | Before peak |

```
Day 1–12:   Rising HCQ demand. Epidemic alert fires ~day 12.
Day 17:     WAVE 1 PEAK.
Day 17–35:  Declining. Lead times normalise.
Day 35–40:  TROUGH. Relative calm. Second wave building silently.
Day 40+:    Epidemic alert re-fires. Second wave rising.
Day 45:     WAVE 2 PEAK — larger than wave 1.
Day 45–60:  Declining. Episode ends before full recovery.
```

---

## Scoring

### Per-Step Reward

```
step_reward = clamp(
    0.50 × prescription_fill_rate_7d
  + 0.30 × critical_sku_fill_rate
  + 0.10 × (1 - mean_expiry_pressure)
  + 0.10 × procurement_budget_ratio,
  0.01, 0.99
)
```

| Term | Weight | Definition |
| :--- | :---: | :--- |
| `prescription_fill_rate_7d` | 0.50 | Fraction of prescriptions filled on time in last 7 days |
| `critical_sku_fill_rate` | 0.30 | Mean fill rate for insulin and BP medication specifically |
| `1 - mean_expiry_pressure` | 0.10 | Penalises over-stocking perishables |
| `procurement_budget_ratio` | 0.10 | Rewards fiscal discipline |

### Final Episode Score

```
final_score = clamp(
    0.5 × mean(step_rewards)
  + 0.3 × prescription_fill_rate_30d
  + 0.2 × (1 - total_backorder_ratio),
  0.01, 0.99
)
```

| Term | Weight | What It Measures |
| :--- | :---: | :--- |
| `mean(step_rewards)` | 0.50 | Decision quality throughout the episode |
| `prescription_fill_rate_30d` | 0.30 | End-of-episode service level |
| `1 - total_backorder_ratio` | 0.20 | Whether accumulated debt was resolved |

**Success threshold: 0.80**

An agent that fills all prescriptions but drains the budget early scores ~0.60. An agent that protects the budget but lets insulin stock out repeatedly scores ~0.50. Only an agent that manages both dimensions consistently scores above 0.80.

---

## Baseline Benchmarks

Evaluated across all 9 task configurations. Scores represent Final Episode Score [0.01–0.99].

| Model | Supply Chain (E/M/H) | Flu Season (E/M/H) | Epidemic (E/M/H) | Success Rate |
| :--- | :---: | :---: | :---: | :---: |
| *Results to be published post-evaluation* | — | — | — | — |

---

## Setup and Usage

### 1. Installation

```bash
uv sync
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Set HF_TOKEN, MODEL_NAME, API_BASE_URL as needed
```

### 3. Run the Server

```bash
uv run server
```

### 4. Run the Benchmark

```bash
python inference.py
```

### Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | — | HuggingFace API key |
| `LOCAL_IMAGE_NAME` | — | Docker image name (optional) |
| `ENV_BASE_URL` | `http://localhost:8000` | Server URL when not using Docker |

---

## Project Structure

```
pharma_env/
├── environment.py   # Core simulation: demand sampling, inventory updates,
│                    # arrivals, disruptions, cold chain breach, LLM prompt
├── models.py        # Pydantic schemas: SKUState, SupplierState, InventoryState,
│                    # PharmaAction
├── tasks.py         # Task constructors (supply_chain_broken, flu_season,
│                    # epidemic_two_wave) + reward functions
├── inference.py     # Benchmark execution
├── client.py        # OpenEnv async client
└── openenv.yaml     # Environment manifest
```

---

## Key Design Decisions

**Three-layer state architecture**: The environment strictly separates hidden stochastic processes (Layer 1), observable history (Layer 2), and current real state (Layer 3). The agent is never shown true demand, true lead times, or true disruption probabilities. It infers risk from consequences — exactly as a real operations manager would.

**Reactive disruption discovery**: The agent discovers supplier disruptions and cold chain breaches AFTER they happen, not before. `disruption_active` flips to `True` on the day the disruption fires. `cold_chain_integrity_flag` flips to `False` on the day of the breach. This makes disruption response a genuine reactive challenge.

**Honest inbound estimates**: `inbound_expected_3d` and `inbound_expected_7d` are computed using `last_observed_lead_time` — the agent's best historical estimate. If the true lead time is longer (due to disruption or stochastic variance), these estimates are silently optimistic. The agent discovers inaccuracy via rising `orders_overdue_count`.

**Epidemic alert from observable history**: `epidemic_alert_flag` fires when the observed 3-day demand average exceeds the 14-day average by 40%. It is triggered by observable history, not by the hidden epidemic process. This gives the agent a 3–5 day warning window but not the true spike magnitude — the agent must decide how aggressively to pre-stock under this uncertainty.

**Asymmetric drug priority**: The penalty hierarchy (insulin 100 >> BP med 60 >> paracetamol 20 >> vitamins 5) and the `cold_chain_certified` hard constraint together force the agent to discover a medically coherent priority ordering from the reward signal alone — not from explicit rules in the prompt.