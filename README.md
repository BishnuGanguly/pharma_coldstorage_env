---
title: InventoryGym
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# InventoryGym

**A configurable inventory environment for evaluating and training LLM agents.**

Describe an inventory in one file: the products, where they are stored, and what goes wrong (supplier delays, demand surges). InventoryGym turns it into seeded, repeatable episodes in which an agent reads a daily report and the news, and decides what to order. It scores the agent against built-in reference agents, and can train a small language model to do the job.

It ships with a complete pharmaceutical warehouse (the **pharma world**, where all published results were measured) and a grocery store example to copy. It is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment, so it also runs as a server and as a Hugging Face Space.

| | |
|---|---|
| **Your own inventory** | Products, any number of storage pools, tasks and events in a YAML file. Mistakes are reported with where they are. |
| **Every seed a different episode** | Amounts, the disrupted product and the timing can be drawn per seed, so no fixed timetable works. The same seed always gives the same episode. |
| **News to read** | Disruptions are announced a few days ahead, in your products' own words. Reading them is what a language model can add to a stock rule. |
| **Reference agents** | From ordering nothing to perfect foresight, including two that read the news perfectly, so a score always has something to be compared with. |
| **Training** | A teacher for a supervised warm start, a reward for GRPO, and a Kaggle notebook for Phi-4-mini. |
| **Safe answer format** | An LLM only adjusts a rule's default order, so a confused model costs little. |

## Quickstart

```bash
git clone https://github.com/BishnuGanguly/InventoryGym
cd InventoryGym
uv sync

# Score a reference agent on the pharma world (5 seeds per task, a few seconds)
uv run python eval.py --agent baseline --episodes 5

# The same on the grocery example, with news, for an agent that reads it
uv run python eval.py --world examples/worlds/grocery.yaml --agent news_plan --news 2 --episodes 5
```

```
task                      n    mean    ±95%     std     min     max
delivery_strike           5   0.966   0.005   0.006   0.958   0.974
holiday_rush              5   0.967   0.017   0.019   0.938   0.986
normal_weeks              5   0.971   0.012   0.014   0.951   0.989
surprise_disruptions      5   0.967   0.016   0.018   0.940   0.986
ALL                      20   0.968   0.006   0.014   0.938   0.989
```

To see an episode play out with charts, start the server and open the dashboard (see [Server and dashboard](#server-and-dashboard)).

## How an episode works

Each day, for 45 to 60 days:

1. **The agent gets a report**: stock on hand, orders on the way, recent demand, recent lead times and how full each storage pool is, per product. With news switched on, it also gets the announcements published so far.
2. **The agent orders.** A rule agent returns quantities. An LLM agent answers in one of three formats; by default it only says how many *extra* days of stock to keep on top of a rule's default (see [answer formats](docs/observation-and-answers.md)).
3. **The day plays out.** Deliveries arrive after a lead time the agent cannot see in advance, and anything that does not fit in its storage pool is wasted. Customers are served from the shelf.
4. **The day is scored**: the share of demand served, weighted by how critical each product is, minus the share of deliveries wasted. The episode score is the average day, from 0 to 1 ([scoring](docs/scoring.md)).

Demand and lead times are noisy, and all of the noise is drawn from the seed when the episode starts. An episode therefore never depends on what the agent does, every agent faces exactly the same world, and one day can be replayed with different answers to compare them.

## Describe your own inventory

A **world file** holds everything that defines an inventory. This is a small but complete one:

```yaml
name: my_store
business: a neighbourhood grocery store        # who the agent works for (used in prompts)
days: 45

storage:                                       # any number of pools, any names
  fridge: {capacity_days: 10, cold_chain: true}
  shelf:  {capacity_days: 21}

products:                                      # amounts are a number or a [low, high] range
  milk:
    name: fresh milk                           # how prompts and news call it
    storage: fridge
    base_demand: [80, 120]                     # units a day
    base_lead_time: [1, 2]                     # days from order to delivery
    starting_inventory: [100, 200]
    stockout_penalty: 50                       # how much a shortage costs, relative to others
    waste_penalty: 8
  rice:
    storage: shelf
    base_demand: [30, 50]
    base_lead_time: [3, 5]
    starting_inventory: [100, 200]
    stockout_penalty: 30
    waste_penalty: 1

news:                                          # the reasons news gives
  delay_causes: [a flooded motorway, a strike at the distribution centre]
  surge_causes: {milk: the holiday season}

tasks:                                         # the same products under different events
  holiday_rush:                                # fixed: the same product and days every episode
    events:
      - {type: demand_wave, product: milk, peak_day: 24, amplitude: [0.5, 1.0], width: 4}
  surprise_strike:                             # drawn: a different product and week every seed
    draw: {struck: any}
    storage_days: {shelf: 14}                  # tighter storage for this task
    events:
      - {type: supplier_delay, product: struck, start_day: [8, 25], duration: [8, 12], multiplier: [2, 4]}
```

```bash
uv run python eval.py --world my_store.yaml --agent baseline --news 2 --episodes 20
```

| Part | What you can set |
|---|---|
| **Storage** | Any number of named pools. Each holds `capacity_days` of its products' demand, and a delivery that does not fit in its own pool is wasted. |
| **Products** | Demand, lead time, starting stock and penalties, as numbers or `[low, high]` ranges drawn per seed. Optional name, description and aliases. |
| **Events** | `supplier_delay` (orders placed in a window arrive slower), `demand_wave` (a bell-shaped rise around a peak day), `demand_step` (a promotion, or a closure below 1). |
| **Drawn tasks** | `draw` picks products per seed (`any`, or one of a list), days can be ranges, and a task can have its own storage size. |
| **Words** | `business`, product names and news causes: prompts and news then speak about your inventory, not a pharmacy. |

A file with a mistake is refused with a message that says where it is:

```
world is not a valid world:
  - products.rice.base_demand: range [9.0, 3.0] has low > high
```

The full reference is at the top of [`world.py`](world.py). [`examples/worlds/grocery.yaml`](examples/worlds/grocery.yaml) is a worked example with three pools and four tasks, and [`worlds/pharma.yaml`](worlds/pharma.yaml) is the built-in world. More detail: [World files](docs/world-files.md).

## News

With `--news 1` or `--news 2`, every disruption is announced 5 to 10 days before it starts. News never changes what happens; it only tells the agent early.

| Level | What the agent reads |
|---|---|
| `0` | No news (the default). |
| `1` | One fixed template with exact days and sizes: *SUPPLIER DELAY - rice: orders placed from day 15 to day 24 will take about 3.2x the normal lead time.* |
| `2` | Varied wording, approximate timing, your products' names and causes: *Planning note: a heat wave is on the way. Sales of ice cream could roughly double in about 7 days and stay high for roughly 2 weeks.* |

A stock rule cannot use this text. An agent that understands it can prepare before the shelves show a problem.

## Agents

All agents are in [`agents.py`](agents.py) and run on any world with `eval.py --agent <name>`.

| Agent | What it does |
|---|---|
| `nothing` | Never orders: the floor. |
| `baseline` | An order-up-to rule: stock for the lead time plus 3 safety days. Ignores news. |
| `baseline_news` | The baseline plus extra days worked out from the exact events behind the news. |
| `news_plan` | An inventory plan from the exact events: covers stock until the next order can arrive, forecasts demand along the announced wave, and respects storage. |
| `oracle` | Perfect foresight of demand and lead times: the ceiling. |
| `llm` | A language model through any OpenAI-compatible API (hosted, or local with Ollama). |
| `llm_news` | A hybrid: the baseline orders, and the model only reads the news. |

`baseline_news`, `news_plan` and `oracle` read information an agent does not get (the events themselves, or the future). They are references: they show what the news is worth and how far a score is from the best possible.

**What reading the news is worth**, with level-2 news, in the two settings where the disrupted product and the timing are drawn per seed:

| Agent | Pharma world, random scenario (50 test seeds × 3 tasks) | Grocery example, `surprise_disruptions` (100 test seeds) |
|---|:---:|:---:|
| "+3 days on every product" (no reading) | 0.869 | 0.939 |
| `baseline` | 0.859 | 0.951 |
| `baseline_news` | 0.877 | 0.965 |
| Teacher (`news_plan` in the LLM's answer format) | 0.887 | 0.968 |
| `news_plan` | 0.892 | 0.969 |
| `oracle` | 0.925 | 0.989 |

Keeping extra stock of everything helps a little where storage is generous and hurts where it is tight. Reading the news helps in both. Results for small LLMs so far, and the measurements behind these references, are in [Results](docs/results.md): no small model has yet shown a reliable gain over a fixed habit, which is what the training below is for.

## Evaluate

```bash
uv run python eval.py --agent news_plan --news 2 --episodes 50          # -> results/news_plan_news2_test_50.jsonl
uv run python eval.py --agent oracle    --news 2 --episodes 50

# A model behind any OpenAI-compatible endpoint, e.g. a local Ollama server
uv run python eval.py --agent llm --model qwen2.5:3b --base-url http://localhost:11434/v1 \
                      --news 2 --episodes 10 --save-steps

# Compare two runs seed by seed; with --oracle, also the share of the gap to the oracle closed
uv run python eval.py --compare results/news_plan_news2_test_50.jsonl results/baseline_test_50.jsonl \
                      --oracle results/oracle_news2_test_50.jsonl
```

- **Seeds are split**: `--split train` uses seeds 0–9,999 and `--split test` 10,000–19,999. Tune and train on train seeds, report test seeds.
- **Comparisons are paired**: the same seed gives every agent the same world, so differences are taken seed by seed, which cancels each seed's luck.
- **Running a model on your own computer** takes one command: see [`local_llm/README.md`](local_llm/README.md).

Details: [Evaluating an agent](docs/evaluation.md).

## Train a small model

Training has two stages, both using the simulator as the judge (no labelled data to collect):

1. **Warm start, supervised.** A teacher (`news_plan`, answering in the model's own format) plays training episodes; each day gives one example: the model's exact prompt and the teacher's answer.

   ```bash
   uv run python -m training.teacher --seeds 0:300 --out data/teacher_adjust.jsonl   # no GPU, seconds
   ```

2. **GRPO, reinforcement learning.** The model then improves on its own answers: several answers to the same day are played forward in the simulator and the better ones are reinforced ([reward spec](training/REWARD.md)).

[`training/grpo_phi4mini_kaggle.ipynb`](training/grpo_phi4mini_kaggle.ipynb) runs both stages for Phi-4-mini on a free Kaggle GPU and reports the trained model against the base model, the teacher and the other references. Details: [Training](docs/training.md).

Training has not yet been run end to end since the warm start was added, and it currently trains on the pharma world only (see [Status](#status)).

## Server and dashboard

```bash
ENABLE_WEB_INTERFACE=true uv run uvicorn server.app:app --port 8000
# open http://localhost:8000/web
```

- **API**: `/reset`, `/step`, `/state`, `/schema` and a WebSocket, following OpenEnv. [`client.py`](client.py) is the Python client.
- **Playground** (`/web`): reset and step an episode by hand.
- **Dashboard** (the *Custom* tab): plays a whole episode with the baseline or an LLM and redraws charts every day, including the hidden demand and lead times, so you can judge each decision.
- **Docker / Hugging Face Space**: the `Dockerfile` builds the server with the web interface on; the front matter of this file configures the Space.

```python
from server.my_env_environment import PharmaEnvironment
from world import load_world

env = PharmaEnvironment()
obs = env.reset(world=load_world("examples/worlds/grocery.yaml"), task_name="holiday_rush", seed=42, news=2)
```

Details: [Server, playground and dashboard](docs/server.md).

## Documentation

| | |
|---|---|
| [World files](docs/world-files.md) | Describing your own inventory |
| [The pharma world](docs/pharma-world.md) | The built-in products, storage and tasks, and its random scenarios |
| [What the agent sees and how it answers](docs/observation-and-answers.md) | The report, and the `adjust`, `days` and `units` answer formats |
| [Scoring](docs/scoring.md) | The daily reward and the episode score |
| [Results](docs/results.md) | Reference agents, what the news is worth, small LLMs |
| [Evaluating an agent](docs/evaluation.md) | `eval.py` in full |
| [Training](docs/training.md) | The teacher, the GRPO reward, the Kaggle notebook |
| [Server, playground and dashboard](docs/server.md) | Running it as a service, environment variables |
| [`local_llm/README.md`](local_llm/README.md) | A local model with Ollama |

## Project layout

```
InventoryGym/
├── world.py               # World files: the format, loading, building episodes
├── worlds/pharma.yaml     # The built-in pharma world
├── examples/worlds/       # Example worlds to copy (grocery.yaml)
├── models.py              # Data models: observation, action, episode config
├── server/
│   ├── my_env_environment.py   # The simulation
│   ├── app.py                  # OpenEnv server (FastAPI)
│   └── dashboard.py            # The live dashboard
├── tasks.py               # The pharma tasks' random scenarios, seeds, scoring
├── news.py                # News about a task's disruptions
├── agents.py              # Reference agents and LLM agents
├── inference.py           # Prompts, the report, answer parsing
├── eval.py                # Evaluate and compare agents
├── training/              # Teacher, GRPO reward, probe exam, Kaggle notebook
├── local_llm/             # Run a local model with Ollama
├── client.py              # OpenEnv client
├── docs/                  # Detailed documentation
└── tests/                 # 308 tests
```

```bash
uv run --extra dev python -m pytest tests     # about a minute
```

## Status

InventoryGym grew out of a pharma cold-chain benchmark, and some parts still show it:

- **Names in the code.** The classes are still `PharmaEnvironment`, `PharmaAction` and `PharmaEnvClient`, and the package is `openenv-my_env`. Renaming them, with the old names kept working, is planned.
- **Pharma only, for now**: training (`training/`), the local LLM runner and the dashboard run the pharma world's tasks. `eval.py`, every agent, news and prompts work on any world.
- **The pharma world keeps its original prompts and news wording**, so published results and trained models stay valid. Every other world, including an edited copy of `pharma.yaml`, gets prompts and news written from its file.
- **Not there yet**: continuous integration, a single command-line tool, and a licence file.
