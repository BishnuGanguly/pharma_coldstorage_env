# Results on the pharma world

Reference agents, what the news is worth, and small LLMs run locally. All numbers are on test seeds of the pharma world.

## Baselines

Mean final score on the first 50 test seeds per task (`python eval.py --agent <agent> --split test --episodes 50`), with the 95% confidence interval of the mean:

| Agent | Supply Chain | Flu Season | Epidemic Two Waves |
|---|:---:|:---:|:---:|
| Order nothing (floor) | 0.003 | 0.003 | 0.003 |
| Baseline (order-up-to rule) | 0.846 ± 0.008 | 0.900 ± 0.004 | 0.856 ± 0.009 |
| Oracle (perfect foresight, ceiling) | 0.931 ± 0.002 | 0.931 ± 0.002 | 0.932 ± 0.002 |

The oracle is not 1.0 because episodes start with almost no stock: the first days are lost for every agent until a first delivery can arrive. Results for small LLMs are in [LLM results](#llm-results).


## News: disruptions announced ahead of time

The baseline reaches about 93% of the oracle because the observation holds only clean numbers, and turning numbers into orders is solved arithmetic. With **news**, the agent also reads text announcing each disruption a few days before it starts. That is information the baseline cannot use, and reading it is something a language model can do.

`news.py` reads each task's own demand and lead-time curves, turns every disruption into an event (a supplier delay where lead times rise, a demand surge where demand reaches 1.5× normal or more), and publishes it **5–10 days before it starts** (drawn per event from the seed, never before day 0). The world itself is unchanged: news never alters demand, lead times or noise, and the baseline scores exactly the same with or without it.

| Level | What the agent reads | Example |
|---|---|---|
| `0` | No news (the original environment, the default) | — |
| `1` | One fixed template with exact days and multiplier | *SUPPLIER DELAY - insulin: orders placed from day 15 to day 25 will take about 3.7x the normal lead time.* |
| `2` | Varied wording; timing relative to publication and approximate (±1–2 days); severity in words or a rounded number; generic product names (*blood-pressure medication* for `bp_medication`) | *Logistics warning: because of a strike at the cold-chain carrier, deliveries of insulin will be severely delayed, starting in about 5 days, for about 10 days.* |

The observation's `news` list holds every item already published whose event has not ended, each prefixed with its publication day (`"Day 9: ..."`). It also appears in the LLM report when not empty. Use it with `eval.py --news 1` or `--news 2`, or `env.reset(task_name=..., seed=..., news=2)`; add `--random-scenarios` (`scenario="random"`) to draw the disrupted SKUs and days per seed (see [Scenarios](pharma-world.md#scenarios-fixed-or-random)). False alarms and irrelevant news are planned for a later level.

**Hybrid agent (`llm_news`).** The baseline does the arithmetic, and the LLM only reads the news and answers with **extra safety days per SKU** (`inference.NEWS_SYSTEM_PROMPT`), e.g. `{"insulin": 6, "paracetamol": 0, ...}`. The extra days are clipped to 0–15 and so that total cover stays within 18 days (storage holds about 21). On days without news the model is not called, so the agent plays exactly like the baseline. A confused model can therefore cost little, and any gain from the news lifts the score *above* the baseline.

**Reference: perfect news reading (`baseline_news`).** This agent reads the exact event behind each published item (not the text) and adds extra days by a fixed rule. Like the oracle, it is a reference: it shows what the news is worth, and it is a natural teacher for fine-tuning. On the first 50 test seeds per task:

| Agent | Supply Chain | Flu Season | Epidemic Two Waves | Gap to oracle closed |
|---|:---:|:---:|:---:|:---:|
| Baseline (ignores news) | 0.846 ± 0.008 | 0.900 ± 0.004 | 0.856 ± 0.009 | — |
| `baseline_news`, news level 1 | 0.898 ± 0.005 | 0.904 ± 0.004 | 0.907 ± 0.004 | 56% |
| Oracle | 0.931 ± 0.002 | 0.931 ± 0.002 | 0.932 ± 0.002 | 100% |

Level 2 gives nearly the same scores (0.903 overall, 55% of the gap), since the rule reads the events, not the text. The open question for an LLM is how much of that +0.036 it recovers from the text alone, especially at level 2.

**Reference: an inventory plan from the news (`news_plan`).** `baseline_news` keeps the baseline and adds days to it. `news_plan` (`agents.news_plan_orders`) instead replaces, for every SKU with published news, the two estimates the baseline can only correct after a disruption shows:

- **Lead time.** An order placed during a published delay takes the normal lead time × the delay's multiplier. Stock has to last until the first later order can arrive, so the plan covers the earliest arrival of any order placed from tomorrow on: the normal lead time before a delay, and up to the delay's end or the slow lead time (whichever is sooner) just before and during one.
- **Demand.** Normal demand is past sales with the published surges divided out, and each coming day is forecast as normal demand × that day's multiple, so the forecast rises ahead of a wave and falls after its peak instead of lagging it.

The target is the forecast demand over that cover period plus 3 safety days, capped at the SKU's share of its storage pool plus what is consumed while the order travels. It uses only the published events' fields and the observation (the normal lead time is the first report's `avg_lead_time`), SKUs without news are ordered exactly as the baseline orders them, and without news the agent *is* the baseline.

| Agent (50 test seeds per task) | Supply Chain | Flu Season | Epidemic Two Waves | Gap to oracle closed |
|---|:---:|:---:|:---:|:---:|
| `baseline_news`, news level 1 | 0.898 ± 0.005 | 0.904 ± 0.004 | 0.907 ± 0.004 | 56% |
| `news_plan`, news level 1 | 0.907 ± 0.003 | 0.911 ± 0.002 | 0.907 ± 0.003 | 64% |
| `baseline_news`, random scenario, level 2 | 0.892 ± 0.006 | 0.877 ± 0.015 | 0.861 ± 0.019 | 27% |
| `news_plan`, random scenario, level 2 | 0.906 ± 0.003 | 0.889 ± 0.009 | 0.881 ± 0.011 | 50% |

Seed by seed, `news_plan` beats `baseline_news` by +0.006 ± 0.002 on the fixed tasks and +0.015 ± 0.005 in the random scenario. The difference is largest where storage is tight and the disrupted SKU is not known in advance, which is where a plan that sizes orders to the delay and to the pool matters most.


## LLM results

Small models run locally with Ollama (`local_llm/run_local_llm.py`, temperature 0.2) on test seeds. Every run below is scored under the **current** reward (fill weighted by `stockout_penalty`): each episode saved the exact order sent every day, and since a seed fixes the world, replaying those orders in the current environment reproduces the run exactly (the same fill rate for every SKU on every day) and gives its score under today's reward. The reference agents are scored on the same seeds. The prompts also changed between runs (each run used the prompt of its answer format at the time); rescoring changes only the reward, not what the model saw.

| Model | Answer format | News | Episodes | Score | Baseline | +2 days for every SKU | Oracle |
|---|---|---|---|:---:|:---:|:---:|:---:|
| Qwen 2.5 1.5B | `units` | — | flu_season, seed 10000 | 0.000 | 0.895 | 0.900 | 0.923 |
| Gemma 2 2B | `units` | — | flu_season, seed 10000 | 0.420 | 0.895 | 0.900 | 0.923 |
| Gemma 2 2B | `days` | — | 3 tasks × seed 10000 | 0.747 | 0.875 | 0.893 | 0.936 |
| Qwen 2.5 3B | `days` | — | 3 tasks × seed 10000 | 0.521 | 0.875 | 0.893 | 0.936 |
| Phi-4-mini (3.8B) | `adjust` | — | 3 tasks × seed 10000 | 0.886 | 0.875 | 0.893 | 0.936 |
| **Phi-4-mini (3.8B)** | **`adjust`** | **level 2** | **3 tasks × seeds 10000–10002** | **0.881** | **0.874** | **0.884** | **0.935** |

"+2 days for every SKU" is a fixed answer in the `adjust` format that never reads the report: a reference for "keep a little more stock". For the news run, perfect news reading (the `news_extra_days` rule on the exact events, answered in the `adjust` format) scores 0.900 on the same 9 episodes.

What each run showed:

- **`units` (Qwen 1.5B, Gemma 2B):** the models copied numbers instead of computing orders. Qwen 1.5B answered `{"insulin": 100}` every day, copying the prompt's example: nothing else was ever ordered, and 100 insulin a day overflowed cold storage on 35 of 60 days, so the waste penalty cancelled what insulin earned (score clipped to 0). Gemma 2B copied the report's `inventory_on_hand` values as order sizes (3-6 units of paracetamol a day against a demand of several hundred). Its 0.420 comes almost entirely from insulin and HCQ, the two SKUs it kept mostly in stock (86% and 73% filled), which carry 45% and 16% of the reward weight; paracetamol was 1% filled.
- **`days` (Gemma 2B, Qwen 3B):** the code does the arithmetic, which removed the copying of report values as quantities, but the models copied `days_of_cover` (what they already had) as their target. Stock ran down until nothing was reordered; Gemma then jumped to 30 days at a time and overflowed storage (17,000-81,000 units wasted per episode), and Qwen let its targets fall to 0: it kept insulin about 89% filled while the other SKUs mostly ran out (189-247 stockout SKU-days per episode; on epidemic_two_wave it ordered nothing but insulin from day 1).
- **`adjust` without news (Phi-4-mini):** valid JSON every day and no collapse, +0.011 over the baseline, but below the fixed "+2 days for every SKU" (0.893): its gain is "keep a little more stock", not reading the report.
- **`adjust` with level-2 news (Phi-4-mini, 9 episodes):** +0.007 ± 0.016 over the baseline (95% CI, not significant), 11% of the gap to the oracle (perfect news reading: 43%). It picks the right SKU and roughly the right timing (insulin from the day a cold-chain strike is announced until it ends), but the amounts are off: +5 where about +14 is needed for an insulin delay, and +10 to +19 days of paracetamol after the flu peak, which wasted 6,700-13,000 units per flu_season episode and made that task worse than the baseline (-0.020). It also adds days with no news (insulin and HCQ on days 1-8) and alternates between +7 and nothing on consecutive days with the same news.

**So far:** the `adjust` format made small models safe (no run in it falls far below the baseline), but none has yet shown a reliable gain from reasoning: every LLM result is at or below a fixed "+2 days for every SKU". These are few episodes; compare runs seed by seed with `eval.py --compare`, and use more seeds (`--episodes 5`) before drawing conclusions.
