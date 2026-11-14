# Evaluating an agent

`eval.py`: run any agent over many seeded episodes and compare agents seed by seed.

## Evaluating an agent

`eval.py` plays an agent over many seeded episodes and reports the mean score per task. It runs the environment in-process (no server needed), and rule-based agents run in parallel.

```bash
uv run python eval.py --agent baseline --split test --episodes 50   # -> results/baseline_test_50.jsonl
uv run python eval.py --agent oracle   --split test --episodes 50

# Any OpenAI-compatible model, e.g. a local Ollama server (no API limits):
uv run python eval.py --agent llm --model qwen2.5:3b --base-url http://localhost:11434/v1 \
                      --split test --episodes 10 --save-steps
```

LLM agents answer in adjustment days by default; `--action days` or `--action units` selects the other formats (see the LLM prompt notes above). With `--save-steps`, each step stores the model's own reply (`reply`) and the units actually sent (`orders_sent`).

**Running a model on your own computer:** see [`local_llm/README.md`](../local_llm/README.md). One command, e.g. `uv run python local_llm/run_local_llm.py --model qwen2.5:3b`, sets up the model with Ollama the first time and runs it against the baseline and oracle on the same seeds.

**Agents** (`agents.py`): `nothing` (the floor), `baseline` (an order-up-to rule), `oracle` (reads the episode's hidden demand and lead times and plans just-in-time deliveries; the ceiling, never something to train against), `llm` (the same prompt as `inference.py`), and, for news (`--news 1` or `2`), `baseline_news` and `news_plan` (perfect news reading with an extra-days rule and with an inventory plan; references) and `llm_news` (the hybrid: the LLM reads the news, the baseline orders).

```bash
uv run python eval.py --agent baseline_news --news 2 --split test --episodes 50
uv run python eval.py --agent news_plan     --news 2 --split test --episodes 50 --random-scenarios
uv run python eval.py --agent llm_news --news 2 --model qwen2.5:3b --base-url http://localhost:11434/v1 \
                      --split test --episodes 10 --save-steps
```

**Seeds:** `--split train` uses seeds 0–9,999 and `--split test` uses 10,000–19,999, always the first `--episodes` of them. Tune prompts and train models on train seeds only, and report test seeds.

**Output:** one JSON line per episode, with its score, average fill rate, stockout SKU-days, overflow days, units wasted and, for LLMs, how many replies contained no JSON and how many calls failed. `--save-steps` also stores every day's reply and reward.

**Comparing agents:** the same seed gives the same world to every agent, so compare seed by seed:

```bash
uv run python eval.py --compare results/llm_qwen2.5-3b_test_10.jsonl results/baseline_test_50.jsonl \
                      --oracle results/oracle_test_50.jsonl
```

This prints the mean per-seed difference with its 95% confidence interval and, with `--oracle`, the **gap closed**: 0% means as good as the baseline, 100% means as good as perfect foresight. Per-seed differences cancel out each seed's luck; comparing a variant of the baseline against the baseline this way gave a confidence interval about 4× tighter than comparing the two averages.
