# Run a local LLM on the Pharma environment

Test the environment with a model running on your own computer through
[Ollama](https://ollama.com): free and without API limits. The default is
**Gemma 2 2B** (`gemma2:2b`, about 2.6 billion parameters, about a 1.6 GB
download), which runs on a normal laptop. Choose any other Ollama model with
`--model`, e.g. `--model qwen2.5:3b`.

## 1. Install Ollama

Download it from https://ollama.com/download and start it. The desktop app starts
the server automatically; otherwise run `ollama serve` in a separate terminal.

## 2. Get the code and dependencies

```bash
git pull
uv sync
```

## 3. Run it

```bash
# 1 episode of flu_season with Gemma 2 2B (60 model calls)
uv run python local_llm/run_local_llm.py

# choose the model with --model
uv run python local_llm/run_local_llm.py --model qwen2.5:3b

# one seed of every task with Qwen 2.5 3B (180 calls)
uv run python local_llm/run_local_llm.py --model qwen2.5:3b --seed 10000 --tasks all

# 3 seeds of every task (540 calls)
uv run python local_llm/run_local_llm.py --tasks all --episodes 3

# one chosen seed (60 calls); the same seed always gives the same world
uv run python local_llm/run_local_llm.py --seed 10042
```

**The first time you use a model** the script sets it up for you: it runs `ollama pull
<model>`, then `ollama create pharma-<model>` (e.g. `pharma-qwen2-5-3b`) from a small
Modelfile that raises the context window to 4,096 tokens. The environment's prompts are
about 1,800 tokens, and Ollama may cut prompts longer than its default window from the
start, which would drop the rules and output format. Later runs reuse the copy; `--setup`
forces the step again.

The script checks that Ollama is running and the model exists, then plays the episodes
(a dot per simulated day), runs the rule-based **baseline** and the perfect-foresight
**oracle** on the same seeds, and prints:

- the mean score per task for the model (named in the heading), baseline and oracle;
- **LLM minus baseline, seed by seed** with a 95% confidence interval, and the **gap closed**
  (0% = as good as the baseline, 100% = as good as the oracle; negative = worse than the baseline);
- how many replies contained no JSON, and how many model calls failed;
- the model's first replies, so you can see what it does.

Results are written to `results/` in the same format as `eval.py`, so you can compare them
later, e.g. `uv run python eval.py --compare results/llm_pharma-gemma2-2b_test_all_3.jsonl results/baseline_test_all_3.jsonl`.

## What the model answers: adjustment days

By default (`--action adjust`) a simple rule already orders every medicine up to a
sensible default: enough for its lead time plus 3 safety days. The model answers, for each
medicine, **how many extra days to keep on top of that default**, from 0 to +10, e.g.
`{"paracetamol": 4, "vitamins": 0, "insulin": 1}`: more paracetamol because demand is
rising, the default for vitamins, a little extra insulin because deliveries are slowing.

So a model that answers 0, leaves a medicine out, copies a number from the report or
writes no JSON at all plays like the baseline (about 0.89), not like an empty warehouse.
Its score comes from the adjustments it gets right; the oracle reaches about 0.94.
(Before this, Qwen 2.5 3B in `days` mode copied `days_of_cover` from the report until its
stock ran out and scored 0.17-0.44, so `days_of_cover` is no longer shown in this mode.)

### Other formats

With `--action days` the model answers with **how many days of stock it wants**, e.g.
`{"paracetamol": 8, "insulin": 6}`, and the script turns that into an order:

```
units ordered = days x recent daily demand - on hand - already inbound
```

One "day of stock" is one day of recent demand: if paracetamol sells about 400 a day,
8 days means 3,200 units in total, so with 1,000 on hand and 1,200 inbound the order is
1,000. Days are capped at 21, the storage size. Small models are bad at that arithmetic but reasonable at
judging "how much cover"; for example, Gemma 2 2B scored 0.24 when it had to write units.

The printout shows both the model's reply and the units sent. With `--action units` the
model writes units itself.

## Other models

Pass any Ollama model tag with `--model` (`--base-model` also works). Each result line
records it as `base_model`, and the result files are named after it, so runs of
different models never overwrite each other:

```bash
uv run python local_llm/run_local_llm.py --model qwen2.5:3b --tasks all --episodes 3
# -> results/llm_pharma-qwen2-5-3b_test_all_3.jsonl
```

Rough guide (approximate download sizes; larger models follow the rules better but are slower):

| `--model` | Parameters | Download |
|---|---|---|
| `qwen2.5:1.5b` | 1.5B | ~1 GB |
| `gemma2:2b` (default) | 2.6B | ~1.6 GB |
| `qwen2.5:3b` | 3B | ~1.9 GB |
| `llama3.2:3b` | 3B | ~2 GB |
| `phi4-mini` | 3.8B | ~2.5 GB |
| `gemma3:4b` | 4B | ~3.3 GB |
| `qwen2.5:7b` | 7B | ~4.7 GB |
| `gemma2:9b` | 9B | ~5.4 GB |

On a laptop CPU each call reads a ~1,800-token prompt, so expect anything from several
seconds (small models) to a minute or more (7B+) per simulated day. Each call may take up
to `--timeout` seconds (default 600) before it counts as failed.

## News and the hybrid agent

With `--news 1` or `--news 2`, disruptions are announced 5–10 days before they start (see
"News" in the main README). `--agent llm_news` runs the hybrid: the baseline places the
orders and the model only reads the news and answers with extra safety days per product.
The model is called only on days with news (about 30–55 of the 60 days), with a short
prompt, so a run is much faster than `--agent llm`:

```bash
uv run python local_llm/run_local_llm.py --model qwen2.5:3b --agent llm_news --news 2 --tasks all --seed 10000
# -> results/llm_news_pharma-qwen2-5-3b_news2_test_all_seed10000.jsonl
```

The score table then also shows `baseline_news`, the same hybrid with perfect news reading:
the target to aim for. Replies on days without news are logged as
`{}  (no news today: baseline only, model not called)`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--model` | `gemma2:2b` | Ollama model to run (also `--base-model`); set up automatically the first time |
| `--setup` | off | Download the model and recreate its `pharma-<model>` copy even if it exists |
| `--tasks` | `flu_season` | Comma-separated task names, or `all` |
| `--episodes` | `1` | Seeds per task (the first N test seeds) |
| `--seed` | none | Run exactly this one seed per task instead (test seeds 10000–19999, train 0–9999) |
| `--timeout` | `600` | Seconds allowed per model call |
| `--agent` | `llm` | `llm`: the model decides every order; `llm_news`: the model reads the news, the baseline orders |
| `--news` | `0` | Announce disruptions ahead: `0` none, `1` exact template, `2` varied wording |
| `--action` | `adjust` | `adjust`: extra days (0 to +10) on top of a sensible default per SKU; `days`: days of stock per SKU; `units`: units to order |
| `--show` | `10` | How many of the model's replies to print |
| `--split` | `test` | `train` or `test` seeds |
| `--base-url` | `http://localhost:11434/v1` | Any OpenAI-compatible server |
| `--server-model` | `pharma-<model>` | Exact model name on the server, e.g. for a non-Ollama server |

## Troubleshooting

| Message | Fix |
|---|---|
| `Cannot reach a model server ...` | Start Ollama (open the app or run `ollama serve`) |
| `Model pharma-... is not available ...` | The `ollama` command was not found, so it could not be set up automatically: install Ollama, or run the `--setup` command it prints |
| `The ollama command was not found` | Install Ollama, or add it to your PATH |
| `The LLM failed its first 3 calls ...` | The server stopped, ran out of memory, or calls are slower than `--timeout`: restart Ollama, use a smaller model, or raise `--timeout` |
| Very slow | Use a smaller model, fewer `--episodes`, or a free GPU notebook (Colab/Kaggle) |
