# Run a local LLM on the Pharma environment

Test the environment with a model running on your own computer through
[Ollama](https://ollama.com): free and without API limits. The default is
**Gemma 2 2B** (`gemma2:2b`, about 2.6 billion parameters, about a 1.6 GB
download), which runs on a normal laptop. Any other Ollama model works with
`--base-model`.

## 1. Install Ollama

Download it from https://ollama.com/download and start it. The desktop app starts
the server automatically; otherwise run `ollama serve` in a separate terminal.

## 2. Get the code and dependencies

```bash
git pull
uv sync
```

## 3. One-time model setup

```bash
uv run python local_llm/run_local_llm.py --setup --episodes 1
```

This runs `ollama pull gemma2:2b`, then `ollama create pharma-gemma2-2b` from a small
Modelfile that raises the context window to 4,096 tokens: the environment's prompts are
about 1,800 tokens, and Ollama may cut prompts longer than its default window from the
start, which would drop the rules and output format. It then runs one test episode.

## 4. Run it

```bash
# 1 episode of flu_season (60 model calls)
uv run python local_llm/run_local_llm.py

# 3 seeds of every task (540 calls)
uv run python local_llm/run_local_llm.py --tasks all --episodes 3

# one chosen seed (60 calls); the same seed always gives the same world
uv run python local_llm/run_local_llm.py --seed 10042
```

The script checks that Ollama is running and the model exists, then plays the episodes
(a dot per simulated day), runs the rule-based **baseline** and the perfect-foresight
**oracle** on the same seeds, and prints:

- the mean score per task for the LLM, baseline and oracle;
- **LLM minus baseline, seed by seed** with a 95% confidence interval, and the **gap closed**
  (0% = as good as the baseline, 100% = as good as the oracle; negative = worse than the baseline);
- how many replies contained no JSON, and how many model calls failed;
- the model's first replies, so you can see what it does.

Results are written to `results/` in the same format as `eval.py`, so you can compare them
later, e.g. `uv run python eval.py --compare results/llm_pharma-gemma2-2b_test_all_3.jsonl results/baseline_test_all_3.jsonl`.

## What the model answers: days of stock

By default (`--action days`) the model does not calculate units. For each medicine it
answers with **how many days of stock it wants**, e.g. `{"paracetamol": 8, "insulin": 6}`,
and the script turns that into an order:

```
units ordered = days x recent daily demand - on hand - already inbound
```

One "day of stock" is one day of recent demand: if paracetamol sells about 400 a day,
8 days means 3,200 units in total, so with 1,000 on hand and 1,200 inbound the order is
1,000. Days are capped at 30. Small models are bad at that arithmetic but reasonable at
judging "how much cover"; for example, Gemma 2 2B scored 0.24 when it had to write units.

The printout shows both the model's reply and the units sent. Use `--action units` to
make the model write units itself.

## Other models

Pass any Ollama model tag with `--base-model`, both for `--setup` and for runs:

```bash
uv run python local_llm/run_local_llm.py --setup --base-model qwen2.5:3b
uv run python local_llm/run_local_llm.py --base-model qwen2.5:3b --tasks all --episodes 3
```

Rough guide (approximate download sizes; larger models follow the rules better but are slower):

| `--base-model` | Parameters | Download |
|---|---|---|
| `qwen2.5:1.5b` | 1.5B | ~1 GB |
| `gemma2:2b` (default) | 2.6B | ~1.6 GB |
| `qwen2.5:3b` | 3B | ~1.9 GB |
| `qwen2.5:7b` | 7B | ~4.7 GB |
| `gemma2:9b` | 9B | ~5.4 GB |

On a laptop CPU each call reads a ~1,800-token prompt, so expect anything from several
seconds (small models) to a minute or more (7B+) per simulated day. Each call may take up
to `--timeout` seconds (default 600) before it counts as failed.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--base-model` | `gemma2:2b` | Ollama model to use |
| `--tasks` | `flu_season` | Comma-separated task names, or `all` |
| `--episodes` | `1` | Seeds per task (the first N test seeds) |
| `--seed` | none | Run exactly this one seed per task instead (test seeds 10000–19999, train 0–9999) |
| `--timeout` | `600` | Seconds allowed per model call |
| `--action` | `days` | `days`: the model answers in days of stock per SKU, converted to units; `units`: the model writes units |
| `--show` | `10` | How many of the model's replies to print |
| `--split` | `test` | `train` or `test` seeds |
| `--base-url` | `http://localhost:11434/v1` | Any OpenAI-compatible server |
| `--model` | `pharma-<base-model>` | Exact model name on the server, e.g. for a non-Ollama server |

## Troubleshooting

| Message | Fix |
|---|---|
| `Cannot reach a model server ...` | Start Ollama (open the app or run `ollama serve`) |
| `Model pharma-... is not available ...` | Run the `--setup` command it prints |
| `The ollama command was not found` | Install Ollama, or add it to your PATH |
| `The LLM failed its first 3 calls ...` | The server stopped, ran out of memory, or calls are slower than `--timeout`: restart Ollama, use a smaller model, or raise `--timeout` |
| Very slow | Use a smaller model, fewer `--episodes`, or a free GPU notebook (Colab/Kaggle) |
