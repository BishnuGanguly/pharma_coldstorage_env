# Run a local LLM on the Pharma environment

Test the environment with **Qwen2.5 1.5B** running on your own computer through
[Ollama](https://ollama.com): free, no API limits, and it runs on a normal laptop
CPU (about 1 GB download, about 2 GB of RAM).

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

This runs `ollama pull qwen2.5:1.5b` and then `ollama create pharma-qwen15 -f local_llm/Modelfile`,
which makes a copy of the model with an 8,192-token context window. The environment's prompts
are about 1,800 tokens, and Ollama may cut prompts longer than its default window from the start,
which would drop the rules and output format. It then runs one test episode.

## 4. Run it

```bash
# 1 episode of flu_season (60 model calls, a few minutes on a laptop CPU)
uv run python local_llm/run_local_llm.py

# 3 seeds of every task (540 calls)
uv run python local_llm/run_local_llm.py --tasks all --episodes 3
```

The script checks that Ollama is running and the model exists, then plays the episodes
(printing progress every 10 days), runs the rule-based **baseline** and the perfect-foresight
**oracle** on the same seeds, and prints:

- the mean score per task for the LLM, baseline and oracle;
- **LLM minus baseline, seed by seed** with a 95% confidence interval, and the **gap closed**
  (0% = as good as the baseline, 100% = as good as the oracle; negative = worse than the baseline);
- how many replies contained no JSON, and how many model calls failed;
- the model's first replies, so you can see how it reasons.

Results are written to `results/` in the same format as `eval.py`, so you can compare them
later, e.g. `uv run python eval.py --compare results/llm_pharma-qwen15_test_all_3.jsonl results/baseline_test_all_3.jsonl`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--tasks` | `flu_season` | Comma-separated task names, or `all` |
| `--episodes` | `1` | Seeds per task (the first N test seeds) |
| `--model` | `pharma-qwen15` | Model name on the server, e.g. another Ollama model |
| `--base-url` | `http://localhost:11434/v1` | Any OpenAI-compatible server |
| `--show` | `10` | How many of the model's replies to print |
| `--split` | `test` | `train` or `test` seeds |

## Troubleshooting

| Message | Fix |
|---|---|
| `Cannot reach a model server ...` | Start Ollama (open the app or run `ollama serve`) |
| `Model pharma-qwen15 is not available ...` | Run the `--setup` command above |
| `The ollama command was not found` | Install Ollama, or add it to your PATH |
| `The LLM failed its first 3 calls ...` | Usually the server stopped or ran out of memory: restart Ollama |
| Very slow | Use fewer `--episodes`, or a free GPU notebook (Colab/Kaggle) |
