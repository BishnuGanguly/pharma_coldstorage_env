# Server, playground and dashboard

Running the environment as an OpenEnv server, the web playground, the live dashboard and the environment variables.

## Web Playground

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


## Live dashboard (Custom tab)

Next to the Playground, `/web` has a **Custom** tab that plays a whole episode and redraws charts after every simulated day:

- **Pick** a task, a seed and an agent, then press **Run episode** (**Stop** cancels).
- **Agents:**
  - *Baseline*: a simple order-up-to rule. It needs no API key and finishes in seconds.
  - *LLM agent*: uses the same prompt and action history as `inference.py`, one model call per day, so a 60-day episode makes 60 calls and takes about 1–5 minutes.
- **Charts:** stock on hand vs. demand, true lead time vs. the agent's estimate, orders and deliveries, step reward, and unmet demand for every SKU and day. **Orange lines are hidden ground truth the agent never sees**, so you can judge its decisions.
- **Tables:** every agent decision (the model's raw reply and the orders accepted), plus a per-day data table for the selected SKU.

To use the LLM agent on the Hugging Face Space, add your token as a **Secret** named `HF_TOKEN` (Space → Settings → Variables and secrets), or paste a token under *LLM settings* for a single run. Calls go through [Inference Providers](https://huggingface.co/docs/inference-providers) and count against that account's credits. `MODEL_NAME` sets the default model and `API_BASE_URL` points it at any OpenAI-compatible endpoint. Each run uses its own environment, and runs are queued one at a time.


## Choosing a task on reset

`reset()` accepts either a registered task name or a full config:

```python
await env.reset(task_name="flu_season", seed=42)
await env.reset(episode_config=get_task_config("flu_season", seed=42).model_dump(mode="json"))
```

With neither, a small 30-day two-SKU default episode is used.


## Environment variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | — | HuggingFace API key |
| `ENV_BASE_URL` | `http://localhost:8000` | Server URL |
| `LOCAL_IMAGE_NAME` | — | Run the server from this Docker image instead of `ENV_BASE_URL` |
| `TASK_SEED` | `42` | Seed for task generation and environment noise |
