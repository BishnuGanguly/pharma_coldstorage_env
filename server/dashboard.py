"""
Live dashboard for the Pharma environment, shown as the "Custom" tab at /web.

Runs one full episode in-process, with either a rule-based baseline or an LLM
reached through an OpenAI-compatible API, and redraws the charts after every
simulated day.

- Each run builds its own PharmaEnvironment, so viewers never share an episode,
  and the Playground tab and the benchmark API are unaffected.
- The agent only ever sees the observation, exactly as in inference.py. The
  charts additionally show the task's hidden ground truth (demand and
  lead-time curves) so a viewer can judge the agent's decisions.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

import inference
from models import EpisodeConfig, InventoryState, PharmaAction
from tasks import SKU_CATALOGUE, TASK_REGISTRY, average_step_reward

try:
    from my_env_environment import PharmaEnvironment
except ModuleNotFoundError:
    from server.my_env_environment import PharmaEnvironment


AGENT_BASELINE = "Baseline (rule-based, no LLM)"
AGENT_LLM = "LLM agent"

# The SKU each task is really about; the detail charts open on it.
FOCUS_SKU = {
    "supply_chain_broken": "insulin",
    "flu_season":          "paracetamol",
    "epidemic_two_wave":   "hydroxychloroquine",
}

# Stop an LLM run after this many failed calls in a row (bad token, wrong model...).
MAX_CONSECUTIVE_LLM_ERRORS = 3

# Chart colours: categorical slots 1-3 of a CVD-validated palette. Blue is what
# the agent sees or does, orange is hidden ground truth, aqua is deliveries.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK_MUTED = "#898781"
GRID = "rgba(137, 135, 129, 0.25)"
# Single-hue sequential ramp for the unmet-demand heatmap (near-zero recedes).
UNMET_SCALE = [[0.0, "#e8f1fc"], [0.25, "#9ec5f4"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"]]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

History = List[Tuple[int, str, str]]


def baseline_policy(obs: InventoryState, safety_days: float = 3.0) -> Dict[str, float]:
    """
    Order-up-to policy that uses only what the agent can observe: keep enough
    stock plus inbound to cover the (most pessimistic) lead time plus a few
    safety days of recent demand.
    """
    inbound: Dict[str, float] = {}
    for sku_id, qty, _ in obs.expected_inbound_orders:
        inbound[sku_id] = inbound.get(sku_id, 0.0) + qty

    orders: Dict[str, float] = {}
    for sku_id, sku in obs.skus.items():
        demand = max(sku.avg_demand_last_5_days, sku.avg_demand_per_day)
        if demand <= 0:
            continue
        lead_time = max([sku.avg_lead_time, *sku.lead_time_last3_orders])
        shortfall = demand * (lead_time + safety_days) - sku.inventory_on_hand - inbound.get(sku_id, 0.0)
        if shortfall > 0:
            orders[sku_id] = round(shortfall)
    return orders


class BaselineAgent:
    def act(self, obs: InventoryState, history: History) -> Tuple[str, Optional[str]]:
        return json.dumps(baseline_policy(obs)), None


class LLMAgent:
    """Same prompt and history format as inference.py, one chat call per day."""

    def __init__(self, model: str, api_key: str, base_url: str, client: Any = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key, timeout=60, max_retries=1)
        self.client = client
        self.model = model

    def act(self, obs: InventoryState, history: History) -> Tuple[str, Optional[str]]:
        obs_dict = inference.observation_to_dict(obs)
        user_prompt = inference.build_user_prompt(obs_dict, history, step=obs.current_date + 1)
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": inference.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=inference.TEMPERATURE,
                max_tokens=inference.MAX_TOKENS,
            )
            text = (completion.choices[0].message.content or "").strip()
            return text or "{}", None
        except Exception as exc:
            return "{}", f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

@dataclass
class EpisodeTrace:
    """Everything the dashboard draws, accumulated one simulated day at a time."""

    task_name: str
    seed: int
    config: EpisodeConfig
    state: InventoryState
    rows: List[Dict[str, Any]] = field(default_factory=list)
    # Every order the environment accepted: (sku_id, qty, order_day, true_arrival_day).
    orders: List[Tuple[str, float, int, int]] = field(default_factory=list)
    llm_errors: int = 0
    finished: bool = False
    stopped_reason: str = ""


def run_episode(task_name: str, seed: int, agent: Any) -> Iterator[EpisodeTrace]:
    """Play one episode, yielding the (same, growing) trace after reset and after every day."""
    env = PharmaEnvironment()
    obs = env.reset(task_name=task_name, seed=seed)
    trace = EpisodeTrace(task_name=task_name, seed=seed, config=env._episode_config, state=obs)
    history: History = []
    consecutive_errors = 0
    yield trace

    while not obs.done:
        day = obs.current_date
        delivered_before = {sku_id: len(h) for sku_id, h in obs.lead_time_history.items()}

        reply, error = agent.act(obs, history)
        obs = env.step(PharmaAction(message=reply))

        placed = [o for o in env._open_orders if o[2] == day]
        trace.orders.extend(placed)
        trace.state = obs
        trace.rows.append({
            "day": day,
            "reply": reply,
            "error": error,
            "orders": {sku_id: qty for sku_id, qty, _, _ in placed},
            "reward": obs.reward,
            "waste": obs.inventory_excess_today,
            "skus": {
                sku_id: {
                    "inventory": sku.inventory_on_hand,
                    "demand": obs.demand_history[sku_id][-1],
                    "fill": sku.demand_fulfilled_today,
                    "agent_lead_time": sku.avg_lead_time,
                    "observed_lead_times": obs.lead_time_history[sku_id][delivered_before[sku_id]:],
                }
                for sku_id, sku in obs.skus.items()
            },
        })
        history.append((day, reply, inference.build_feedback(obs)))

        if error:
            trace.llm_errors += 1
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_LLM_ERRORS:
                trace.stopped_reason = (
                    f"Stopped: the LLM call failed {consecutive_errors} times in a row. "
                    f"Last error: {error}"
                )
                yield trace
                return
        else:
            consecutive_errors = 0
        yield trace

    trace.finished = True
    yield trace


def score_so_far(trace: EpisodeTrace) -> float:
    """The final-score formula over the days played so far (equals the final score at the end)."""
    return average_step_reward(trace.state, len(trace.rows))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _figure(trace: EpisodeTrace, y_title: str, height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        # Gradio draws the chart title as a badge in the top-left corner, so the
        # legend goes below the x-axis instead.
        margin=dict(l=56, r=16, t=40, b=96),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK_MUTED),
        legend=dict(orientation="h", x=0, y=-0.22, xanchor="left", yanchor="top"),
        hovermode="x unified",
        xaxis=dict(
            title="Day", range=[-0.5, trace.config.no_of_days - 0.5],
            showgrid=False, zeroline=False, showline=True, linecolor=GRID,
        ),
        yaxis=dict(title=y_title, gridcolor=GRID, zeroline=False, rangemode="tozero"),
    )
    return fig


def _mark_today(fig: go.Figure, trace: EpisodeTrace) -> None:
    if trace.rows and not trace.finished:
        fig.add_vline(x=trace.rows[-1]["day"], line_width=1, line_color=INK_MUTED, opacity=0.6)


def stock_figure(trace: EpisodeTrace, sku_id: str) -> go.Figure:
    cfg = trace.config.skus[sku_id]
    curve = cfg.demand_curve or [1.0] * trace.config.no_of_days
    fig = _figure(trace, "Units")
    fig.add_scatter(
        x=list(range(len(curve))), y=[cfg.base_demand * c for c in curve],
        name="Daily demand (hidden from agent)", mode="lines",
        line=dict(color=ORANGE, width=2), hovertemplate="%{y:,.0f}",
    )
    fig.add_scatter(
        x=[r["day"] for r in trace.rows], y=[r["skus"][sku_id]["inventory"] for r in trace.rows],
        name="Stock on hand", mode="lines", line=dict(color=BLUE, width=2), hovertemplate="%{y:,.0f}",
    )
    _mark_today(fig, trace)
    return fig


def lead_time_figure(trace: EpisodeTrace, sku_id: str) -> go.Figure:
    cfg = trace.config.skus[sku_id]
    curve = cfg.lead_time_curve or [1.0] * trace.config.no_of_days
    fig = _figure(trace, "Days")
    fig.add_scatter(
        x=list(range(len(curve))), y=[max(1, round(cfg.base_lead_time * c)) for c in curve],
        name="True lead time if ordered that day (hidden)", mode="lines", line_shape="hv",
        line=dict(color=ORANGE, width=2), hovertemplate="%{y:.0f} days",
    )
    fig.add_scatter(
        x=[r["day"] for r in trace.rows], y=[r["skus"][sku_id]["agent_lead_time"] for r in trace.rows],
        name="Agent's lead-time estimate", mode="lines", line_shape="hv",
        line=dict(color=BLUE, width=2), hovertemplate="%{y:.1f} days",
    )
    delivered = [(r["day"], lt) for r in trace.rows for lt in r["skus"][sku_id]["observed_lead_times"]]
    fig.add_scatter(
        x=[d for d, _ in delivered], y=[lt for _, lt in delivered],
        name="Lead time seen on delivery", mode="markers",
        marker=dict(color=AQUA, size=9), hovertemplate="delivered after %{y:.0f} days",
    )
    _mark_today(fig, trace)
    return fig


def orders_figure(trace: EpisodeTrace, sku_id: str) -> go.Figure:
    today = trace.rows[-1]["day"] if trace.rows else -1
    ordered: Dict[int, float] = {}
    delivered: Dict[int, float] = {}
    for sku, qty, order_day, arrival_day in trace.orders:
        if sku != sku_id:
            continue
        ordered[order_day] = ordered.get(order_day, 0.0) + qty
        if arrival_day <= today:
            delivered[arrival_day] = delivered.get(arrival_day, 0.0) + qty
    fig = _figure(trace, "Units")
    fig.add_bar(
        x=list(ordered), y=list(ordered.values()), name="Ordered",
        marker=dict(color=BLUE, cornerradius=4), hovertemplate="%{y:,.0f}",
    )
    fig.add_bar(
        x=list(delivered), y=list(delivered.values()), name="Delivered",
        marker=dict(color=AQUA, cornerradius=4), hovertemplate="%{y:,.0f}",
    )
    fig.update_layout(barmode="group", bargap=0.15, bargroupgap=0.1)
    return fig


def reward_figure(trace: EpisodeTrace) -> go.Figure:
    rewards = [r["reward"] for r in trace.rows]
    fig = _figure(trace, "Reward")
    fig.add_scatter(
        x=[r["day"] for r in trace.rows], y=rewards, name="Step reward", mode="lines",
        line=dict(color=BLUE, width=2), hovertemplate="%{y:.3f}", showlegend=False,
    )
    fig.update_yaxes(range=[min([-0.05, *rewards]) - 0.05, 1.05], rangemode="normal")
    fig.add_hline(y=0, line_width=1, line_color=GRID)
    return fig


def unmet_heatmap(trace: EpisodeTrace) -> go.Figure:
    sku_ids = list(trace.config.skus)
    days = trace.config.no_of_days
    z: List[List[Optional[float]]] = [[None] * days for _ in sku_ids]
    for r in trace.rows:
        for i, sku_id in enumerate(sku_ids):
            z[i][r["day"]] = round((1.0 - r["skus"][sku_id]["fill"]) * 100, 1)
    fig = _figure(trace, "", height=100 + 36 * len(sku_ids))
    fig.add_heatmap(
        z=z, x=list(range(days)), y=sku_ids, zmin=0, zmax=100, colorscale=UNMET_SCALE,
        xgap=2, ygap=2, hoverongaps=False,
        colorbar=dict(title="Unmet %", thickness=12, outlinewidth=0),
        hovertemplate="%{y}, day %{x}: %{z:.0f}% of demand unmet<extra></extra>",
    )
    fig.update_layout(hovermode="closest", margin=dict(b=48))
    fig.update_yaxes(title=None, gridcolor="rgba(0,0,0,0)", autorange="reversed")
    return fig


def kpi_html(trace: EpisodeTrace) -> str:
    rows = trace.rows
    sku_days = [s for r in rows for s in r["skus"].values()]
    fill = sum(s["fill"] for s in sku_days) / len(sku_days) if sku_days else 0.0
    stockouts = sum(1 for s in sku_days if s["fill"] < 0.999)
    waste = sum(r["waste"] for r in rows)
    tiles = [
        ("Day", f"{len(rows)} / {trace.config.no_of_days}"),
        ("Score so far", f"{score_so_far(trace):.3f}"),
        ("Avg fill rate", f"{fill:.0%}"),
        ("Stockout SKU-days", f"{stockouts}"),
        ("Units wasted", f"{waste:,.0f}"),
    ]
    if trace.llm_errors:
        tiles.append(("LLM errors", f"{trace.llm_errors}"))
    cells = "".join(
        f'<div class="pd-tile"><div class="pd-label">{label}</div><div class="pd-value">{value}</div></div>'
        for label, value in tiles
    )
    return f"""
<style>
.pd-kpis {{ display: flex; flex-wrap: wrap; gap: 12px; }}
.pd-tile {{ flex: 1 1 140px; padding: 12px 16px; border-radius: 8px;
           border: 1px solid var(--border-color-primary); background: var(--block-background-fill); }}
.pd-label {{ font-size: 12px; color: var(--body-text-color-subdued); }}
.pd-value {{ font-size: 24px; font-weight: 600; color: var(--body-text-color); margin-top: 4px; }}
</style>
<div class="pd-kpis">{cells}</div>"""


def decisions_table(trace: EpisodeTrace) -> pd.DataFrame:
    records = []
    for r in reversed(trace.rows):
        reply = " ".join(r["reply"].split())
        records.append({
            "Day": r["day"],
            "Agent reply": reply if len(reply) <= 200 else reply[:197] + "...",
            "Orders accepted": ", ".join(f"{k}: {v:,.0f}" for k, v in r["orders"].items()) or "none",
            "Worst fill": f"{min(s['fill'] for s in r['skus'].values()):.0%}",
            "Reward": round(r["reward"], 3),
            "Error": r["error"] or "",
        })
    return pd.DataFrame(records, columns=["Day", "Agent reply", "Orders accepted", "Worst fill", "Reward", "Error"])


def sku_table(trace: EpisodeTrace, sku_id: str) -> pd.DataFrame:
    """Table view of the detail charts, so no value is only readable by colour or hover."""
    ordered: Dict[int, float] = {}
    for sku, qty, order_day, _ in trace.orders:
        if sku == sku_id:
            ordered[order_day] = ordered.get(order_day, 0.0) + qty
    records = []
    for r in trace.rows:
        s = r["skus"][sku_id]
        records.append({
            "Day": r["day"],
            "Demand": round(s["demand"], 1),
            "Stock on hand": round(s["inventory"], 1),
            "Fill %": round(s["fill"] * 100, 1),
            "Ordered": round(ordered.get(r["day"], 0.0), 1),
            "Agent lead-time estimate": round(s["agent_lead_time"], 2),
            "Lead times seen on delivery": ", ".join(f"{x:.0f}" for x in s["observed_lead_times"]),
        })
    return pd.DataFrame(records)


def status_text(trace: EpisodeTrace, agent_label: str) -> str:
    header = f"**{trace.task_name}** · seed {trace.seed} · {agent_label}"
    if trace.stopped_reason:
        return f"{header}\n\n⚠️ {trace.stopped_reason}"
    if trace.finished:
        return f"{header}\n\n✅ Episode finished. Final score **{score_so_far(trace):.3f}** (success threshold {inference.SUCCESS_THRESHOLD})."
    return f"{header}\n\n⏳ Running day {len(trace.rows)} of {trace.config.no_of_days}..."


def render_detail(trace: EpisodeTrace, sku_id: str) -> Tuple[go.Figure, go.Figure, go.Figure, pd.DataFrame]:
    return stock_figure(trace, sku_id), lead_time_figure(trace, sku_id), orders_figure(trace, sku_id), sku_table(trace, sku_id)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

INTRO = """
### Watch an agent run the warehouse
Pick a task and an agent, then **Run episode**. The charts update after every simulated day.
**Orange lines are hidden ground truth the agent never sees**; blue is what the agent sees or does.
The *Baseline* agent is a simple order-up-to rule and needs no API key. The *LLM agent* uses the
same prompt as `inference.py` and calls `API_BASE_URL` (Hugging Face router by default) with your token.
"""


def build_dashboard(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md) -> gr.Blocks:
    """gradio_builder for openenv's create_app; the arguments are part of its signature and unused here."""
    sku_choices = list(SKU_CATALOGUE)
    default_task = "flu_season"

    with gr.Blocks() as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            task = gr.Dropdown(list(TASK_REGISTRY), value=default_task, label="Task")
            seed = gr.Number(value=42, precision=0, label="Seed")
            agent = gr.Radio([AGENT_BASELINE, AGENT_LLM], value=AGENT_BASELINE, label="Agent")
            delay = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Pause per day (seconds)")
        with gr.Accordion("LLM settings", open=False):
            model = gr.Textbox(value=inference.MODEL_NAME, label="Model")
            token = gr.Textbox(
                type="password", label="API token (optional)",
                placeholder="Leave blank to use the server's HF_TOKEN secret",
            )
            gr.Markdown(f"Endpoint: `{inference.API_BASE_URL}` (set `API_BASE_URL` to change it). "
                        "A 60-day episode makes 60 model calls.")
        with gr.Row():
            run_btn = gr.Button("Run episode", variant="primary")
            stop_btn = gr.Button("Stop")
            focus = gr.Dropdown(sku_choices, value=FOCUS_SKU[default_task], label="SKU shown in the detail charts")

        status = gr.Markdown()
        kpis = gr.HTML()
        with gr.Row():
            stock_plot = gr.Plot(label="Stock on hand vs. demand")
            lead_plot = gr.Plot(label="Lead time: truth vs. the agent's estimate")
        with gr.Row():
            orders_plot = gr.Plot(label="Orders placed and delivered")
            reward_plot = gr.Plot(label="Step reward")
        heat_plot = gr.Plot(label="Unmet demand by SKU and day (all SKUs)")
        log = gr.Dataframe(
            label="Agent decisions (newest first)", wrap=True,
            column_widths=["6%", "38%", "28%", "9%", "8%", "11%"],
        )
        with gr.Accordion("Data table for the selected SKU", open=False):
            table = gr.Dataframe()
        trace_state = gr.State(None)

        run_outputs = [status, kpis, stock_plot, lead_plot, orders_plot, table, reward_plot, heat_plot, log, trace_state]

        def on_run(task_name, seed_value, agent_choice, pause, model_name, api_token, sku_id):
            seed_int = int(seed_value) if seed_value is not None else 42
            if agent_choice == AGENT_LLM:
                api_key = (api_token or "").strip() or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
                if not api_key:
                    yield ("⚠️ The LLM agent needs a token: paste one under **LLM settings**, "
                           "or set the `HF_TOKEN` secret on the Space.",
                           *[gr.skip()] * (len(run_outputs) - 1))
                    return
                runner = LLMAgent(
                    model=(model_name or "").strip() or inference.MODEL_NAME,
                    api_key=api_key, base_url=inference.API_BASE_URL,
                )
            else:
                runner = BaselineAgent()

            for trace in run_episode(task_name, seed_int, runner):
                yield (
                    status_text(trace, agent_choice), kpi_html(trace), *render_detail(trace, sku_id),
                    reward_figure(trace), unmet_heatmap(trace), decisions_table(trace), trace,
                )
                if pause and trace.rows and not (trace.finished or trace.stopped_reason):
                    time.sleep(pause)

        def on_focus(trace, sku_id):
            if trace is None:
                return [gr.skip()] * 4
            return render_detail(trace, sku_id)

        run_event = run_btn.click(
            on_run, inputs=[task, seed, agent, delay, model, token, focus], outputs=run_outputs,
        )
        stop_btn.click(lambda: "⏹️ Stopped.", outputs=status, cancels=[run_event])
        task.change(lambda t: FOCUS_SKU.get(t, sku_choices[0]), inputs=task, outputs=focus)
        focus.change(on_focus, inputs=[trace_state, focus], outputs=[stock_plot, lead_plot, orders_plot, table])

    return demo
