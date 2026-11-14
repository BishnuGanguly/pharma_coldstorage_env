"""Tests for the teacher (training/teacher.py) and the reward options it needs (training/grpo_reward.py)."""

from __future__ import annotations

import json
import statistics as st

import pytest

import eval as E
import inference
from models import PharmaAction
from server.my_env_environment import PharmaEnvironment
from tasks import TASK_REGISTRY, TEST_SEEDS
from training import episodes as EP
from training import grpo_reward as G
from training import probe as P
from training import teacher as T

TASKS = sorted(TASK_REGISTRY)
LOW, HIGH = inference.ADJUST_DAYS_RANGE


def days_of(task="supply_chain_broken", seed=4):
    """(env, obs) on every day of an episode the teacher plays."""
    env = PharmaEnvironment()
    obs = env.reset(task_name=task, seed=seed, news=2, scenario="random")
    agent = T.TeacherAgent()
    agent.start_episode(env)
    while not obs.done:
        yield env, obs, agent
        obs = env.step(PharmaAction(message=agent.act(obs, [])[0]))


# ---------------------------------------------------------------------------
# The teacher's answer
# ---------------------------------------------------------------------------

def test_answers_are_whole_days_in_range_for_every_sku_and_zero_without_visible_news():
    saw_extra = False
    for env, obs, _ in days_of():
        events = env._episode_config.news
        answer = T.teacher_answer(obs, events, T.normal_lead_times(env))
        assert list(answer) == list(obs.skus)
        assert all(isinstance(v, int) and LOW <= v <= HIGH for v in answer.values())
        visible = {e.sku_id for e in events if e.announce_day <= obs.current_date <= e.end_day}
        assert all(v == 0 for k, v in answer.items() if k not in visible)
        assert bool(visible) == bool(obs.news)
        saw_extra |= any(answer.values())
    assert saw_extra


def test_normal_lead_times_are_the_first_reports():
    env = PharmaEnvironment()
    obs = env.reset(task_name="flu_season", seed=1, news=2, scenario="random")
    assert T.normal_lead_times(env) == {k: s.avg_lead_time for k, s in obs.skus.items()}


def test_teacher_agent_sends_what_a_model_giving_its_answer_would_send():
    for _, obs, agent in days_of(seed=6):
        order, error = agent.act(obs, [])
        assert error is None and order == EP.answer_to_order(obs, agent.last_model_reply)
        assert set(json.loads(agent.last_model_reply)) == set(obs.skus)
        if obs.current_date > 25:
            break


def test_teacher_beats_the_baseline_and_the_extra_days_rule():
    seeds = list(TEST_SEEDS[:6])
    teacher = st.fmean(r["score"] for r in EP.run_rule_episodes(T.TeacherAgent(), "teacher", TASKS, seeds))
    rule = st.fmean(E.run_episode("baseline_news", t, s, news=2, scenario="random")["score"] for t in TASKS for s in seeds)
    baseline = st.fmean(E.run_episode("baseline", t, s, news=2, scenario="random")["score"] for t in TASKS for s in seeds)
    assert teacher > rule > baseline


# ---------------------------------------------------------------------------
# Supervised examples
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rows():
    return T.sft_examples(["supply_chain_broken", "flu_season"], range(2, 6))


def test_examples_are_the_models_prompt_with_the_teachers_answer(rows):
    assert {r["states"] for r in rows} == {"teacher", "baseline"}             # mixed: by seed parity
    assert all(r["states"] == ("teacher" if r["seed"] % 2 == 0 else "baseline") for r in rows)
    for row in (rows[0], rows[len(rows) // 2], rows[-1]):
        system, user, assistant = row["messages"]
        assert (system["role"], user["role"], assistant["role"]) == ("system", "user", "assistant")
        prompt = G.DayPrompt(row["task"], row["seed"], row["day"], states=row["states"])
        assert [system, user] == G.prompt_messages(prompt)                    # exactly what GRPO and eval send
        env, obs, _ = G.build_state(prompt)
        assert json.loads(assistant["content"]) == T.teacher_answer(obs, env._episode_config.news, T.normal_lead_times(env))


def test_every_news_day_is_kept_and_a_share_of_the_quiet_days(rows):
    news_days = len(G.news_day_prompts(["supply_chain_broken", "flu_season"], [3, 5]))        # baseline-played seeds
    assert sum(r["has_news"] and r["states"] == "baseline" for r in rows) == news_days
    quiet = [r for r in rows if not r["has_news"]]
    assert quiet and all(not any(json.loads(r["messages"][-1]["content"]).values()) for r in quiet)
    none = T.sft_examples(["flu_season"], [2], quiet_share=0.0)
    every = T.sft_examples(["flu_season"], [2], quiet_share=1.0)
    assert all(r["has_news"] for r in none) and len(every) == 60 and len(none) < len(every)
    assert T.sft_examples(["flu_season"], [2]) == T.sft_examples(["flu_season"], [2])          # deterministic


def test_states_option_and_label_stats(rows):
    teacher_only = T.sft_examples(["flu_season"], [3], states="teacher")
    assert {r["states"] for r in teacher_only} == {"teacher"}
    with pytest.raises(ValueError, match="states must be"):
        T.sft_examples(["flu_season"], [3], states="model")
    stats = T.label_stats(rows)
    assert stats["examples"] == len(rows) and stats["with_news"] == sum(r["has_news"] for r in rows)
    assert 0 < stats["all_zero_with_news"] < stats["with_news"]
    assert all(LOW < v <= HIGH for v in stats["nonzero_values"])


def test_examples_only_come_from_train_seeds_outside_the_probe():
    assert T.PROBE_SEEDS == P.PROBE_SEEDS and (T.SCENARIO, T.NEWS_LEVEL) == (G.SCENARIO, G.NEWS_LEVEL)
    for bad in ([TEST_SEEDS[0]], [P.PROBE_SEEDS[0]], [-1]):
        with pytest.raises(ValueError, match="train seeds outside the probe seeds"):
            T.sft_examples(["flu_season"], bad)


def test_command_line_writes_examples(tmp_path, capsys):
    out = tmp_path / "t.jsonl"
    T.main(["--seeds", "0:2", "--tasks", "flu_season", "--out", str(out)])
    lines = out.read_text().splitlines()
    assert len(lines) > 20 and json.loads(lines[0])["messages"][2]["role"] == "assistant"
    assert '"examples"' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        T.main(["--seeds", "9000:9002", "--out", str(out)])


class FakeTokenizer:
    """One token per character, with a Phi-style chat template."""

    eos_token = "#"

    def __init__(self, appends_answer=True):
        self.appends_answer = appends_answer

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if not self.appends_answer and messages[-1]["role"] == "assistant":
            return "<other layout>" + messages[-1]["content"]
        text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        return text + ("<|assistant|>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}


def test_encode_example_trains_on_the_answer_only(rows):
    tok, messages = FakeTokenizer(), rows[0]["messages"]
    enc = T.encode_example(tok, messages, max_tokens=100_000)
    prompt = tok.apply_chat_template(messages[:2], add_generation_prompt=True)
    answer = messages[2]["content"] + "<|end|>"
    assert "".join(map(chr, enc["input_ids"])) == prompt + answer            # the generation prompt, then the answer
    assert enc["labels"][:len(prompt)] == [T.IGNORE_INDEX] * len(prompt)
    assert "".join(map(chr, enc["labels"][len(prompt):])) == answer
    assert len(enc["input_ids"]) == len(enc["labels"]) == len(enc["attention_mask"]) == len(prompt) + len(answer)
    # A template that does not append the answer: the answer and the end token are added.
    other = T.encode_example(FakeTokenizer(appends_answer=False), messages, max_tokens=100_000)
    assert "".join(map(chr, other["input_ids"])) == prompt + messages[2]["content"] + "#"
    with pytest.raises(ValueError, match="more than 50"):
        T.encode_example(tok, messages, max_tokens=50)


# ---------------------------------------------------------------------------
# Reward options: who played before, who plays after
# ---------------------------------------------------------------------------

def test_prompts_can_come_from_the_teachers_play():
    by_teacher = G.news_day_prompts(["supply_chain_broken"], [4], states="teacher")
    assert by_teacher and {p.states for p in by_teacher} == {"teacher"}
    last = by_teacher[-1]
    _, obs_t, hist_t = G.build_state(last)
    _, obs_b, hist_b = G.build_state(G.DayPrompt(last.task, last.seed, last.day))
    assert len(hist_t) == len(hist_b) == last.day
    assert [h[1] for h in hist_t] != [h[1] for h in hist_b]                   # different orders were placed
    assert G.dataset_rows([last])[0]["states"] == "teacher"
    with pytest.raises(ValueError, match="player must be one of"):
        G.news_day_prompts(["flu_season"], [1], states="oracle")


def test_reward_reads_the_states_column_and_defaults_to_the_baseline():
    prompt = G.news_day_prompts(["supply_chain_broken"], [4], states="teacher")[5]
    reward = G.GRPOReward()
    text = json.dumps({"insulin": 4, "paracetamol": 4, "bp_medication": 4, "vitamins": 4, "hydroxychloroquine": 4})
    with_column = reward([text], task=[prompt.task], seed=[prompt.seed], day=[prompt.day], states=["teacher"])
    assert with_column == [reward.score_answer(prompt, text)["reward"]]
    without = reward([text], task=[prompt.task], seed=[prompt.seed], day=[prompt.day])
    assert without == [reward.score_answer(G.DayPrompt(prompt.task, prompt.seed, prompt.day), text)["reward"]]


def test_continuation_is_validated_and_the_default_is_unchanged():
    with pytest.raises(ValueError, match="continuation must be one of"):
        G.GRPOReward(continuation="oracle")
    prompt = G.news_day_prompts(["flu_season"], [7])[10]
    env, obs, _ = G.build_state(prompt)
    extra = {"paracetamol": 5.0}
    assert G.score(env, obs, prompt, extra) == G.score(env, obs, prompt, extra, continuation="baseline")
    assert G.GRPOReward(continuation="teacher").score_answer(prompt, "{}")["reward"] == 0.0     # answering 0 is still 0


def test_with_the_teacher_playing_on_its_own_answer_beats_stockpiling():
    # Judged as one step of good play (the teacher continues), the teacher's answer beats
    # +10 on the news SKUs; and +10 earns more when the baseline continues, since then
    # nobody else will prepare. (On 300 days: +0.27 +- 0.18 and +0.46 +- 0.22 points.)
    prompts = [p for t in TASKS for p in G.news_day_prompts([t], range(20, 32), states="teacher")[::7]]
    then_teacher, then_baseline = G.GRPOReward(continuation="teacher"), G.GRPOReward()
    teacher, plus10, plus10_baseline = [], [], []
    for p in prompts:
        env, obs, _ = G.build_state(p)
        answer = T.answer_text(T.teacher_answer(obs, env._episode_config.news, T.normal_lead_times(env)))
        flat = json.dumps({k: 10 for k in P.news_skus(p)})
        teacher.append(then_teacher.score_answer(p, answer)["reward"])
        plus10.append(then_teacher.score_answer(p, flat)["reward"])
        plus10_baseline.append(then_baseline.score_answer(p, flat)["reward"])
    assert len(prompts) > 50
    assert st.fmean(teacher) > st.fmean(plus10) + 0.1
    assert st.fmean(plus10_baseline) > st.fmean(plus10) + 0.1


def test_probe_exam_can_use_teacher_states_and_the_training_reward():
    reward = G.GRPOReward(continuation="teacher")
    items = P.build_probe(seeds=range(9000, 9004), count=6, reward=reward, states="teacher")
    assert len(items) == 6 and {i.prompt.states for i in items} == {"teacher"}
    results = P.score_probe(items, ["{}"] * 6, reward=reward)
    assert all(r["reward"] == 0.0 for r in results)
