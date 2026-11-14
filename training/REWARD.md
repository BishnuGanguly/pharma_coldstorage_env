# GRPO reward for the `adjust` answer format

Implemented in [`grpo_reward.py`](grpo_reward.py); every rule below has a test in
[`tests/test_grpo_reward.py`](../tests/test_grpo_reward.py).

## What one training prompt is

One day of one episode: `(task, seed, day)` in the **random scenario**
(`scenario="random"`) with **level-2 news**, on **train seeds** only. The state of
that day is rebuilt exactly by resetting the episode and letting the baseline play
every earlier day (so the state comes from the baseline's play). The prompt is
exactly what `agents.LLMAgent` sends in the `adjust` format: the system prompt and a
user turn with the report (no `days_of_cover`) and the results of the last 3 days.

Only **days with visible news** become prompts. On days without news every answer
scores the same (measured: 0.001 difference between answers), so they would teach
nothing.

## How one answer is scored

```
reward = (S(answer) - S(answer 0)) x 100 / HORIZON  +  format penalty
```

`S(x)` is the mean, over `FUTURES` possible futures, of the sum of the daily rewards
over the next `HORIZON` days when the answer's extra days are `x`:

1. Copy the environment as it is on that day.
2. Redraw its noise from today on (`resample_future_noise`) with future `i`'s
   generator. Today is included: the agent does not know today's demand or the lead
   time of today's order when it answers.
3. Today: order `baseline_policy(obs, extra_days=x)`, i.e. the baseline's default
   (lead time + 3 safety days) plus the answer's extra days, clipped to 0..10.
   **The answer applies today only.**
4. The next `HORIZON - 1` days: the baseline orders (extra 0).
5. Add up the `HORIZON` daily rewards (the environment's own reward:
   `stockout_penalty`-weighted fill minus waste).

| Setting | Value | Why (measured on 90 news-day states) |
|---|---|---|
| `HORIZON` | 15 days | best signal-to-noise: 8 days 4.6, **15 days 5.2**, 30 days 4.4 |
| `FUTURES` | 4 | the best answer matches the 16-future best on 91% of states (real future only: 74%; 8 futures: 92%) |
| answer applies | today only | holding it longer changes the gain by under 10% (0.217 vs 0.229-0.233) and matches how the model is used: a new answer every day |

The futures are fixed per prompt (seeded by task, seed, day and future index), so
**every answer to the same prompt is scored on the same futures**: luck that hits one
answer hits them all. Scores are deterministic: the same answer to the same prompt
always gets the same reward.

### Units

`(S(answer) - S(0)) x 100 / HORIZON` is how much better than answering 0 the answer
does **per day, in hundredths of a day's reward ("points")**. Answering 0 scores
exactly 0. On the measured states the best of 0, +2, +4, +7, +10 averaged about +1.5
points; on about half of the news days answering 0 was best.

### Format penalties

| Answer | Treated as | Penalty |
|---|---|---|
| No JSON object at all | answer 0 | -1.0 |
| A SKU left out, an unknown name, a non-number | that SKU gets 0 | none |
| A value below 0 or above 10 | clipped to 0..10 | -0.25 (once per answer) |

The penalties are smaller than a typical good answer, so getting the decision right
always matters more than the format.

## Who plays before and after today

Two choices decide what the reward teaches. Both default to the baseline, the original
behaviour; training after a warm start on the teacher sets both to the teacher
(`GRPOReward(continuation="teacher")`, `news_day_prompts(..., states="teacher")`).

| Choice | Setting | Values |
|---|---|---|
| Who played the days before the prompt | `DayPrompt.states` | `"baseline"` (default), `"teacher"` |
| Who orders on the days after the answer | `GRPOReward(continuation=...)` | `"baseline"` (default), `"teacher"` |

The teacher is [`teacher.py`](teacher.py): the news plan (`agents.news_plan_targets`)
answering in the `adjust` format. Played over full episodes it scores 0.887 on test
seeds (baseline 0.859, "+3 days on every SKU" 0.869, `baseline_news` 0.877, oracle 0.925).

**Why the continuation matters.** With the baseline ordering afterwards, adding many
days today looks good because nobody will prepare later: the answer is judged as the
only chance to react. A model that will answer again tomorrow should instead be judged
as one step of good play. Measured on 300 news days (probe seeds 9000-9059, states
from the teacher's play; mean reward in points, 95% interval for the differences):

| Answer | Baseline continues | Teacher continues |
|---|:---:|:---:|
| The teacher's answer | +0.235 | **+0.228** |
| +4 on the news SKUs | +0.381 | +0.152 |
| +7 on the news SKUs | **+0.476** | +0.069 |
| +10 on the news SKUs | +0.419 | -0.043 |
| Teacher's answer minus +10 | -0.184 ± 0.289 | **+0.270 ± 0.184** |
| Best of 0/+2/+4/+7/+10, chosen per day | 0.913 | 0.448 |

With the baseline continuing, the reward prefers flat stockpiling to the teacher's
answer, although over full episodes the teacher is clearly better than stockpiling.
GRPO with that reward would pull a warm-started model away from what it learned. With
the teacher continuing, the teacher's answer beats every flat answer, and the best flat
answer per day (0.448) is still above the teacher's (0.228): that difference is what
GRPO can add on top of the warm start. States from the baseline's play give the same
picture (teacher's answer minus +10: +0.373 ± 0.202 with the teacher continuing).

The differences between answers are smaller with the teacher continuing (the reference
answers differ by 0.25 points or more on 34% of news days, against 48%), so more
prompts are dropped by a spread filter and each kept prompt carries less signal.

## GRPO settings this reward assumes

- Group size 8 (the comparison between answers to one prompt is the signal).
- After a warm start, **no KL penalty** (`beta=0`): with a LoRA adapter TRL measures the
  KL against the base model (adapter off), which would pull the model back to where it
  was before the warm start.
- **No reward scaling by the group's standard deviation** (TRL `scale_rewards=False`,
  "Dr. GRPO"): when the answers to a prompt score nearly the same, dividing by their
  tiny spread would turn noise into large updates.
- Sampling temperature around 1.0, so that the 8 answers differ.

## Known simplifications

- After today a fixed rule orders (the baseline, or the teacher), not the model itself.
  With the teacher continuing an answer is judged as one step of good play, but still
  not as a step of the model's own play.
- States come from a rule's play (the baseline's or the teacher's), not the model's own.
