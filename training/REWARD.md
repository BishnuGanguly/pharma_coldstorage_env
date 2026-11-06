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

## GRPO settings this reward assumes

- Group size 8 (the comparison between answers to one prompt is the signal).
- **No reward scaling by the group's standard deviation** (TRL `scale_rewards=False`,
  "Dr. GRPO"): when the answers to a prompt score nearly the same, dividing by their
  tiny spread would turn noise into large updates.
- Sampling temperature around 1.0, so that the 8 answers differ.

## Known simplifications

- After today the baseline orders, not the model, so the reward cannot teach plans
  that need several coordinated days.
- States come from the baseline's play, not the model's own.
