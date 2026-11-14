# Training a small model

The teacher (a supervised warm start), the GRPO reward, and the Kaggle notebook for Phi-4-mini.

## Training reward (GRPO)

`training/grpo_reward.py` scores answers in the `adjust` format for GRPO training, using the simulator only (no model calls). One prompt is one day with visible news in the random scenario (level-2 news, train seeds), rebuilt by letting the baseline play the earlier days. An answer's reward is how much better than answering 0 it does over the next 15 days (the answer sets today's order, then the baseline orders), averaged over 4 possible futures that every answer to the same prompt shares, in points (hundredths of a day's reward, per day), with small penalties for an answer without JSON (-1) or with values outside 0..10 (-0.25). The rules and the measurements behind each one are in [`training/REWARD.md`](../training/REWARD.md). Scoring a group of 8 answers takes about 0.1 s.

### The teacher: a warm start before GRPO

GRPO only compares the answers the model happens to give, so it starts slowly from a model that has no idea how many days a delay is worth. `training/teacher.py` gives it a head start. The **teacher** is the news plan (`news_plan`, see [News](results.md#news-disruptions-announced-ahead-of-time)) answering in the model's own `adjust` format: whole extra days from 0 to 10 for the SKUs in today's news, 0 for the rest. Played through the model's answer path it scores:

| 50 test seeds per task, level-2 news | Fixed tasks | Random scenario |
|---|:---:|:---:|
| Baseline | 0.867 | 0.859 |
| "+3 days on every SKU" | 0.886 | 0.869 |
| `baseline_news` (extra-days rule) | 0.903 | 0.877 |
| **Teacher (news plan in the `adjust` format)** | **0.909** | **0.887** |
| `news_plan` (exact orders) | 0.909 | 0.892 |
| Oracle | 0.931 | 0.925 |

The format costs the plan 0.005 in the random scenario, mostly because it cannot go below the baseline's default when demand falls after a peak (allowing -3 would recover 0.003; rounding to whole days costs nothing).

```bash
uv run python -m training.teacher --seeds 0:300 --out data/teacher_adjust.jsonl   # supervised examples, no GPU
```

Each example is the exact chat the model gets on one day plus the teacher's answer. Every news day is kept and 15% of the days without news (answer: all zeros); half of the episodes are played by the teacher and half by the baseline, so the model sees both kinds of state. About half of the news-day answers are all zeros, which teaches holding back as well as reacting. Examples only come from train seeds outside the probe exam's (checked). 120 episodes give about 5,500 examples in 6 seconds.

**The GRPO reward had to change with it.** Judged by the original reward (the answer today, then the *baseline* for 15 days), the teacher's answers score below flat "+7 on the news SKUs", because with the baseline continuing the answer is the only chance to react and stockpiling pays. GRPO would then un-teach the warm start. `GRPOReward(continuation="teacher")` lets the teacher order on the following days instead, so an answer is judged as one step of good play: the teacher's answer then beats every flat answer (+0.270 ± 0.184 points over +10 on 300 news days), while the best answer per day is still better than the teacher's, which is what GRPO can add. Prompts can likewise come from the teacher's play (`news_day_prompts(..., states="teacher")`). Both default to the baseline, the original behaviour; the measurements are in [`training/REWARD.md`](../training/REWARD.md).


## Training Phi-4-mini with GRPO (Kaggle)

`training/grpo_phi4mini_kaggle.ipynb` is the full run, in clearly separated cells:

1. setup;
2. Phi-4-mini in 4-bit with LoRA (Unsloth);
3. training prompts (news days, train seeds 0–8999) and a probe exam on held-out seeds 9000–9099;
4. the untrained model on test seeds;
   - 4b. the **warm start** (`WARM_START = True`, the default): supervised fine-tuning on the teacher's answers, loss on the answer tokens only, then the probe exam again;
5. GRPO with health metrics and a probe exam every 25 steps (after a warm start: prompts from the teacher's play, the teacher continuing in the reward, no KL penalty);
6. saving the LoRA adapter;
7. after-training evaluation: test-seed probe, full 60-day episodes against the baseline, "+3 everywhere", perfect news reading, the teacher and the oracle with paired 95% intervals (and what the warm start and GRPO each contributed), plus news-0 and fixed-scenario checks;
8. a verdict, a GGUF export for Ollama, and a results zip.

**To run it on Kaggle:**
1. Create a new notebook and import this `.ipynb` (File → Import Notebook).
2. Turn on GPU T4 ×1 and Internet.
3. Choose *Save Version → Save & Run All*.

It takes about 6–8 h with the default config, plus the warm start (not yet timed on a T4; expect roughly an hour). For a quick first run, lower `TRAIN_STEPS`, `EVAL_SEEDS` and `SFT_EXAMPLES` in the config cell. `WARM_START = False` runs GRPO alone, as before. The notebook clones the `main` branch of `BishnuGanguly/InventoryGym` (`REPO` and `BRANCH` in the setup cell).

The notebook's GPU-free helpers are tested in `tests/test_training_helpers.py`:
- `training/probe.py`: the probe exam;
- `training/episodes.py`: batched full episodes with a local model;
- `training/monitor.py`: the training-health numbers.

`tests/test_notebook_dry_run.py` also executes every code cell of the notebook with stand-ins for the model, tokenizer and trainers, with and without the warm start, so the notebook's own logic is checked without a GPU. The training itself has not been run since the warm start was added.
