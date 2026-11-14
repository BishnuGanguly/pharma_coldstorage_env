"""
Dry run of training/grpo_phi4mini_kaggle.ipynb without a GPU: every code cell is executed
with stand-ins for the model, tokenizer and trainers, and tiny settings. It checks the
notebook's own logic (prompts, probe exams, warm-start examples, the reward with the
dataset's columns, comparisons, the verdict), not the training itself.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "training" / "grpo_phi4mini_kaggle.ipynb"
SMALL = dict(TRAIN_EPISODES_PER_TASK=3, TRAIN_STEPS=2, PROBE_EVERY=1, PROBE_COUNT=6, EVAL_SEEDS=1, CHECK_SEEDS=1,
             SFT_EPISODES_PER_TASK=2, SFT_EXAMPLES=24, EXPORT_GGUF=False, MIN_SPREAD=0.0)
ANSWER = '{"insulin": 2, "paracetamol": 0, "bp_medication": 1, "vitamins": 0, "hydroxychloroquine": 3}'


class FakeTokenizer:
    eos_token = pad_token = "#"
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        text += "<|assistant|>" if add_generation_prompt else ""
        return [ord(c) for c in text[::8]] if tokenize else text          # ~1 token per 8 characters

    def __call__(self, text, add_special_tokens=True, **kwargs):
        return {"input_ids": [ord(c) for c in text[::8]] or [ord("#")]}

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


class FakeModel:
    def named_modules(self):
        return [("layers.0.qkv_proj", None), ("layers.0.o_proj", None)]

    def print_trainable_parameters(self):
        print("trainable: fake")

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter.bin").write_text("fake")


class FakeDataset(list):
    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    def select(self, indices):
        return FakeDataset(self[i] for i in indices)


@dataclass
class FakeGRPOConfig:
    output_dir: str = ""
    learning_rate: float = 0.0
    max_steps: int = 0
    num_generations: int = 8
    beta: float = 0.0
    scale_rewards: bool = True


class FakeState:
    def __init__(self):
        self.global_step, self.log_history = 0, [{"loss": 1.0}, {"loss": 0.4}]


class FakeTrainer:
    """transformers.Trainer: remembers what it was given; train() reads every example through the collator."""

    def __init__(self, model=None, args=None, train_dataset=None, data_collator=None, **kwargs):
        self.dataset, self.collator, self.state = train_dataset, data_collator, FakeState()

    def train(self):
        for i in range(len(self.dataset)):
            batch = self.collator([self.dataset[i]])
            assert set(batch) == {"input_ids", "attention_mask", "labels"}


class FakeGRPOTrainer:
    """TRL's GRPOTrainer: each step scores one group with the reward functions, as TRL calls them."""

    def __init__(self, model=None, processing_class=None, reward_funcs=None, args=None, train_dataset=None,
                 callbacks=()):
        self.reward_funcs, self.args, self.dataset, self.callbacks = reward_funcs, args, train_dataset, callbacks
        self.rewards = []

    def train(self):
        state = FakeState()
        for step in range(self.args.max_steps):
            row = self.dataset[step % len(self.dataset)]
            n = self.args.num_generations
            columns = {k: [v] * n for k, v in row.items() if k != "prompt"}
            answers = [ANSWER] * (n - 1) + ["{}"]
            for fn in self.reward_funcs:
                self.rewards += fn(prompts=[row["prompt"]] * n, completions=answers, **columns)
            state.global_step = step + 1
            for cb in self.callbacks:
                cb.on_log(self.args, state, None, logs={"loss": 0.1, "reward": 0.2, "kl": 0.0})
                cb.on_step_end(self.args, state, None)


class _Callback:
    def on_log(self, *args, **kwargs):
        pass

    def on_step_end(self, *args, **kwargs):
        pass


def fake_modules(monkeypatch):
    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    class FastLanguageModel:
        from_pretrained = staticmethod(lambda **kw: (FakeModel(), FakeTokenizer()))
        get_peft_model = staticmethod(lambda model, **kw: model)
        for_inference = staticmethod(lambda model: None)
        for_training = staticmethod(lambda model: None)

    module("unsloth", FastLanguageModel=FastLanguageModel, is_bfloat16_supported=lambda: False)
    data = types.SimpleNamespace(Dataset=object)
    module("torch", __version__="fake", manual_seed=lambda s: None, tensor=lambda v: v,
           cuda=types.SimpleNamespace(is_available=lambda: False), utils=types.SimpleNamespace(data=data))
    module("transformers", __version__="fake", Trainer=FakeTrainer, TrainerCallback=_Callback,
           TrainingArguments=lambda **kw: types.SimpleNamespace(**kw))
    module("datasets", Dataset=FakeDataset)
    module("trl", __version__="fake", GRPOConfig=FakeGRPOConfig, GRPOTrainer=FakeGRPOTrainer)
    module("peft", __version__="fake")


def run_notebook(monkeypatch, tmp_path, capsys, **overrides):
    fake_modules(monkeypatch)
    monkeypatch.chdir(ROOT)
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    ns = {"WORK": str(tmp_path), "CODE_DIR": str(ROOT), "__name__": "notebook"}
    for source in code:
        shell = any(line.startswith("!") for line in source.splitlines())
        if shell or "git\", \"clone\"" in source or "import matplotlib" in source:
            continue                                      # install, clone and the plots
        exec(compile(source, "<notebook cell>", "exec"), ns)
        if "MODEL_NAME =" in source:                      # right after the config cell
            ns.update({**SMALL, **overrides})
            ns["PLAYER"] = "teacher" if ns["WARM_START"] else "baseline"
            ns["BETA_KL"] = 0.0 if ns["WARM_START"] else 0.04
        if source.startswith("def generate("):            # the model's answers
            ns["generate"] = lambda batch, **kw: [ANSWER] * len(batch)
            ns["back_to_training"] = lambda: None
    return ns, capsys.readouterr().out


def test_every_code_cell_is_valid_python():
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] == "code":
            source = "\n".join(line for line in "".join(cell["source"]).splitlines() if not line.startswith("!"))
            compile(source, "<notebook cell>", "exec")


@pytest.mark.parametrize("warm_start", [True, False])
def test_notebook_runs_end_to_end_with_stand_ins(monkeypatch, tmp_path, capsys, warm_start):
    ns, out = run_notebook(monkeypatch, tmp_path, capsys, WARM_START=warm_start)
    player = "teacher" if warm_start else "baseline"
    # Prompts, probe exams and the reward all use the same player.
    assert {row["states"] for row in ns["train_dataset"]} == {player}
    assert {i.prompt.states for i in ns["probe_items"] + ns["test_probe_items"]} == {player}
    assert ns["reward_fn"].continuation == ns["reward_ref"].continuation == player
    assert ns["config"].beta == (0.0 if warm_start else 0.04)
    assert len(ns["trainer"].rewards) == ns["TRAIN_STEPS"] * ns["NUM_GENERATIONS"]
    # The verdict and the saved results.
    assert "==== VERDICT ====" in out and "trained - teacher" in out
    summary = json.loads((tmp_path / "grpo_run" / "summary.json").read_text())
    assert summary["warm_start"] is warm_start and summary["player"] == player
    assert {"teacher", "news_plan", "baseline", "oracle"} <= set(summary["mean_scores"])
    assert (tmp_path / "grpo_results.zip").exists()
    if warm_start:
        assert len(ns["sft_encoded"]) == SMALL["SFT_EXAMPLES"] and summary["sft"]["examples"] == SMALL["SFT_EXAMPLES"]
        assert "after warm start" in summary["mean_scores"] and "after warm start - teacher" in out
        assert ns["probe_cb"].history[0]["reward"] == ns["sft_probe_summary"]["reward"]
        assert (tmp_path / "grpo_run" / "sft_summary.json").exists()
        assert (tmp_path / "grpo_run" / "phi4mini_pharma_lora_sft").is_dir()
    else:
        assert summary["sft"] == {} and "after warm start" not in summary["mean_scores"]
        assert ns["probe_cb"].history[0]["reward"] == ns["base_probe_summary"]["reward"]
