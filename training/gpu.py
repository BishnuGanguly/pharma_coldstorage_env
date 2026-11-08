"""
Loading Phi-4-mini with Unsloth and answering prompts with it, shared by the Kaggle notebook
and the evaluation worker (training/eval_worker.py). Needs a GPU; unsloth and torch are
imported only when these functions run.
"""

from __future__ import annotations

import gc
from typing import Any, Callable, Dict, List, Optional, Tuple

Messages = List[Dict[str, str]]


def load_model(name_or_path: str, max_seq_length: int) -> Tuple[Any, Any]:
    """A model in 4-bit for answering prompts: a base model name, or a saved LoRA folder or checkpoint."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(model_name=name_or_path, max_seq_length=max_seq_length,
                                                         load_in_4bit=True, dtype=None)
    prepare_tokenizer(tokenizer)
    return model, tokenizer


def prepare_tokenizer(tokenizer: Any) -> None:
    tokenizer.padding_side = "left"          # batched generation pads on the left
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def make_generate(model: Any, tokenizer: Any, batch_size: int, max_new_tokens: int) -> Callable[..., List[str]]:
    """generate(list of chats, temperature=0.0) -> one answer per chat. Greedy by default."""
    import torch
    from unsloth import FastLanguageModel

    def generate(batch: List[Messages], temperature: float = 0.0, max_tokens: Optional[int] = None) -> List[str]:
        FastLanguageModel.for_inference(model)
        answers: List[str] = []
        for i in range(0, len(batch), batch_size):
            chunk = batch[i:i + batch_size]
            texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in chunk]
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            sampling = dict(do_sample=True, temperature=temperature, top_p=1.0) if temperature > 0 else dict(do_sample=False)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_tokens or max_new_tokens,
                                     pad_token_id=tokenizer.pad_token_id, **sampling)
            answers += tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [a.strip() for a in answers]

    return generate


def free(*objects: Any) -> None:
    """Release GPU memory held by models that are no longer used."""
    del objects
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass
