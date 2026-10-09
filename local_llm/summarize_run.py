"""
Print a short summary of an LLM results file written by run_local_llm.py or eval.py
--save-steps: each episode's score, then only the days the model asked for something
(non-zero values) or had news to read. Small enough to paste or read at a glance.

    uv run python local_llm/summarize_run.py results/llm_pharma-phi4-mini_news2_test_all_3.jsonl
    uv run python local_llm/summarize_run.py results/llm_*.jsonl       # several files

Example output:

    flu_season 10000 score 0.9157 stockout_sku_days 26 wasted 0
      day  6 news {}
      day 25 news {'paracetamol': 7}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference import parse_json_object  # noqa: E402


def summarize(path: Path) -> Iterator[str]:
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        yield (f"{record['task']} {record['seed']} score {record['score']} "
               f"stockout_sku_days {record['stockout_sku_days']} wasted {round(record['units_wasted'])}")
        steps = record.get("steps")
        if not steps:
            yield "  (no steps saved: run with save_steps / --save-steps to see the replies)"
            continue
        for step in steps:
            extra = {k: v for k, v in parse_json_object(step.get("reply") or "").items() if v}
            if extra or step.get("news"):
                yield f"  day {step['day']:2d} {'news' if step.get('news') else '    '} {extra}"


def main(argv: Optional[Sequence[str]] = None) -> None:
    paths: List[str] = list(argv if argv is not None else sys.argv[1:])
    if not paths:
        sys.exit(__doc__)
    for i, name in enumerate(paths):
        path = Path(name)
        if not path.is_file():
            sys.exit(f"No such file: {path}. List the results with `ls results/`.")
        if len(paths) > 1:
            print(("\n" if i else "") + f"== {path.name} ==")
        for line in summarize(path):
            print(line)


if __name__ == "__main__":
    main()
