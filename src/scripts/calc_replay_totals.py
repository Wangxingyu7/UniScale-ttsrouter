#!/usr/bin/env python3
"""Calculate per-task total runtime from replay_timing.out logs.

Example:
  python ./scripts/calc_replay_totals.py --input ../replay_timing.out
"""

from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path

STEP_RE = re.compile(
    r"^\[(?P<task>[^\]]+)\]\s+step\s+(?P<idx>\d+)/(?P<total>\d+)\s+\|.*\|\s+(?P<sec>[0-9]+(?:\.[0-9]+)?)s\s+\|"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sum per-task runtime from replay output log")
    parser.add_argument("--input", required=True, help="Path to replay_timing.out")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Number of warmup steps; post-warmup runtime is computed after this boundary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    totals = OrderedDict()
    counts = OrderedDict()
    post_warmup_totals = OrderedDict()
    post_warmup_counts = OrderedDict()

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = STEP_RE.search(line)
            if not m:
                continue
            task = m.group("task")
            idx = int(m.group("idx"))
            total = int(m.group("total"))
            sec = float(m.group("sec"))
            totals[task] = totals.get(task, 0.0) + sec
            counts[task] = counts.get(task, 0) + 1

            # "Post warmup" section: defaults to last 160 steps when total=210 and warmup=50.
            post_warmup_start = max(1, args.warmup_steps + 1)
            if idx >= post_warmup_start and idx <= total:
                post_warmup_totals[task] = post_warmup_totals.get(task, 0.0) + sec
                post_warmup_counts[task] = post_warmup_counts.get(task, 0) + 1

    if not totals:
        print("No step timing lines found.")
        return

    print(f"Input: {input_path}")
    print("Per-task totals:")
    for task, total in totals.items():
        step_cnt = counts[task]
        post_total = post_warmup_totals.get(task, 0.0)
        post_cnt = post_warmup_counts.get(task, 0)
        print(
            f"- {task}: total={total:.3f}s ({total/60.0:.2f} min), steps={step_cnt}; "
            f"post_warmup={post_total:.3f}s ({post_total/60.0:.2f} min), steps={post_cnt}"
        )

    grand_total = sum(totals.values())
    grand_steps = sum(counts.values())
    grand_post_total = sum(post_warmup_totals.values())
    grand_post_steps = sum(post_warmup_counts.values())
    print(f"Overall: {grand_total:.3f}s ({grand_total/60.0:.2f} min), steps={grand_steps}")
    print(
        f"Overall post_warmup: {grand_post_total:.3f}s ({grand_post_total/60.0:.2f} min), "
        f"steps={grand_post_steps}"
    )


if __name__ == "__main__":
    main()
