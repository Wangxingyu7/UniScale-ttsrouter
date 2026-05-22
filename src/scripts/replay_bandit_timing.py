#!/usr/bin/env python3
"""Replay bandit logs step-by-step via deployed TTS service and annotate timing.

Usage example (run from ttsrouter-v1.1 root):

nohup python -u ./scripts/replay_bandit_timing.py \
  --log-files ./src/bandit_process_routing_full_LinUCB_alpha1_w0.1_0.1_0.8_lin_a1.0_b1.0_l1.0_lr0.0005_h32x32_iter30_f_ave.log \
             ./src/bandit_process_routing_full_LinUCB_alpha1_w0.4_0.4_0.2_lin_a1.0_b1.0_l1.0_lr0.0005_h32x32_iter30_f_ave.log \
  --dataset ./src/envs/MATH/dataset/combined_dataset_210.jsonl \
  --service-url http://127.0.0.1:7777/tts-router-json \
  > ./replay_timing.out 2>&1 &
echo $! > ./replay_timing.pid

"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


PROBLEM_RE = re.compile(r"^Problem ID:\s*(\d+)/(\d+)\s*\(QueryID:\s*([^)]+)\)")
SELECTED_RE = re.compile(
    r"^Selected Model:\s*([^,]+),\s*QP:\s*(\d+),\s*CP:\s*(\d+),\s*BS:\s*(\d+),"
)


@dataclass
class StepSpec:
    idx: int
    total: int
    query_id: str
    model: str
    qp: int
    cp: int
    bs: int
    problem_line: int
    answer_line: Optional[int]


def _repo_root() -> Path:
    # .../ttsrouter-v1.1/src/scripts/replay_bandit_timing.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def _resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_question_map(dataset_path: Path) -> Dict[str, str]:
    qmap: Dict[str, str] = {}
    with dataset_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Dataset JSONL parse error at line {line_no}: {exc}") from exc
            qid = obj.get("id")
            prob = obj.get("problem")
            if isinstance(qid, str) and isinstance(prob, str):
                qmap[qid] = prob
    if not qmap:
        raise ValueError(f"No id->problem mapping loaded from {dataset_path}")
    return qmap


def parse_steps(log_lines: List[str]) -> List[StepSpec]:
    steps: List[StepSpec] = []
    i = 0
    n = len(log_lines)

    while i < n:
        line = log_lines[i].rstrip("\n")
        m = PROBLEM_RE.match(line.strip())
        if not m:
            i += 1
            continue

        idx = int(m.group(1))
        total = int(m.group(2))
        query_id = m.group(3).strip()

        selected_line_idx: Optional[int] = None
        answer_line_idx: Optional[int] = None
        model = ""
        qp = cp = bs = 0

        j = i + 1
        while j < n:
            cur = log_lines[j].rstrip("\n").strip()
            if cur.startswith("Problem ID:"):
                break
            if cur.startswith("Selected Model:"):
                selected_line_idx = j
                sm = SELECTED_RE.match(cur)
                if sm:
                    model = sm.group(1).strip()
                    qp = int(sm.group(2))
                    cp = int(sm.group(3))
                    bs = int(sm.group(4))
            if cur.startswith("Answer:"):
                answer_line_idx = j
            if cur.startswith("-" * 80):
                # end of block
                break
            j += 1

        if selected_line_idx is None or not model:
            raise ValueError(f"Failed to parse Selected Model/QP/CP/BS near Problem ID {idx}")

        steps.append(
            StepSpec(
                idx=idx,
                total=total,
                query_id=query_id,
                model=model,
                qp=qp,
                cp=cp,
                bs=bs,
                problem_line=i,
                answer_line=answer_line_idx,
            )
        )
        i = j + 1

    if not steps:
        raise ValueError("No step blocks parsed from log")
    return steps


def call_single_step(
    service_url: str,
    timeout_s: int,
    question: str,
    model: str,
    qp: int,
    cp: int,
    bs: int,
) -> Tuple[float, bool, str]:
    payload = {
        "problems": [
            {
                "problem": question,
                "solution": "None",
                "lm": model,
                "beam": {
                    "QP": float(qp),
                    "CP": float(cp),
                    "BS": int(bs),
                },
            }
        ],
        "eval_config": {
            "method": "beam_search",
            "num_sequence": int(bs),
            "tree_max_width": int(cp),
            "question_parallel_num": int(qp),
        },
    }

    start = time.perf_counter()
    ok = False
    msg = ""
    try:
        resp = requests.post(service_url, json=payload, timeout=timeout_s)
        ok = resp.status_code == 200
        if ok:
            msg = "ok"
        else:
            msg = f"http_{resp.status_code}: {resp.text[:200]}"
    except requests.RequestException as exc:
        msg = f"request_error: {exc}"
    elapsed = time.perf_counter() - start
    return elapsed, ok, msg


def annotate_log(
    original_lines: List[str],
    steps: List[StepSpec],
    step_times: List[float],
    step_status: List[str],
    total_time: float,
    success_cnt: int,
    service_url: str,
) -> List[str]:
    out = list(original_lines)

    # Insert per-step time line after each "Answer:" line if available, otherwise after Problem line.
    inserts: List[Tuple[int, str]] = []
    for s, t, st in zip(steps, step_times, step_status):
        target = s.answer_line if s.answer_line is not None else s.problem_line
        inserts.append((target + 1, f"Step Runtime: {t:.3f}s | Replay Status: {st}\n"))

    # Apply inserts from bottom to top to keep indices stable.
    for pos, text in sorted(inserts, key=lambda x: x[0], reverse=True):
        out.insert(pos, text)

    header = [
        "[Replay Timing Summary]\n",
        f"Total Replay Inference Time: {total_time:.3f}s\n",
        f"Successful Steps: {success_cnt}/{len(steps)}\n",
        f"Service URL: {service_url}\n",
        "=" * 80 + "\n",
    ]

    # Put summary near top (after first title+separator if present).
    insert_pos = 0
    if len(out) >= 2 and out[1].strip("\n") == "=" * 80:
        insert_pos = 2
    out[insert_pos:insert_pos] = header
    return out


def process_one_log(
    log_path: Path,
    qmap: Dict[str, str],
    service_url: str,
    timeout_s: int,
    output_dir: Path,
) -> Dict[str, float]:
    with log_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    steps = parse_steps(lines)

    step_times: List[float] = []
    step_status: List[str] = []
    total_time = 0.0
    success_cnt = 0

    for s in steps:
        question = qmap.get(s.query_id)
        if not question:
            elapsed = 0.0
            ok = False
            msg = f"missing_question_for_{s.query_id}"
        else:
            elapsed, ok, msg = call_single_step(
                service_url=service_url,
                timeout_s=timeout_s,
                question=question,
                model=s.model,
                qp=s.qp,
                cp=s.cp,
                bs=s.bs,
            )

        step_times.append(elapsed)
        step_status.append(msg)
        total_time += elapsed
        if ok:
            success_cnt += 1

        print(
            f"[{log_path.name}] step {s.idx}/{s.total} | model={s.model} qp={s.qp} cp={s.cp} bs={s.bs} "
            f"| {elapsed:.3f}s | {msg}"
        )

    annotated = annotate_log(
        original_lines=lines,
        steps=steps,
        step_times=step_times,
        step_status=step_status,
        total_time=total_time,
        success_cnt=success_cnt,
        service_url=service_url,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{log_path.stem}_with_timing.log"
    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(annotated)

    # Also save a compact CSV for downstream analysis.
    csv_path = output_dir / f"{log_path.stem}_step_timing.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("step_idx,step_total,query_id,model,qp,cp,bs,elapsed_sec,status\n")
        for s, t, st in zip(steps, step_times, step_status):
            f.write(
                f"{s.idx},{s.total},{s.query_id},{s.model},{s.qp},{s.cp},{s.bs},{t:.6f},{st}\n"
            )

    print(f"Saved annotated log: {out_path}")
    print(f"Saved step timing csv: {csv_path}")
    print(
        f"Task summary [{log_path.name}]: total={total_time:.3f}s, "
        f"success={success_cnt}/{len(steps)}"
    )
    return {
        "total_time": total_time,
        "success_cnt": float(success_cnt),
        "step_cnt": float(len(steps)),
    }


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    default_dataset = "../../supplementary_material/TTSRouter/cache/combined_dataset_210.jsonl"

    parser = argparse.ArgumentParser(description="Replay bandit process logs and annotate per-step inference timing")
    parser.add_argument(
        "--log-files",
        nargs="+",
        required=True,
        help="One or more bandit process log paths (absolute or repo-relative)",
    )
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help="Dataset JSONL with id/problem fields (absolute or repo-relative)",
    )
    parser.add_argument(
        "--service-url",
        default="http://127.0.0.1:7777/tts-router-json",
        help="TTS service endpoint",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="HTTP timeout (seconds) per step",
    )
    parser.add_argument(
        "--output-dir",
        default="output/replay_timing",
        help="Output directory (absolute or repo-relative)",
    )
    parser.add_argument(
        "--dedupe-inputs",
        action="store_true",
        help="If set, duplicate log paths will be de-duplicated",
    )

    args = parser.parse_args()

    args.repo_root = root
    args.dataset = _resolve_path(args.dataset, root)
    args.output_dir = _resolve_path(args.output_dir, root)
    args.log_files = [_resolve_path(p, root) for p in args.log_files]
    if args.dedupe_inputs:
        seen = set()
        deduped = []
        for p in args.log_files:
            if str(p) not in seen:
                deduped.append(p)
                seen.add(str(p))
        args.log_files = deduped
    return args


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    for p in args.log_files:
        if not p.exists():
            raise FileNotFoundError(f"Log file not found: {p}")

    qmap = load_question_map(args.dataset)
    print(f"Loaded question map entries: {len(qmap)} from {args.dataset}")

    overall_time = 0.0
    overall_success = 0.0
    overall_steps = 0.0

    for i, log_path in enumerate(args.log_files, start=1):
        print(f"\n=== [{i}/{len(args.log_files)}] Replaying {log_path} ===")
        task_result = process_one_log(
            log_path=log_path,
            qmap=qmap,
            service_url=args.service_url,
            timeout_s=args.timeout,
            output_dir=args.output_dir,
        )
        overall_time += task_result["total_time"]
        overall_success += task_result["success_cnt"]
        overall_steps += task_result["step_cnt"]

    print(
        f"\nOverall summary: total={overall_time:.3f}s, "
        f"success={int(overall_success)}/{int(overall_steps)}"
    )

    print("\nAll tasks finished.")


if __name__ == "__main__":
    main()
