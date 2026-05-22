#!/usr/bin/env python3
"""
Scan output directories and compute per-question accuracy for beam-search runs.

```bash
# scan one prefix and verbose
python3 scripts/compute_tts_accuracy.py \
    --output-root output \
    --prefixes aime_Qwen3 \
    --save results_aime_qwen3.csv \
    --verbose

# scan multiple prefixes
python3 scripts/compute_tts_accuracy.py \
    --output-root output \
    --prefixes aime_Qwen3 \
    --save results_multi.csv
```
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import List, Optional
import csv
import math


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # basic normalizations
    s = s.replace("$", "")
    s = s.strip().lower()
    # remove trailing dots/commas/semicolon
    s = s.rstrip(r".\",;:\)")
    s = s.lstrip("(\"")
    return s


def try_parse_number(s: str) -> Optional[float]:
    try:
        # remove commas
        s2 = s.replace(",", "")
        return float(s2)
    except Exception:
        return None


def extract_answer_from_groundtruth(gt: str) -> Optional[str]:
    """Try to extract a concise answer from a LaTeX-style groundtruth.
    Look for \boxed{} or \framebox{}, otherwise take the last numeric token if present.
    """
    if not gt:
        return None
    s = str(gt)
    # look for \boxed{...} or \framebox{...}
    boxed = re.findall(r'\\(?:boxed|framebox)\{([^}]*)\}', s)
    if boxed:
        return normalize_text(boxed[-1])
    # strip dollar signs and LaTeX spacing
    s2 = s.replace('$', ' ')
    # find numbers
    nums = re.findall(r'([+-]?\d+(?:\.\d+)?)', s2)
    if nums:
        return nums[-1]
    # fallback: return normalized whole
    return normalize_text(s)


def answers_match(gt: str, cand: str) -> bool:
    if gt is None:
        return False
    # try to extract concise answer from groundtruth first
    gt_ex = extract_answer_from_groundtruth(gt)
    gt_n = normalize_text(gt_ex if gt_ex is not None else gt)
    cand_n = normalize_text(cand)

    if not gt_n or not cand_n:
        return False

    gt_num = try_parse_number(gt_n)
    cand_num = try_parse_number(cand_n)
    if gt_num is not None and cand_num is not None:
        # numeric comparison: allow small tolerance
        if math.isclose(gt_num, cand_num, rel_tol=1e-6, abs_tol=1e-6):
            return True
        # also allow integer-equality if both near-integers
        if abs(round(gt_num) - gt_num) < 1e-9 and abs(round(cand_num) - cand_num) < 1e-9:
            return int(round(gt_num)) == int(round(cand_num))
        return False

    # otherwise exact normalized string compare
    return gt_n == cand_n


def find_record_files(run_dir: Path) -> List[Path]:
    # only match files named exactly like `record_<num>.jsonl`
    files = []
    for p in run_dir.rglob('record_*.jsonl'):
        if p.name and re.match(r'^record_\d+\.jsonl$', p.name):
            files.append(p)
    return files


def load_last_json(path: Path) -> Optional[dict]:
    try:
        with path.open('r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return None
        return json.loads(lines[-1])
    except Exception:
        return None


def score_run_dir(run_dir: Path) -> dict:
    # returns { 'run_dir': str, 'n_questions': int, 'n_correct': int }
    rec_files = find_record_files(run_dir)
    # group record files by their question directory (e.g., question_0)
    groups: dict[str, List[Path]] = {}
    for p in rec_files:
        # find ancestor named question_<num>
        qdir = None
        for anc in p.parents:
            if re.match(r'^question_\d+$', anc.name):
                qdir = anc
                break
        if qdir is None:
            qdir = p.parent
        groups.setdefault(str(qdir), []).append(p)

    total_questions = 0
    correct_questions = 0
    n_q_with_signal1 = 0
    total_num_signal1 = 0
    signal_fields = [
        'majority_vote', 'prm_min_max', 'prm_min_vote',
        'prm_last_max', 'prm_last_vote', 'prm_avg_max', 'prm_avg_vote'
    ]

    for qdir, files in groups.items():
        total_questions += 1
        question_matched = False
        question_cnt_ones = 0
        for rf in files:
            record = load_last_json(rf)
            if not record:
                continue
            gt = record.get('groundtruth') or record.get('solution') or record.get('ground_truth')
            outputs = record.get('output') or []

            # count ones in this record's result
            res = record.get('result') or record.get('results') or {}
            cnt_ones = 0
            if isinstance(res, dict):
                for f in signal_fields:
                    v = res.get(f)
                    try:
                        if int(v) == 1:
                            cnt_ones += 1
                    except Exception:
                        if str(v) == '1':
                            cnt_ones += 1
            question_cnt_ones += cnt_ones

            # check any output in this record
            for out in outputs:
                cand = None
                if isinstance(out, dict):
                    cand = out.get('extracted_answer') or out.get('answer') or out.get('text')
                    ea = out.get('extracted_answers') or out.get('gen_answers') or out.get('generated_answers')
                    if not cand and isinstance(ea, list) and ea:
                        for item in reversed(ea):
                            if item:
                                cand = item
                                break
                    if cand and try_parse_number(str(cand)) is None and isinstance(ea, list) and ea:
                        for item in reversed(ea):
                            if not item:
                                continue
                            nums = re.findall(r'([+-]?\d+(?:\.\d+)?)', str(item))
                            if nums:
                                cand = nums[-1]
                                break
                else:
                    if isinstance(out, str):
                        cand = out
                if cand and answers_match(gt, cand):
                    question_matched = True
                    break
            if question_matched:
                # no need to check further records for this question
                break

        if question_cnt_ones > 0:
            n_q_with_signal1 += 1
            total_num_signal1 += question_cnt_ones
        if question_matched:
            correct_questions += 1

    return {
        'run_dir': str(run_dir),
        'n_questions': total_questions,
        'n_correct': correct_questions,
        'n_q_with_signal1': n_q_with_signal1,
        'total_num_signal1': total_num_signal1,
    }


def scan_prefix(root: Path, prefix: str) -> List[dict]:
    # find subdirectories under root that start with prefix
    if not root.exists():
        return []
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    results = []
    for c in candidates:
        # each candidate may contain multiple run_* subdirs; scan those
        run_subdirs = [d for d in c.iterdir() if d.is_dir()]
        if not run_subdirs:
            # if no subdirs, treat candidate itself as run dir
            res = score_run_dir(c)
            res['top_dir'] = c.name
            results.append(res)
            continue
        for rd in run_subdirs:
            res = score_run_dir(rd)
            res['top_dir'] = c.name
            results.append(res)
    return results


def write_csv(rows: List[dict], out_path: Path):
    fieldnames = [
        'top_dir', 'run_dir', 'n_questions', 'n_correct', 'accuracy',
        'n_q_with_signal1', 'total_num_signal1'
    ]
    with out_path.open('w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            acc = (r['n_correct'] / r['n_questions']) if r['n_questions'] > 0 else 0.0
            writer.writerow({
                'top_dir': r.get('top_dir', ''),
                'run_dir': r.get('run_dir', ''),
                'n_questions': r.get('n_questions', 0),
                'n_correct': r.get('n_correct', 0),
                'accuracy': f"{acc:.6f}",
                'n_q_with_signal1': r.get('n_q_with_signal1', 0),
                'total_num_signal1': r.get('total_num_signal1', 0)
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=str, default='src/output', help='Root output dir')
    parser.add_argument('--prefixes', type=str, required=True, help='Comma-separated directory prefixes to scan (e.g. aime_Qwen3)')
    parser.add_argument('--save', type=str, default='tts_accuracy_results.csv', help='CSV file to save results')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    root = Path(args.output_root)
    prefixes = [p.strip() for p in args.prefixes.split(',') if p.strip()]
    all_rows = []
    for pref in prefixes:
        if args.verbose:
            print(f"Scanning prefix {pref} under {root}")
        rows = scan_prefix(root, pref)
        for r in rows:
            if args.verbose:
                n = r['n_questions']
                c = r['n_correct']
                q_with = r.get('n_q_with_signal1', 0)
                total_ones = r.get('total_num_signal1', 0)
                print(f"{pref}: {r['run_dir']} -> {c}/{n} = { (c/n if n>0 else 0):.4f }, q_with_signal1={q_with}, total_ones={total_ones}")
        all_rows.extend(rows)

    outp = Path(args.save)
    write_csv(all_rows, outp)
    print(f"Wrote CSV to {outp} with {len(all_rows)} rows")


if __name__ == '__main__':
    main()
