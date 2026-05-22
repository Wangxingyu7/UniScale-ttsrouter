#!/usr/bin/env python3
"""Sample subsets of record files to emulate smaller QP settings and score accuracy.

Example:
    # Quick copy/paste (single run per question sampling)
    python3 scripts/sample_qp_accuracy_per_question.py --source-run output_aime/aime_Qwen3-32B_QP32_CP2_BS1_beam_search/Qwen3-32B/Skywork-o1-Open-PRM-Qwen-2.5-1.5B/40_4_2 --sample-sizes 64,32,16 --trials 3 --seed 42 --save question_trials_aime.csv --save-summary question_summary_aime.csv

# Quick copy/paste (wildcard across runs)
python3 scripts/sample_qp_accuracy_per_question.py \
    --source-run "output_aime/aime_Qwen3*" \
     --sample-sizes 64,32,16,8,4,2,1 \
     --trials 3 \
     --seed 42 \
     --save question_trials_aime.csv \
     --save-summary question_summary_aime.csv 
"""


from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import re
from glob import glob

import math
from compute_eflops import ExperimentCostCalculator, get_model_config, MODEL_CONFIGS

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

try:
    from compute_tts_accuracy import (  # type: ignore
        answers_match,
        extract_answer_from_groundtruth,
        normalize_text,
        try_parse_number,
    )
except Exception:

    def normalize_text(s: str) -> str:
        if s is None:
            return ""
        s = str(s).strip()
        s = s.replace("$", "")
        s = s.strip().lower()
        s = s.rstrip(".\"\\,;:)")
        s = s.lstrip("(")
        s = s.lstrip("\"")
        return s

    def try_parse_number(s: str) -> Optional[float]:
        try:
            return float(str(s).replace(",", ""))
        except Exception:
            return None

    def extract_answer_from_groundtruth(gt: str) -> Optional[str]:
        if not gt:
            return None
        boxed = re.findall(r"\\(?:boxed|framebox)\{([^}]*)\}", str(gt))
        if boxed:
            return normalize_text(boxed[-1])
        s2 = str(gt).replace("$", " ")
        nums = re.findall(r"([+-]?\d+(?:\.\d+)?)", s2)
        if nums:
            return nums[-1]
        return normalize_text(gt)

    def answers_match(gt: str, cand: str) -> bool:
        if gt is None:
            return False
        gt_ex = extract_answer_from_groundtruth(gt)
        gt_n = normalize_text(gt_ex if gt_ex is not None else gt)
        cand_n = normalize_text(cand)
        if not gt_n or not cand_n:
            return False
        gt_num = try_parse_number(gt_n)
        cand_num = try_parse_number(cand_n)
        if gt_num is not None and cand_num is not None:
            if math.isclose(gt_num, cand_num, rel_tol=1e-6, abs_tol=1e-6):
                return True
            if (
                abs(round(gt_num) - gt_num) < 1e-9
                and abs(round(cand_num) - cand_num) < 1e-9
            ):
                return int(round(gt_num)) == int(round(cand_num))
            return False
        return gt_n == cand_n

SIGNAL_FIELDS = [
    "majority_vote",
    "prm_min_max",
    "prm_min_vote",
    "prm_last_max",
    "prm_last_vote",
    "prm_avg_max",
    "prm_avg_vote",
]

RECORD_PATTERN = re.compile(r"record_(\d+)\.jsonl$")
RUN_META_PATTERN = re.compile(
    r"(?P<prefix>.+?)_QP(?P<qp>\d+)_CP(?P<cp>\d+)_BS(?P<bs>\d+)",
    re.IGNORECASE,
)
QUESTION_PATTERN = re.compile(r"question_(\d+)")
MODEL_SIZE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*([BbMm])", re.IGNORECASE)


def parse_model_size(prefix: str) -> float:
    if not prefix:
        return 0.0
    match = MODEL_SIZE_PATTERN.search(prefix)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "b":
        return value
    if unit == "m":
        return value / 1000.0
    return value


def parse_question_index(name: str, fallback: int) -> int:
    match = QUESTION_PATTERN.search(name)
    if match:
        return int(match.group(1))
    digits = re.findall(r"\d+", name)
    if digits:
        return int(digits[0])
    return fallback


def resolve_source_paths(spec: str) -> List[Path]:
    parts = [chunk.strip() for chunk in spec.split(",") if chunk.strip()]
    if not parts:
        return []
    resolved: List[Path] = []
    for part in parts:
        matches = glob(part)
        if matches:
            resolved.extend(Path(m) for m in matches)
        else:
            resolved.append(Path(part))
    unique_paths = []
    seen = set()
    for path in resolved:
        try:
            real = path.resolve()
        except Exception:
            real = path
        if real in seen:
            continue
        seen.add(real)
        unique_paths.append(real)
    return unique_paths


def find_question_run_dirs(base_path: Path) -> List[Path]:
    if not base_path.exists() or not base_path.is_dir():
        return []
    try:
        children = list(base_path.iterdir())
    except Exception:
        children = []
    has_questions = any(
        child.is_dir() and child.name.startswith("question_") for child in children
    )
    if has_questions:
        return [base_path]

    parents = set()
    try:
        for qdir in base_path.glob("**/question_*"):
            if qdir.is_dir():
                parents.add(qdir.parent.resolve())
    except Exception:
        return []
    return sorted(parents)


def load_last_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        if not lines:
            return None
        return json.loads(lines[-1])
    except Exception:
        return None


def collect_question_records(run_dir: Path) -> Dict[str, List[Path]]:
    def question_sort_key(path: Path) -> Tuple[int, str]:
        suffix = path.name.split("_")[-1]
        if suffix.isdigit():
            return (int(suffix), path.name)
        return (10**9, path.name)

    question_dirs = sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("question_")],
        key=question_sort_key,
    )
    records: Dict[str, List[Path]] = {}
    for qdir in question_dirs:
        recs = []
        for rec in qdir.iterdir():
            if not rec.is_file():
                continue
            match = RECORD_PATTERN.match(rec.name)
            if match:
                recs.append(rec)
        if recs:
            recs.sort(key=lambda p: int(RECORD_PATTERN.match(p.name).group(1)))
            records[str(qdir)] = recs
    return records


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_last_numeric(data: Any) -> Optional[float]:
    if isinstance(data, (list, tuple)):
        for item in reversed(data):
            val = _coerce_float(item)
            if val is not None:
                return val
        return None
    return _coerce_float(data)


def _sigmoid(value: float) -> float:
    if math.isnan(value):
        return 0.0
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def extract_candidate_tokens(out: Dict[str, Any]) -> int:
    for key in (
        "completion_tokens",
        "total_completion_tokens",
        "total_tokens",
        "token_count",
    ):
        value = out.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue

    token_history = out.get("token_history")
    if isinstance(token_history, Iterable) and not isinstance(token_history, (str, bytes)):
        total = 0
        for item in token_history:
            try:
                total += int(item)
            except (TypeError, ValueError):
                continue
        if total:
            return total
    return 0


def extract_best_reward_and_tokens(record: dict) -> Tuple[Optional[float], Optional[int]]:
    outputs = record.get("output")
    if not isinstance(outputs, Iterable) or isinstance(outputs, (str, bytes)):
        return None, None

    best_reward: Optional[float] = None
    best_tokens: Optional[int] = None
    for out in outputs:
        if not isinstance(out, dict):
            continue

        candidate: Optional[float] = None
        for key in ("reward_history", "value", "values"):
            candidate = _maybe_last_numeric(out.get(key))
            if candidate is not None:
                break
        if candidate is None:
            for key in ("reward", "score"):
                candidate = _maybe_last_numeric(out.get(key))
                if candidate is not None:
                    break
        if candidate is None:
            continue

        candidate_sigmoid = _sigmoid(candidate)
        if best_reward is None or candidate_sigmoid > best_reward:
            best_reward = candidate_sigmoid
            best_tokens = extract_candidate_tokens(out)

    return best_reward, best_tokens


def extract_best_reward(record: dict) -> Optional[float]:
    best_reward, _ = extract_best_reward_and_tokens(record)
    return best_reward


def extract_total_tokens(record: dict) -> int:
    res = record.get("result") or record.get("results") or {}
    if isinstance(res, dict):
        for key in (
            "total_completion_tokens",
            "total_tokens",
            "completion_tokens",
            "completion_token_count",
        ):
            value = res.get(key)
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                continue

    outputs = record.get("output")
    token_sum = 0
    if isinstance(outputs, Iterable) and not isinstance(outputs, (str, bytes)):
        for out in outputs:
            if not isinstance(out, dict):
                continue
            for key in (
                "total_completion_tokens",
                "total_tokens",
                "completion_tokens",
                "token_count",
            ):
                value = out.get(key)
                try:
                    if value is not None:
                        token_sum += int(value)
                        continue
                except (TypeError, ValueError):
                    continue
            token_history = out.get("token_history")
            if isinstance(token_history, Iterable) and not isinstance(token_history, (str, bytes)):
                for item in token_history:
                    try:
                        token_sum += int(item)
                    except (TypeError, ValueError):
                        continue
    return token_sum


def evaluate_question(record_paths: Sequence[Path]) -> Tuple[bool, int, List[float], int, int]:
    question_matched = False
    question_cnt_ones = 0
    record_rewards: List[float] = []
    total_tokens = 0
    best_answer_reward: Optional[float] = None
    best_answer_tokens = 0
    for path in record_paths:
        record = load_last_json(path)
        if not record:
            continue
        total_tokens += extract_total_tokens(record)
        best_reward, candidate_tokens = extract_best_reward_and_tokens(record)
        if best_reward is not None:
            record_rewards.append(best_reward)
            if best_answer_reward is None or best_reward > best_answer_reward:
                best_answer_reward = best_reward
                if candidate_tokens is not None:
                    best_answer_tokens = candidate_tokens
        res = record.get("result") or record.get("results") or {}
        if isinstance(res, dict):
            for field in SIGNAL_FIELDS:
                value = res.get(field)
                try:
                    if int(value) == 1:
                        question_cnt_ones += 1
                        continue
                except Exception:
                    if str(value) == "1":
                        question_cnt_ones += 1
        outputs = record.get("output") or []
        gt = record.get("groundtruth") or record.get("solution") or record.get("ground_truth")
        for out in outputs if isinstance(outputs, Iterable) else []:
            cand: Optional[str] = None
            if isinstance(out, dict):
                cand = (
                    out.get("extracted_answer")
                    or out.get("answer")
                    or out.get("text")
                )
                ea = out.get("extracted_answers") or out.get("gen_answers") or out.get("generated_answers")
                if not cand and isinstance(ea, list) and ea:
                    for item in reversed(ea):
                        if item:
                            cand = item
                            break
                if cand and try_parse_number(str(cand)) is None and isinstance(ea, list) and ea:
                    for item in reversed(ea):
                        if not item:
                            continue
                        nums = re.findall(r"([+-]?\d+(?:\.\d+)?)", str(item))
                        if nums:
                            cand = nums[-1]
                            break
            elif isinstance(out, str):
                cand = out
            if cand and answers_match(gt, cand):
                question_matched = True
                break
        if question_matched:
            break
    return question_matched, question_cnt_ones, record_rewards, total_tokens, best_answer_tokens


def get_question_length(qdir: str, token_lens: Dict[str, int]) -> Optional[int]:
    path_obj = Path(qdir)
    q_name = path_obj.name
    if q_name in token_lens:
        return token_lens[q_name]
    
    # Improved Heuristic
    # 1. Try to find a key that ends with "_{q_name}"
    # 2. Extract prefix (e.g., "aime")
    # 3. Check if prefix is in the FULL path (ignoring case)
    
    path_str_lower = str(path_obj.resolve()).lower()
    
    for key, val in token_lens.items():
        # Check if key ends with _{q_name} (e.g. aime_question_1 ends with _question_1)
        suffix = f"_{q_name}"
        if key.endswith(suffix):
            prefix = key[:-len(suffix)] # e.g. "aime"
            # Remove trailing underscores from prefix if any, though key construction suggests standard
            clean_prefix = prefix.rstrip('_')
            
            if not clean_prefix:
                continue
                
            # Check if clean_prefix is in path
            if clean_prefix.lower() in path_str_lower:
                return val
                
    return None


def score_sample(
    question_records: Dict[str, List[Path]],
    sample_size: int,
    rng: random.Random,
    question_limit: Optional[int] = None,
    eflops_calc: Optional[ExperimentCostCalculator] = None,
    question_token_lens: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    total_questions = 0
    correct_questions = 0
    n_q_with_signal1 = 0
    total_num_signal1 = 0
    total_reward = 0.0
    reward_questions = 0
    total_tokens_across_questions = 0
    total_answer_tokens = 0

    total_prefill_eflops = 0.0
    total_decoding_eflops = 0.0
    total_verification_eflops = 0.0
    total_eflops = 0.0
    total_memory = 0.0

    items = sorted(question_records.items(), key=lambda kv: kv[0])
    for q_idx, (qdir, recs) in enumerate(items):
        if question_limit is not None and total_questions >= question_limit:
            break
        if len(recs) < sample_size:
            raise ValueError(
                f"Question {qdir} only has {len(recs)} records, cannot sample {sample_size}."
            )
        chosen = rng.sample(recs, sample_size)
        matched, cnt_ones, record_rewards, token_count, answer_tokens = evaluate_question(chosen)
        if record_rewards:
            question_reward = max(record_rewards)
            total_reward += question_reward
            reward_questions += 1
        total_tokens_across_questions += token_count
        total_answer_tokens += answer_tokens

        if eflops_calc:
            beam_files = []
            for p in chosen:
                beam_p = p.parent / p.name.replace(".jsonl", "_beam.json")
                if beam_p.exists():
                    beam_files.append(str(beam_p))
            if beam_files:
                try:
                    # Get question token length if available
                    q_len = None
                    if question_token_lens:
                        q_len = get_question_length(qdir, question_token_lens)
                        
                    cost = eflops_calc.calculate_from_question_files(beam_files, question_token_len=q_len)
                    if cost:
                        total_prefill_eflops += cost["prefill"]
                        total_decoding_eflops += cost["decoding"]
                        total_verification_eflops += cost["verification"]
                        total_eflops += cost["total_eflops"]
                        total_memory += cost.get("total_memory", 0.0)
                except Exception:
                    pass

        total_questions += 1
        if cnt_ones > 0:
            n_q_with_signal1 += 1
            total_num_signal1 += cnt_ones
        if matched:
            correct_questions += 1

    accuracy = (correct_questions / total_questions) if total_questions else 0.0
    avg_reward = (total_reward / reward_questions) if reward_questions else 0.0

    result: Dict[str, float] = {
        "n_questions": total_questions,
        "n_correct": correct_questions,
        "accuracy": accuracy,
        "n_q_with_signal1": n_q_with_signal1,
        "total_num_signal1": total_num_signal1,
        "avg_reward": avg_reward,
        "answer_tokens": total_answer_tokens,
        "total_tokens": total_tokens_across_questions,
    }

    if eflops_calc and total_questions > 0:
        result.update(
            {
                "avg_prefill_eflops": total_prefill_eflops / total_questions,
                "avg_decoding_eflops": total_decoding_eflops / total_questions,
                "avg_verification_eflops": total_verification_eflops / total_questions,
                "avg_total_eflops": total_eflops / total_questions,
                "avg_memory": total_memory / total_questions,
            }
        )

    return result


def summarize_trials(trial_rows: List[Dict[str, float]]) -> Dict[str, float]:
    accuracies = [row["accuracy"] for row in trial_rows]
    mean_acc = statistics.fmean(accuracies) if accuracies else 0.0
    stdev_acc = statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0
    return {
        "mean_accuracy": mean_acc,
        "stdev_accuracy": stdev_acc,
        "min_accuracy": min(accuracies) if accuracies else 0.0,
        "max_accuracy": max(accuracies) if accuracies else 0.0,
    }


def parse_sample_sizes(raw: str) -> List[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        val = int(chunk)
        if val <= 0:
            raise ValueError("Sample sizes must be positive integers.")
        values.append(val)
    if not values:
        raise ValueError("Provide at least one sample size.")
    return values


def derive_sample_sizes(
    qp: Optional[int],
    requested: Sequence[int],
    min_available: int,
) -> List[int]:
    if min_available <= 0:
        return []
    limit = min_available
    if qp is not None:
        limit = min(limit, qp)
    allowed = {size for size in requested if size <= limit}
    start = qp if qp is not None else min_available
    current = max(1, start)
    seen = set()
    while current >= 1 and current not in seen:
        seen.add(current)
        if current <= limit:
            allowed.add(current)
        if current == 1:
            break
        next_val = current // 2
        if next_val < 1:
            next_val = 1
        current = next_val
    if not allowed and limit >= 1:
        allowed.add(limit)
    return sorted(allowed, reverse=True)


def extract_run_metadata(run_dir: Path) -> Dict[str, Optional[int]]:
    current = run_dir
    visited = set()
    while True:
        name = current.name
        match = RUN_META_PATTERN.search(name)
        if match:
            prefix = match.group("prefix")
            return {
                "group_name": name,
                "model_prefix": prefix,
                "qp": int(match.group("qp")),
                "cp": int(match.group("cp")),
                "bs": int(match.group("bs")),
            }
        if current.parent == current or current in visited:
            break
        visited.add(current)
        current = current.parent
    return {
        "group_name": run_dir.name,
        "model_prefix": run_dir.name,
        "qp": None,
        "cp": None,
        "bs": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        required=True,
        help="Directory (or glob) containing run folders with question_* subdirectories.",
    )
    parser.add_argument(
        "--sample-sizes",
        required=True,
        help=(
            "Comma-separated list of target sample sizes (e.g. 64,32,16). "
            "Per run we derive downward-compatible sizes based on its QP and available records."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of random trials for each (question, model, sample size) combination (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="Optional cap on number of questions processed (per run).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional CSV file to save per-trial statistics.",
    )
    parser.add_argument(
        "--save-summary",
        type=str,
        default=None,
        help="CSV file to save aggregated per-question results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-trial details.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen3-32b",
        help="Model name key for eFLOPs parameters (policy model).",
    )
    parser.add_argument(
        "--verifier-name",
        type=str,
        default="skywork-o1-prm-1.5b",
        help="Verifier name key for eFLOPs parameters.",
    )
    args = parser.parse_args()

    sample_sizes = parse_sample_sizes(args.sample_sizes)
    source_paths = resolve_source_paths(args.source_run)
    if not source_paths:
        parser.error(f"Source path specification '{args.source_run}' resolved to nothing")

    run_dirs: List[Path] = []
    missing: List[str] = []
    for candidate in source_paths:
        found = find_question_run_dirs(candidate)
        if not found:
            missing.append(str(candidate))
            continue
        run_dirs.extend(found)
    if missing:
        print(
            "Warning: skipped paths without question_* directories -> " + ", ".join(missing),
            file=sys.stderr,
        )
    if not run_dirs:
        detail = "; ".join(missing) if missing else args.source_run
        parser.error(f"No question_* directories found under: {detail}")

    unique_dirs: List[Path] = []
    seen_dirs = set()
    for rd in run_dirs:
        if rd in seen_dirs:
            continue
        seen_dirs.add(rd)
        unique_dirs.append(rd)
    run_dirs = unique_dirs

    run_infos: List[Dict[str, Any]] = []
    skipped_runs: List[str] = []
    for rd in run_dirs:
        collected = collect_question_records(rd)
        if not collected:
            skipped_runs.append(f"{rd} (no question records)")
            continue
        meta = extract_run_metadata(rd)
        model_prefix = meta.get("model_prefix", rd.name)
        model_size = parse_model_size(model_prefix)
        min_available = min(len(recs) for recs in collected.values())
        derived_sizes = derive_sample_sizes(meta.get("qp"), sample_sizes, min_available)
        if not derived_sizes:
            skipped_runs.append(f"{rd} (no compatible sample sizes)")
            continue
        run_infos.append(
            {
                "path": rd,
                "records": collected,
                "meta": meta,
                "sample_sizes": derived_sizes,
                "model_prefix": model_prefix,
                "model_size": model_size,
            }
        )

    if not run_infos:
        detail = "; ".join(skipped_runs) if skipped_runs else "no valid runs"
        parser.error(
            f"No record_*.jsonl files found under resolved run directories ({detail})"
        )

    if skipped_runs:
        print(
            "Warning: skipped runs -> " + ", ".join(skipped_runs),
            file=sys.stderr,
        )

    # Load question token lengths if available
    question_token_lens = {}
    try:
        q_len_path = Path(__file__).with_suffix('').parent.parent / 'envs/MATH/question_token_len.json'
        if q_len_path.exists():
            with open(q_len_path, 'r') as f:
                question_token_lens = json.load(f)
            if args.verbose:
                print(f"Loaded {len(question_token_lens)} question token lengths from {q_len_path}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to load question token lengths: {e}", file=sys.stderr)

    eflops_calc = None
    try:
        gen_config = get_model_config(args.model_name)
        ver_config = get_model_config(args.verifier_name)
        eflops_calc = ExperimentCostCalculator(gen_config, ver_config)
    except ValueError as e:
        print(f"Warning: eFLOPs calculator init failed (check model names): {e}", file=sys.stderr)

    rows_for_csv: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for run_idx, info in enumerate(run_infos):
        meta = info["meta"]
        model_prefix = info["model_prefix"]
        
        # Dynamically determine model params based on model_prefix
        # Try to find a matching key in MODEL_CONFIGS
        detected_model_name = args.model_name
        prefix_lower = model_prefix.lower()
        # Sort keys by length descending to match most specific first
        for key in sorted(MODEL_CONFIGS.keys(), key=len, reverse=True):
            if key in prefix_lower:
                detected_model_name = key
                break
        
        current_eflops_calc = eflops_calc
        # If detected model differs from default, re-init calculator
        if detected_model_name != args.model_name:
             try:
                gen_config = get_model_config(detected_model_name)
                # Assuming verifier stays same or needs similar logic? 
                # For now keeping verifier fixed as per args
                ver_config = get_model_config(args.verifier_name)
                current_eflops_calc = ExperimentCostCalculator(gen_config, ver_config)
             except ValueError:
                # Fallback to default calculator if specific model config fails
                pass

        qp_value = meta.get("qp")
        cp_value = meta.get("cp")
        bs_value = meta.get("bs")
        sample_sizes_for_run = info["sample_sizes"]
        records = info["records"]
        sorted_questions = sorted(records.items(), key=lambda kv: kv[0])

        if args.question_limit is not None:
            sorted_questions = sorted_questions[: args.question_limit]

        for question_index, (qdir, recs) in enumerate(sorted_questions):
            question_name = Path(qdir).name
            question_idx = parse_question_index(question_name, question_index)
            question_records = {qdir: recs}
            for sample_size in sample_sizes_for_run:
                if len(recs) < sample_size:
                    # Skip this combination if insufficient records for the question.
                    continue
                stats_list: List[Dict[str, float]] = []
                for trial_idx in range(args.trials):
                    trial_seed = (
                        args.seed
                        + sample_size * 1000
                        + trial_idx
                        + question_idx * 100000
                        + run_idx * 10_000_000
                    )
                    rng = random.Random(trial_seed)
                    stats = score_sample(
                        question_records,
                        sample_size,
                        rng,
                        question_limit=1,
                        eflops_calc=current_eflops_calc,
                        question_token_lens=question_token_lens,
                    )
                    stats.update(
                        {
                            "question": question_name,
                            "sample_size": sample_size,
                            "trial": trial_idx,
                            "seed": trial_seed,
                            "run_path": str(info["path"]),
                            "model_prefix": model_prefix,
                            "qp": qp_value,
                            "cp": cp_value,
                            "bs": bs_value,
                        }
                    )
                    stats_list.append(stats)
                    rows_for_csv.append(stats)
                    if args.verbose:
                        print(
                            f"[{question_name}][{model_prefix}] sample={sample_size} trial={trial_idx} "
                            f"accuracy={stats['accuracy']:.6f} reward={stats.get('avg_reward', 0.0):.4f} "
                            f"tokens={stats.get('total_tokens', 0.0):.2f} "
                            f"eflops={stats.get('avg_total_eflops', 0.0):.2e}",
                            file=sys.stderr,
                        )
                if not stats_list:
                    continue
                summary = summarize_trials(stats_list)
                mean_reward = statistics.fmean(s.get("avg_reward", 0.0) for s in stats_list)
                mean_answer_tokens = statistics.fmean(s.get("answer_tokens", 0.0) for s in stats_list)
                mean_total_tokens = statistics.fmean(s.get("total_tokens", 0.0) for s in stats_list)
                mean_avg_eflops = statistics.fmean(
                    s.get("avg_total_eflops", 0.0) for s in stats_list
                )
                mean_avg_memory = statistics.fmean(s.get("avg_memory", 0.0) for s in stats_list)
                question_count = stats_list[0].get("n_questions", 0)
                summary_rows.append(
                    {
                        "序号": 0,
                        "question{idx}_model_sample_size": f"{question_name}_{model_prefix}_S{sample_size}",
                        "qp": qp_value,
                        "cp": cp_value if cp_value is not None else "",
                        "bs": bs_value if bs_value is not None else "",
                        "n_trials": len(stats_list),
                        "n_question": question_count,
                        "准确率": summary["mean_accuracy"],
                        "平均reward": mean_reward,
                        "answer_tokens": mean_answer_tokens,
                        "total_tokens": mean_total_tokens,
                        "mean_eflops": mean_avg_eflops,
                        "mean_memory": mean_avg_memory,
                        "model_size": info.get("model_size", 0.0),
                        "_sample_size": sample_size,
                        "_question_idx": question_idx,
                    }
                )

    if not summary_rows:
        parser.error("No summary rows produced. Check sample sizes and source data.")

    summary_rows.sort(
        key=lambda row: (
            row.get("_question_idx", 10**9),
            -float(row.get("model_size", 0.0)),
            -float(row.get("qp") or 0),
            -int(row.get("_sample_size", 0)),
        )
    )
    for idx, row in enumerate(summary_rows, start=1):
        row["序号"] = idx

    summary_fieldnames = [
        "序号",
        "question{idx}_model_sample_size",
        "qp",
        "cp",
        "bs",
        "n_trials",
        "n_question",
        "准确率",
        "平均reward",
        "answer_tokens",
        "total_tokens",
        "mean_eflops",
        "mean_memory",
    ]

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=summary_fieldnames)
            writer.writeheader()
            for idx, row in enumerate(rows_for_csv, start=1):
                output_row = {
                    "序号": idx,
                    "question{idx}_model_sample_size": f"{row.get('question', '')}_{row.get('model_prefix', '')}_S{row.get('sample_size', '')}",
                    "qp": row.get("qp", ""),
                    "cp": row.get("cp", ""),
                    "bs": row.get("bs", ""),
                    "n_trials": 1,
                    "n_question": row.get("n_questions", ""),
                    "准确率": row.get("accuracy", ""),
                    "平均reward": row.get("avg_reward", ""),
                    "answer_tokens": row.get("answer_tokens", ""),
                    "total_tokens": row.get("total_tokens", ""),
                    "mean_eflops": row.get("avg_total_eflops", ""),
                    "mean_memory": row.get("avg_memory", ""),
                }
                cleaned = {k: ("" if v is None else v) for k, v in output_row.items()}
                writer.writerow(cleaned)
        print(f"Wrote per-trial stats to {out_path}")

    if args.save_summary:
        summary_path = Path(args.save_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=summary_fieldnames)
            writer.writeheader()
            for row in summary_rows:
                cleaned = {}
                for key in summary_fieldnames:
                    value = row.get(key, "")
                    cleaned[key] = "" if value is None else value
                writer.writerow(cleaned)
        print(f"Wrote summary stats to {summary_path}")


if __name__ == "__main__":
    main()
