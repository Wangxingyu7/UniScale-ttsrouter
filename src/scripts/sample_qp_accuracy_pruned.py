#!/usr/bin/env python3
"""Sample subsets of record files to emulate smaller QP settings and score accuracy with early stopping.

This script implements a pruning mechanism based on expected rewards to improve average batch size.

Quick run: 

python3 scripts/sample_qp_accuracy_pruned.py \
   --source-run "output_aime/aime_Qwen3*" \
   --sample-sizes 64,32,16,8,4,2,1 \
    --trials 3  \
    --seed 42 \
   --save pruned_trials_aime.csv \
   --save-summary pruned_summary_aime.csv

"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
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
        s = s.lstrip('(')
        s = s.lstrip('"')
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
MODEL_SIZE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*([BbMm])", re.IGNORECASE)


def parse_model_size(prefix: str) -> float:
    """Extract model parameter size in billions from the prefix."""
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
    
    # Pruned paths should not be evaluated for correctness, but we need their rewards if available?
    # Actually, if a path is pruned, it doesn't contribute to the final answer in a "best of N" sense usually,
    # unless we just take the best reward found SO FAR.
    # The logic here assumes record_paths contains ONLY the paths that were "completed" or "selected" 
    # BEFORE pruning stopped them? 
    # Wait, the user wants to simulate the pruning process.
    # So we need to simulate the step-by-step generation and prune 'live'.
    # But we only have the final records.
    # We can use the corresponding _beam.json files to reconstruct the step-by-step progress.
    # However, to keep it simple and consistent with the existing structure,
    # we can process the "chosen" records.
    # But wait, pruning decides WHICH records get to finish.
    # If we just take the final records, we are assuming they all finished.
    # We need to filter `record_paths` based on the pruning logic.
    
    # This function `evaluate_question` evaluates the final set of paths.
    # We should first prune `record_paths` (or rather, simulate which ones survive)
    # and then pass the survivors to this function.
    # Or, we modify this function to handle pruning.
    
    # Since we need step-by-step info for pruning, let's defer pruning logic to `score_sample_with_pruning`.
    # This function will just evaluate whatever list is passed to it.
    
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


def load_beam_data(path: Path) -> Optional[dict]:
    """Load the corresponding _beam.json file for a record."""
    beam_path = path.parent / path.name.replace(".jsonl", "_beam.json")
    if not beam_path.exists():
        return None
    try:
        with beam_path.open("r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return None
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try reading last line if it's jsonl format
                lines = content.split('\n')
                valid_lines = [l for l in lines if l.strip()]
                if not valid_lines:
                    return None
                return json.loads(valid_lines[-1])
    except Exception:
        return None

def simulate_pruning(
    record_paths: List[Path], 
    eflops_calc: Optional[ExperimentCostCalculator]
) -> Tuple[List[Path], Dict[str, Any]]:
    """
    Simulate the generation process with pruning.
    
    Returns:
        - List of paths that survived (or were completed before pruning).
        - eFLOPs cost dictionary (if calculator provided).
    """
    
    # 1. Load beam data for all paths to reconstruct steps
    # We need to align steps across all paths.
    # Structure: paths_data = [ { 'path': p, 'nodes': [...], 'steps': { step_idx: node } } ]
    
    active_paths = []
    
    for p in record_paths:
        beam_data = load_beam_data(p)
        if not beam_data or 'nodes' not in beam_data:
            # If no beam data, we assume it's a single step or we can't prune it step-by-step properly
            # so we treat it as "always active until end"? 
            # Or skip? Let's assume valid data for simulation.
            continue
            
        nodes = beam_data['nodes']
        # Organize nodes by depth/step
        # Assume root is depth 0.
        # We need to trace the path from root to leaf.
        # But wait, beam file might contain a tree.
        # For a single "record", is it a single path or a tree?
        # The prompt says "sample subsets of record files".
        # Usually one record file = one attempt (one path or one tree search).
        # If it's a tree search, "pruning a path" is complex.
        # Assuming one record = one independent "path" (or trial) in the context of "sample_size".
        # If the record contains a tree, we treat the whole tree as "one sample unit".
        # BUT the user says: "When A PATH is completed...".
        # This implies we are managing multiple PATHS (samples).
        # So we assume each record corresponds to one main path (or we just look at the best path in it).
        # Let's extract the longest/best path from the nodes to simulate "Step-by-step" generation.
        
        # Find leaf nodes
        parent_map = {n['node_id']: n.get('parent_id') for n in nodes}
        children_map = {}
        for n in nodes:
            pid = n.get('parent_id')
            if pid not in children_map: children_map[pid] = []
            children_map[pid].append(n['node_id'])
            
        leaves = [n for n in nodes if n['node_id'] not in children_map]
        
        # We assume the record represents ONE solution path. 
        # If there are multiple leaves, we pick the one with highest reward or just the last one?
        # Usually MCTS/Beam search records have many nodes.
        # But `sample_qp_accuracy` treats each FILE as a sample unit.
        # Let's assume we track the "main" path of this file.
        # For simplicity, let's take the leaf with the highest reward (or just the first leaf if no reward).
        
        best_leaf = None
        best_leaf_reward = -float('inf')
        
        for leaf in leaves:
            # Find reward for this leaf
            # In some formats, reward is in the node.
            # In others, it's in the 'output' field of the record jsonl.
            # Let's look at the node 'value' or 'reward'
            r = leaf.get('value') or leaf.get('reward') or 0.0
            # If r is list, take last
            if isinstance(r, list): r = r[-1] if r else 0.0
            try:
                r = float(r)
            except:
                r = 0.0
            r = _sigmoid(r)
            
            if r > best_leaf_reward:
                best_leaf_reward = r
                best_leaf = leaf
        
        if not best_leaf:
            continue

        # Reconstruct path from root to best_leaf
        path_nodes = []
        curr = best_leaf
        while curr:
            path_nodes.append(curr)
            pid = curr.get('parent_id')
            if pid is None:
                break
            # Find parent node
            parent = next((n for n in nodes if n['node_id'] == pid), None)
            curr = parent
        path_nodes.reverse() # Root -> Leaf
        
        # Extract reward at each step (if available) or assume final reward is accumulated?
        # User says: "current total reward + future total reward".
        # This implies reward is additive or we have a value function?
        # "1*(Hbest*1.2 - j)" suggests each step gives +1 reward? 
        # "current total reward" -> likely the accumulated reward so far.
        # In many math/reasoning tasks, reward is only at the end (0 or 1).
        # But the user formula `1*(Hbest*1.2 - j)` implies a dense reward or a heuristic.
        # `Hbest` is "number of steps of best path".
        # `Vbest` is "best path value".
        # `j` is current step.
        # Formula: `Predicted = Current_Reward + (H_best * 1.2 - j) * 1`
        # This assumes each future step gives reward 1? 
        # Or maybe "Current Total Reward" is just the value so far?
        # If it's a sparse reward task (0/1), "Current Total Reward" might be 0 until end.
        # Let's assume standard PRM or Value at each step if available.
        # If not available, we assume 0?
        # BUT, if the user defines "Future Total Reward" as `(Hbest*1.2 - j)`, 
        # it looks like they are assuming a "Length Penalty" or "Time Cost" or "Step Reward = 1".
        # Let's stick to the user's formula strictly.
        # We need `Current_Reward` for the path at step j.
        # We will check `node['value']` or `node['reward']` for this.
        
        active_paths.append({
            'path_obj': p,
            'nodes': path_nodes,
            'length': len(path_nodes), # Steps = length - 1 (excluding root?) or just length? 
                                       # Root is prompt. Step 1 is first thought.
                                       # Let's say Step j corresponds to index j in path_nodes (0-based).
            'completed': False,
            'survived': True,
            'final_reward': best_leaf_reward
        })

    # 2. Simulate step-by-step
    # We proceed time step j = 0, 1, 2, ...
    # At each step, we:
    #   a. Identify paths that JUST finished at this step.
    #   b. Update Global Best (Hbest, Vbest).
    #   c. Prune incomplete paths.
    
    completed_paths = [] # List of {H, V}
    Hbest = None
    Vbest = -float('inf')
    
    # Max steps across all paths
    max_steps = max(p['length'] for p in active_paths) if active_paths else 0
    
    # For eFLOPs calculation
    # We need to track which paths are active at each step.
    # To use `eFLOPsCalculator.calculate_from_question_files` logic, we need to pass the files.
    # But here we are dynamically pruning.
    # So we need to calculate eFLOPs manually step-by-step or construct a "virtual" set of files
    # where pruned paths are truncated.
    
    # We will record the "effective length" of each path for eFLOPs.
    # Initially all effective lengths = actual length.
    # If pruned at step j, effective length = j.
    
    for j in range(max_steps):
        # 1. Identify newly completed paths
        # A path is completed at step j if its length is j+1 (assuming 0-indexed nodes, last node is at index length-1)
        # So if we are processing step j (meaning we just generated node at index j),
        # paths with length == j+1 are done.
        
        newly_completed = []
        for p in active_paths:
            if not p['completed'] and p['length'] == j + 1:
                p['completed'] = True
                newly_completed.append(p)
        
        # 2. Update Best
        if newly_completed:
            for p in newly_completed:
                # V is final reward
                v = p['final_reward']
                h = p['length'] # Steps
                
                # Logic for "Best Path": Max Reward?
                # User says "calculate optimal path... corresponding Hbest and Vbest".
                # Usually "Optimal" means highest V.
                if v > Vbest:
                    Vbest = v
                    Hbest = h
                elif v == Vbest:
                    # If rewards equal, prefer shorter? Or longer?
                    # Let's assume shorter is better for efficiency, but user didn't specify.
                    # We'll just keep the first one or update if strictly greater.
                    # Let's stick to > Vbest.
                    pass
            
        # 3. Prune
        # "对当前仍未完成的路径..."
        if Hbest is not None:
            for p in active_paths:
                if not p['completed'] and p['survived']:
                    # Current step is j. We just finished generating node j.
                    # Current accumulated reward?
                    # We need the reward of node at index j.
                    curr_node = p['nodes'][j]
                    # Try to get value/reward
                    # Some datasets use 'value', some 'reward'.
                    # User said "current total reward".
                    # If the nodes store cumulative reward (value), we use it directly.
                    # If they store instantaneous reward, we might need to sum?
                    # Given it's likely a PRM value (0-1) or similar, it's usually "expected future success" or similar.
                    # Let's assume the node value IS the "Current Total Reward" state.
                    curr_val = curr_node.get('value') or curr_node.get('reward') or 0.0
                    if isinstance(curr_val, list): curr_val = curr_val[-1] if curr_val else 0.0
                    try:
                        curr_val = float(curr_val)
                    except:
                        curr_val = 0.0
                    curr_val = _sigmoid(curr_val)
                        
                    # Formula: Expected = Current + (Hbest * 1.2 - j)
                    # Note: j in user prompt likely refers to "current step count" (1-based?).
                    # If we are at index j (0-based), we have taken j+1 steps?
                    # "Suppose this is step j... 1*(... - j)"
                    # Let's assume j is 1-based step count.
                    # Current node index is j. So we have taken j+1 steps?
                    # Let's use `step_count = j + 1`.
                    
                    step_count = j + 1
                    
                    # Wait, "Hbest * 1.2 - j" -> if j is large, this term is small/negative.
                    # If j > Hbest * 1.2, penalty is negative?
                    # Yes, logic: if you are taking too long, expected gain decreases.
                    
                    term = math.ceil(Hbest * 1.2) - step_count
                    expected_max_reward = curr_val + term
                    
                    if expected_max_reward < Vbest:
                        # Prune
                        p['survived'] = False
                        p['pruned_at'] = step_count # Effectively stopped after this step
                        # print(f"Pruned path at step {step_count}. Exp: {expected_max_reward:.2f} < Best: {Vbest:.2f}")

    # Collect survivors
    survived_paths = [p['path_obj'] for p in active_paths if p['survived']]
    
    # Calculate eFLOPs
    # We need to construct a "virtual" set of beam files where pruned paths are truncated.
    # Since we can't easily modify files on disk efficiently, we can use the eFLOPs calculator's
    # internal logic but adapted.
    # OR, we can just calculate the cost for each path based on its "effective length" and
    # sum them up, BUT we need to account for Batch Size.
    # The eFLOPs calculator `calculate_from_question_files` assumes all files run in parallel
    # and calculates layer-wise costs.
    # We can pass the `active_paths` with their `nodes` truncated to `eFLOPsCalculator`.
    # But `eFLOPsCalculator` expects file paths.
    
    # We will modify `eFLOPsCalculator` usage or subclass it? 
    # Better: We can reconstruct the "batch size schedule".
    # We know for each step t, how many paths are active.
    # Active count at step t = count(p for p in active_paths if p.effective_length >= t)
    
    # Determine effective length for each path
    for p in active_paths:
        if p['survived']:
            p['effective_length'] = p['length']
        else:
            p['effective_length'] = p.get('pruned_at', p['length'])

    cost_dict = None
    if eflops_calc:
        # We perform the calculation manually using the schedule
        # 1. Prefill
        # All paths share prompt? Assuming yes.
        # Cost = 1 * Prefill_Cost (Broadcasting)
        # We need prompt length. Take from first path's root.
        prompt_len = 0
        if active_paths:
             root = active_paths[0]['nodes'][0]
             state_before = root.get('state_before', "")
             prompt_len = len(state_before.split()) if state_before else 0
        
        c_prefill_cost, c_prefill_mem = eflops_calc.gen_model.calculate_prefill(prompt_len)
        
        # 2. Decoding
        c_dec_total_cost = 0.0
        c_dec_total_mem = 0.0
        
        # 3. Verification
        c_ver_total_cost = 0.0
        c_ver_total_mem = 0.0
        
        # Max effective length
        max_eff_len = max((p['effective_length'] for p in active_paths), default=0)
        
        # We assume step 1 (first generated token/thought) corresponds to index 1 in nodes list?
        # Root is index 0.
        # Yes.
        
        for t in range(1, max_eff_len): # t is node index, from 1 to end
            # Find active paths at this step
            # A path is active at step t if effective_length > t
            # (i.e. it has a node at index t)
            
            current_active = [p for p in active_paths if p['effective_length'] > t]
            b_t = len(current_active)
            
            if b_t == 0:
                break
            
            step_delta_ls = []
            step_l_inits = []
            
            for p in current_active:
                node = p['nodes'][t]
                # delta_l
                if node.get('num_generated_token', 0) > 0:
                    d = node['num_generated_token']
                elif node.get('action'):
                    d = len(node['action'].split())
                elif node.get('state_after') and node.get('state_before'):
                    d = len(node['state_after'].split()) - len(node['state_before'].split())
                else:
                    d = 1
                d = max(1, d)
                
                # context_len
                # We need accumulated context length from previous steps (including prompt)
                # But here we just estimate from state_before?
                # For consistency with `compute_eflops.py` (which tracks accumulated length), 
                # we should ideally do the same.
                # `compute_eflops.py` does: l_init = parent_cum_len
                # Here `state_before` includes everything.
                sb = node.get('state_before', "")
                ctx = len(sb.split()) if sb else 0
                
                step_delta_ls.append(d)
                step_l_inits.append(ctx)
            
            if step_delta_ls:
                # Decoding Cost
                step_cost, step_mem = eflops_calc.gen_model.calculate_incremental_step(step_delta_ls, step_l_inits)
                c_dec_total_cost += step_cost
                c_dec_total_mem += step_mem
                
                # Verification Cost
                # Calculates cost for each branch's full sequence verification
                # `calculate_verification` expects list of final lengths for each branch
                l_final_list = [step_l_inits[i] + step_delta_ls[i] for i in range(len(step_delta_ls))]
                ver_cost, ver_mem = eflops_calc.ver_model.calculate_verification(l_final_list)
                c_ver_total_cost += ver_cost
                c_ver_total_mem += ver_mem
            
        total_eflops = c_prefill_cost + c_dec_total_cost + c_ver_total_cost
        total_memory = c_prefill_mem + c_dec_total_mem + c_ver_total_mem
        
        cost_dict = {
            "prefill": {"total": c_prefill_cost, "memory": c_prefill_mem},
            "decoding": {"total": c_dec_total_cost, "memory": c_dec_total_mem},
            "verification": {"total": c_ver_total_cost, "memory": c_ver_total_mem},
            "summary": {
                "total_eflops": total_eflops,
                "total_memory": total_memory,
            }
        }

    return survived_paths, cost_dict


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
    
    # eFLOPs stats accumulation
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
        
        # Apply Pruning Simulation
        survived, cost = simulate_pruning(chosen, eflops_calc)
        
        # Override prefill cost with accurate L_in if available
        if cost and question_token_lens:
            q_len = get_question_length(qdir, question_token_lens)
            if q_len is not None:
                # Recalculate prefill with correct length
                # Note: simulate_pruning already calculated a prefill cost based on state_before
                # We need to replace it.
                # But simulate_pruning returns a dict. We can just recalculate here if we have eflops_calc.
                if eflops_calc:
                    new_prefill, new_m_prefill = eflops_calc.gen_model.calculate_prefill(q_len)
                    old_prefill = cost['prefill']['total']
                    old_prefill_mem = cost['prefill'].get('memory', 0.0)

                    cost['prefill']['total'] = new_prefill
                    cost['prefill']['memory'] = new_m_prefill
                    # Update total summary aggregates
                    cost['summary']['total_eflops'] = (
                        cost['summary']['total_eflops'] - old_prefill + new_prefill
                    )
                    if 'total_memory' in cost['summary']:
                        cost['summary']['total_memory'] = (
                            cost['summary']['total_memory'] - old_prefill_mem + new_m_prefill
                        )

        # Evaluate only survived paths
        matched, cnt_ones, record_rewards, token_count, answer_tokens = evaluate_question(survived)
        if record_rewards:
            question_reward = max(record_rewards)
            total_reward += question_reward
            reward_questions += 1
        total_tokens_across_questions += token_count
        total_answer_tokens += answer_tokens
        
        # Accumulate eFLOPs
        if cost:
            total_prefill_eflops += cost['prefill']['total']
            total_decoding_eflops += cost['decoding']['total']
            total_verification_eflops += cost['verification']['total']
            total_eflops += cost['summary']['total_eflops']
            total_memory += cost['summary'].get('total_memory', 0.0)

        total_questions += 1
        if cnt_ones > 0:
            n_q_with_signal1 += 1
            total_num_signal1 += cnt_ones
        if matched:
            correct_questions += 1

    accuracy = (correct_questions / total_questions) if total_questions else 0.0
    avg_reward = (total_reward / reward_questions) if reward_questions else 0.0
    
    result = {
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
        result.update({
            "avg_prefill_eflops": total_prefill_eflops / total_questions,
            "avg_decoding_eflops": total_decoding_eflops / total_questions,
            "avg_verification_eflops": total_verification_eflops / total_questions,
            "avg_total_eflops": total_eflops / total_questions,
            "avg_memory": total_memory / total_questions if total_memory else 0.0,
        })
        
    return result


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        required=True,
        help="Directory containing question_* subdirectories from the high-QP run.",
    )
    parser.add_argument(
        "--sample-sizes",
        required=True,
        help=(
            "Comma-separated list of target sample sizes (e.g. 32,16,8). "
            "The script prunes incompatible sizes per run and automatically adds halved sizes down to 1 based on the run's QP."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of random trials to run for each sample size (default: 3).",
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
        help="Optional cap on number of questions to score (for quick checks).",
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
        help="Optional CSV file to save per-sample aggregated statistics.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-trial details in addition to summary.",
    )
    parser.add_argument("--model-name", type=str, default="qwen3-32b", help="Model name key for eFLOPs parameters")
    parser.add_argument("--verifier-name", type=str, default="skywork-o1-prm-1.5b", help="Verifier name key for eFLOPs parameters")
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

    # dedupe while preserving order
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
        effective_qp = meta.get("qp") or min_available
        run_infos.append(
            {
                "path": rd,
                "records": collected,
                "meta": meta,
                "sample_sizes": derived_sizes,
                "question_count": len(collected),
                "min_available": min_available,
                "effective_qp": effective_qp,
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

    if args.verbose:
        total_questions = sum(info["question_count"] for info in run_infos)
        print(
            f"Resolved {len(run_infos)} run dirs spanning {total_questions} questions",
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
    summary_index = 1

    for run_idx, info in enumerate(run_infos):
        meta = info["meta"]
        model_prefix = meta.get("model_prefix", info["path"].name)
        
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

        group_name = meta.get("group_name", model_prefix)
        qp_value = info["effective_qp"]
        cp_value = meta.get("cp")
        bs_value = meta.get("bs")
        for sample_size in info["sample_sizes"]:
            stats_list: List[Dict[str, Any]] = []
            for trial_idx in range(args.trials):
                trial_seed = (
                    args.seed
                    + sample_size * 1000
                    + trial_idx
                    + run_idx * 100000
                )
                rng = random.Random(trial_seed)
                stats = score_sample(
                    info["records"],
                    sample_size,
                    rng,
                    question_limit=args.question_limit,
                    eflops_calc=current_eflops_calc,
                )
                stats.update(
                    {
                        "sample_size": sample_size,
                        "trial": trial_idx,
                        "seed": trial_seed,
                        "run_path": str(info["path"]),
                        "group_name": group_name,
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
                        f"[{model_prefix}] sample={sample_size} trial={trial_idx} "
                        f"accuracy={stats['accuracy']:.6f} reward={stats.get('avg_reward', 0.0):.4f} "
                        f"eflops={stats.get('avg_total_eflops', 0.0):.2e} "
                        f"memory={stats.get('avg_memory', 0.0):.2e}",
                        file=sys.stderr,
                    )
            if not stats_list:
                continue
            summary = summarize_trials(stats_list)
            mean_reward = statistics.fmean(
                s.get("avg_reward", 0.0) for s in stats_list
            )
            mean_answer_tokens = statistics.fmean(
                s.get("answer_tokens", 0.0) for s in stats_list
            )
            mean_total_tokens = statistics.fmean(
                s.get("total_tokens", 0.0) for s in stats_list
            )
            mean_avg_eflops = statistics.fmean(
                s.get("avg_total_eflops", 0.0) for s in stats_list
            )
            mean_avg_memory = statistics.fmean(
                s.get("avg_memory", 0.0) for s in stats_list
            )
            question_count = stats_list[0].get("n_questions", 0)
            print(
                f"[{model_prefix}] sample={sample_size} mean_acc={summary['mean_accuracy']:.6f} "
                f"reward={mean_reward:.4f} answer_tokens={mean_answer_tokens:.2f} "
                f"tokens={mean_total_tokens:.2f} "
                f"mean_eflops={mean_avg_eflops:.2e} memory={mean_avg_memory:.2e}"
            )
            summary_rows.append(
                {
                    "序号": summary_index,
                    "model_sample_size": f"{model_prefix}_S{sample_size}",
                    "qp": qp_value,
                    "cp": cp_value if cp_value is not None else "",
                    "bs": bs_value if bs_value is not None else "",
                    "n_trials": len(stats_list),
                    "n_question": question_count,
                    "准确性": summary["mean_accuracy"],
                    "平均reward": mean_reward,
                    "answer_tokens": mean_answer_tokens,
                    "total_tokens": mean_total_tokens,
                    "mean_eflops": mean_avg_eflops,
                    "访存量": mean_avg_memory,
                    "model_size": info.get("model_size", 0.0),
                    "_sample_size": sample_size,
                }
            )
            summary_index += 1

    if summary_rows:
        summary_rows.sort(
            key=lambda row: (
                row.get("model_size", 0.0),
                row.get("qp") if isinstance(row.get("qp"), (int, float)) else 0,
                row.get("_sample_size", 0),
            ),
            reverse=True,
        )
        for idx, row in enumerate(summary_rows, start=1):
            row["序号"] = idx

    summary_fieldnames = [
        "序号",
        "model_sample_size",
        "qp",
        "cp",
        "bs",
        "n_trials",
        "n_question",
        "准确性",
        "平均reward",
        "answer_tokens",
        "total_tokens",
        "mean_eflops",
        "访存量",
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
                    "model_sample_size": f"{row.get('model_prefix', '')}_S{row.get('sample_size', '')}",
                    "qp": row.get("qp", ""),
                    "cp": row.get("cp", ""),
                    "bs": row.get("bs", ""),
                    "n_trials": 1,
                    "n_question": row.get("n_questions", ""),
                    "准确性": row.get("accuracy", ""),
                    "平均reward": row.get("avg_reward", ""),
                    "answer_tokens": row.get("answer_tokens", ""),
                    "total_tokens": row.get("total_tokens", ""),
                    "mean_eflops": row.get("avg_total_eflops", ""),
                    "访存量": row.get("avg_memory", ""),
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
                for k in summary_fieldnames:
                    value = row.get(k, "")
                    cleaned[k] = "" if value is None else value
                writer.writerow(cleaned)
        print(f"Wrote summary stats to {summary_path}")


if __name__ == "__main__":
    main()
