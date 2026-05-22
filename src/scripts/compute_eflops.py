import json
import os
import glob
import math
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# ---- eFLOPs 计算相关模型参数与常量 ---- 

# 结构配置：直接使用用户提供的参数
MODEL_CONFIGS = {
    # 0.6B和1.7B的相同：N_layer=28, N_q=16, N_kv=8
    "qwen3-0.6b": {"P": 0.752e9, "N_layer": 28, "N_q": 16, "N_kv": 8, "d_head": 128},
    "qwen3-1.7b": {"P": 2.03e9, "N_layer": 28, "N_q": 16, "N_kv": 8, "d_head": 128},
    
    # 4B和8B的相同：N_layer=36, N_q=32, N_kv=8
    "qwen3-4b":   {"P": 4.02e9, "N_layer": 36, "N_q": 32, "N_kv": 8, "d_head": 128},
    "qwen3-8b":   {"P": 8.19e9, "N_layer": 36, "N_q": 32, "N_kv": 8, "d_head": 128},
    
    # 14B: N_layer=40, N_q=40, N_kv=8
    "qwen3-14b":  {"P": 14.77e9, "N_layer": 40, "N_q": 40, "N_kv": 8, "d_head": 128},
    
    # 32B: N_layer=64, N_q=64, N_kv=8
    "qwen3-32b":  {"P": 32.76e9, "N_layer": 64, "N_q": 64, "N_kv": 8, "d_head": 128},

    # skywork-o1-prm-1.5b: 假设使用 Qwen2.5-1.5B 的配置 (GQA=6.0, N_q=12, N_kv=2)
    "skywork-o1-prm-1.5b": {"P": 1.54e9, "N_layer": 28, "N_q": 12, "N_kv": 2, "d_head": 128},
}

# 算术强度 I（例如 B200 取 562.5） 
ARITHMETIC_INTENSITY_I = 156

class InferenceCostModel: 
    def __init__(self, P, N_layer, N_q, N_kv, d_head, prec_p, prec_kv, I): 
        """ 
        初始化模型参数与硬件算力强度 
        :param P: 总参数量 (e.g., 7e9 for 7B) 
        :param N_layer: 层数 
        :param N_q: Query Head 数量 
        :param N_kv: KV Head 数量 
        :param d_head: Head 维度 
        :param prec_p: 参数存储精度 (Bytes, e.g., 2 for FP16) 
        :param prec_kv: KV Cache 存储精度 (Bytes) 
        :param I: 硬件算数强度 (Peak FLOPS / Memory Bandwidth) 
        """ 
        self.P = P 
        self.N_layer = N_layer 
        self.N_q = N_q 
        self.N_kv = N_kv 
        self.d_head = d_head 
        self.prec_p = prec_p 
        self.prec_kv = prec_kv 
        self.I = I 

    # --- 原子组件计算 --- 
    def f_p_comp(self, b): 
        return 2 * self.P * b 

    def f_p_mem(self): 
        return self.P * self.prec_p 

    def f_a_comp(self, b, l): 
        return 4 * b * l * self.N_layer * self.N_q * self.d_head 

    def f_a_mem(self, b, l): 
        return 2 * b * l * self.N_layer * self.N_kv * self.d_head * self.prec_kv 

    # --- 核心阶段计算 --- 
    def calculate_prefill(self, L_in): 
        """Proposition 3.1: Prefill Phase Cost""" 
        comp = L_in * self.f_p_comp(1) + sum(self.f_a_comp(1, i) for i in range(1, L_in + 1)) 
        mem = self.f_p_mem() + self.f_a_mem(1, L_in) 
        return comp + mem * self.I, mem

    def calculate_incremental_step(self, states_delta_L, L_init_list): 
        """ 
        Proposition 3.2: 单个推理步 j 的增量解码成本 
        :param states_delta_L: 当前步内各分支生成的 token 长度列表 [delta_L_1, delta_L_2, ...] 
        :param L_init_list: 各分支进入该步前的初始 context 长度列表 
        """ 
        N_j = max(states_delta_L) 
        total_step_cost = 0 
        total_step_mem = 0
        
        # 遍历该步中的每一个 token 位置 n 
        for n in range(1, N_j + 1): 
            # 筛选出在位置 n 仍在生成的活跃分支索引 
            active_indices = [idx for idx, length in enumerate(states_delta_L) if length >= n] 
            b_j_n = len(active_indices) 
            
            if b_j_n == 0: continue 
            
            # 计算平均 context 长度 L_bar_j(n) 
            L_bar_j_n = sum(L_init_list[idx] + n for idx in active_indices) / b_j_n 
            
            # eFLOPs 公式 
            comp = self.f_p_comp(b_j_n) + self.f_a_comp(b_j_n, L_bar_j_n) 
            mem = self.f_p_mem() + self.f_a_mem(b_j_n, L_bar_j_n) 
            total_step_cost += comp + mem * self.I 
            total_step_mem += mem
            
        return total_step_cost, total_step_mem

    def calculate_verification(self, L_final_list): 
        """Proposition 3.3: Verification Cost (Discriminative Mode)""" 
        # 计算所有分支的总计算量 
        total_comp = 0 
        total_a_mem = 0 
        for L_f in L_final_list: 
            comp_branch = L_f * self.f_p_comp(1) + sum(self.f_a_comp(1, n) for n in range(1, L_f + 1)) 
            total_comp += comp_branch 
            total_a_mem += self.f_a_mem(1, L_f) 
            
        # 验证器权重加载一次，KV cache 每个分支独立加载 
        total_mem = self.f_p_mem() + total_a_mem 
        return total_comp + total_mem * self.I, total_mem

def get_model_config(model_name, prec_p=2, prec_kv=2, I=ARITHMETIC_INTENSITY_I):
    """
    Construct configuration dictionary from MODEL_CONFIGS
    """
    if model_name not in MODEL_CONFIGS:
        # Fallback or error? For now, raise error or use default
        # Assuming all valid models are in MODEL_CONFIGS now
        raise ValueError(f"Unknown model config for: {model_name}")
        
    base_config = MODEL_CONFIGS[model_name]
    
    config = base_config.copy()
    config.update({
        "prec_p": prec_p,
        "prec_kv": prec_kv,
        "I": I
    })
    return config

def get_question_length(qdir: str, token_lens: Dict[str, int]) -> Optional[int]:
    path_obj = Path(qdir)
    q_name = path_obj.name
    if q_name in token_lens:
        return token_lens[q_name]
    
    path_str_lower = str(path_obj.resolve()).lower()
    
    for key, val in token_lens.items():
        suffix = f"_{q_name}"
        if key.endswith(suffix):
            prefix = key[:-len(suffix)]
            clean_prefix = prefix.rstrip('_')
            if not clean_prefix:
                continue
            if clean_prefix.lower() in path_str_lower:
                return val
    return None

class ExperimentCostCalculator:
    def __init__(self, gen_model_config, ver_model_config):
        self.gen_model = InferenceCostModel(**gen_model_config)
        self.ver_model = InferenceCostModel(**ver_model_config)

    def calculate_from_question_files(self, json_files, verbose=False, question_token_len=None):
        """
        Merge nodes from multiple record files (for the same question) 
        and calculate eFLOPs using the new InferenceCostModel.
        
        :param question_token_len: Optional integer for initial prompt length (L_in). 
                                   If provided, overrides length estimation from state_before.
        """
        if not json_files:
            return None

        # 1. Load all nodes from all files
        all_nodes_by_depth = {} # depth -> list of nodes
        
        roots = []
        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    try:
                        content = f.read().strip()
                        if not content:
                            continue
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        lines = content.split('\n')
                        valid_lines = [l for l in lines if l.strip()]
                        if not valid_lines:
                            continue
                        data = json.loads(valid_lines[-1])
                    
                    if 'nodes' not in data:
                        continue
                        
                    file_nodes = data['nodes']
                    file_tree = {}
                    for node in file_nodes:
                        pid = node['parent_id']
                        if pid not in file_tree: file_tree[pid] = []
                        file_tree[pid].append(node)
                    
                    root = next((n for n in file_nodes if n['parent_id'] is None), None)
                    if root:
                        roots.append((root, file_tree))
            except Exception as e:
                print(f"Error reading {jf}: {e}")
                continue

        if not roots:
            return None

        # 2. Calculate Prefill (Generator)
        # 如果提供了 question_token_len，直接使用；否则从第一个 root 的 state_before 估算
        if question_token_len is not None:
            l_in_global = question_token_len
        else:
            root_node_0 = roots[0][0]
            state_before = root_node_0.get('state_before', "")
            l_in_global = len(state_before.split()) if state_before else 0
            
        c_prefill, m_prefill = self.gen_model.calculate_prefill(l_in_global)

        if verbose:
            print(f"\n[Detailed Calculation Report]")
            print(f"--- Prefill Phase ---")
            print(f"Initial Context Length (L_in): {l_in_global} ({'Provided' if question_token_len is not None else 'Estimated'})")
            print(f"Prefill Cost: {c_prefill:.4e} eFLOPs")
            print(f"Prefill Memory: {m_prefill:.4e} Bytes")

        # 3. Step-by-step Decoding & Verification
        current_level_nodes = []
        for r_node, r_tree in roots:
            # 无论是否提供 L_in，我们都用 l_in_global 作为初始累积长度
            # 因为 Prefill 已经计算了 prompt 部分
            root_len = l_in_global
            
            children = r_tree.get(r_node['node_id'], [])
            for child in children:
                # 传入累积长度 (parent_cum_len)
                current_level_nodes.append((child, r_tree, root_len))
        
        c_dec_total = 0
        c_ver_total = 0
        m_dec_total = 0
        m_ver_total = 0
        
        step_j = 1
        
        while current_level_nodes:
            step_deltas = []
            step_l_inits = []
            step_node_ids = []
            
            # 准备下一层节点时，需要同时传递更新后的累积长度
            next_level_nodes = []
            
            for node, tree_ctx, parent_cum_len in current_level_nodes:
                # 强制检查 num_generated_token
                if 'num_generated_token' not in node:
                    raise ValueError(f"Missing 'num_generated_token' in node {node.get('node_id')}")
                
                delta_l = node['num_generated_token']
                # 当前节点的输入上下文长度 = 父节点的累积长度
                l_init = parent_cum_len
                
                step_deltas.append(delta_l)
                step_l_inits.append(l_init)
                step_node_ids.append(node.get('node_id'))
                
                # 计算当前节点结束后的累积长度，传递给子节点
                current_final_len = l_init + delta_l
                
                children = tree_ctx.get(node['node_id'], [])
                for child in children:
                    next_level_nodes.append((child, tree_ctx, current_final_len))
            
            if step_deltas:
                # Decoding Cost (Generator)
                step_dec_cost, step_dec_mem = self.gen_model.calculate_incremental_step(step_deltas, step_l_inits)
                c_dec_total += step_dec_cost
                m_dec_total += step_dec_mem
                
                # Verification Cost (Verifier)
                # Calculates cost for each branch's full sequence verification
                l_final_list = [step_l_inits[i] + step_deltas[i] for i in range(len(step_deltas))]
                step_ver_cost, step_ver_mem = self.ver_model.calculate_verification(l_final_list)
                c_ver_total += step_ver_cost
                m_ver_total += step_ver_mem

                if verbose:
                    print(f"\n--- Step {step_j} ---")
                    print(f"Active Branches (b_j): {len(step_deltas)}")
                    print(f"Branch Details:")
                    for i in range(len(step_deltas)):
                        print(f"  Branch {i+1} (Node: {step_node_ids[i]}): L_init={step_l_inits[i]}, Delta_L={step_deltas[i]}, L_final={l_final_list[i]}")
                    print(f"Step Decoding Cost: {step_dec_cost:.4e} eFLOPs")
                    print(f"Step Verification Cost: {step_ver_cost:.4e} eFLOPs")

            current_level_nodes = next_level_nodes
            step_j += 1

        total_eflops = c_prefill + c_dec_total + c_ver_total
        total_memory = m_prefill + m_dec_total + m_ver_total

        if verbose:
            print(f"\n--- Final Summary ---")
            print(f"Total Prefill: {c_prefill:.4e}")
            print(f"Total Decoding: {c_dec_total:.4e}")
            print(f"Total Verification: {c_ver_total:.4e}")
            print(f"Grand Total eFLOPs: {total_eflops:.4e}")
            print(f"Grand Total Memory: {total_memory:.4e} Bytes")
            print(f"--------------------------\n")

        return {
            "prefill": c_prefill,
            "decoding": c_dec_total,
            "verification": c_ver_total,
            "total_eflops": total_eflops,
            "total_memory": total_memory
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=str, 
                        default="/root/autodl-fs/tts/ttsrouter-v1.1/src/output/test150_Qwen3-32B_QP2_CP32_BS4_beam_search/Qwen3-32B/Skywork-o1-Open-PRM-Qwen-2.5-1.5B/1000_4_2",
                        help="Target directory containing question folders")
    parser.add_argument("--model-name", type=str, default="qwen3-32b", help="Model name key for parameters")
    parser.add_argument("--verifier-name", type=str, default="skywork-o1-prm-1.5b", help="Verifier name key for parameters")
    parser.add_argument("--filter-json", type=str, default=None, 
                        help="Optional JSON file mapping question names to list of record indices to use.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed calculation report")
    parser.add_argument("--question-name", type=str, default=None, help="Specific question name to calculate (e.g., question_0)")
    args = parser.parse_args()

    gen_model = args.model_name
    ver_model = args.verifier_name

    # Load filter config if provided
    record_filter = {}
    if args.filter_json:
        try:
            with open(args.filter_json, 'r') as f:
                record_filter = json.load(f)
        except Exception as e:
            print(f"Error loading filter JSON: {e}")
            return

    # Load question token lengths if available
    question_token_lens = {}
    try:
        q_len_path = "/root/autodl-fs/tts/ttsrouter-v1.1/src/envs/MATH/question_token_len.json"
        if os.path.exists(q_len_path):
            with open(q_len_path, 'r') as f:
                question_token_lens = json.load(f)
            if args.verbose:
                print(f"Loaded {len(question_token_lens)} question token lengths from {q_len_path}")
    except Exception as e:
        if args.verbose:
            print(f"Warning: Failed to load question token lengths: {e}")

    try:
        gen_config = get_model_config(gen_model)
        ver_config = get_model_config(ver_model)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return
    
    calc = ExperimentCostCalculator(gen_config, ver_config)
    target_dir = args.target_dir
    
    results = {}
    
    if args.question_name:
        question_dirs = [os.path.join(target_dir, args.question_name)]
    else:
        question_dirs = glob.glob(os.path.join(target_dir, "question_*"))
        
    for q_dir in question_dirs:
        q_name = os.path.basename(q_dir)
        
        all_json_files = glob.glob(os.path.join(q_dir, "record_*_beam.json"))
        
        if record_filter and q_name in record_filter:
            allowed_indices = set(record_filter[q_name])
            filtered_files = []
            for jf in all_json_files:
                basename = os.path.basename(jf)
                try:
                    idx_str = basename.split('_')[1]
                    idx = int(idx_str)
                    if idx in allowed_indices:
                        filtered_files.append(jf)
                except:
                    continue
            json_files = filtered_files
        else:
            if record_filter and q_name not in record_filter:
                 json_files = all_json_files # Default to all if not filtered
            else:
                 json_files = all_json_files

        if not json_files:
            continue
            
        q_len = None
        if question_token_lens:
            q_len = get_question_length(q_dir, question_token_lens)

        cost = calc.calculate_from_question_files(json_files, verbose=args.verbose, question_token_len=q_len)
        if cost:
            results[q_name] = cost
            
    if not args.verbose:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
