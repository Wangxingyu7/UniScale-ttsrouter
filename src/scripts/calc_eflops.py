import json
import os
import glob
import math
import argparse

# ---- eFLOPs 计算相关模型参数与常量 ---- 
MODEL_PARAM_P = { 
    "qwen3-0.6b": 0.6e9, 
    "qwen3-1.7b": 1.7e9, 
    "qwen3-4b": 4e9, 
    "qwen3-8b": 8.2e9, 
    "qwen3-14b": 14.8e9, 
    "qwen3-32b": 32.8e9, 
    "skywork-o1-prm-1.5b": 1.54e9, 
} 
MODEL_GQA_RATIO_R = { 
    "qwen3-0.6b": 0.5, 
    "qwen3-1.7b": 0.5, 
    "qwen3-4b": 0.25, 
    "qwen3-8b": 0.25, 
    "qwen3-14b": 0.2, 
    "qwen3-32b": 1.0, 
    "skywork-o1-prm-1.5b": 2.0, 
} 

# KV 大小 D（每 token 的 KV 维度；缺省统一设为 128） 
MODEL_KV_SIZE_D = { 
    "qwen3-0.6b": 57344, 
    "qwen3-1.7b": 114688, 
    "qwen3-4b": 73728, 
    "qwen3-8b": 147456, 
    "qwen3-14b": 196608, 
    "qwen3-32b": 262144, 
    "skywork-o1-prm-1.5b": 28672, 
} 

# 算术强度 I（例如 B200 取 562.5） 
ARITHMETIC_INTENSITY_I = 156

class eFLOPsCalculator:
    def __init__(self, P, Pv, I, r, D):
        """
        P: 生成模型参数量 (Model parameters)
        Pv: 验证模型参数量 (Verifier parameters)
        I: 硬件算术强度 (Arithmetic Intensity, e.g., A100 ~= 150-200)
        r: GQA 因子 (Query heads / KV heads)
        D: 每 token KV Cache 大小 (Bytes)
        """
        self.P = P
        self.Pv = Pv
        self.I = I
        self.r = r
        self.D = D

    def g_func(self, b, l):
        """General cost function g(b, l) from Assumption 4.3"""
        comp_param = 2 * self.P * b
        mem_param = 2 * self.P * self.I
        comp_attn = 2 * self.r * b * l * self.D
        mem_kv = 2 * b * l * self.D * self.I
        
        cost_compute = comp_param + comp_attn
        cost_memory = mem_param + mem_kv
        return cost_compute, cost_memory

    def calculate_from_question_files(self, json_files):
        """
        Merge nodes from multiple record files (for the same question) 
        and calculate eFLOPs as if processed in a single parallel batch.
        """
        if not json_files:
            return None

        # 1. Load all nodes from all files
        # We need to organize them by depth: level_nodes = {depth: [node, ...]}
        # Also need to handle tree structure for each file independently first?
        # Actually, if we assume parallel processing, we just need to know
        # how many active branches are there at each step t.
        
        # Structure:
        # We need to traverse "Step 1", "Step 2", ... across ALL files.
        # Step j corresponds to depth j (assuming root is depth 0).
        # Depth 0 is the root (Prompt).
        # Depth 1 is the first thought step.
        
        all_nodes_by_depth = {} # depth -> list of nodes
        
        # To avoid duplicate counting of the root node (Prompt Prefill),
        # we will take the root from the first file for prefill calculation.
        # But we need to check if multiple files imply multiple distinct roots?
        # Usually for the same question, the root (prompt) is identical.
        # So Prefill is done ONCE (batch=1) or MULTIPLE times (batch=N)?
        # If we simulate "parallel generation", it's one large batch of N prompts.
        # But since prompts are identical, maybe prefix caching is used?
        # Let's assume standard parallel batching: Batch Size = Number of files.
        # BUT, the user says "30+28=58 thoughts".
        # This implies at Depth 1, we have 58 active branches.
        
        # Let's collect all nodes from all files.
        roots = []
        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # We need to reconstruct the tree structure per file
                    # to know which nodes are children of which.
                    # But actually, our calculation only depends on "current level nodes".
                    # So we can just collect all nodes and group by depth?
                    # NO. We need to follow the tree structure to know when a branch ends.
                    
                    file_nodes = data['nodes']
                    file_tree = {}
                    for node in file_nodes:
                        pid = node['parent_id']
                        if pid not in file_tree: file_tree[pid] = []
                        file_tree[pid].append(node)
                    
                    # Root is usually id 0 or parent_id=None
                    # In this dataset, root_id is in data['root_id'] or we find node with parent_id=None
                    root = next((n for n in file_nodes if n['parent_id'] is None), None)
                    if root:
                        roots.append((root, file_tree))
            except Exception as e:
                print(f"Error reading {jf}: {e}")
                continue

        if not roots:
            return None

        # 2. Calculate Prefill
        # Assumption: All records share the same prompt.
        # If we treat them as a single batch of size B=len(roots) sharing the same prompt?
        # Or B=1 with shared prompt?
        # Usually in serving, if multiple requests come with same prompt, 
        # we might just process it once (if hit cache) or B times.
        # Given "parallel generation" context, let's assume we process the prompt ONCE 
        # (or B times but it's identical so maybe 1 effective prefill + broadcasting?)
        # Let's stick to the conservative assumption: 
        # The prompt is processed once to generate the KV cache, which is then expanded/shared.
        # So Prefill Batch = 1.
        
        root_node_0 = roots[0][0]
        state_before = root_node_0.get('state_before', "")
        l_in = len(state_before.split()) if state_before else 0
        
        c_prefill_compute = 2 * self.P * l_in
        c_prefill_memory = 2 * self.P * self.I
        c_prefill_total = c_prefill_compute + c_prefill_memory

        # 3. Step-by-step Decoding & Verification
        # We maintain a list of "current active nodes" across ALL files.
        # Initially, it's all the children of the roots from all files.
        
        current_level_nodes = []
        for r_node, r_tree in roots:
            # Add children of root from this file's tree
            children = r_tree.get(r_node['node_id'], [])
            # We need to attach the 'tree' context to the node so we can find its children later
            for child in children:
                current_level_nodes.append((child, r_tree))
        
        c_dec_compute = 0
        c_dec_memory = 0
        c_dec_total = 0
        
        c_ver_compute = 0
        c_ver_memory = 0
        c_ver_total = 0
        
        step_j = 1
        
        while current_level_nodes:
            # current_level_nodes contains pairs of (node, file_tree)
            # This level corresponds to Step j.
            
            step_deltas = []
            step_l_inits = []
            
            for node, _ in current_level_nodes:
                if node.get('num_generated_token', 0) > 0:
                    delta_l = node['num_generated_token']
                elif node.get('action'):
                    delta_l = len(node['action'].split())
                elif node.get('state_after') and node.get('state_before'):
                    delta_l = len(node['state_after'].split()) - len(node['state_before'].split())
                else:
                    delta_l = 1
                delta_l = max(1, delta_l)
                
                sb = node.get('state_before', "")
                l_init = len(sb.split()) if sb else 0
                
                step_deltas.append(delta_l)
                step_l_inits.append(l_init)
            
            # --- Decoding Cost ---
            # Now we have a large batch combining all files.
            # e.g. 30 nodes from file 1 + 28 nodes from file 2 = 58 nodes.
            if step_deltas:
                t_j = max(step_deltas)
                c_dec_j_compute = 0
                c_dec_j_memory = 0
                
                for t in range(1, t_j + 1):
                    active_branches = [i for i, d in enumerate(step_deltas) if d >= t]
                    b_jt = len(active_branches) # This is the aggregated batch size
                    if b_jt > 0:
                        l_bar_jt = sum(step_l_inits[i] + t for i in active_branches) / b_jt
                        cost_comp, cost_mem = self.g_func(b_jt, l_bar_jt)
                        c_dec_j_compute += cost_comp
                        c_dec_j_memory += cost_mem
                
                c_dec_compute += c_dec_j_compute
                c_dec_memory += c_dec_j_memory
                c_dec_total += (c_dec_j_compute + c_dec_j_memory)
                
                # --- Verification Cost ---
                # Verify all nodes in this level in one go (or conceptually parallel)
                # Verification is done on the final state of each node.
                sum_l_final = sum(step_l_inits[i] + step_deltas[i] for i in range(len(step_deltas)))
                
                c_ver_j_compute = 2 * self.Pv * sum_l_final
                # Memory cost for verification:
                # Weights are loaded once per verification "batch".
                # If we assume all these 58 nodes are verified in one batch (or pipelined such that weights stay):
                c_ver_j_memory = 2 * self.Pv * self.I
                
                c_ver_compute += c_ver_j_compute
                c_ver_memory += c_ver_j_memory
                c_ver_total += (c_ver_j_compute + c_ver_j_memory)

            # Prepare next level
            next_level_nodes = []
            for node, tree_ctx in current_level_nodes:
                children = tree_ctx.get(node['node_id'], [])
                for child in children:
                    next_level_nodes.append((child, tree_ctx))
            
            current_level_nodes = next_level_nodes
            step_j += 1

        total_compute = c_prefill_compute + c_dec_compute + c_ver_compute
        total_memory = c_prefill_memory + c_dec_memory + c_ver_memory
        total_eflops = total_compute + total_memory

        return {
            "prefill": {
                "compute": c_prefill_compute,
                "memory": c_prefill_memory,
                "total": c_prefill_total
            },
            "decoding": {
                "compute": c_dec_compute,
                "memory": c_dec_memory,
                "total": c_dec_total
            },
            "verification": {
                "compute": c_ver_compute,
                "memory": c_ver_memory,
                "total": c_ver_total
            },
            "summary": {
                "total_compute": total_compute,
                "total_memory": total_memory,
                "total_eflops": total_eflops
            }
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=str, 
                        default="/root/autodl-fs/tts/ttsrouter-v1.1/src/output/test150_Qwen3-32B_QP2_CP32_BS4_beam_search/Qwen3-32B/Skywork-o1-Open-PRM-Qwen-2.5-1.5B/1000_4_2",
                        help="Target directory containing question folders")
    parser.add_argument("--model-name", type=str, default="qwen3-32b", help="Model name key for parameters")
    parser.add_argument("--verifier-name", type=str, default="skywork-o1-prm-1.5b", help="Verifier name key for parameters")
    parser.add_argument("--filter-json", type=str, default=None, 
                        help="Optional JSON file mapping question names to list of record indices to use. "
                             "Format: {'question_0': [0, 1], 'question_1': [0]}")
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

    # 参数从全局字典获取
    params = {
        "P": MODEL_PARAM_P[gen_model], 
        "Pv": MODEL_PARAM_P[ver_model], 
        "I": ARITHMETIC_INTENSITY_I, 
        "r": MODEL_GQA_RATIO_R[gen_model],   
        "D": MODEL_KV_SIZE_D[gen_model]
    }
    
    calc = eFLOPsCalculator(**params)
    target_dir = args.target_dir
    
    results = {}
    
    # 遍历 question_? 文件夹
    question_dirs = glob.glob(os.path.join(target_dir, "question_*"))
    for q_dir in question_dirs:
        q_name = os.path.basename(q_dir)
        
        # 收集该 question 下的所有 record 文件
        all_json_files = glob.glob(os.path.join(q_dir, "record_*_beam.json"))
        
        # 如果有过滤器配置，筛选特定的 record
        if record_filter and q_name in record_filter:
            allowed_indices = set(record_filter[q_name])
            filtered_files = []
            for jf in all_json_files:
                # 解析文件名中的 index, e.g. record_0_beam.json -> 0
                basename = os.path.basename(jf)
                try:
                    # 假设文件名格式固定为 record_{index}_beam.json
                    idx_str = basename.split('_')[1]
                    idx = int(idx_str)
                    if idx in allowed_indices:
                        filtered_files.append(jf)
                except:
                    continue
            json_files = filtered_files
        else:
            # 如果没有指定过滤器，默认行为（比如使用全部，或者你可以修改这里为默认只用 record_0）
            # 这里保持默认使用全部，除非 filter_json 显式给出了空列表（意味着跳过该问题）
            if record_filter and q_name not in record_filter:
                 # 如果提供了过滤器但该问题不在其中，可以选择跳过，或者默认处理
                 # 这里假设未提及的问题使用全部
                 json_files = all_json_files
            else:
                 json_files = all_json_files

        if not json_files:
            continue
            
        cost = calc.calculate_from_question_files(json_files)
        if cost:
            results[q_name] = cost
            
    # 输出结果
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()