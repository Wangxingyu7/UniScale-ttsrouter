#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版beam search树可视化 - 按iteration分组显示
"""

import json
import argparse
from collections import defaultdict


def load_jsonl(filepath: str) -> dict:
    """加载JSONL文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"文件 {filepath} 为空或格式错误")


def visualize_by_iteration(data: dict) -> str:
    """按iteration分组可视化"""
    lines = []
    
    question = data.get('question', '')
    total_iterations = data.get('total_iterations', 0)
    
    lines.append("=" * 80)
    lines.append("Beam Search 树结构可视化")
    lines.append("=" * 80)
    if question:
        lines.append(f"\n问题: {question[:200]}...")
    lines.append(f"总迭代次数: {total_iterations}")
    lines.append("\n" + "=" * 80 + "\n")
    
    records = data.get('complete_latency_record', [])
    
    # 构建节点映射和父子关系
    node_map = {}  # node_id -> node_info
    parent_child_map = defaultdict(list)  # parent_id -> [child_ids]
    
    for iter_idx, record in enumerate(records):
        beam_details = record.get('beam_details', [])
        parent_child_mapping = record.get('parent_child_mapping', {})
        
        # 收集节点信息
        for beam in beam_details:
            node_id = str(beam.get('node_id', ''))
            node_map[node_id] = {
                'beam_idx': beam.get('beam_idx', -1),
                'value': beam.get('value', 0.0),
                'kept': beam.get('kept', False),
                'is_terminal': beam.get('is_terminal', False),
                'num_tokens': beam.get('num_tokens', 0),
                'iteration': iter_idx,
                'last_action': beam.get('last_action', '')[:80],
                # 延迟信息
                'total_time': beam.get('total_time', 0.0),
                'lm_latency': beam.get('lm_latency', 0.0),
                'rm_latency': beam.get('rm_latency', 0.0),
                'step_wait': beam.get('step_wait', 0.0),
                'other_latency': beam.get('other_latency', 0.0),
                'lm_time_per_token': beam.get('lm_time_per_token', 0.0),
                'rm_time_per_token': beam.get('rm_time_per_token', 0.0),
            }
        
        # 构建父子关系
        for parent_id, child_ids in parent_child_mapping.items():
            parent_id = str(parent_id)
            for child_id in child_ids:
                child_id = str(child_id)
                parent_child_map[parent_id].append(child_id)
    
    # 按iteration显示
    for iter_idx, record in enumerate(records):
        step_latency = record.get('step_latency', 0.0)
        step_wait = record.get('step_wait', 0.0)
        num_active = record.get('num_active_beams', 0)
        num_expanded = record.get('num_expanded_beams', 0)
        
        lines.append(f"\n{'='*80}")
        lines.append(f"Iteration {iter_idx} | 步骤延迟: {step_latency:.3f}s | 等待时间: {step_wait:.3f}s | 活跃beams: {num_active} | 扩展beams: {num_expanded}")
        lines.append('='*80)
        
        # 获取当前iteration的节点
        iter_nodes = {nid: ninfo for nid, ninfo in node_map.items() 
                     if ninfo.get('iteration', -1) == iter_idx}
        
        if not iter_nodes:
            lines.append("(无节点)")
            continue
        
        # 按beam_idx排序
        sorted_nodes = sorted(iter_nodes.items(), 
                            key=lambda x: x[1].get('beam_idx', 0))
        
        # 显示节点信息
        for node_id, node_info in sorted_nodes:
            beam_idx = node_info['beam_idx']
            value = node_info['value']
            tokens = node_info['num_tokens']
            kept = node_info['kept']
            is_terminal = node_info['is_terminal']
            
            # 状态标记
            if is_terminal:
                status = "[TERMINAL]"
            elif kept:
                status = "[KEPT →]"
            else:
                status = "[DROP]"
            
            # 显示节点
            lines.append(f"\n  {status} Beam-{beam_idx} (Node: {node_id[:12]}...)")
            lines.append(f"      Value: {value:.4f} | Tokens: {tokens}")
            
            # 显示延迟信息
            total_time = node_info.get('total_time', 0.0)
            lm_latency = node_info.get('lm_latency', 0.0)
            rm_latency = node_info.get('rm_latency', 0.0)
            step_wait = node_info.get('step_wait', 0.0)
            other_latency = node_info.get('other_latency', 0.0)
            lm_time_per_token = node_info.get('lm_time_per_token', 0.0)
            rm_time_per_token = node_info.get('rm_time_per_token', 0.0)
            
            lines.append(f"      延迟信息:")
            if total_time > 0:
                lines.append(f"        总时间: {total_time:.3f}s")
            if lm_latency > 0:
                lines.append(f"        LM延迟: {lm_latency:.3f}s (每token: {lm_time_per_token*1000:.2f}ms)")
            if rm_latency > 0:
                lines.append(f"        RM延迟: {rm_latency:.3f}s (每token: {rm_time_per_token*1000:.2f}ms)")
            if other_latency > 0:
                lines.append(f"        其他延迟: {other_latency:.3f}s")
            if step_wait > 0:
                lines.append(f"        等待时间: {step_wait:.3f}s")
            
            # 显示父节点
            parent_id = None
            for pid, children in parent_child_map.items():
                if node_id in children:
                    parent_id = pid
                    break
            
            if parent_id:
                parent_info = node_map.get(parent_id)
                if parent_info:
                    lines.append(f"      ← Parent: Beam-{parent_info['beam_idx']} (Iter {parent_info['iteration']})")
            
            # 显示子节点
            children = parent_child_map.get(node_id, [])
            if children:
                child_info_list = []
                for child_id in children:
                    child_info = node_map.get(child_id)
                    if child_info:
                        child_info_list.append(f"Beam-{child_info['beam_idx']} (Iter {child_info['iteration']})")
                
                if child_info_list:
                    lines.append(f"      → Children: {', '.join(child_info_list)}")
            
            # 显示last_action的前50个字符
            if node_info.get('last_action'):
                action_preview = node_info['last_action'][:80].replace('\n', ' ')
                lines.append(f"      Action: {action_preview}...")
        
        lines.append("")
    
    # 显示延迟统计摘要
    lines.append("\n" + "="*80)
    lines.append("延迟统计摘要")
    lines.append("="*80 + "\n")
    
    total_question_latency = data.get('question_latency', 0.0)
    lines.append(f"总问题延迟: {total_question_latency:.3f}s\n")
    
    for iter_idx, record in enumerate(records):
        step_latency = record.get('step_latency', 0.0)
        step_wait = record.get('step_wait', 0.0)
        beam_details = record.get('beam_details', [])
        
        if beam_details:
            # 计算延迟分析：当前总延迟和只考虑KEPT的beam的总延迟
            total_latency_all_beams = 0.0  # 所有beam的总延迟
            total_latency_kept_beams = 0.0  # 只考虑KEPT的beam的总延迟
            
            total_lm_latency_all = 0.0
            total_rm_latency_all = 0.0
            total_other_latency_all = 0.0
            total_wait_all = 0.0
            
            total_lm_latency_kept = 0.0
            total_rm_latency_kept = 0.0
            total_other_latency_kept = 0.0
            total_wait_kept = 0.0
            
            for beam_detail in beam_details:
                lm_lat = beam_detail.get('lm_latency', 0.0)
                rm_lat = beam_detail.get('rm_latency', 0.0)
                other_lat = beam_detail.get('other_latency', 0.0)
                wait_lat = beam_detail.get('step_wait', 0.0)
                beam_total = lm_lat + rm_lat + other_lat + wait_lat
                
                # 累加所有beam的延迟
                total_latency_all_beams += beam_total
                total_lm_latency_all += lm_lat
                total_rm_latency_all += rm_lat
                total_other_latency_all += other_lat
                total_wait_all += wait_lat
                
                # 只累加KEPT的beam的延迟
                if beam_detail.get('kept', False):
                    total_latency_kept_beams += beam_total
                    total_lm_latency_kept += lm_lat
                    total_rm_latency_kept += rm_lat
                    total_other_latency_kept += other_lat
                    total_wait_kept += wait_lat
            
            num_all_beams = len(beam_details)
            num_kept_beams = sum(1 for bd in beam_details if bd.get('kept', False))
            
            lines.append(f"Iteration {iter_idx}:")
            lines.append(f"  步骤延迟: {step_latency:.3f}s | 等待时间: {step_wait:.3f}s")
            lines.append(f"  Beam数量: 总计={num_all_beams}, KEPT={num_kept_beams}, 丢弃={num_all_beams - num_kept_beams}")
            lines.append("")
            
            # 显示所有beam的延迟分析
            lines.append(f"  【所有Beam延迟分析】")
            lines.append(f"    总延迟: {total_latency_all_beams:.3f}s")
            lines.append(f"      分解: LM={total_lm_latency_all:.3f}s, RM={total_rm_latency_all:.3f}s, "
                        f"Other={total_other_latency_all:.3f}s, Wait={total_wait_all:.3f}s")
            lines.append("")
            
            # 显示KEPT beam的延迟分析
            lines.append(f"  【KEPT Beam延迟分析】")
            lines.append(f"    总延迟: {total_latency_kept_beams:.3f}s")
            lines.append(f"      分解: LM={total_lm_latency_kept:.3f}s, RM={total_rm_latency_kept:.3f}s, "
                        f"Other={total_other_latency_kept:.3f}s, Wait={total_wait_kept:.3f}s")
            lines.append("")
            
            # 计算被丢弃beam的开销
            dropped_latency = total_latency_all_beams - total_latency_kept_beams
            if dropped_latency > 0:
                lines.append(f"  【被丢弃Beam开销】")
                lines.append(f"    额外延迟: {dropped_latency:.3f}s ({dropped_latency/total_latency_all_beams*100:.1f}%)")
                lines.append("")
            
            # 计算该iteration的平均延迟（保留原有统计）
            lm_latencies = [b.get('lm_latency', 0.0) for b in beam_details if b.get('lm_latency', 0.0) > 0]
            rm_latencies = [b.get('rm_latency', 0.0) for b in beam_details if b.get('rm_latency', 0.0) > 0]
            other_latencies = [b.get('other_latency', 0.0) for b in beam_details if b.get('other_latency', 0.0) > 0]
            total_times = [b.get('total_time', 0.0) for b in beam_details if b.get('total_time', 0.0) > 0]
            
            if lm_latencies:
                avg_lm = sum(lm_latencies) / len(lm_latencies)
                max_lm = max(lm_latencies)
                min_lm = min(lm_latencies)
                lines.append(f"  LM延迟: 平均={avg_lm:.3f}s, 最大={max_lm:.3f}s, 最小={min_lm:.3f}s")
            
            if rm_latencies:
                avg_rm = sum(rm_latencies) / len(rm_latencies)
                max_rm = max(rm_latencies)
                min_rm = min(rm_latencies)
                lines.append(f"  RM延迟: 平均={avg_rm:.3f}s, 最大={max_rm:.3f}s, 最小={min_rm:.3f}s")
            
            if other_latencies:
                avg_other = sum(other_latencies) / len(other_latencies)
                max_other = max(other_latencies)
                min_other = min(other_latencies)
                lines.append(f"  其他延迟: 平均={avg_other:.3f}s, 最大={max_other:.3f}s, 最小={min_other:.3f}s")
            
            if total_times:
                avg_total = sum(total_times) / len(total_times)
                max_total = max(total_times)
                min_total = min(total_times)
                lines.append(f"  总时间: 平均={avg_total:.3f}s, 最大={max_total:.3f}s, 最小={min_total:.3f}s")
            
            lines.append("")
    
    # 显示树结构连接关系
    lines.append("\n" + "="*80)
    lines.append("树结构连接关系 (Parent → Children)")
    lines.append("="*80 + "\n")
    
    # 按iteration分组显示连接
    for iter_idx in range(len(records)):
        iter_connections = []
        for parent_id, children in parent_child_map.items():
            parent_info = node_map.get(parent_id)
            if parent_info and parent_info.get('iteration') == iter_idx:
                child_list = []
                for child_id in children:
                    child_info = node_map.get(child_id)
                    if child_info:
                        child_list.append(f"Beam-{child_info['beam_idx']}")
                
                if child_list:
                    iter_connections.append(
                        f"  Iter {iter_idx}: Beam-{parent_info['beam_idx']} → {', '.join(child_list)}"
                    )
        
        if iter_connections:
            lines.extend(iter_connections)
            lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='可视化beam search树结构（简化版）')
    parser.add_argument('input_file', type=str, help='输入的JSONL文件路径')
    parser.add_argument('-o', '--output', type=str, help='输出文件路径（可选）')
    
    args = parser.parse_args()
    
    print(f"正在加载数据: {args.input_file}")
    data = load_jsonl(args.input_file)
    
    print("正在生成可视化...")
    result = visualize_by_iteration(data)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\n可视化结果已保存到: {args.output}")
    else:
        print("\n" + result)


if __name__ == '__main__':
    main()

