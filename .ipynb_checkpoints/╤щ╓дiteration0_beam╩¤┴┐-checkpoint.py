#!/usr/bin/env python3
"""
验证iteration 0的beam数量
"""
import json

def analyze_iteration0_beams():
    """分析iteration 0的beam数量"""
    
    # 读取latency记录
    with open('src/output/AMC23_t1_beam_search/Qwen3-0.6B/Skywork-o1-Open-PRM-Qwen-2.5-1.5B/40_4_2/question_9/record_0_latency.jsonl', 'r') as f:
        data = json.loads(f.read())
    
    # 分析iteration 0
    iter0 = data['complete_latency_record'][0]
    
    print("=" * 80)
    print("Iteration 0 分析")
    print("=" * 80)
    print(f"总延迟: {iter0['step_latency']:.3f}秒")
    print(f"活跃Beam数: {iter0['num_active_beams']}")
    print(f"完成的节点数: {iter0['num_end_nodes']}")
    print(f"记录的Beam详细信息数: {len(iter0['beam_details'])}")
    print()
    
    print("Beam详细信息:")
    print("-" * 80)
    for beam in iter0['beam_details']:
        print(f"Beam {beam['beam_idx']}:")
        print(f"  - Value: {beam['value']:.6f}")
        print(f"  - 总时间: {beam['total_time']:.3f}s")
        print(f"  - LM延迟: {beam['lm_latency']:.3f}s")
        print(f"  - RM延迟: {beam['rm_latency']:.3f}s")
        print(f"  - Tokens: {beam['num_tokens']}")
        print(f"  - LM每token: {beam['lm_time_per_token']:.4f}s")
        print(f"  - RM每token: {beam['rm_time_per_token']:.4f}s")
        print()
    
    # 分析配置
    print("=" * 80)
    print("配置参数")
    print("=" * 80)
    print(f"Beam Size (从配置推断): 2")
    print()
    
    # 验证：为什么只有2个beam？
    print("=" * 80)
    print("验证逻辑")
    print("=" * 80)
    print("根据代码分析 (tree.py:256-347):")
    print("1. Root节点被expand后，会创建多个子节点（基于legal_actions）")
    print("2. 从这些子节点中选择top-k个（k = beam_size）")
    print("3. 这里beam_size = 2，所以只选择了2个beam")
    print("4. 这解释了为什么num_active_beams=2，且只有2个beam_details")
    print()
    
    # 对比所有iteration的beam数量
    print("=" * 80)
    print("所有Iteration的Beam数量变化")
    print("=" * 80)
    for i, iter_data in enumerate(data['complete_latency_record']):
        num_beams = iter_data['num_active_beams']
        num_details = len(iter_data['beam_details'])
        print(f"Iteration {i}: {num_beams} 活跃beams, {num_details} 详细记录")

if __name__ == "__main__":
    analyze_iteration0_beams()



