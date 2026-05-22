#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import argparse
from typing import List, Dict


def build_ascii_tree(latency: Dict) -> str:
    lines: List[str] = []

    q = latency.get("question", "")
    total_iter = latency.get("total_iterations", len(latency.get("complete_latency_record", [])))
    lines.append(f"# Beam Search 树结构图\n")
    lines.append(f"问题: {q}\n")
    lines.append(f"总 Iterations: {total_iter}\n")
    lines.append("")

    recs = latency["complete_latency_record"]

    # 顶层根节点描述
    def fmt_iter_header(iter_idx: int, iter_data: Dict) -> str:
        step_lat = iter_data.get("step_latency", 0.0)
        num_beams = iter_data.get("num_active_beams", 0)
        return f"Iter {iter_idx} ({step_lat:.2f}s) [{num_beams} beams]"

    # 逐 iteration 展开，画成分层树（不跨 iteration 追踪具体beam id），展示本 iteration 的活跃数与 top-k 详情；
    # 对未被记录（非 top-k）的活跃 beam，用占位标注“未记录，非top-k”。
    for i, it in enumerate(recs):
        header = fmt_iter_header(i, it)
        lines.append(header)
        lines.append("|-")
        details = it.get("beam_details", [])
        total_active = it.get("num_active_beams", 0)

        # 输出所有 beam；kept=True 标注 [TOP] 并格式化数值；kept=False 标注 [DROP] 并使用 N/A
        recorded_cnt = 0
        if details:
            for b in details:
                bid = b.get("beam_idx")
                val = b.get("value", 0.0)
                tok = b.get("num_tokens", 0)
                kept = b.get("kept", False)

                lm = b.get("lm_latency")
                rm = b.get("rm_latency")

                if kept:
                    lm_str = f"{lm:.2f}s" if isinstance(lm, (int, float)) else "0.00s"
                    rm_str = f"{rm:.2f}s" if isinstance(rm, (int, float)) else "0.00s"
                    tag = "[TOP]"
                    recorded_cnt += 1
                else:
                    lm_str = "N/A"
                    rm_str = "N/A"
                    tag = "[DROP]"

                lines.append(
                    f"|  {tag} Beam-{bid}: value={val:.6f}, tokens={tok}, LM={lm_str}, RM={rm_str}"
                )
        else:
            lines.append("|  (无详细beam记录)")

        # 若活跃数大于记录数，补充占位的未记录条目
        missing = max(0, total_active - recorded_cnt)
        for j in range(missing):
            lines.append("|  Beam-? (未记录，非top-k)")
        lines.append("")

    # 简要的数量变化摘要
    trend = " → ".join(str(it.get("num_active_beams", 0)) for it in recs)
    lines.append("数量变化: " + trend)
    lines.append("")

    # 备注提示
    lines.append("备注: 这里的 beam_idx 为每个 iteration 内部的局部索引，不跨 iteration 对应；[TOP] 为被保留到下一轮的 top-k。")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="src/output/AMC23_t1_beam_search/Qwen3-0.6B/Skywork-o1-Open-PRM-Qwen-2.5-1.5B/40_4_2/question_9/record_0_latency.jsonl")
    parser.add_argument("--output", type=str, default="BeamSearch_树结构图.md")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    with open(input_path, "r", encoding="utf-8") as f:
        latency = json.loads(f.read())

    md = build_ascii_tree(latency)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"已生成: {output_path}")


if __name__ == "__main__":
    main()


