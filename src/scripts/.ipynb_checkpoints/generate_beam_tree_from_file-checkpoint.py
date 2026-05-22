#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import List, Dict


def build_ascii_tree(latency: Dict, title: str = "Beam Search 树结构图") -> str:
    lines: List[str] = []

    q = latency.get("question", "")
    total_iter = latency.get("total_iterations", len(latency.get("complete_latency_record", [])))
    lines.append(f"# {title}\n")
    if q:
        lines.append(f"问题: {q}\n")
    lines.append(f"总 Iterations: {total_iter}\n")
    lines.append("")

    recs = latency["complete_latency_record"]

    def fmt_iter_header(iter_idx: int, iter_data: Dict) -> str:
        step_lat = iter_data.get("step_latency", 0.0)
        num_beams = iter_data.get("num_active_beams", 0)
        return f"Iter {iter_idx} ({step_lat:.2f}s) [{num_beams} beams]"

    for i, it in enumerate(recs):
        header = fmt_iter_header(i, it)
        lines.append(header)
        lines.append("|-")
        details = it.get("beam_details", [])
        total_active = it.get("num_active_beams", 0)

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

        missing = max(0, total_active - recorded_cnt)
        for _ in range(missing):
            lines.append("|  Beam-? (未记录，非top-k)")
        lines.append("")

    trend = " → ".join(str(it.get("num_active_beams", 0)) for it in recs)
    lines.append("数量变化: " + trend)
    lines.append("")
    lines.append("备注: beam_idx 为每个 iteration 的局部索引；[TOP] 表示进入下一轮；[DROP] 为被裁掉候选（LM/RM未记录）。")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="record_*_latency.jsonl 路径")
    parser.add_argument("--output", required=False, help="输出的 Markdown 文件路径")
    parser.add_argument("--title", required=False, default="Beam Search 树结构图")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        latency = json.loads(f.read())

    md = build_ascii_tree(latency, title=args.title)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.abspath(f"BeamSearch_树结构图_{base}.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"已生成: {output_path}")


if __name__ == "__main__":
    main()




