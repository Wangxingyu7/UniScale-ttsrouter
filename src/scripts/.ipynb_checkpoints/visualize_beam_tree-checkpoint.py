#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化beam search树结构
支持多种输出格式：图形化（PNG/SVG）、ASCII文本、HTML交互式
"""

import json
import argparse
import os
from typing import Dict, List, Set, Optional
from collections import defaultdict


def load_jsonl(filepath: str) -> Dict:
    """加载JSONL文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"文件 {filepath} 为空或格式错误")


def build_tree_structure(data: Dict) -> Dict:
    """构建树结构"""
    tree = {
        'nodes': {},  # node_id -> node_info
        'edges': [],  # [(parent_id, child_id)]
        'root_nodes': set(),  # 根节点集合
        'iterations': []  # 每个iteration的信息
    }
    
    records = data.get('complete_latency_record', [])
    
    # 收集所有节点信息
    all_parents = set()
    all_children = set()
    
    for iter_idx, record in enumerate(records):
        beam_details = record.get('beam_details', [])
        parent_child_mapping = record.get('parent_child_mapping', {})
        
        iteration_info = {
            'iteration': iter_idx,
            'step_latency': record.get('step_latency', 0.0),
            'num_active_beams': record.get('num_active_beams', 0),
            'nodes': []
        }
        
        # 收集节点信息
        for beam in beam_details:
            node_id = str(beam.get('node_id', ''))
            parent_id = str(beam.get('parent_node_id', ''))
            
            node_info = {
                'node_id': node_id,
                'parent_id': parent_id,
                'beam_idx': beam.get('beam_idx', -1),
                'value': beam.get('value', 0.0),
                'kept': beam.get('kept', False),
                'is_terminal': beam.get('is_terminal', False),
                'num_tokens': beam.get('num_tokens', 0),
                'last_action': beam.get('last_action', '')[:100],  # 截断前100字符
                'iteration': iter_idx,
                'lm_latency': beam.get('lm_latency', 0.0),
                'rm_latency': beam.get('rm_latency', 0.0),
            }
            
            tree['nodes'][node_id] = node_info
            iteration_info['nodes'].append(node_id)
            
            if parent_id:
                all_parents.add(parent_id)
                all_children.add(node_id)
        
        # 构建边
        for parent_id, child_ids in parent_child_mapping.items():
            parent_id = str(parent_id)
            for child_id in child_ids:
                child_id = str(child_id)
                tree['edges'].append((parent_id, child_id))
                all_parents.add(parent_id)
                all_children.add(child_id)
        
        tree['iterations'].append(iteration_info)
    
    # 找出根节点（没有父节点的节点）
    tree['root_nodes'] = all_parents - all_children
    
    return tree


def visualize_with_graphviz(tree: Dict, output_path: str, format: str = 'png'):
    """使用graphviz可视化树结构"""
    try:
        from graphviz import Digraph
    except ImportError:
        print("错误: 需要安装graphviz库。运行: pip install graphviz")
        print("同时需要安装graphviz系统包: sudo apt-get install graphviz")
        return False
    
    dot = Digraph(comment='Beam Search Tree', format=format)
    dot.attr(rankdir='TB', size='12,16')
    dot.attr('node', shape='box', style='rounded,filled')
    
    nodes = tree['nodes']
    edges = tree['edges']
    
    # 按iteration分组节点
    iteration_nodes = defaultdict(list)
    for node_id, node_info in nodes.items():
        iter_idx = node_info.get('iteration', 0)
        iteration_nodes[iter_idx].append(node_id)
    
    # 创建子图按iteration分组
    for iter_idx in sorted(iteration_nodes.keys()):
        with dot.subgraph(name=f'cluster_iter_{iter_idx}') as c:
            c.attr(label=f'Iteration {iter_idx}', style='filled', color='lightgrey')
            c.attr(rank='same')
            
            for node_id in iteration_nodes[iter_idx]:
                node_info = nodes[node_id]
                label = f"Beam-{node_info['beam_idx']}\\n"
                label += f"Value: {node_info['value']:.3f}\\n"
                label += f"Tokens: {node_info['num_tokens']}\\n"
                
                if node_info['kept']:
                    label += "[KEPT]"
                    color = 'lightgreen'
                elif node_info['is_terminal']:
                    label += "[TERMINAL]"
                    color = 'lightblue'
                else:
                    label += "[DROPPED]"
                    color = 'lightcoral'
                
                c.node(node_id, label=label, fillcolor=color)
    
    # 添加边
    for parent_id, child_id in edges:
        if parent_id in nodes and child_id in nodes:
            dot.edge(parent_id, child_id)
    
    # 渲染
    try:
        dot.render(output_path, cleanup=True)
        print(f"图形已保存到: {output_path}.{format}")
        return True
    except Exception as e:
        print(f"渲染图形时出错: {e}")
        return False


def visualize_with_matplotlib(tree: Dict, output_path: str):
    """使用matplotlib可视化树结构"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("错误: 需要安装matplotlib库。运行: pip install matplotlib")
        return False
    
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    nodes = tree['nodes']
    edges = tree['edges']
    
    # 简单的层次布局
    iteration_positions = defaultdict(list)
    max_iter = max(n.get('iteration', 0) for n in nodes.values()) if nodes else 0
    
    # 为每个iteration分配y坐标
    y_positions = {}
    for iter_idx in range(max_iter + 1):
        iter_nodes = [nid for nid, ninfo in nodes.items() if ninfo.get('iteration', 0) == iter_idx]
        if iter_nodes:
            y = 9 - (iter_idx * 1.5)
            x_spacing = 8.0 / max(len(iter_nodes), 1)
            for i, node_id in enumerate(iter_nodes):
                x = 1 + i * x_spacing
                y_positions[node_id] = (x, y)
    
    # 绘制边
    for parent_id, child_id in edges:
        if parent_id in y_positions and child_id in y_positions:
            x1, y1 = y_positions[parent_id]
            x2, y2 = y_positions[child_id]
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)
    
    # 绘制节点
    for node_id, (x, y) in y_positions.items():
        node_info = nodes[node_id]
        
        if node_info['kept']:
            color = 'lightgreen'
        elif node_info['is_terminal']:
            color = 'lightblue'
        else:
            color = 'lightcoral'
        
        # 绘制节点框
        box = FancyBboxPatch((x-0.3, y-0.2), 0.6, 0.4,
                            boxstyle="round,pad=0.05",
                            facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        
        # 添加文本
        label = f"B{node_info['beam_idx']}\\nv:{node_info['value']:.2f}"
        ax.text(x, y, label, ha='center', va='center', fontsize=6)
    
    # 添加图例
    kept_patch = mpatches.Patch(color='lightgreen', label='Kept')
    terminal_patch = mpatches.Patch(color='lightblue', label='Terminal')
    dropped_patch = mpatches.Patch(color='lightcoral', label='Dropped')
    ax.legend(handles=[kept_patch, terminal_patch, dropped_patch], loc='upper right')
    
    plt.title('Beam Search Tree Visualization', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图形已保存到: {output_path}")
    plt.close()
    return True


def visualize_ascii(tree: Dict) -> str:
    """生成ASCII格式的树结构"""
    lines = []
    nodes = tree['nodes']
    edges = tree['edges']
    
    # 构建邻接表
    children_map = defaultdict(list)
    for parent_id, child_id in edges:
        children_map[parent_id].append(child_id)
    
    # 找出根节点 - 使用第一个iteration的节点作为起始点
    root_nodes = tree['root_nodes']
    if not root_nodes:
        # 如果没有明确的根节点，找第一个iteration的节点
        first_iter_nodes = [nid for nid, ninfo in nodes.items() if ninfo.get('iteration', 0) == 0]
        if first_iter_nodes:
            root_nodes = {first_iter_nodes[0]}
    
    # 如果根节点不在nodes中，从第一个iteration开始
    if root_nodes:
        root_list = list(root_nodes)
        if root_list[0] not in nodes:
            # 根节点不在nodes中，从第一个iteration的所有节点开始
            first_iter_nodes = [nid for nid, ninfo in nodes.items() if ninfo.get('iteration', 0) == 0]
            root_nodes = set(first_iter_nodes[:1]) if first_iter_nodes else set()
    
    def print_node(node_id: str, prefix: str = "", is_last: bool = True):
        """递归打印节点"""
        if node_id not in nodes:
            return
        
        node_info = nodes[node_id]
        connector = "└── " if is_last else "├── "
        
        status = "[KEPT]" if node_info['kept'] else ("[TERMINAL]" if node_info['is_terminal'] else "[DROP]")
        label = f"Beam-{node_info['beam_idx']} (v={node_info['value']:.3f}, t={node_info['num_tokens']}) {status}"
        lines.append(prefix + connector + label)
        
        children = sorted(children_map[node_id], 
                         key=lambda cid: nodes.get(cid, {}).get('beam_idx', 0))
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child_id in enumerate(children):
            print_node(child_id, new_prefix, i == len(children) - 1)
    
    # 从根节点开始打印，如果没有根节点，从第一个iteration开始
    if root_nodes:
        root_list = list(root_nodes)
        if root_list[0] in nodes:
            # 根节点在nodes中，直接打印
            for root_id in sorted(root_nodes):
                print_node(root_id)
        else:
            # 根节点不在nodes中，从第一个iteration的所有节点开始
            first_iter_nodes = sorted([nid for nid, ninfo in nodes.items() if ninfo.get('iteration', 0) == 0],
                                      key=lambda nid: nodes[nid].get('beam_idx', 0))
            if first_iter_nodes:
                lines.append(f"Root: {root_list[0]} (not in nodes)")
                lines.append("First iteration nodes:")
                for i, root_id in enumerate(first_iter_nodes):
                    print_node(root_id, "", i == len(first_iter_nodes) - 1)
    else:
        # 从第一个iteration的所有节点开始
        first_iter_nodes = sorted([nid for nid, ninfo in nodes.items() if ninfo.get('iteration', 0) == 0],
                                  key=lambda nid: nodes[nid].get('beam_idx', 0))
        for i, root_id in enumerate(first_iter_nodes):
            print_node(root_id, "", i == len(first_iter_nodes) - 1)
    
    return "\n".join(lines)


def visualize_html_interactive(tree: Dict, output_path: str):
    """生成交互式HTML可视化"""
    # 转换set为list以便JSON序列化
    tree_serializable = {
        'nodes': tree['nodes'],
        'edges': tree['edges'],
        'root_nodes': list(tree['root_nodes']),
        'iterations': tree['iterations']
    }
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Beam Search Tree Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .node { cursor: pointer; }
        .node circle { fill: #fff; stroke: #333; stroke-width: 2px; }
        .node.kept circle { fill: #90EE90; }
        .node.terminal circle { fill: #87CEEB; }
        .node.dropped circle { fill: #FFB6C1; }
        .link { fill: none; stroke: #ccc; stroke-width: 1.5px; }
        .node text { font-size: 10px; }
        .tooltip { position: absolute; text-align: left; padding: 8px; font-size: 12px; 
                   background: rgba(0,0,0,0.8); color: white; border-radius: 4px; 
                   pointer-events: none; opacity: 0; }
    </style>
</head>
<body>
    <h1>Beam Search Tree Visualization</h1>
    <div id="tree"></div>
    
    <script>
        const data = """ + json.dumps(tree_serializable, indent=2, ensure_ascii=False) + """;
        
        const width = 1200;
        const height = 800;
        const margin = {top: 20, right: 120, bottom: 20, left: 120};
        
        const svg = d3.select("#tree")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);
        
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip");
        
        // 构建树形数据
        const nodesMap = new Map();
        const edges = data.edges || [];
        const nodes = data.nodes || {};
        
        // 创建节点映射
        Object.keys(nodes).forEach(id => {
            nodesMap.set(id, {id: id, ...nodes[id]});
        });
        
        // 构建层次结构
        const rootNodes = Array.from(data.root_nodes || []);
        if (rootNodes.length === 0 && nodesMap.size > 0) {
            rootNodes.push(nodesMap.keys().next().value);
        }
        
        function buildTree(rootId) {
            const node = nodesMap.get(rootId);
            if (!node) return null;
            
            const children = edges
                .filter(e => e[0] === rootId)
                .map(e => buildTree(e[1]))
                .filter(n => n !== null);
            
            return {
                ...node,
                children: children.length > 0 ? children : null
            };
        }
        
        // 如果有多个根节点，创建一个虚拟根
        let treeData;
        if (rootNodes.length === 1) {
            treeData = buildTree(rootNodes[0]);
        } else {
            treeData = {
                id: "root",
                children: rootNodes.map(id => buildTree(id)).filter(n => n !== null)
            };
        }
        
        const treeLayout = d3.tree()
            .size([height - margin.top - margin.bottom, width - margin.left - margin.right]);
        
        const root = d3.hierarchy(treeData);
        treeLayout(root);
        
        // 绘制链接
        const links = g.selectAll(".link")
            .data(root.links())
            .enter().append("path")
            .attr("class", "link")
            .attr("d", d3.linkHorizontal()
                .x(d => d.y)
                .y(d => d.x));
        
        // 绘制节点
        const node = g.selectAll(".node")
            .data(root.descendants())
            .enter().append("g")
            .attr("class", d => {
                const nodeData = nodesMap.get(d.data.id);
                if (!nodeData) return "node";
                if (nodeData.is_terminal) return "node terminal";
                if (nodeData.kept) return "node kept";
                return "node dropped";
            })
            .attr("transform", d => `translate(${d.y},${d.x})`)
            .on("mouseover", function(event, d) {
                const nodeData = nodesMap.get(d.data.id);
                if (nodeData) {
                    tooltip.transition().duration(200).style("opacity", .9);
                    tooltip.html(`
                        <strong>Beam-${nodeData.beam_idx}</strong><br/>
                        Value: ${nodeData.value.toFixed(4)}<br/>
                        Tokens: ${nodeData.num_tokens}<br/>
                        Iteration: ${nodeData.iteration}<br/>
                        LM Latency: ${nodeData.lm_latency.toFixed(3)}s<br/>
                        RM Latency: ${nodeData.rm_latency.toFixed(3)}s<br/>
                        ${nodeData.last_action ? `<br/>Action: ${nodeData.last_action.substring(0, 100)}...` : ''}
                    `)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
                }
            })
            .on("mouseout", function() {
                tooltip.transition().duration(500).style("opacity", 0);
            });
        
        node.append("circle")
            .attr("r", 6);
        
        node.append("text")
            .attr("dy", ".35em")
            .attr("x", d => d.children ? -13 : 13)
            .style("text-anchor", d => d.children ? "end" : "start")
            .text(d => {
                const nodeData = nodesMap.get(d.data.id);
                return nodeData ? `B${nodeData.beam_idx}` : d.data.id;
            });
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"交互式HTML已保存到: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='可视化beam search树结构')
    parser.add_argument('input_file', type=str, help='输入的JSONL文件路径')
    parser.add_argument('-o', '--output', type=str, help='输出文件路径（不含扩展名）')
    parser.add_argument('-f', '--format', type=str, 
                       choices=['png', 'svg', 'html', 'ascii', 'matplotlib'],
                       default='html', help='输出格式')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"正在加载数据: {args.input_file}")
    data = load_jsonl(args.input_file)
    
    # 构建树结构
    print("正在构建树结构...")
    tree = build_tree_structure(data)
    
    print(f"找到 {len(tree['nodes'])} 个节点, {len(tree['edges'])} 条边")
    print(f"根节点: {tree['root_nodes']}")
    
    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_path = base_name + '_tree'
    
    # 根据格式生成可视化
    if args.format == 'ascii':
        result = visualize_ascii(tree)
        output_file = output_path + '.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"ASCII树已保存到: {output_file}")
        print("\\n" + result)
    
    elif args.format == 'html':
        output_file = output_path + '.html'
        visualize_html_interactive(tree, output_file)
    
    elif args.format == 'png' or args.format == 'svg':
        output_file = output_path + '.' + args.format
        visualize_with_graphviz(tree, output_path, args.format)
    
    elif args.format == 'matplotlib':
        output_file = output_path + '.png'
        visualize_with_matplotlib(tree, output_file)
    
    print("\\n可视化完成！")


if __name__ == '__main__':
    main()

