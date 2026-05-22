"""
The Node and MCTS class for AlphaZero.
"""

import copy
import json
import math
import traceback

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
from utils import print_rank_0, print_with_rank
from envs.base_env import CoTEnv
import heapq
from loguru import logger
from metrics.records import BeamDetail, IterationLatencyRecord
from metrics.recorder import SearchLatencyRecorder


class Node(object):
    """
    Overview:
        The node base class for tree_search.
    """

    def __init__(self, parent: "Node" = None, prior_p: float = 1.0, initial_value: float = 0.0, parent_value: float = 0.0) -> None:
        self._parent = parent
        self._children = {}
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = prior_p
        self.prior_p_ori = prior_p

        self._initial_value = initial_value
        self._parent_value = parent_value
        self._terminated = False

    def __lt__(self, other):
        return self._initial_value < other._initial_value

    @property
    def terminated(self):
        return self._terminated

    def set_as_terminate_node(self):
        self._terminated = True

    @property
    def value(self) -> float:
        """
        Overview:
            The value of the current node.
        Returns:
            - output (:obj:`Int`): Current value, used to compute ucb score.
        """
        if self._visit_count == 0:
            # if not visited, return the initial value
            return self._initial_value
        return self._value_sum / self._visit_count

    def update(self, value: float) -> None:
        """
        Overview:
            Update the current node information, such as visit_count and value_sum.
        Arguments:
            - value (:obj:`Int`): The value of the node.
        """
        self._visit_count += 1
        self._value_sum += value

    def update_recursive(self, leaf_value: float, mcts_mode: str) -> None:
        """
        Overview:
            Update node information recursively.
        Arguments:
            - leaf_value (:obj:`Int`): The value of the node.
        """
        if mcts_mode == "self_play_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(-leaf_value, mcts_mode)
        if mcts_mode == "play_with_bot_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(leaf_value, mcts_mode)

    def is_leaf(self) -> bool:
        """
        Overview:
            Check if the current node is a leaf node or not.
        Returns:
            - output (:obj:`Dict`): Dict type children node.
        """
        return self._children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
        Returns:
            - output (:obj:`Bool`): Whether it is the parent node.
        """
        return self._parent is None

    @property
    def parent(self) -> None:
        return self._parent

    @property
    def children(self) -> None:
        return self._children

    @property
    def visit_count(self) -> None:
        return self._visit_count

    def get_info(self):
        # return [
        #     "visit_cnt: {}, value: {:.6f}, prior: {:.6f}".format(
        #         self.visit_count, self.value, self.prior_p)
        # ]
        return {
            "visit_cnt": self.visit_count,
            "value": self.value,
            "prior_p": float(self.prior_p_ori),
            "initial_value": self._initial_value,
            "terminated": self.terminated,
        }

    def clear(self):
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = self.prior_p_ori

    def to_json(self):
        childrens = {}
        for name, child_node in self.children.items():
            childrens[name] = child_node.to_json()

        rets = {"children": childrens, "info": self.get_info()}
        return rets

    def __str__(self) -> str:
        if self.is_root():
            return "root"
        else:
            return "child: value: {:.3f}, prior: {:.3f}".format(self.last_action, self.value, self.prior_p)


class LanguageNode(Node):
    text_state: Optional[str] = None
    last_action: Optional[str] = None
    num_generated_token: Optional[int] = None

    def __init__(
        self,
        parent: Node = None,
        prior_p: float = 1.0,
        prm_value: Optional[float] = None,
        text_state: Optional[str] = None,
        last_action: Optional[str] = None,
        initial_value: float = 0.0,
        parent_value: float = 0.0,
        num_generated_token: Optional[int] = None,
        model_name: str = "",
        finish_reason: Optional[str] = None,
        raw_action: Optional[str] = None,
    ) -> None:
        super().__init__(parent, prior_p, initial_value, parent_value)
        self.text_state = text_state
        self.last_action = last_action
        self.prm_value = prm_value

        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False

        self.model_name = model_name
        self.finish_reason = finish_reason
        self.raw_action = raw_action

        self.node_id: Optional[int] = None
        self.step_metrics: Optional[Dict[str, Any]] = None
        self.state_after: Optional[str] = None

    def get_path(self):
        ans = []
        node = self
        while not node.is_root():
            ans.append(node.last_action)
            node = node.parent
        return "\n".join(reversed(ans))

    def get_info(self):
        info_dict = super().get_info()
        if not self.is_root():
            info_dict["last_action"] = self.last_action
            info_dict["prm_value"] = self.prm_value
        else:
            info_dict["text_state"] = self.text_state
        return info_dict

    def __str__(self):
        if self.is_root():
            return "root: {}".format(self.text_state)
        else:
            return "action: {}, value: {:.3f}, prior: {:.3f}".format(self.last_action, self.value, self.prior_p)


def get_root(node: Node):
    while not node.is_root():
        node = node.parent
    return node


class SearchTree:
    """
    Overview:
        MCTS search process.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg

        self._num_simulations = self._cfg.get("num_simulations", 20)

        # UCB formula
        self._pb_c_base = self._cfg.get("pb_c_base", 19652)  # 19652
        self._pb_c_init = self._cfg.get("pb_c_init", 1.25)  # 1.25

        # Root prior exploration noise.
        self._root_dirichlet_alpha = self._cfg.get("root_dirichlet_alpha", 0.3)  # 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self._root_noise_weight = self._cfg.get("root_noise_weight", 0.25)  # 0.25

        self.root = None

        self.answers = set()
        self.wrong_answers = set()
        self.visited_paths = None

        self.no_terminal_reward = self._cfg.get("no_terminal_reward", True)
        self.mask_non_terminal_node_value = self._cfg.get("mask_non_terminal_node_value", False)

        self._init_critic_value = self._cfg.get("init_critic_value", True)

        self._completion_tokens = 0

        self.model_names = self._cfg.get("model_names", [])
        self.direct_io = self._cfg.get("direct_io", 0)
        self.max_actions = self._cfg.get("max_actions", 0)

        self._tree_records: Dict[int, Dict[str, Any]] = {}
        self._tree_snapshot: Optional[Dict[str, Any]] = None
        self._next_node_id: int = 0
        self._root_node_id: Optional[int] = None
        self._tree_metadata: Dict[str, Any] = {}

    @property
    def num_generated_token(self):
        return self._completion_tokens

    def _reset_tree_tracking(self) -> None:
        self._tree_records = {}
        self._tree_snapshot = None
        self._next_node_id = 0
        self._root_node_id = None
        self._tree_metadata = {}

    def _allocate_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _register_root(self, node: LanguageNode, env: CoTEnv) -> None:
        if getattr(node, "node_id", None) is not None:
            return
        node_id = self._allocate_node_id()
        node.node_id = node_id
        self._root_node_id = node_id
        state_repr = node.text_state
        record = {
            "node_id": node_id,
            "parent_id": None,
            "action": None,
            "raw_action": None,
            "prob": 1.0,
            "initial_value": float(node._initial_value),
            "parent_value": float(node._parent_value),
            "num_generated_token": 0,
            "model_name": "",
            "finish_reason": None,
            "terminated": False,
            "state_before": state_repr,
            "state_after": state_repr,
            "depth": 0,
            "children": [],
            "step_metrics": None,
            "answer": env.answer,
            "reward_history": [],
            "token_history": [],
            "prob_history": [],
            "model_history": [],
            "question": getattr(env, "question", None),
        }
        self._tree_records[node_id] = record

    def _register_child_node(
        self,
        child: LanguageNode,
        parent: LanguageNode,
        action_dict: Dict[str, Any],
        state_before: Optional[str],
    ) -> None:
        if getattr(child, "node_id", None) is not None:
            return
        child_id = self._allocate_node_id()
        child.node_id = child_id
        parent_id = getattr(parent, "node_id", None)
        parent_depth = self._tree_records.get(parent_id, {}).get("depth", -1) if parent_id is not None else -1
        record = {
            "node_id": child_id,
            "parent_id": parent_id,
            "action": child.last_action,
            "raw_action": action_dict.get("raw_action"),
            "prob": float(action_dict.get("prob", child.prior_p)),
            "initial_value": float(child._initial_value),
            "parent_value": float(child._parent_value),
            "num_generated_token": int(child.num_generated_token or 0),
            "model_name": child.model_name,
            "finish_reason": child.finish_reason,
            "terminated": bool(child.terminated),
            "state_before": state_before,
            "state_after": None,
            "depth": parent_depth + 1,
            "children": [],
            "step_metrics": None,
            "answer": None,
            "reward_history": [],
            "token_history": [],
            "prob_history": [],
            "model_history": [],
        }
        self._tree_records[child_id] = record
        if parent_id is not None and parent_id in self._tree_records:
            self._tree_records[parent_id]["children"].append(child_id)

    def _update_node_after_step(self, node: LanguageNode, env: CoTEnv) -> None:
        node_id = getattr(node, "node_id", None)
        if node_id is None:
            return
        record = self._tree_records.get(node_id)
        if record is None:
            return
        record["terminated"] = bool(node.terminated)
        record["state_after"] = env.get_state(model_name="raw")
        record["answer"] = env.answer
        record["reward_history"] = [float(x) for x in env.reward_history]
        record["token_history"] = [int(x) for x in env.token_history]
        record["prob_history"] = [float(x) for x in env.prob_history]
        record["model_history"] = list(env.model_history)
        record["depth"] = len(env.action_history)
        last_record = env.latency_recorder.last_step_record
        if last_record is not None:
            record["step_metrics"] = {
                "total": float(last_record.total),
                "lm": float(last_record.lm),
                "rm": float(last_record.rm),
                "wait": float(last_record.wait),
                "num_tokens": int(last_record.num_tokens),
                "prob": float(last_record.prob),
                "model": last_record.model,
            }
        else:
            record["step_metrics"] = None

    def _finalize_tree_snapshot(self, beam_size: int, max_step: int) -> None:
        nodes = [copy.deepcopy(record) for _, record in sorted(self._tree_records.items(), key=lambda x: x[0])]
        metadata = {
            "beam_size": beam_size,
            "max_step": max_step,
            "config": {
                "max_actions": self.max_actions,
                "direct_io": self.direct_io,
                "model_names": list(self.model_names),
            },
        }
        self._tree_snapshot = {
            "root_id": self._root_node_id,
            "nodes": nodes,
            "metadata": metadata,
        }

    def get_tree_snapshot(self) -> Optional[Dict[str, Any]]:
        if self._tree_snapshot is None:
            return None
        return copy.deepcopy(self._tree_snapshot)

    def clear_node(self, node):
        assert node is not None
        node.clear()
        for child in node.children.values():
            self.clear_node(child)
    
    def _group_by_prefix(self, nodes_and_envs):
        """按前缀分组节点，为KV Cache复用做准备"""
        from collections import defaultdict
        prefix_groups = defaultdict(list)
        
        for node, env in nodes_and_envs:
            # 使用环境的前缀key进行分组
            prefix_key = env.get_prefix_key()
            prefix_groups[prefix_key].append((node, env))
        
        return prefix_groups
    
    def _batch_expand_nodes(self, nodes_to_expand, reward_model_fn):
        """批量扩展节点（未来可实现真正的批量LM/RM调用）"""
        expanded_results = []
        
        # 当前实现：串行处理，但结构为批量处理准备
        # TODO: 实现真正的批量LM调用
        for node, env in nodes_to_expand:
            try:
                # 未来这里可以改为批量调用
                self._expand_leaf_node(node, env, reward_model_fn)
                expanded_results.append((node, env, True))
            except Exception as e:
                import traceback
                traceback.print_exc()
                expanded_results.append((node, env, False))
        
        return expanded_results

    def beam_search(
        self,
        simulate_env: CoTEnv,
        beam_size: int,
        max_step: int,
        reward_model_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        """Beam Search implementation
        Args:
            simulate_env: The environment to simulate the search.
            beam_size: beam_size
            max_step: The maximum number of steps to search.
            reward_model_fn: The reward model function to evaluate the state.
        """
        if max_step == 1:
            assert self.direct_io
        
        import time
        
        self._reset_tree_tracking()
        self.root = None

        api_call_completion_tokens = 0
        _, info = simulate_env.reset(update_legal_action=True)
        api_call_completion_tokens += info["api_completion_token"]
        root = LanguageNode(text_state=simulate_env.get_state(model_name='raw'))
        self._register_root(root, simulate_env)
        self._expand_leaf_node(root, simulate_env, reward_model_fn)
        self.root = root

        # 优化：明确分离completed和active beams（参考VLLM）
        completed_beams = []  # 已完成的beams
        active_beams = [(-root._initial_value, -root._initial_value, -root._parent_value, root, simulate_env.copy())]  # 活跃的beams

        search_latency_recorder = SearchLatencyRecorder()

        for i in range(max_step + 1):
            # Start measuring step latency for this iteration
            step_start_time = time.time()
            
            if not active_beams:
                break
            
            # 准备候选池
            candidate_pool = []
            
            # 第一步：收集所有子节点到候选池
            for cur_neg_q_plus_a, cur_neg_v, cur_neg_parent_v, cur_node, cur_env in active_beams:
                if cur_node.terminated:
                    # 终止节点移到completed列表
                    completed_beams.append((cur_neg_q_plus_a, cur_neg_v, cur_neg_parent_v, cur_node, cur_env))
                else:
                    # 收集所有子节点
                    assert (len(cur_node.children) > 0), "in beam search you should expand this non-terminal node at first."

                    if self.direct_io:
                        for child_idx, child in cur_node.children.items():
                            new_env = cur_env.copy()
                            candidate_pool.append((
                                -child._initial_value,  # q_plus_a
                                -child._initial_value,  # value
                                -child._parent_value,   # parent_value
                                child,
                                new_env
                            ))
                    else:
                        for action, child in cur_node.children.items():
                            new_env = cur_env.copy()
                            candidate_pool.append((
                                -child._initial_value,  # q_plus_a
                                -child._initial_value,  # value
                                -child._parent_value,   # parent_value
                                child,
                                new_env
                            ))
            
            # 第二步：从候选池中选择top-k作为新的active beams
            # Snapshot all candidates before trimming (for full beam recording)
            candidates_snapshot = list(candidate_pool)
            
            if len(candidate_pool) > beam_size:
                active_beams = heapq.nsmallest(beam_size, candidate_pool)
            else:
                active_beams = candidate_pool
                        
            if not active_beams:
                break

            # 第三步：批量扩展active beams（长期优化）
            # 收集需要扩展的节点信息
            nodes_to_expand = []
            stepped_nodes = []
            kept_env_ids = set()
            
            for q_plus_a, value, parent_value, node, new_env in active_beams:
                kept_env_ids.add(id(new_env))
                if not node.terminated:
                    nodes_to_expand.append((node, new_env))
            
            # 优化：按前缀分组（为KV Cache复用准备）
            prefix_groups = self._group_by_prefix(nodes_to_expand)
            
            # 批量执行step和expand
            expanded_env_copies = []
            if nodes_to_expand:
                # 分组处理：相同前缀的节点可以共享KV Cache
                for prefix_key, group_nodes in prefix_groups.items():
                    # TODO: 未来可以为同一前缀组实现共享KV Cache的批量生成
                    # shared_kv_cache = lm.prefill(prefix_key)
                    
                    for node, new_env in group_nodes:
                        _, _, terminated, truncated, info = new_env.step(
                            node.last_action, 
                            update_legal_action=self.direct_io == 0, 
                            model_name=node.model_name,
                            reward=node._initial_value, 
                            num_token=node.num_generated_token, 
                            prob=node.prior_p,
                        )
                        api_call_completion_tokens += info["api_completion_token"]
                        if terminated or truncated:
                            node.set_as_terminate_node()
                        else:
                            self._expand_leaf_node(node, new_env, reward_model_fn)
                        expanded_env_copies.append(new_env)
                        stepped_nodes.append((node, new_env))

            # 优化：更准确的延迟计算和性能追踪
            step_latency_for_iter = time.time() - step_start_time
            
            # 收集所有扩展节点的模型时间
            lm_latencies = []
            rm_latencies = []
            for env_copy in expanded_env_copies:
                last_record = env_copy.latency_recorder.last_step_record
                if last_record is not None:
                    lm_latencies.append(float(last_record.lm))
                    rm_latencies.append(float(last_record.rm))
            
            # 计算模型时间（使用最大值而非平均值，因为串行执行）
            max_lm_time = max(lm_latencies) if lm_latencies else 0.0
            max_rm_time = max(rm_latencies) if rm_latencies else 0.0
            total_model_time = sum(lm_latencies) + sum(rm_latencies)  # 实际消耗的总模型时间
            
            # 计算并行度和等待时间
            # 如果是真正的批量处理，理想时间应该是max而非sum
            ideal_batch_time = max_lm_time + max_rm_time
            actual_wait_time = max(0.0, step_latency_for_iter - ideal_batch_time)
            
            # 用于记录的step_latency（实际消耗）
            adjusted_step_latency = step_latency_for_iter
            
            # Store step latency in all active env copies for this iteration
            for q_plus_a, value, parent_value, node, env_copy in active_beams:
                tokens = int(env_copy.token_history[-1]) if env_copy.token_history else 0
                prob = float(env_copy.prob_history[-1]) if env_copy.prob_history else 0.0
                model_name = env_copy.model_history[-1] if env_copy.model_history else getattr(node, "model_name", "")
                env_copy.latency_recorder.record_step(
                    total=adjusted_step_latency,
                    wait=actual_wait_time,
                    tokens=tokens,
                    prob=prob,
                    model=model_name or "",
                )

            for node, env_copy in stepped_nodes:
                self._update_node_after_step(node, env_copy)

            # Record complete iteration latency information only for meaningful iterations
            if len(expanded_env_copies) > 0 or len(candidates_snapshot) > 0:
                beam_details: List[BeamDetail] = []
                parent_child_mapping: Dict[str, List[str]] = {}

                kept_env_set = set(id(env) for (_, _, _, _, env) in active_beams)

                for beam_idx, (q_plus_a, value, parent_value, node, env) in enumerate(candidates_snapshot):
                    kept = id(env) in kept_env_set

                    parent_node_id = str(id(node.parent)) if node.parent else "root"
                    current_node_id = str(id(node))
                    parent_child_mapping.setdefault(parent_node_id, []).append(current_node_id)

                    last_record = env.latency_recorder.last_step_record if kept else None
                    if last_record is None:
                        total_time = 0.0
                        lm_latency = 0.0
                        rm_latency = 0.0
                        step_wait = 0.0
                        num_tokens = 0
                        prob = 0.0
                        model_name = ""
                    else:
                        total_time = float(last_record.total)
                        lm_latency = float(last_record.lm)
                        rm_latency = float(last_record.rm)
                        step_wait = float(last_record.wait)
                        num_tokens = int(last_record.num_tokens)
                        prob = float(last_record.prob)
                        model_name = last_record.model

                    lm_tokens = num_tokens
                    lm_time_per_token = float(lm_latency / max(1, lm_tokens)) if lm_tokens else 0.0
                    rm_time_per_token = float(rm_latency / max(1, lm_tokens)) if lm_tokens else 0.0

                    beam_details.append(
                        BeamDetail(
                            beam_idx=beam_idx,
                            node_id=current_node_id,
                            parent_node_id=parent_node_id,
                            value=float(value),
                            parent_value=float(parent_value),
                            total_time=total_time,
                            lm_latency=lm_latency,
                            rm_latency=rm_latency,
                            step_wait=step_wait,
                            num_tokens=num_tokens,
                            prob=prob,
                            lm_tokens=lm_tokens,
                            lm_time_per_token=lm_time_per_token,
                            rm_time_per_token=rm_time_per_token,
                            kept=kept,
                            is_terminal=node.terminated,
                            text_state=node.text_state if hasattr(node, "text_state") else "",
                        )
                    )

                iteration_record = IterationLatencyRecord(
                    iteration=i,
                    step_latency=adjusted_step_latency,
                    step_wait=actual_wait_time,
                    num_active_beams=len(active_beams),
                    num_completed_beams=len(completed_beams),
                    num_expanded_beams=len(expanded_env_copies),
                    num_prefix_groups=len(prefix_groups),
                    total_model_time=total_model_time,
                    ideal_batch_time=ideal_batch_time,
                    parallelism_efficiency=(ideal_batch_time / total_model_time * 100) if total_model_time > 0 else 0.0,
                    beam_details=beam_details,
                    parent_child_mapping=parent_child_mapping,
                )
                search_latency_recorder.record(iteration_record)
            else:
                # 如果没有活跃节点，提前结束循环
                break

        # Calculate question latency via centralized recorder for consistency
        question_latency = simulate_env.latency_recorder.finish_question()
        simulate_env.question_latency = question_latency
        complete_latency_record = search_latency_recorder.to_dicts()

        # 优化：合并completed和active beams，选择最优的beam_size个（参考VLLM）
        all_beams = completed_beams + active_beams
        
        # 从所有beams中选择top beam_size个
        if len(all_beams) > beam_size:
            end_nodes = heapq.nsmallest(beam_size, all_beams)
        else:
            end_nodes = all_beams
        
        self._finalize_tree_snapshot(beam_size, max_step)

        # 如果仍然没有结果（极端情况），返回空列表
        if not end_nodes:
            return []
        
        traj_list = []
        for i, (neg_e_q_plus_a, neg_e_v, neg_e_parent_v, e_node, e_env) in enumerate(end_nodes):
            # compute per-trajectory api completion tokens from env token history
            try:
                per_api_completion_tokens = int(sum(e_env.token_history)) if hasattr(e_env, 'token_history') and e_env.token_history is not None else 0
            except Exception:
                per_api_completion_tokens = 0

            # compute per-trajectory tree completion tokens by traversing node parents and summing num_generated_token
            tree_tokens = 0
            try:
                node_ptr = e_node
                while node_ptr is not None and not node_ptr.is_root():
                    if hasattr(node_ptr, 'num_generated_token') and node_ptr.num_generated_token is not None:
                        tree_tokens += int(node_ptr.num_generated_token)
                    node_ptr = node_ptr.parent
            except Exception:
                tree_tokens = 0

            # Calculate step wait time (step_latency - step_lm_latency - step_rm_latency)
            recorder = e_env.latency_recorder
            step_latency = list(recorder.step_latency_history)
            step_lm_latency = list(recorder.step_lm_latency_history)
            step_rm_latency = list(recorder.step_rm_latency_history)
            step_wait = list(recorder.step_wait_history)
            
            # 验证时间一致性
            for j in range(len(step_latency)):
                lm_time = step_lm_latency[j] if j < len(step_lm_latency) else 0.0
                rm_time = step_rm_latency[j] if j < len(step_rm_latency) else 0.0
                wait_time = step_wait[j] if j < len(step_wait) else 0.0
                calculated_total = lm_time + rm_time + wait_time
                
                # 确保一致性，如果有微小差异则调整
                if abs(step_latency[j] - calculated_total) > 1e-6:
                    step_latency[j] = calculated_total

            traj_list.append({
                "path_idx": i,
                "text": e_env.answer,
                "value": -neg_e_v,
                "parent_value": -neg_e_parent_v,
                "q_plus_a": -neg_e_q_plus_a,
                "api_completion_tokens": per_api_completion_tokens,
                "tree_completion_tokens": tree_tokens,
                "step_latency": step_latency,
                "step_lm_latency": step_lm_latency,
                "step_rm_latency": step_rm_latency,
                "step_wait": step_wait,
                "question_latency": question_latency,
                "total_unit_latency": float(sum(step_latency)),
                "reward_history": e_env.reward_history,
                "token_history": e_env.token_history,
                "prob_history": e_env.prob_history,
                "model_history": e_env.model_history,
                # num_generated_token is hard to compute for each single answer
            })

        # note: api_call_completion_tokens and self._completion_tokens represent global totals
        # If desired, they can be exposed separately; currently per-trajectory fields are set above.
        
        # Add complete latency record to each trajectory
        for traj in traj_list:
            traj["complete_latency_record"] = complete_latency_record
            traj["question_latency"] = question_latency
        
        return traj_list

    def _select_child(self, node: LanguageNode, simulate_env: CoTEnv) -> Tuple[Union[int, float], Node]:
        """
        Overview:
            Select the child with the highest UCB score.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        Returns:
            - action (:obj:`Int`): choose the action with the highest ucb score.
            - child (:obj:`Node`): the child node reached by executing the action with the highest ucb score.
        """

        action = None
        child = None
        best_score = -9999999

        for action_tmp, child_tmp in node.children.items():
            ucb_score = self._ucb_score(node, child_tmp)
            score = ucb_score
            if score > best_score:
                best_score = score
                action = action_tmp
                child = child_tmp

        if child is None:
            child = node  # child==None, node is leaf node in play_with_bot_mode.

        return action, child

    def _select_by_prior(self, node: Node, simulate_env: CoTEnv):
        data_tmp = [(x_action, x_node.prior_p) for x_action, x_node in node.children.items()]
        action_list, prior_list = list(zip(*data_tmp))
        chosen_action = np.random.choice(action_list, p=np.array(prior_list))
        chosen_node = node.children[chosen_action]

        return chosen_action, chosen_node

    def _expand_leaf_node(
        self,
        node: Node,
        simulate_env: CoTEnv,
        rm_call: Optional[Callable] = None,
    ) -> float:
        """
        Overview:
            expand the node with the rm_call.
        Arguments:
            - node (:obj:`Class Node`): current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): the class of simulate env.
            - rm_call (:obj:`Function`): the Callable to compute the state value.
        Returns:
            - leaf_value (:obj:`Bool`): the leaf node's value.
        """
        """
        action_probs_dict, leaf_value = rm_call(simulate_env)
        for action, prior_p in action_probs_dict.items():
            if action in simulate_env.legal_actions:
                node.children[action] = Node(parent=node, prior_p=prior_p)
        """

        text_state = simulate_env.get_state(model_name='raw')
        if not self._init_critic_value:
            leaf_value = rm_call(text_state)
        else:
            leaf_value = node._initial_value
            assert len(simulate_env.legal_actions) > 0
            if self.direct_io:
                prms = [[0.0] for _ in simulate_env.legal_actions]
            else:
                prm_inputs = [(simulate_env.question, simulate_env.answer + x["action"]) for x in simulate_env.legal_actions]
                for i in range(2):
                    try:
                        # Measure RM latency
                        import time
                        rm_start_time = time.time()
                        prms = rm_call(prm_inputs)
                        rm_latency = time.time() - rm_start_time
                        # Store RM latency in environment
                        simulate_env.latency_recorder.record_rm_latency(rm_latency)
                        break
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        # prms = [[0.0] for _ in simulate_env.legal_actions]
            child_values = []
            for act, rs in zip(simulate_env.legal_actions, prms):
                if len(simulate_env.action_history) + 1 != len(rs):
                    logger.warning(f"PRM value length not match with action history. len(prm)={len(rs)}, "
                                   f"len(action_history)={len(simulate_env.action_history)}\ns:\n{text_state}\na:\n{act}\nrs:{rs}")
                    try:
                        prm = rm_call([(simulate_env.question, simulate_env.answer + x["action"]) for x in [act]], verbose=False, legal_action=[act])
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                    child_values.append(0.0)
                elif len(rs) == 0:
                    logger.warning(f"Empty PRM value for: \nState: \n{text_state} \naction: \n{act}, will be set to 0.0")
                    child_values.append(0.0)
                else:
                    # prm-last
                    child_values.append(rs[-1])  # PRM get last r as single reward, [0.9783847332000732, 0.9621075391769409]
                    # # prm-min
                    # child_values.append(min(rs))
                    # # prob-prm
                    # child_values.append(act['prob'])

        assert len(node.children) == 0
        for i, action_dict in enumerate(simulate_env.legal_actions):
            action, prob = action_dict["action"], action_dict["prob"]
            model_name = action_dict["model_name"]

            if self._init_critic_value:
                child_value = child_values[i]
            else:
                # XXX(ziyu): consider turn off this branch, i.e. always assume
                #  `self._init_critic=True`, since with LLM
                child_value = 0.0

            child_node = LanguageNode(
                parent=node,
                prior_p=prob,
                text_state=text_state,
                last_action=action,
                initial_value=child_value,
                parent_value=leaf_value,
                num_generated_token=action_dict["num_token"],
                model_name=model_name,
                finish_reason=action_dict.get("finish_reason"),
                raw_action=action_dict.get("raw_action"),
            )

            if self.direct_io:
                node.children[i] = child_node
            else:
                node.children[action] = child_node

            if simulate_env._next_state_terminated[action]:
                child_node.set_as_terminate_node()

            self._register_child_node(child_node, node, action_dict, text_state)
        if len(node.children) == 0:
            print_rank_0("Prune all current children at node {}".format(node.last_action))

        # collect num tokens
        if not node.has_collected_token_num:
            self._completion_tokens += sum(c.num_generated_token for c in node.children.values())
            node.has_collected_token_num = True
        else:
            raise RuntimeError("Token number has been collected again.")

        return leaf_value

    def _ucb_score(self, parent: Node, child: Node) -> float:
        """
        Overview:
            Compute UCB score. The score for a node is based on its value, plus an exploration bonus based on the prior.
        Arguments:
            - parent (:obj:`Class Node`): Current node.
            - child (:obj:`Class Node`): Current node's child.
        Returns:
            - score (:obj:`float`): UCB score.
        """
        pb_c = math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base) + self._pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior_p
        value_score = child.value
        return prior_score + value_score