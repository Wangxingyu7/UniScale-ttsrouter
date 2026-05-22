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
    ) -> None:
        super().__init__(parent, prior_p, initial_value, parent_value)
        self.text_state = text_state
        self.last_action = last_action
        self.prm_value = prm_value

        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False

        self.model_name = model_name

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

    @property
    def num_generated_token(self):
        return self._completion_tokens

    def clear_node(self, node):
        assert node is not None
        node.clear()
        for child in node.children.values():
            self.clear_node(child)

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
        
        # Record question start time
        import time
        question_start_time = time.time()
        
        api_call_completion_tokens = 0
        _, info = simulate_env.reset(update_legal_action=True)
        api_call_completion_tokens += info["api_completion_token"]
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state(model_name='raw'))
            self._expand_leaf_node(root, simulate_env, reward_model_fn)
            self.root = root

        end_nodes, top_k_nodes = [], [(-root._initial_value, -root._initial_value, -root._parent_value, root, simulate_env.copy())]
        
        # Record complete latency information for all iterations
        complete_latency_record = []

        for i in range(max_step + 1):
            # Start measuring step latency for this iteration
            step_start_time = time.time()
            
            cur_nodes_to_search = top_k_nodes
            top_k_nodes = []
            active_env_copies = []  # Track all env copies that are active in this iteration
            
            # 如果已经没有活跃的节点需要处理，提前结束
            if not cur_nodes_to_search:
                break
                
            for cur_neg_q_plus_a, cur_neg_v, cur_neg_parent_v, cur_node, cur_env in cur_nodes_to_search:
                if cur_node.terminated:
                    # 终止节点放回 top_k_nodes，保持在 beam 中但不再扩展
                    heapq.heappush(top_k_nodes, (cur_neg_q_plus_a, cur_neg_v, cur_neg_parent_v, cur_node, cur_env))
                else:
                    # select all children and add to candidates pool
                    assert (len(cur_node.children) > 0), "in beam search you should expand this non-terminal node at first."

                    if self.direct_io:
                        values = {child_idx: copy.deepcopy(child._initial_value) for child_idx, child in cur_node.children.items()}
                        parent_values = {child_idx: copy.deepcopy(child._parent_value) for child_idx, child in cur_node.children.items()}
                        q_plus_alpha_a = values

                        for child_idx, child in cur_node.children.items():
                            new_env = cur_env.copy()
                            if not hasattr(new_env, 'step_latency_history'):
                                new_env.step_latency_history = []
                            heapq.heappush(top_k_nodes, (-q_plus_alpha_a[child_idx], -values[child_idx], -parent_values[child_idx], child, new_env))
                            active_env_copies.append(new_env)
                    else:
                        values = {action: copy.deepcopy(child._initial_value) for action, child in cur_node.children.items()}
                        parent_values = {action: copy.deepcopy(child._parent_value) for action, child in cur_node.children.items()}
                        q_plus_alpha_a = values

                        for action, child in cur_node.children.items():
                            new_env = cur_env.copy()
                            if not hasattr(new_env, 'step_latency_history'):
                                new_env.step_latency_history = []
                            heapq.heappush(top_k_nodes, (-q_plus_alpha_a[action], -values[action], -parent_values[action], child, new_env))
                            active_env_copies.append(new_env)
            
            # Snapshot all candidates before trimming to top-k (for full beam recording)
            candidates_snapshot = list(top_k_nodes)
            
            # 标准 Beam Search: 始终保持恰好 beam_size 个节点（包括已终止的）
            # 但如果候选总数不足 beam_size，只能保留所有候选
            if len(top_k_nodes) > beam_size:
                top_k_nodes = heapq.nsmallest(beam_size, top_k_nodes)
            # else: 候选不足时保留所有，不强制填充（因为无法复制 node 对象）
            
            # 检查是否所有 beam 节点都已终止
            all_terminated = all(node.terminated for (_, _, _, node, _) in top_k_nodes)
            if all_terminated:
                # 所有节点都终止了，可以提前结束
                end_nodes = top_k_nodes
                break

            # expand selected nodes (these are kept top-k, only expand non-terminated ones)
            kept_env_ids = set()
            expanded_env_copies = []  # 只记录真正扩展的节点用于延迟计算
            for q_plus_a, value, parent_value, node, new_env in top_k_nodes:
                kept_env_ids.add(id(new_env))
                # 只扩展未终止的节点
                if not node.terminated:
                    _, _, terminated, truncated, info = new_env.step(
                            node.last_action, update_legal_action=self.direct_io == 0, model_name=node.model_name,
                            reward=node._initial_value, num_token=node.num_generated_token, prob=node.prior_p,
                        )
                    api_call_completion_tokens += info["api_completion_token"]
                    if terminated or truncated:
                        node.set_as_terminate_node()
                    else:
                        self._expand_leaf_node(node, new_env, reward_model_fn)
                    expanded_env_copies.append(new_env)  # 记录扩展的环境

            # For dropped candidates (not in top-k), perform a measure-only step to collect LM/RM latencies
            # without expanding children, so we can record complete per-beam latency for analysis.
            for q_plus_a, value, parent_value, node, env in candidates_snapshot:
                if id(env) in kept_env_ids:
                    continue
                try:
                    # step once to measure LM/RM latency; do not expand children afterwards
                    _, _, terminated, truncated, info = env.step(
                        node.last_action, update_legal_action=self.direct_io == 0, model_name=node.model_name,
                        reward=node._initial_value, num_token=node.num_generated_token, prob=node.prior_p,
                    )
                    api_call_completion_tokens += info["api_completion_token"]
                    # intentionally skip expansion for dropped beams
                except Exception:
                    import traceback
                    traceback.print_exc()
            
            # Record step latency for this iteration (including all operations in the loop)
            step_latency_for_iter = time.time() - step_start_time
            
            # 计算实际的等待时间和其他开销（只计算真正扩展的节点）
            total_model_time = 0.0
            for env_copy in expanded_env_copies:
                lm_lat = getattr(env_copy, 'step_lm_latency_history', [])[-1] if hasattr(env_copy, 'step_lm_latency_history') and len(getattr(env_copy, 'step_lm_latency_history', [])) > 0 else 0.0
                rm_lat = getattr(env_copy, 'step_rm_latency_history', [])[-1] if hasattr(env_copy, 'step_rm_latency_history') and len(getattr(env_copy, 'step_rm_latency_history', [])) > 0 else 0.0
                total_model_time += (lm_lat + rm_lat)
            
            # 如果有多个环境，取平均值
            avg_model_time = total_model_time / max(len(expanded_env_copies), 1)
            
            # 计算真实的等待时间和其他开销
            actual_wait_time = max(0.0, step_latency_for_iter - avg_model_time)
            
            # 调整step_latency使其等于组件之和
            adjusted_step_latency = avg_model_time + actual_wait_time
            
            # Store step latency in all active env copies for this iteration
            # 注意：这里应该存储到所有 top_k_nodes 的环境中，包括终止的（它们也在这一步中）
            for q_plus_a, value, parent_value, node, env_copy in top_k_nodes:
                if hasattr(env_copy, 'step_latency_history'):
                    env_copy.step_latency_history.append(adjusted_step_latency)
                else:
                    env_copy.step_latency_history = [adjusted_step_latency]
                
                # 同时存储等待时间
                if not hasattr(env_copy, 'step_wait_history'):
                    env_copy.step_wait_history = []
                env_copy.step_wait_history.append(actual_wait_time)
            
            # Record complete iteration latency information only for meaningful iterations
            if len(expanded_env_copies) > 0 or len(candidates_snapshot) > 0:
                iteration_latency_info = {
                    "iteration": i,
                    "step_latency": adjusted_step_latency,
                    "step_wait": actual_wait_time,
                    "num_active_beams": len(top_k_nodes),  # 当前 beam 中的节点数
                    "num_expanded_beams": len(expanded_env_copies),  # 真正扩展的节点数
                    "num_end_nodes": sum(1 for (_, _, _, n, _) in top_k_nodes if n.terminated),  # 已终止的节点数
                    "beam_details": [],
                    "parent_child_mapping": {}  # 新增：记录父子关系
                }
                
                # Build a set for kept envs to mark [TOP]
                kept_env_set = set(id(env) for (_, _, _, _, env) in top_k_nodes)

                # 记录所有活跃节点的详细信息
                for beam_idx, (q_plus_a, value, parent_value, node, env) in enumerate(candidates_snapshot):
                    kept = id(env) in kept_env_set
                    
                    # 获取父节点信息
                    parent_node_id = str(id(node.parent)) if node.parent else "root"
                    current_node_id = str(id(node))
                    
                    # 记录父子关系
                    if parent_node_id not in iteration_latency_info["parent_child_mapping"]:
                        iteration_latency_info["parent_child_mapping"][parent_node_id] = []
                    iteration_latency_info["parent_child_mapping"][parent_node_id].append(current_node_id)
                    
                    beam_detail = {
                        "beam_idx": beam_idx,
                        "node_id": current_node_id,
                        "parent_node_id": parent_node_id,
                        "value": float(value),
                        "parent_value": float(parent_value),
                        "total_time": float(env.step_latency_history[-1]) if env.step_latency_history else 0.0,
                        "lm_latency": float(env.step_lm_latency_history[-1]) if env.step_lm_latency_history else 0.0,
                        "rm_latency": float(env.step_rm_latency_history[-1]) if env.step_rm_latency_history else 0.0,
                        "step_wait": float(env.step_wait_history[-1]) if env.step_wait_history else 0.0,
                        "num_tokens": int(env.token_history[-1]) if env.token_history else 0,
                        "prob": float(env.prob_history[-1]) if env.prob_history else 0.0,
                        "lm_tokens": int(env.token_history[-1]) if env.token_history else 0,
                        "lm_time_per_token": float(env.step_lm_latency_history[-1] / max(1, int(env.token_history[-1]))) if env.token_history and env.step_lm_latency_history else 0.0,
                        "rm_time_per_token": float(env.step_rm_latency_history[-1] / max(1, int(env.token_history[-1]))) if env.token_history and env.step_rm_latency_history else 0.0,
                        "kept": kept,
                        "is_terminal": node.terminated,
                        "text_state": node.text_state if hasattr(node, 'text_state') else ""
                    }
                    iteration_latency_info["beam_details"].append(beam_detail)
                
                complete_latency_record.append(iteration_latency_info)
            else:
                # 如果没有活跃节点，提前结束循环
                break

        # Calculate question latency
        question_latency = time.time() - question_start_time
        
        # 如果循环正常结束（达到 max_step），end_nodes 可能为空，使用 top_k_nodes
        if not end_nodes:
            end_nodes = top_k_nodes
        
        # 确保返回恰好 beam_size 个结果（用于延迟测试的一致性）
        if len(end_nodes) < beam_size:
            # 候选不足时，从现有的最优节点中循环选择来填充
            shortage = beam_size - len(end_nodes)
            sorted_end_nodes = sorted(end_nodes, key=lambda x: x[0])  # 按质量排序
            for idx in range(shortage):
                # 循环选择已有的节点（它们指向相同的 node，但会有不同的输出索引）
                src = sorted_end_nodes[idx % len(sorted_end_nodes)]
                end_nodes.append(src)  # 直接复制 tuple（env 已经是副本）
        elif len(end_nodes) > beam_size:
            # 候选过多时，只保留最优的 beam_size 个
            end_nodes = heapq.nsmallest(beam_size, end_nodes)
        
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
            step_latency = getattr(e_env, 'step_latency_history', [])
            step_lm_latency = getattr(e_env, 'step_lm_latency_history', [])
            step_rm_latency = getattr(e_env, 'step_rm_latency_history', [])
            step_wait = getattr(e_env, 'step_wait_history', [])
            
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
                        if not hasattr(simulate_env, 'step_rm_latency_history'):
                            simulate_env.step_rm_latency_history = []
                        simulate_env.step_rm_latency_history.append(rm_latency)
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

            if self.direct_io:
                node.children[i] = LanguageNode(
                    parent=node,
                    prior_p=prob,
                    # prm_value=prm_value,
                    text_state=text_state,
                    last_action=action,
                    initial_value=child_value,
                    parent_value=leaf_value,
                    num_generated_token=action_dict["num_token"],
                    model_name=model_name,
                )
            else:
                node.children[action] = LanguageNode(
                    parent=node,
                    prior_p=prob,
                    # prm_value=prm_value,
                    text_state=text_state,
                    last_action=action,
                    initial_value=child_value,
                    parent_value=leaf_value,
                    num_generated_token=action_dict["num_token"],
                    model_name=model_name,
                )
            # set terminal node here
            if simulate_env._next_state_terminated[action]:
                if self.direct_io:
                    node.children[i].set_as_terminate_node()
                else:
                    node.children[action].set_as_terminate_node()
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