import os
import copy
import json
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from textwrap import dedent

import jsonlines
import numpy as np
import tree
from transformers import AutoTokenizer
import ray
from ray.util.actor_pool import ActorPool


from reason.evaluation.evaluator import RemoteMathEvaluator, TreeSearchSolutionOutput
from reason.evaluation.methods import BeamSearchConfig, beam_search, Task
from reason.inference.lm_call import LMCallingConfig, VLLMRemoteCaller
from reason.inference.rm_call import (
    RMRemoteCaller,
    DummyRewardModelCaller,
    RemoteRewardModelConfig,
    get_prm_special_tokens,
)
from utils import check_process_cnt, assign_tasks, get_model_name, setup_seed, check_lock_timeout

from reason.guided_search.tree import SearchTree

cot_prompt_dict = {
    'llama_official': dedent(
        """\
        Solve the following math problem efficiently and clearly:

        - For simple problems (2 steps or fewer):
        Provide a concise solution with minimal explanation.

        - For complex problems (3 steps or more):
        Use this step-by-step format:

        ## Step 1: [Concise description]
        [Brief explanation and calculations]

        ## Step 2: [Concise description]
        [Brief explanation and calculations]

        ...

        Regardless of the approach, always conclude with:

        Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

        Where [answer] is just the final number or expression that solves the problem.
        """
    ),
    'qwen': dedent(
        """\
        You MUST follow every formatting rule below.  
        If you cannot follow them exactly, reply with `UNABLE TO COMPLY`.
        1. **Reason step-by-step** in clear, complete sentences.\n
        2. **Final Answer**: Place the final result inside `\\boxed{}` using LaTeX. No line breaks inside. If it is a multiple-choice question, choose the correct option(e.g. A or B or C or D).\n
        3. **End Signal**: After the boxed answer, output exactly one line: `FINISHED`.\n\n
        """
    ),
    'default': dedent(
        """\
        You MUST follow every formatting rule below.  
        If you cannot follow them exactly, reply with `UNABLE TO COMPLY`.
        1. **Reason step-by-step** in clear, complete sentences.\n
        2. **Final Answer**: Place the final result inside `\\boxed{}` using LaTeX. No line breaks inside.\n
        3. **End Signal**: After the boxed answer, output exactly one line: `FINISHED`.\n\n
        """
    ),
}

llm_step_tag_dict = {
    'llama': "## Step ",
    'qwen': "\nStep ",
    'default': "\nStep ",
}

sep_dict = {
    'llama': ["## Step"],
    'qwen': ["\nStep"],
    'default': ["\nStep"],
}

stop_str_dict = {
    'llama': ["\\boxed{"],
    'qwen': ["FINISHED"],
    'default': ["FINISHED"],
}

if __name__ == "__main__":
    parser = ArgumentParser()
    # LLM config
    parser.add_argument("--LM", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--add_step_prompt", action="store_true")
    parser.add_argument("--serve_type", type=str, default="vllm", choices=["fastchat", "vllm"])
    parser.add_argument("--cot_prompt", type=str, default="")
    parser.add_argument("--llm_step_tag", type=str, default="")
    parser.add_argument("--stop_str", default=[])
    parser.add_argument("--sep", default=[])
    parser.add_argument("--double_line_break", type=int, default=0)
    # RM config
    parser.add_argument("--RM", type=str, default="dummy")
    parser.add_argument("--rm_device", type=str, default="cuda")
    parser.add_argument("--good_tag", type=str, default="+")
    parser.add_argument("--bad_tag", type=str, default="-")
    parser.add_argument("--prm_step_tag", type=str, default="ки\n")
    parser.add_argument("--prm_format_str", type=str, default="{question} {answer}")
    parser.add_argument("--rm_serve_type", type=str, default="vllm", choices=["fastchat", "vllm"])
    # method config
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--num_sequence", type=int, default=1)
    parser.add_argument("--tree_max_depth", type=int, default=None)
    parser.add_argument("--tree_max_width", type=int, default=None)
    # other config
    parser.add_argument("--task_name", type=str, default="MATH")
    parser.add_argument("--is_few_shot", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--controller_addr", type=str, default="http://localhost:21001")
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--local", type=int, default=0)
    parser.add_argument("--question_parallel_num", type=int, default=0)
    parser.add_argument("--question_max_num", type=int, default=0)
    parser.add_argument("--lock_dir", type=str, default="lock_dir")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--max_time", type=int, default=0)

    args = parser.parse_args()

    if 'llama-3' in args.LM.lower():
        args.cot_prompt = cot_prompt_dict['llama_official']
        args.llm_step_tag = llm_step_tag_dict['llama']
        args.sep = sep_dict['llama']
        args.stop_str = stop_str_dict['llama']
    elif 'qwen' in args.LM.lower():
        args.cot_prompt = cot_prompt_dict['qwen']
        args.llm_step_tag = llm_step_tag_dict['qwen']
        args.sep = sep_dict['qwen']
        args.stop_str = stop_str_dict['qwen']
    else:
        args.cot_prompt = cot_prompt_dict['default']
        args.llm_step_tag = llm_step_tag_dict['default']
        args.sep = sep_dict['default']
        args.stop_str = stop_str_dict['default']


    if args.double_line_break == 1:
        args.sep = ["\n\n"]

    os.makedirs(args.save_dir, exist_ok=True)

    if "dummy" in args.RM:
        assert args.method in ["cot", "best_of_n"]
    if args.tree_max_depth is not None and args.tree_max_width is not None:
        assert args.tree_max_width % args.num_sequence == 0

    args.LM = args.LM.split(',')

    setup_seed(args.seed)
    if args.local:
        print("run in pure local mode for debug only")
        args.num_worker = 1
        ray.init(local_mode=True)
    else:
        ray.init()

    if args.RM.endswith("/"):
        args.RM = args.RM[:-1]
    rm_model_name = args.RM
    rm_model_path = args.RM
    if "dummy" in args.RM:
        rm_config = RemoteRewardModelConfig(
            prm_step_tag=args.prm_step_tag, format_str=args.prm_format_str, model_name=args.RM, controller_addr=args.controller_addr,
            step_tag_id=None, returned_token_ids=None, rm_serve_type=args.rm_serve_type, multi_gpu=args.multi_gpu
        )
        rm_call = DummyRewardModelCaller(rm_config)
    else:
        if args.rm_serve_type == "vllm":
            tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)
            step_tag_id, returned_token_ids = get_prm_special_tokens(rm_model_name, tokenizer)
            if 'pqm' in args.RM:
                prm_format_str = "{question}\n{answer}"
            else:
                prm_format_str = "{question} {answer}"
            rm_config = RemoteRewardModelConfig(
                prm_step_tag=args.prm_step_tag, format_str=prm_format_str, model_name=args.RM, controller_addr=args.controller_addr,
                step_tag_id=step_tag_id, returned_token_ids=returned_token_ids, rm_serve_type=args.rm_serve_type, multi_gpu=args.multi_gpu,
            )
            rm_call = RMRemoteCaller(rm_config, tokenizer=tokenizer)
        else:
            raise NotImplementedError

    llm_step_tags = []
    llm_gen_fns = []
    for i, lm in enumerate(args.LM):
        llm_step_tags.append(args.llm_step_tag)
        model_path = lm
        llm_gen_fns.append(
            VLLMRemoteCaller(
                lm, model_path, args.controller_addr, args.llm_step_tag, apply_chat_template=True, multi_gpu=args.multi_gpu,
                serve_type=args.serve_type, double_line_break=args.double_line_break, model_idx=i
            )
        )

    rm_call = partial(rm_call, model_names=args.LM)

    task = Task(task_name=args.task_name, is_few_shot=args.is_few_shot, model_names=args.LM)


    def route_for_problem(problem_inst):
        lm_idx = problem_inst.get("lm_idx")
        lm_name = problem_inst.get("lm")
        if lm_idx is None and lm_name is not None:
            try:
                lm_idx = next((i for i, m in enumerate(args.LM) if lm_name.lower() in m.lower()), None)
                if lm_idx is None:
                    print(f"Warning: Could not find model {lm_name} in {args.LM}, using default model 0")
                    lm_idx = 0
            except Exception as e:
                print(f"Error matching model {lm_name}: {e}, using default model 0")
                lm_idx = 0
        if lm_idx is None:
            lm_idx = 0  

        beam = problem_inst.get("beam", {})
        dynamic_params = {}
        
        if "QP" in beam:
            dynamic_params["question_parallel_num"] = int(beam["QP"])
        if "CP" in beam:
            dynamic_params["tree_max_width"] = int(beam["CP"])
        if "BS" in beam:
            dynamic_params["beam_size"] = int(beam["BS"])

        return lm_idx, dynamic_params


    def dynamic_solver_fn(problem_inst, lm_calls, rm_call):
        lm_idx, dynamic_params = route_for_problem(problem_inst)
        print(f"Dynamic solver: lm_idx={lm_idx}, dynamic_params={dynamic_params}")

        dyn_gen_config = LMCallingConfig(
            n=dynamic_params.get("beam_size", args.num_sequence),  # 只动态修改beam_size
            temperature=args.temperature,  
            top_k=args.top_k,  
            top_p=args.top_p,  
            max_new_tokens=args.max_new_tokens,  
        )

        dyn_tree_max_width = dynamic_params.get("tree_max_width", args.tree_max_width or 4)
        dyn_beam_size = dynamic_params.get("beam_size", args.num_sequence)

        def _noop_rm_call(qa_pairs, *_, **__):
            # QP=CP=BS=1时，不调用RM，全部奖励为0
            if isinstance(qa_pairs, tuple):
                qa_pairs = [qa_pairs]
            zero_rewards = []
            for pair in qa_pairs:
                answer = ""
                if isinstance(pair, (list, tuple)) and len(pair) > 1:
                    answer = pair[1] or ""
                steps = answer.count(" ки\n")
                if steps <= 0:
                    steps = answer.count("\nStep")
                steps = max(1, steps)
                zero_rewards.append([0.0] * steps)
            return zero_rewards

        # 只有BS=1且CP=1且QP=1时，才用_noop_rm_call
        qp = dynamic_params.get("question_parallel_num", args.question_parallel_num)
        #effective_rm_call = _noop_rm_call if (dyn_beam_size == 1 and dyn_tree_max_width == 1 and qp == 1) else rm_call
        effective_rm_call = rm_call

        dyn_beam_config = BeamSearchConfig(
            task_name=args.task_name,
            tree_max_depth=args.tree_max_depth or 40,  
            tree_max_width=dyn_tree_max_width,  
            beam_size=dyn_beam_size,  
            model_names=args.LM,  
            is_few_shot=args.is_few_shot,
            add_step_prompt=args.add_step_prompt,
            cot_prompt=args.cot_prompt,
            stop_str=args.stop_str if "beam_search" not in args.method else None,
            sep=args.sep,
            direct_io=0 if "beam_search" in args.method else 0,
            double_line_break=args.double_line_break,
        )

        dynamic_question_parallel_num = dynamic_params.get("question_parallel_num", args.question_parallel_num)

        env = task.env_fn(
            config={
                "max_actions": dyn_beam_config.tree_max_width,
                "max_length": dyn_beam_config.tree_max_depth,
                "beam_size": dyn_beam_config.beam_size,
                "cot_prompt": dyn_beam_config.cot_prompt,
                "stop_str": dyn_beam_config.stop_str,
                "sep": dyn_beam_config.sep,
                "generation_config": {
                    "max_new_tokens": dyn_gen_config.max_new_tokens,
                    "temperature": dyn_gen_config.temperature,
                    "top_p": dyn_gen_config.top_p,
                    "top_k": dyn_gen_config.top_k,
                },
                "is_few_shot": dyn_beam_config.is_few_shot,
                "add_step_prompt": dyn_beam_config.add_step_prompt,
                "direct_io": dyn_beam_config.direct_io,
                "double_line_break": dyn_beam_config.double_line_break,
                "model_names": dyn_beam_config.model_names,
                "selected_model_idx": lm_idx,  # 关键：将选中的LM索引传给环境
            },
            math_problems=[{
                "question": problem_inst["question"],
                "answer": problem_inst.get("extracted_groundtruth") or task.extract_groundtruth(problem_inst["answer"]),
            }],
            llm_gen_fns=lm_calls,  # 多LM列表
            rm_call=effective_rm_call,
            update_legal_action=False,
        )

        # 5. 执行beam搜索并返回结果
        search_tree = SearchTree(
            cfg={"model_names": dyn_beam_config.model_names, "direct_io": dyn_beam_config.direct_io,
                 "max_actions": dyn_beam_config.tree_max_width})
        traj_list = search_tree.beam_search(env, dyn_beam_config.beam_size, dyn_beam_config.tree_max_depth, effective_rm_call)

        # 6. 返回结果，包含动态的question_parallel_num信息
        result = TreeSearchSolutionOutput(
            solutions=[t["text"] for t in traj_list],
            completion_tokens=[t["api_completion_tokens"] for t in traj_list],
            tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
            reward_history=[t["reward_history"] for t in traj_list],
            token_history=[t["token_history"] for t in traj_list],
            prob_history=[t["prob_history"] for t in traj_list],
            model_history=[t["model_history"] for t in traj_list],
            step_latency=[t.get("step_latency", []) for t in traj_list],
            step_lm_latency=[t.get("step_lm_latency", []) for t in traj_list],
            step_rm_latency=[t.get("step_rm_latency", []) for t in traj_list],
            step_wait=[t.get("step_wait", []) for t in traj_list],
            question_latency=[t.get("question_latency", 0.0) for t in traj_list],
            total_unit_latency=[t.get("total_unit_latency", 0.0) for t in traj_list],
            complete_latency_record=traj_list[0].get("complete_latency_record", []) if traj_list else [],
            tree_snapshot=search_tree.get_tree_snapshot(),
        )

        # 将动态question_parallel_num附加到结果中（用于调试和记录）
        result.dynamic_question_parallel_num = dynamic_question_parallel_num
        return result


    def parallel_evaluate_test_dataset(actor_pool, raw_test_ds, method_name, solver_fn, save_dir, question_parallel_num):
        results = []
        question2id = {problem_inst["question"]: i for i, problem_inst in enumerate(raw_test_ds)}

        test_ds, _ = assign_tasks(
            raw_test_ds, question_parallel_num, args.num_sequence, save_dir, args.lock_dir, args.batch_size, args.max_time
        )

        res_q = actor_pool.map_unordered(lambda p, x: p.evaluate_problem.remote(x, solver_fn), test_ds)
        start_time = time.time()
        last_time = start_time

        for i, (problem_inst, result, output, tree_snapshot) in enumerate(res_q):
            status = result.get("_status") if isinstance(result, dict) else None
            if status == "skipped":
                q_idx = question2id.get(problem_inst["question"], -1)
                skip_obj = {
                    "question": problem_inst.get("question"),
                    "groundtruth": problem_inst.get("answer"),
                    "status": "skipped",
                    "error": result.get("_error"),
                    "attempts": result.get("_attempts"),
                    "timestamp": datetime.now().isoformat(),
                    "output": [],
                }
                file_path = problem_inst.get("file_path")
                if file_path:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    try:
                        record_writer = jsonlines.open(file_path, mode="w", flush=True)
                        record_writer.write(skip_obj)
                    except Exception as e:
                        print(f"Skip Save Error: {e}")
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')[2:]}]   Skip question idx={q_idx} after attempts={result.get('_attempts')}"
                )
                continue

            if len(output) == 0:
                continue

            q_idx = question2id[problem_inst["question"]]
            
            obj = {"question": problem_inst["question"], "groundtruth": problem_inst["answer"], "result": result, "output": output}
            question_path = os.path.join(save_dir, f"question_{q_idx}")
            os.makedirs(question_path, exist_ok=True)

            file_path = problem_inst["file_path"]
            idx = int(file_path.split("_")[-1].split(".")[0])
            try:
                record_writer = jsonlines.open(file_path, mode="w", flush=True)
                record_writer.write(obj)
            except Exception as e:
                print(f"Save Error: {e}")

            if tree_snapshot is not None:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                beam_file_path = os.path.join(question_path, f"{base_name}_beam.json")
                try:
                    with open(beam_file_path, "w", encoding="utf-8") as beam_file:
                        json.dump(tree_snapshot, beam_file, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Beam Save Error: {e}")
            
            # Save complete latency record to a separate file
            try:
                complete_latency_record = problem_inst.get("_complete_latency_record", None)
                
                if complete_latency_record:
                    latency_file = file_path.replace(".jsonl", "_latency.jsonl")
                    latency_obj = {
                        "question": problem_inst["question"],
                        "complete_latency_record": complete_latency_record,
                        "total_iterations": len(complete_latency_record),
                        "question_latency": output[0].get("question_latency", 0.0) if output else 0.0
                    }
                    latency_writer = jsonlines.open(latency_file, mode="w", flush=True)
                    latency_writer.write(latency_obj)
            except Exception as e:
                print(f"Latency Save Error: {e}")

            temp_time = time.time()
            delta_time = temp_time - last_time
            total_time = temp_time - start_time
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")[2:]
            last_time = temp_time

            cnt = check_process_cnt(raw_test_ds, question_parallel_num, save_dir)
            total = len(raw_test_ds) * question_parallel_num if question_parallel_num else len(raw_test_ds)
            print(
                f"[{time_str}]   Cnt: {i + 1:>3} / {len(test_ds):>3}  |  Q: {q_idx:>3}  |  Idx: {idx:>3}  |  "
                f"Del T: {delta_time:>6.1f}s  |  Tot T: {total_time:>7.1f}s  |  Avg T: {total_time / (i + 1):>6.1f}s/it  |  "
                f"Pct: {cnt:>5} / {total:>5} = {cnt / total * 100:.2f}%"
            )

            if not question_parallel_num:
                results.append(result)

        if not question_parallel_num:
            try:
                avg_res = (tree.map_structure(lambda *xs: np.mean(xs), *results),)
                json.dump(avg_res, open(os.path.join(save_dir, f"avg_result.json"), "w"))
                print("Method: {}. Average result: {}".format(method_name, avg_res))
            except Exception as e:
                pass

        return results


    cfg_dict_record = dict()
    gen_config = LMCallingConfig(
        n=args.num_sequence,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    cfg_dict_record["gen_config"] = gen_config.__dict__

    if args.method == "cot":
        direct_io = 2
    elif args.method == "best_of_n":
        direct_io = 1
    else:
        direct_io = 0

    if args.method in ["cot", "best_of_n"]:
        args.tree_max_depth = 1
        if args.method == "cot":
            args.num_sequence = 1
            args.tree_max_width = 1
            if 'deepseek-r1' in args.LM[0].lower():
                top_p = 0.95
            else:
                top_p = 1.0
            gen_config = LMCallingConfig(
                n=1,
                temperature=args.temperature,
                top_k=-1,
                top_p=top_p,
                max_new_tokens=gen_config.max_new_tokens,
            )
        elif args.method == "best_of_n":
            args.num_sequence = args.tree_max_width
        method_config = BeamSearchConfig(
            task_name=args.task_name,
            tree_max_depth=1,
            tree_max_width=args.tree_max_width,
            beam_size=args.num_sequence,
            model_names=args.LM,
            is_few_shot=args.is_few_shot,
            add_step_prompt=args.add_step_prompt,
            cot_prompt=args.cot_prompt,
            stop_str=None,
            sep=args.sep,
            direct_io=direct_io,
            double_line_break=args.double_line_break,
        )
        solver_fn = partial(beam_search, method_config, gen_config)
    elif "beam_search" in args.method:
        method_config = BeamSearchConfig(
            task_name=args.task_name,
            tree_max_depth=args.tree_max_depth,
            tree_max_width=args.tree_max_width,
            beam_size=args.num_sequence,
            model_names=args.LM,
            is_few_shot=args.is_few_shot,
            add_step_prompt=args.add_step_prompt,
            cot_prompt=args.cot_prompt,
            stop_str=args.stop_str,
            sep=args.sep,
            direct_io=direct_io,
            double_line_break=args.double_line_break,
        )
        #  solver_fn = partial(beam_search, method_config, gen_config)
        solver_fn = dynamic_solver_fn # 动态beam search参数
    else:
        raise ValueError(f"Unknown method: {args.method}")

    cfg_dict_record["method"] = args.method
    cfg_dict_record["method_config"] = method_config.__dict__

    params = f'{args.tree_max_depth}_{args.tree_max_width}_{args.num_sequence}'
    model_name = get_model_name(args.LM[0])
    rm_model_name = get_model_name(args.RM)
    save_dir = f'{args.save_dir}/{args.task_name}_{args.method}/{model_name}/{rm_model_name}/{params}'
    print(f"Auto set dir as {save_dir}")

    try:
        os.makedirs(save_dir, exist_ok=True)
        if args.lock_dir:
            os.makedirs(os.path.join(save_dir, args.lock_dir), exist_ok=True)
    except Exception as e:
        print(f"Error: {e}")

    cfg_dict_record["llm_step_tags"] = llm_step_tags
    cfg_dict_record["prm_step_tag"] = args.prm_step_tag
    cfg_dict_record["good_tag"] = args.good_tag
    cfg_dict_record["bad_tag"] = args.bad_tag
    cfg_dict_record["stop_str"] = args.stop_str
    cfg_dict_record["sep"] = args.sep
    cfg_dict_record["LM"] = args.LM
    cfg_dict_record["RM"] = args.RM
    try:
        json.dump(cfg_dict_record, open(os.path.join(save_dir, f"config.json"), "w"))
    except Exception as e:
        print(f"Error: {e}")

    actor_pool = ActorPool(
        [RemoteMathEvaluator.remote(args.task_name, llm_gen_fns, rm_call, direct_io=direct_io) for _ in range(args.num_worker)]
    )

    test_ds = task.test_ds(args.task_name)

    returned_temp = parallel_evaluate_test_dataset(
        actor_pool, test_ds, args.method, solver_fn, save_dir, question_parallel_num=args.question_parallel_num,
    )
    check_lock_timeout(test_ds, args.question_parallel_num, save_dir, args.lock_dir, args.max_time)
