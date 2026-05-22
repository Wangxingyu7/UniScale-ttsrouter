import copy
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import requests

from reason.inference.infer_fns import (
    _skywork_infer_fn,
    _qwen_infer_fn,
)


def get_prm_special_tokens(model_name, tokenizer):
    step_tag_id, returned_token_ids = None, None
    if 'qwen2.5-math' in model_name.lower():  
        prm_step_tag = "<extra_0>"
        step_tag_id = tokenizer.encode(prm_step_tag)[0]
        returned_token_ids = []
    elif 'skywork' in model_name.lower():  # models--Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B
        prm_step_tag = "\n"
        step_tag_id = tokenizer.encode(prm_step_tag)[-1]
        returned_token_ids = []
    else:
        raise ValueError("Model path: {} not recognized".format(model_name))
    return step_tag_id, returned_token_ids


def get_infer_fn(model_path, rm_serve_type='fastchat'):
    if rm_serve_type == 'vllm':
        raise ValueError("vLLM RM inference should use remote worker, not local inference")
    if "qwen2.5-math" in model_path.lower():
        return _qwen_infer_fn
    elif "skywork" in model_path.lower():
        return _skywork_infer_fn
    else:
        raise ValueError("Model path: {} not recognized".format(model_path))


@dataclass
class RewardModelBaseConfig:
    prm_step_tag: str
    format_str: str  # a format string that takes in question and answer need to have {question} and {answer} in the string

    rm_serve_type: str
    step_tag_id: int
    returned_token_ids: List[int]


class RewardModelCallingFunction:

    def __init__(self, config: RewardModelBaseConfig):
        self.config = config
        self.prm_step_tag = config.prm_step_tag
        self.format_str = config.format_str

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        model_names: List[str],
    ) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError

    def replace_step_tag(self, answer: str):
        if self.prm_step_tag not in answer:
            answer += f" {self.prm_step_tag}"
        splits = answer.split(f" {self.prm_step_tag}")
        splits = [s.strip() for s in splits]
        response = f" {self.prm_step_tag}".join([s for s in splits if s != ""])
        response += f" {self.prm_step_tag}"
        return response


class DummyRewardModelCaller(RewardModelCallingFunction):
    # a dummy rm caller that always return 0

    def __init__(self, config: RewardModelBaseConfig):
        super().__init__(config)

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        model_names: List[str],
    ) -> Union[List[int], List[List[int]]]:

        def fn(s):
            steps = s.split(self.prm_step_tag)
            steps = [s for s in steps if s.strip() != ""]
            return list(range(len(steps)))

        if isinstance(question_answer_pairs[0], str):
            return fn(
                self.format_str.format(
                    question=question_answer_pairs[0],
                    answer=self.replace_step_tag(question_answer_pairs[1]),
                )
            )
        else:
            return [
                fn(
                    self.format_str.format(
                        question=s[0],
                        answer=self.replace_step_tag(s[1]),
                    )
                ) for s in question_answer_pairs
            ]


@dataclass
class RemoteRewardModelConfig(RewardModelBaseConfig):
    model_name: str
    controller_addr: str
    multi_gpu: bool


def _reward_inference_vllm(input_str, model_name, controller_addr="http://localhost:21001", multi_gpu=True, timeout=0):
    import time
    rm_start_time = time.time()
    # Query controller for a worker address that hosts the desired RM model.
    try:
        ret = requests.post(controller_addr + "/get_worker_address", json={"model": model_name}, timeout=5)
        worker_addr = ret.json().get("address", "")
        if not worker_addr:
            raise ValueError(f"Model name {model_name} does not have any registered worker.")
    except Exception as e:
        raise

    headers = {"User-Agent": "RM-Client"}
    gen_params = {"input_str": input_str}

    try:
        if timeout > 0:
            response = requests.post(worker_addr + "/worker_reward_inference", headers=headers, json=gen_params, stream=True, timeout=timeout)
        else:
            response = requests.post(worker_addr + "/worker_reward_inference", headers=headers, json=gen_params, stream=True)
        results = response.json()
        reward = results["reward"]
        
    except Exception as e:
        # 打印调试信息
        if isinstance(input_str, list):
            for i in range(len(input_str)):
                print(f'input_str {i}: {input_str[i]}')
        else:
            print(f'input_str: {input_str}')

        error_info = traceback.format_exc()
        print(f'Error in _reward_inference_vllm: {error_info}')
        traceback.print_exc()

        # 根据input_str的类型返回对应格式的默认值
        if isinstance(input_str, list):
            reward = [[0.01] for _ in range(len(input_str))]
        else:
            reward = [0.01]

        print("_reward_inference_vllm ERROR - using default reward values")

    rm_latency = time.time() - rm_start_time
    # Store RM latency in a global variable or return it
    if not hasattr(_reward_inference_vllm, 'rm_latency_history'):
        _reward_inference_vllm.rm_latency_history = []
    _reward_inference_vllm.rm_latency_history.append(rm_latency)

    return reward


class RMRemoteCaller(RewardModelCallingFunction):

    def __init__(self, config: RemoteRewardModelConfig, tokenizer):
        self.model_name = config.model_name
        self.controller_addr = config.controller_addr
        self.tokenizer = tokenizer

        self.prm_step_tag = config.prm_step_tag
        self.step_tag_id = config.step_tag_id
        self.returned_token_ids = config.returned_token_ids

        self.multi_gpu = config.multi_gpu

        super().__init__(config)

    def process_input(self, qa_pairs, model_names, verbose, legal_action=[]):
        if isinstance(qa_pairs[0], str):
            raise ValueError("The input of PRM should be a list of tuples")
        if 'skywork' in self.model_name.lower():
            temp_qa_pairs = copy.deepcopy(qa_pairs)
            for i in range(len(temp_qa_pairs)):
                raw_splits = temp_qa_pairs[i][1].split(f" ки\n")
                splits = []
                for s in raw_splits:
                    temp = s.replace("\n", " ").strip()
                    if temp:
                        splits.append(temp)
                if len(splits) == 1:
                    answer = splits[0]
                else:
                    answer = f"\n".join(splits)
                temp_qa_pairs[i] = (temp_qa_pairs[i][0], answer)
            return temp_qa_pairs
        elif 'qwen2.5-math' in self.model_name.lower():
            conversations = []
            temp_qa_pairs = copy.deepcopy(qa_pairs)
            for i in range(len(temp_qa_pairs)):
                conversation = [
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": temp_qa_pairs[i][0]},
                ]
                assistant_content = ""
                raw_splits = temp_qa_pairs[i][1].split(f" ки\n")
                for j in range(len(raw_splits)):
                    if raw_splits[j].strip() == "":
                        continue
                    text = raw_splits[j].strip()
                    assistant_content += f"{text}<extra_0>"
                conversation.append({"role": "assistant", "content": assistant_content})
                conversations.append(conversation)
            return conversations
        else:
            input_str = []
            for i in range(len(qa_pairs)):
                answer = self.replace_step_tag(qa_pairs[i][1])
                if 'llama3.1-math-prm' in self.model_name.lower():
                    answer = answer.replace(" ки\n", " ки")
                elif 'pqm' in self.model_name.lower():
                    answer = answer.replace(" ки", " [PRM]")  # Each step ends with: " [PRM]\n"
                format_str = self.format_str.format(question=qa_pairs[i][0], answer=answer)
                input_str.append(format_str)
            return input_str

    def __call__(
        self,
        qa_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        model_names: List[str],
        verbose: Optional[bool] = False,
        local: Optional[bool] = False,
        legal_action: Optional[List[str]] = [],
        process: Optional[bool] = True,
        timeout: Optional[int] = 0,
    ) -> Union[List[int], List[List[int]]]:
        if process:
            input_str = self.process_input(qa_pairs, model_names, verbose=verbose, legal_action=legal_action)
        else:
            input_str = qa_pairs

        if local:
            infer_fn = get_infer_fn(self.model_name, rm_serve_type=self.config.rm_serve_type)
            return infer_fn(input_str)

        result = _reward_inference_vllm(
            input_str=input_str, model_name=self.model_name, controller_addr=self.controller_addr, timeout=timeout
        )

        # Get RM latency from the global history
        if hasattr(_reward_inference_vllm, 'rm_latency_history') and _reward_inference_vllm.rm_latency_history:
            rm_latency = _reward_inference_vllm.rm_latency_history.pop(0)  # Get and remove the first latency
            # Store RM latency in a way that can be accessed by the environment
            if not hasattr(self, 'rm_latency_history'):
                self.rm_latency_history = []
            self.rm_latency_history.append(rm_latency)
        
        return result
