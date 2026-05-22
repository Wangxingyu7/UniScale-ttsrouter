import traceback
from dataclasses import dataclass
from typing import List

import requests
from transformers import AutoTokenizer


@dataclass
class ConcatedLMGenResult:
    text: List[str]
    prompt_tokens: List[int]
    num_tokens: List[int]
    cumulative_logprob: List[float]
    logp_avg_by_len: List[float]
    finish_reason: List[str]

    def __post_init__(self):
        self.completion_tokens = sum(self.num_tokens)


def process_prompt(prompt: str, tokenizer: AutoTokenizer, model_name, double_line_break=0, first_generation=False):
    eos_token = tokenizer.eos_token

    if prompt.endswith(f"{eos_token}\n"):
        prompt = prompt[:-len(f"{eos_token}\n")]
    elif prompt.endswith(eos_token):
        prompt = prompt[:-len(eos_token)]

    if double_line_break == 1:
        if 'llama-3' in model_name.lower() and 'meta-llama' in model_name.lower():
            if not prompt.endswith('\n\n'):
                prompt += '\n\n'
    elif double_line_break == 2:
        if 'llama-3' in model_name.lower() and 'meta-llama' in model_name.lower():
            if not prompt.endswith('\n\n'):
                prompt += '\n\n'

    return prompt


def _generate_vllm(
    messages,
    model_name,
    n,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    stop_token_ids,
    stop_str,
    include_stop_str_in_output,
    controller_addr,
    tokenizer,
    apply_chat_template=False,
    worker_addr="",
    multi_gpu=False,
    double_line_break=0,
    first_generation=False,
    model_idx=0,  
) -> ConcatedLMGenResult:
    if multi_gpu:
        ret = requests.post(controller_addr + "/get_worker_address", json={"model": model_name})  # get worker address by model name
        worker_addr = ret.json()["address"]
        if not worker_addr:
            raise ValueError("Language Model name {} does not exist.".format(model_name))
    else:
        base_port = 10082
        port = base_port + model_idx
        worker_addr = f"http://0.0.0.0:{port}"

    if apply_chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = process_prompt(prompt, tokenizer, model_name, double_line_break, first_generation)
    else:
        prompt = messages

    headers = {"User-Agent": "vLLM Client"}
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "top_k": top_k,
        "stop_token_ids": stop_token_ids,
        "max_new_tokens": max_new_tokens,
        "stop": stop_str,
        "echo": False,
        "include_stop_str_in_output": include_stop_str_in_output,
    }

    try:
        # print(f"Sending request to: {worker_addr}/worker_generate") #一些调试信息
        # print(f"Request params: {gen_params}")
        response = requests.post(worker_addr + "/worker_generate", headers=headers, json=gen_params, stream=False)
        results = response.json()
    except Exception as e:
        print(f'Error in _generate_vllm: {e}')
        print(f'Response status: {response.status_code if "response" in locals() else "No response"}')
        print(f'Response text: {response.text if "response" in locals() else "No response"}')
        raise e

    output_token_lens = results["output_token_len"]
    cum_logps = results["cumulative_logprob"]
    avg_len_logps = [clp / max(1, otl) for clp, otl in zip(cum_logps, output_token_lens)]

    return ConcatedLMGenResult(
        text=results["text"],
        prompt_tokens=results["usage"]["prompt_tokens"],
        num_tokens=results["output_token_len"],
        cumulative_logprob=cum_logps,
        logp_avg_by_len=avg_len_logps,
        finish_reason=results["finish_reason"],
    )


