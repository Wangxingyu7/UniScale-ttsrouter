"""
FastAPI worker wrapping a vLLM AsyncLLM engine for text generation.

Primary HTTP endpoints expected by `rm_call.py` and controller services:
- `/worker_generate_stream`: stream tokens for a single request.
- `/worker_generate`: run batched generation and return the final payload.
- `/worker_get_status`: expose worker health and metadata.
- `/count_token`: count tokens for a prompt via the underlying tokenizer.
- `/worker_get_conv_template`: return the conversation template bound to
    the worker.
- `/model_details`: surface model configuration such as context length.

Runtime helpers manage worker registration, concurrency limits, and request
lifecycles (acquire/abort/release) for vLLM-backed inference.
"""

import asyncio
import json
import time
from typing import List
import uuid
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.utils.argparse_utils import FlexibleArgumentParser

from reason.llm_service.workers.base_model_worker import BaseModelWorker, build_logger

worker_id = str(uuid.uuid4())[:8]
time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))[2:]
logger = build_logger("model_worker", f"vllm_{time_str}.log")

app = FastAPI()

class VLLMWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        llm_engine: AsyncLLM,
        conv_template: str,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        logger.info(f"Loading the model {self.model_names} on worker {worker_id}, worker type: vLLM worker...")
        self.tokenizer = llm_engine.tokenizer
        self.context_len = llm_engine.model_config.max_model_len

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        n = params.get("n", 1)
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = params.get("top_k", -1.0)
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        echo = params.get("echo", True)
        # use_beam_search = params.get("use_beam_search", False)
        best_of = params.get("best_of", None)
        include_stop_str_in_output = params.get("include_stop_str_in_output", False)

        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            stop=list(stop),
            stop_token_ids=stop_token_ids,
            max_tokens=max_new_tokens,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            best_of=best_of,
            logprobs=1,
            include_stop_str_in_output=include_stop_str_in_output,
        )
        results_generator = engine.generate(context, sampling_params, request_id)

        async for request_output in results_generator:
            prompt = request_output.prompt
            if echo:
                text_outputs = [prompt + output.text for output in request_output.outputs]
            else:
                text_outputs = [output.text for output in request_output.outputs]
            # text_outputs = " ".join(text_outputs)
            # Note: usage is not supported yet
            prompt_tokens = len(request_output.prompt_token_ids)
            completion_tokens = sum(len(output.token_ids) for output in request_output.outputs)
            
            ret = {
                "text": text_outputs,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "cumulative_logprob": [output.cumulative_logprob for output in request_output.outputs],
                "output_token_len": [len(output.token_ids) for output in request_output.outputs],
                "finish_reason": [output.finish_reason for output in request_output.outputs],
                "indices": [output.index for output in request_output.outputs],
            }
            yield (json.dumps(ret) + "\0").encode()

    async def generate(self, params):
        outputs = {}
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        async for chunk in self.generate_stream(params):
            data = json.loads(chunk[:-1].decode())

            usage = data.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                total_tokens = usage.get("total_tokens", total_tokens)

            texts = data.get("text", [])
            logprobs = data.get("cumulative_logprob", [])
            token_lens = data.get("output_token_len", [])
            finish_reasons = data.get("finish_reason", [])
            indices = data.get("indices")
            if indices is None:
                indices = list(range(len(texts)))

            for idx, text, logprob, token_len, finish in zip(indices, texts, logprobs, token_lens, finish_reasons):
                outputs[idx] = {
                    "text": text,
                    "logprob": logprob,
                    "token_len": token_len,
                    "finish_reason": finish,
                }

        sorted_indices = sorted(outputs.keys())
        final_texts = [outputs[idx]["text"] for idx in sorted_indices]
        final_logprobs = [outputs[idx]["logprob"] for idx in sorted_indices]
        final_token_lens = [outputs[idx]["token_len"] for idx in sorted_indices]
        final_finish_reasons = [outputs[idx]["finish_reason"] for idx in sorted_indices]

        return {
            "text": final_texts,
            "error_code": 0,
            "usage": {
                "prompt_tokens": prompt_tokens or 0,
                "completion_tokens": completion_tokens or 0,
                "total_tokens": total_tokens or 0,
            },
            "cumulative_logprob": final_logprobs,
            "output_token_len": final_token_lens,
            "finish_reason": final_finish_reasons,
        }

def release_worker_semaphore():
    worker.semaphore.release()

def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()

def create_background_tasks(request_id):
    async def abort_request() -> None:
        await engine.abort(request_id)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks

@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)

@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    output = await worker.generate(params)
    release_worker_semaphore()
    await engine.abort(request_id)
    return JSONResponse(output)

@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()

@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)

@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()

@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}

if __name__ == "__main__":
    # vLLM expects a FlexibleArgumentParser (supports extra kwargs like
    # `deprecated` when adding arguments).
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")

    parser.add_argument("--conv-template", type=str, default=None, help="Conversation prompt template.")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-model-length", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)

    parser.add_argument(
        "--trust_remote_code",
        action="store_false",
        default=True,
        help="Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.",
    )

    parser.add_argument("--max-num-sequences", type=int, default=0)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    if args.model_path:
        args.model = args.model_path
    if args.num_gpus > 1:
        args.tensor_parallel_size = args.num_gpus
    if args.max_model_length > 0:
        args.max_model_len = args.max_model_length
    if args.max_num_sequences > 0:
        args.max_num_seqs = args.max_num_sequences

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLM.from_engine_args(engine_args)

    worker = VLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        engine,
        args.conv_template,
    )
    logger.info(f"LM worker started: worker_id={worker_id}, model_path={args.model_path}, worker_addr={args.worker_address}, controller={args.controller_address}")
    print(args)
    print('*' * 50)
    print(engine_args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
