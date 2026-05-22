"""
vLLM-based Reward Model Worker
Exposes `/worker_reward_inference` endpoint expected by `rm_call.py`.
This worker uses vLLM AsyncLLMEngine when available and falls back to
"""

import time
import uuid
from typing import List
from fastapi import FastAPI, Request
import uvicorn

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.config.pooler import PoolerConfig
from transformers import AutoTokenizer

from reason.llm_service.workers.base_model_worker import BaseModelWorker, build_logger
from reason.inference.vllm_rm_infer_fns import get_vllm_rm_infer_fn

worker_id = str(uuid.uuid4())[:8]
time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))[2:]
logger = build_logger("reward_model_worker", f"rm_{time_str}.log")

app = FastAPI()

class VLLMRMWorker(BaseModelWorker):
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

        self.tokenizer = llm_engine.tokenizer

        self.rm_infer_fn = get_vllm_rm_infer_fn(model_path, llm_engine, self.tokenizer)

        if not no_register:
            self.init_heart_beat()

    async def reward_inference_gate(self, params):
        # Expect params: {"input_str": <str or list>}
        input_str = params.get("input_str")
        if input_str is None:
            return {"reward": []}

        try:
            rewards = await self.rm_infer_fn(input_str)
        except Exception as e:
            logger.exception("RM inference failed")
            if isinstance(input_str, list):
                return {"reward": [[0.01] for _ in input_str]}
            else:
                return {"reward": [0.01]}

        return {"reward": rewards}

@app.post("/worker_reward_inference")
async def reward_inference(request: Request):
    params = await request.json()
    
    output = await request.app.state.worker.reward_inference_gate(params)

    return output

if __name__ == "__main__":
    # Use vLLM's FlexibleArgumentParser so AsyncEngineArgs.add_cli_args can
    # pass vLLM-specific kwargs (like `deprecated`) to add_argument.
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=10081)
    parser.add_argument("--worker-address", type=str, default="http://localhost:10081")
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="gpt2")
    parser.add_argument("--model-names", type=lambda s: s.split(","), help="Optional display comma separated names")
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--max-model-length", type=int, default=4096, help="Max context length for the model.")

    parser.add_argument(
        "--trust_remote_code",
        action="store_false",
        default=True,
        help="Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    parser.set_defaults(trust_remote_code=True)
    args = parser.parse_args()
    if args.model_path:
        args.model = args.model_path
    
    if args.max_model_length > 0:
        args.max_model_len = args.max_model_length
    
    # Set step_tag_id for Skywork PRM: load tokenizer to get newline token ID
    temp_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    newline_token_id = temp_tokenizer.encode("\n", add_special_tokens=False)[-1]
    
    # Use pooler_config instead of deprecated override_pooler_config
    if not hasattr(args, 'pooler_config') or args.pooler_config is None:
        args.pooler_config = PoolerConfig(step_tag_id=newline_token_id)
    elif hasattr(args.pooler_config, 'step_tag_id'):
        args.pooler_config.step_tag_id = newline_token_id
    
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLM.from_engine_args(engine_args)

    worker_obj = VLLMRMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        engine,
        args.conv_template if hasattr(args, "conv_template") else None,
    )

    # store worker on app state so endpoint can access it
    app.state.worker = worker_obj

    logger.info(f"RM worker started: worker_id={worker_id}, model_path={args.model_path}, worker_addr={args.worker_address}, controller={args.controller_address}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")