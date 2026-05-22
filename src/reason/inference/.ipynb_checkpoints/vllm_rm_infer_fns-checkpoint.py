import asyncio
import uuid
import logging
import time
from typing import List, Union, Any

import torch
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreRequest

logger = logging.getLogger(__name__)


def get_vllm_rm_infer_fn(model_path: str, engine, tokenizer):
    step_tag_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
    
    async def infer_fn(obj: Union[Any, List[Any]]):
        single = not isinstance(obj, list)
        batch = [obj] if single else obj
        
        results = await asyncio.gather(*[
            _process_sample(engine, sample, tokenizer, step_tag_id) for sample in batch
        ])
        
        return results[0] if single else results
    
    return infer_fn


async def _process_sample(engine, sample: Any, tokenizer, step_tag_id: int) -> List[float]:
    try:
        from reason.inference.skywork_o1_prm_inference.io_utils import prepare_input, prepare_batch_input_for_model
        
        question, response = _extract_qa(sample)
        input_ids, _, reward_flags = prepare_input(question, response, tokenizer, step_token="\n")
        
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        batch_ids, attention_mask, batch_flags = prepare_batch_input_for_model(
            [input_ids], [reward_flags], pad_token_id=pad_id
        )
        
        input_token_ids = batch_ids[0].tolist() if hasattr(batch_ids[0], "tolist") else batch_ids[0]
        
        params = PoolingParams()
        params.task = "token_classify"
        params.step_tag_id = step_tag_id
        params.output_kind = RequestOutputKind.FINAL_ONLY
        
        rid = f"rm_{uuid.uuid4().hex}"
        
        # Ensure the background output handler is running
        if hasattr(engine, "_run_output_handler"):
            engine._run_output_handler()
        
        # Construct EngineCoreRequest to avoid deprecated Processor path
        request = EngineCoreRequest(
            request_id=rid,
            prompt_token_ids=input_token_ids,
            mm_features=None,
            sampling_params=None,
            pooling_params=params,
            eos_token_id=tokenizer.eos_token_id,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        
        queue = await engine.add_request(
            request_id=rid,
            prompt=request,
            params=params
        )
        out = await asyncio.wait_for(queue.get(), timeout=60.0)
        # print("out", out)
        
        step_rewards = out.outputs.data.squeeze().tolist()
        return step_rewards if isinstance(step_rewards, list) else [step_rewards]
        
    except Exception as e:
        logger.error(f"RM error: {type(e).__name__}: {e}")
        return [0.0]


def _extract_qa(item: Any) -> tuple[str, str]:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return str(item[0]), str(item[1])
    raise ValueError(f"Invalid input: {type(item)}")