from fastapi import FastAPI, File, UploadFile, Body, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from typing import Union, List, Optional, Dict
import asyncio
import json
import os
from pathlib import Path
import shutil
import uuid
from datetime import datetime, timedelta
import logging
import aiofiles
import time
import heapq
from contextlib import asynccontextmanager
import contextlib
import copy
import re
from scipy.special import expit

import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 7777
    cleanup_delay: int = 3000
    default_rm: str = "/root/autodl-pub/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    default_lm: str = "/root/autodl-pub/policy_models/Qwen3-0.6B,/root/autodl-pub/policy_models/Qwen3-1.7B"
    bash_path: Optional[str] = None
    max_cleanup_queue_size: int = 10000
    default_timeout: int = 7200
    class Config:
        env_prefix = "TTS_"

config = ServiceConfig()

class BeamConfig(BaseModel):
    QP: float = Field(default=1.0, gt=0)
    CP: float = Field(default=8.0, gt=0)
    BS: int = Field(default=4, gt=0)

class Problem(BaseModel):
    problem: str
    solution: Union[str, float]
    lm: str
    beam: BeamConfig

    @field_validator('solution', mode='before')
    @classmethod
    def to_str(cls, v):
        return str(v)

class EvaluationConfig(BaseModel):
    method: str = Field(default="beam_search")
    temperature: Optional[float] = Field(default=0.7, gt=0, le=1)
    top_k: Optional[int] = Field(default=-1)
    top_p: Optional[float] = Field(default=1.0, gt=0, le=1)
    max_new_tokens: Optional[int] = Field(default=2048, gt=0)
    num_sequence: Optional[int] = Field(default=1, gt=0)
    tree_max_depth: Optional[int] = Field(default=None)
    tree_max_width: Optional[int] = Field(default=None)
    question_parallel_num: Optional[int] = Field(default=1, gt=0)
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        if v not in ["beam_search"]:
            raise ValueError("Unsupported method")
        return v

class TTSRouter:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.input_base = self.base_dir / "envs" / "MATH" / "dataset"
        self.output_base = self.base_dir / "output"
        self.active_tasks = {}
        self.task_lock = asyncio.Lock()
        self.cleanup_queue = []
        self.input_base.mkdir(parents=True, exist_ok=True)
        self.output_base.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _expand_variants(raw: str) -> List[str]:
        if not raw:
            return []
        expanded = os.path.expandvars(raw)
        variants = [expanded]
        home_prefix = str(Path.home())
        if "/root/" in expanded:
            variants.append(expanded.replace("/root/", f"{home_prefix}/"))
        return list(dict.fromkeys([os.path.expanduser(v.strip()) for v in variants if v.strip()]))

    @staticmethod
    def _normalize_dir(path_str: str) -> Optional[str]:
        candidate = os.path.expanduser(os.path.expandvars(path_str.strip()))
        if not candidate:
            return None
        path = Path(candidate)
        if path.exists() and path.is_dir():
            return str(path)
        return None

    @staticmethod
    def _normalize_lm_spec(spec: str) -> Optional[str]:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if not parts:
            return None
        normalized_parts: List[str] = []
        for part in parts:
            normalized = TTSRouter._normalize_dir(part)
            if not normalized:
                return None
            normalized_parts.append(normalized)
        return ",".join(normalized_parts)

    @staticmethod
    def _collect_numbered_policy_models() -> List[str]:
        collected: List[str] = []
        index = 1
        while True:
            key = f"POLICY_MODEL_{index}_PATH"
            if key not in os.environ:
                break
            value = os.environ.get(key, "").strip()
            if value:
                collected.append(value)
            index += 1
        return collected

    @staticmethod
    def _hint_keys(hint: str) -> List[str]:
        if not hint:
            return []
        slug = re.sub(r"[^A-Z0-9]+", "_", hint.upper()).strip("_")
        if not slug:
            return []
        return [
            f"POLICY_MODEL_{slug}_PATH",
            f"LM_PATH_{slug}",
            f"MODEL_PATH_{slug}",
            hint,
        ]

    def _resolve_rm_path(self) -> str:
        candidates: List[tuple[str, str]] = []
        for key in ["VALUE_MODEL_PATH", "TTS_VALUE_MODEL_PATH", "RM_PATH"]:
            val = os.environ.get(key)
            if val:
                candidates.append((key, val))
        candidates.append(("config.default_rm", config.default_rm))
        if config.default_rm:
            home_variant = config.default_rm.replace("/root/", f"{Path.home()}/")
            if home_variant != config.default_rm:
                candidates.append(("config.default_rm(home)", home_variant))

        for source, raw in candidates:
            for variant in self._expand_variants(raw):
                normalized = self._normalize_dir(variant)
                if normalized:
                    logger.info(f"Using RM path from {source}: {normalized}")
                    return normalized
                logger.warning(f"Skipping RM path candidate from {source}: {variant} (missing)")

        raise HTTPException(status_code=500, detail="Please set VALUE_MODEL_PATH or TTS_DEFAULT_RM")

    def _resolve_lm_spec(self, hint: Optional[str]) -> str:
        candidates: List[tuple[str, str]] = []

        for key in self._hint_keys(hint or ""):
            val = os.environ.get(key)
            if val:
                candidates.append((key, val))

        env_spec = os.environ.get("POLICY_MODEL_PATH")
        if env_spec:
            candidates.append(("POLICY_MODEL_PATH", env_spec))

        numbered = self._collect_numbered_policy_models()
        if numbered:
            candidates.append(("POLICY_MODEL_{i}_PATH", ",".join(numbered)))

        candidates.append(("config.default_lm", config.default_lm))
        if config.default_lm:
            home_variant = config.default_lm.replace("/root/", f"{Path.home()}/")
            if home_variant != config.default_lm:
                candidates.append(("config.default_lm(home)", home_variant))

        tried: List[str] = []
        for source, raw in candidates:
            for variant in self._expand_variants(raw):
                normalized = self._normalize_lm_spec(variant)
                if normalized:
                    logger.info(f"Using LM spec from {source}: {normalized}")
                    return normalized
                tried.append(variant)
                logger.warning(f"Skipping LM spec candidate from {source}: {variant} (part or all path lacking)")

        detail = "No valid policy model path found. Please set POLICY_MODEL_PATH or POLICY_MODEL_<N>_PATH"
        if tried:
            detail += f". Tried: {', '.join(tried)}"
        raise HTTPException(status_code=500, detail=detail)

    @staticmethod
    def infer_lm_hint(input_data: Union[Problem, List[Problem], UploadFile]) -> Optional[str]:
        def extract(obj) -> Optional[str]:
            if isinstance(obj, Problem):
                return obj.lm
            if isinstance(obj, dict):
                value = obj.get("lm")
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None

        if isinstance(input_data, list):
            for item in input_data:
                hint = extract(item)
                if hint:
                    return hint
            return None
        return extract(input_data)

    async def start_cleanup_worker(self):
        while True:
            try:
                await asyncio.sleep(1000)
                current_time = time.time()
                while self.cleanup_queue and self.cleanup_queue[0][0] <= current_time:
                    cleanup_time, task_id = heapq.heappop(self.cleanup_queue)
                    await self._cleanup_task_files(task_id)
                if len(self.cleanup_queue) > config.max_cleanup_queue_size:
                    logger.warning(f"Cleanup queue size ({len(self.cleanup_queue)}) exceeded limit, removing oldest entries")
                    self.cleanup_queue = heapq.nsmallest(config.max_cleanup_queue_size // 2, self.cleanup_queue)
                    heapq.heapify(self.cleanup_queue)
                    
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}", exc_info=True)
                await asyncio.sleep(5)

    def create_task_dir(self) -> tuple[str, Path]:
        task_id = str(uuid.uuid4())
        input_dir = self.input_base / task_id
        input_dir.mkdir(parents=True, exist_ok=True)
        return task_id, input_dir

    async def schedule_cleanup(self, task_id: str):
        cleanup_time = time.time() + config.cleanup_delay
        heapq.heappush(self.cleanup_queue, (cleanup_time, task_id))
        logger.info(f"Scheduled cleanup for task {task_id} at {datetime.fromtimestamp(cleanup_time)}")

    async def _cleanup_task_files(self, task_id: str):
        try:
            async with self.task_lock:
                task_meta = self.active_tasks.get(task_id, {})
                run_dir = task_meta.get('run_dir')
                self.active_tasks.pop(task_id, None)
            await asyncio.to_thread(self._cleanup_files_sync, task_id, run_dir)
            logger.info(f"Cleaned up task {task_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up task {task_id}: {e}", exc_info=True)
            
    @staticmethod
    def _sigmoid(x: List[float]) -> List[float]:
        return expit(x).tolist()
    
    def _cleanup_files_sync(self, task_id: str, run_dir: Optional[str]):
        input_dir = self.input_base / task_id
        shutil.rmtree(input_dir, ignore_errors=True)
        if run_dir:
            run_dir_path = Path(run_dir)
            if run_dir_path.exists() and self.output_base in run_dir_path.parents:
                shutil.rmtree(run_dir_path, ignore_errors=True)
                logger.debug(f"Removed run directory: {run_dir}")

    async def prepare_input(self, problems: Union[Problem, List[Problem], UploadFile], task_id: str) -> Path:
        input_file = self.input_base / task_id / "input.jsonl"
        try:
            if isinstance(problems, UploadFile):
                async with aiofiles.open(input_file, 'wb') as f:
                    while chunk := await problems.read(65536):
                        await f.write(chunk)
            else:
                async with aiofiles.open(input_file, 'w') as f:
                    items = problems if isinstance(problems, list) else [problems]
                    for p in items:
                        data = p.model_dump() if isinstance(p, Problem) else p
                        await f.write(json.dumps(data) + '\n')
            return input_file
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error preparing input for task {task_id}: {e}", exc_info=True)
            await self.schedule_cleanup(task_id)
            raise HTTPException(status_code=400, detail=f"Failed to prepare input: {str(e)}")

    async def run_evaluation(
        self, 
        task_id: str, 
        eval_config: EvaluationConfig, 
        timeout_seconds: Optional[int] = None,
        save_root_dir: Optional[Path] = None, 
        input_path: Optional[Path] = None,
        lm_hint: Optional[str] = None
    ) -> tuple[bytes, bytes, str]:
        if timeout_seconds is None:
            timeout_seconds = config.default_timeout
            
        input_dir = self.input_base / task_id
        if save_root_dir is None:
            save_root_dir = self.output_base / "query_beam_search"
        run_dir = save_root_dir / f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{task_id[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        src_root = self.base_dir
        
        bash_cmd = self._get_bash_command()
        script_path = src_root / "scripts" / "run_t1.sh"
        if not script_path.exists():
            raise HTTPException(status_code=500, detail=f"Not found: {script_path}")

        lm_spec = self._resolve_lm_spec(lm_hint)
        rm_path = self._resolve_rm_path()

        cmd = [
            bash_cmd, str(script_path),
            "--method", eval_config.method,
            "--LM", lm_spec,
            "--RM", rm_path,
            "--task_name", "input"
        ]
        
        try:
            env_vars = os.environ.copy()
            effective_input = input_path if input_path is not None else (input_dir / "input.jsonl")
            env_vars["TTS_INPUT_PATH"] = str(effective_input)
            env_vars["TTS_SAVE_DIR"] = str(run_dir)
            env_vars.setdefault("POLICY_MODEL_PATH", lm_spec)
            env_vars.setdefault("VALUE_MODEL_PATH", rm_path)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(src_root),
                env=env_vars
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                raise HTTPException(status_code=504, detail=f"Evaluation timed out (>{timeout_seconds}s)")
            
            if proc.returncode != 0:
                err_tail = (stderr.decode(errors='ignore') if stderr else '')[-2000:]
                logger.error(f"Evaluation failed for task {task_id}: {err_tail}")
                raise HTTPException(status_code=500, detail=f"Evaluation failed: {err_tail}")
                
            return stdout, stderr, str(run_dir)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during evaluation for task {task_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")
    
    def _get_bash_command(self) -> str:
        if os.name != 'nt':
            return "bash"
        
        bash_path = config.bash_path or os.environ.get("BASH_PATH")
        if bash_path and Path(bash_path).exists():
            return bash_path
        
        raise HTTPException(
            status_code=400, 
            detail="Bash not found: please use WSL/Git Bash or set the environment variable BASH_PATH to point to the bash executable"
        )

    async def get_results(self, task_id: str, save_root_dir: Optional[Path] = None) -> Optional[Dict]:
        result_root = save_root_dir if save_root_dir else (self.output_base / "query_beam_search")
        try:
            aggregated = await self.collect_all_outputs(result_root)
            if not aggregated.get("records"):
                logger.warning(f"No record data collected for task {task_id}")
                return None
            return aggregated
        except Exception as e:
            logger.error(f"Error reading results for task {task_id}: {e}", exc_info=True)
            return None
    
    def _find_record_file(self, result_root: Path) -> Optional[Path]:
        candidate_records = []
        for root, _, files in os.walk(result_root):
            for name in files:
                if name.endswith(".jsonl") and name.startswith("record_"):
                    p = Path(root) / name
                    try:
                        m = p.stat().st_mtime
                    except OSError:
                        m = 0
                    candidate_records.append((m, p))
        
        if not candidate_records:
            return None
        for _, p in candidate_records:
            if p.name == "record_0.jsonl":
                return p
        
        candidate_records.sort(reverse=True)
        return candidate_records[0][1]

    async def collect_all_outputs(self, run_dir: Path) -> Dict[str, Union[str, Dict[str, str], List[Dict]]]:
        return await asyncio.to_thread(self._collect_all_outputs_sync, Path(run_dir))

    def _collect_all_outputs_sync(self, run_dir: Path) -> Dict[str, Union[str, Dict[str, str], List[Dict]]]:
        records: List[Dict] = []
        outputs: List[Dict] = []
        question: Optional[str] = None
        groundtruth: Optional[str] = None
        questions_map: Dict[str, str] = {}

        run_dir = Path(run_dir)
        if not run_dir.exists():
            return {
                "question": None,
                "groundtruth": None,
                "records": [],
                "output": [],
                "questions": {},
            }

        for root, _, files in os.walk(run_dir):
            for name in files:
                if not (name.startswith("record_") and name.endswith(".jsonl")):
                    continue
                path = Path(root) / name
                question_idx = Path(root).name
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    if not lines:
                        continue
                    record = json.loads(lines[-1])
                except Exception as exc:
                    logger.warning(f"Failed to parse record file {path}: {exc}")
                    continue

                record_copy = copy.deepcopy(record)
                record_copy.setdefault("question_idx", question_idx)
                records.append(record_copy)
                question_text = record.get("question")
                if question_text and question_idx not in questions_map:
                    questions_map[question_idx] = question_text
                question = question or question_text
                groundtruth = groundtruth or record.get("groundtruth")

                record_outputs = record.get("output", []) or []
                for item in record_outputs:
                    item_copy = copy.deepcopy(item)
                    if "question_idx" not in item_copy:
                        item_copy["question_idx"] = question_idx
                    outputs.append(item_copy)

        return {
            "question": question,
            "groundtruth": groundtruth,
            "records": records,
            "output": outputs,
            "questions": questions_map,
        }

    @staticmethod
    def _average_reward(entry: Dict) -> float:
        history = entry.get("reward_history") or []
        values: List[float] = []
        for value in history:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if not values:
            return float("-inf")
        return float(sum(values) / len(values))

    @staticmethod
    def _response_length(entry: Dict) -> int:
        completion_tokens = entry.get("completion_tokens")
        if completion_tokens is not None:
            try:
                return int(completion_tokens)
            except (TypeError, ValueError):
                pass

        total = 0
        for token in entry.get("token_history", []) or []:
            try:
                total += int(token)
            except (TypeError, ValueError):
                continue
        return total

    @staticmethod
    def _extract_answer(entry: Dict) -> str:
        answer = entry.get("extracted_answer")
        if isinstance(answer, str) and answer.strip():
            return answer
        text = entry.get("text")
        return text if isinstance(text, str) else ""

router = TTSRouter()

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(router.start_cleanup_worker())
    yield

app = FastAPI(lifespan=lifespan)


async def _process_evaluation_request(
    task_id: str,
    input_data: Union[Problem, List[Problem], UploadFile],
    eval_config: EvaluationConfig,
    save_root_dir: Path
) -> Dict:
    cleanup_time = datetime.now() + timedelta(seconds=config.cleanup_delay)
    
    try:
        async with router.task_lock:
            router.active_tasks[task_id] = {
                'status': 'preparing',
                'start_time': datetime.now(),
                'cleanup_scheduled': cleanup_time
            }
        
        await router.prepare_input(input_data, task_id)
        
        async with router.task_lock:
            router.active_tasks[task_id]['status'] = 'running'
        
        lm_hint = router.infer_lm_hint(input_data)

        stdout, stderr, run_dir = await router.run_evaluation(
            task_id, eval_config, save_root_dir=save_root_dir, lm_hint=lm_hint
        )
        
        async with router.task_lock:
            router.active_tasks[task_id]['run_dir'] = run_dir
        
        results = await router.get_results(task_id, save_root_dir=Path(run_dir))
        
        if results is None:
            raise HTTPException(status_code=500, detail="Failed to get results")
        outputs = results.get("output", []) if isinstance(results, dict) else []

        per_question_best: Dict[str, Dict] = {}
        per_question_score: Dict[str, float] = {}
        per_question_total_tokens: Dict[str, int] = {}

        def _safe_list(values) -> List[float]:
            out: List[float] = []
            for v in values or []:
                try:
                    out.append(float(v))
                except (TypeError, ValueError):
                    continue
            return out

        def _safe_sum(values) -> float:
            return float(sum(_safe_list(values)))

        for entry in outputs:
            question_idx = str(entry.get("question_idx", "unknown"))
            score = router._average_reward(entry)
            current_best_score = per_question_score.get(question_idx, float("-inf"))
            if score > current_best_score:
                per_question_score[question_idx] = score
                per_question_best[question_idx] = {
                    "score": router._sigmoid(entry.get("reward_history", [])),
                    "answer": router._extract_answer(entry),
                    "response_len": entry.get("completion_tokens", 0),
                    "reward_history": entry.get("reward_history", []),
                    "token_history": entry.get("token_history", []),
                    "model_history": entry.get("model_history", []),
                    "question_latency": float(entry.get("question_latency", 0.0) or 0.0),
                    "total_unit_latency": float(entry.get("total_unit_latency", 0.0) or 0.0),
                    "step_latency": entry.get("step_latency", []),
                    "step_lm_latency": entry.get("step_lm_latency", []),
                    "step_rm_latency": entry.get("step_rm_latency", []),
                    "step_wait": entry.get("step_wait", []),
                    "step_latency_sum": _safe_sum(entry.get("step_latency", [])),
                    "step_lm_latency_sum": _safe_sum(entry.get("step_lm_latency", [])),
                    "step_rm_latency_sum": _safe_sum(entry.get("step_rm_latency", [])),
                    "step_wait_sum": _safe_sum(entry.get("step_wait", [])),
                }
            # Accumulate total tokens for each question
            if question_idx not in per_question_total_tokens:
                per_question_total_tokens[question_idx] = 0
            per_question_total_tokens[question_idx] += entry.get("completion_tokens", 0)

        questions_map = results.get("questions", {}) if isinstance(results, dict) else {}

        def _sort_key(question_idx: str):
            parts = question_idx.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return (0, int(parts[1]))
            if question_idx.isdigit():
                return (0, int(question_idx))
            return (1, question_idx)

        best_results = []
        for question_idx in sorted(per_question_best.keys(), key=_sort_key):
            item = per_question_best[question_idx]
            if question_idx in questions_map:
                item = {"question": questions_map[question_idx], **item}
            item["question_idx"] = question_idx
            item["response_len"] = per_question_total_tokens[question_idx]  # Use total tokens for the question
            best_results.append(item)
        
        # Simplify the output
        simplified_result = {
            "task_id": task_id,
            "status": "completed",
            "questions": [q for q in questions_map.values()],
            "best_results": best_results,
            "cleanup_scheduled": cleanup_time.isoformat()
        }
        
        return simplified_result
    
    except HTTPException:
        async with router.task_lock:
            if task_id in router.active_tasks:
                router.active_tasks[task_id]['status'] = 'failed'
        raise
    except Exception as e:
        async with router.task_lock:
            if task_id in router.active_tasks:
                router.active_tasks[task_id]['status'] = 'failed'
                router.active_tasks[task_id]['error'] = str(e)
        logger.error(f"Evaluation request failed for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await router.schedule_cleanup(task_id)


@app.post("/tts-router")
async def process_request(
    problems: Optional[Union[Problem, List[Problem]]] = None,
    file: Optional[UploadFile] = File(None),
    eval_config: EvaluationConfig = Body(...)
):
    task_id, _ = router.create_task_dir()
    
    if problems is not None:
        input_data = problems
    elif file is not None:
        input_data = file
    else:
        raise HTTPException(status_code=400, detail="No input provided")
    
    save_root = router.output_base / "loadfile_beam_search"
    result = await _process_evaluation_request(task_id, input_data, eval_config, save_root)
    return JSONResponse(content=result)

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    async with router.task_lock:
        if task_id not in router.active_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        return router.active_tasks[task_id]


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_tasks": len(router.active_tasks),
        "cleanup_queue_size": len(router.cleanup_queue)
    }

@app.post("/tts-router-json")
async def process_request_json(
    problems: Union[Problem, List[Problem]],
    eval_config: EvaluationConfig = Body(...)
):
    task_id, _ = router.create_task_dir()
    save_root = router.output_base / "query_beam_search"
    result = await _process_evaluation_request(task_id, problems, eval_config, save_root)
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=1,
        log_level="info"
    )
    