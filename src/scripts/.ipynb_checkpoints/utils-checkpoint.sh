# for serve_gpu1_t1.sh
cd src
export VALUE_MODEL_PATH=/root/autodl-pub/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
export POLICY_MODEL_1_PATH=/root/autodl-pub/policy_models/Qwen3-0.6B
export POLICY_MODEL_2_PATH=/root/autodl-pub/policy_models/Qwen3-1.7B && LOGDIR="${PWD}/logs_vllm"
export HOST_ADDR=0.0.0.0 && export CONTROLLER_PORT=21001 && export WORKER_BASE_PORT=10081


bash scripts/serve_gpu1_t1.sh 2 $POLICY_MODEL_1_PATH $POLICY_MODEL_2_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT

bash scripts/run_t1.sh --method beam_search --RM $VALUE_MODEL_PATH --task_name AMC23_t1 

python -m service.serve

# 单问题
curl -X POST http://localhost:7777/tts-router-json \
  -H "Content-Type: application/json" \
  -d '{
    "problems": {
      "problem": "Cities $A$ and $B$ are $45$ miles apart. Alicia lives in $A$ and Beth lives in $B$. Alicia bikes towards $B$ at 18 miles per hour. Leaving at the same time, Beth bikes toward $A$ at 12 miles per hour. How many miles from City $A$ will they be when they meet?",
      "solution": "27.0",
      "lm": "Qwen3-0.6B",
      "beam": { "QP": 2.0, "CP": 8.0, "BS": 4 }
    },
    "eval_config": {
      "method": "beam_search"
    }
  }'

# 多问题
curl -X POST http://localhost:7777/tts-router-json \
  -H "Content-Type: application/json" \
  --data-binary @- <<'JSON'
{
  "problems": [
    {
      "problem": "How many digits are in the base-ten representation of $8^5 \\cdot 5^{10} \\cdot 15^5$?",
      "solution": "18.0",
      "lm": "Qwen3-0.6B",
      "beam": { "QP": 2.0, "CP": 8.0, "BS": 4 }
    },
    {
      "problem": "Cities $A$ and $B$ are $45$ miles apart. Alicia lives in $A$ and Beth lives in $B$. Alicia bikes towards $B$ at 18 miles per hour. Leaving at the same time, Beth bikes toward $A$ at 12 miles per hour. How many miles from City $A$ will they be when they meet?",
      "solution": "27.0",
      "lm": "Qwen3-0.6B",
      "beam": { "QP": 2.0, "CP": 8.0, "BS": 4 }
    }
  ],
  "eval_config": { "method": "beam_search" }
}
JSON

# 文件类
curl -X POST http://localhost:7777/tts-router \
  -F "file=@./test.jsonl;type=application/jsonl" \
  -F "eval_config=@-;type=application/json" <<'JSON'
{"method":"beam_search"}
JSON