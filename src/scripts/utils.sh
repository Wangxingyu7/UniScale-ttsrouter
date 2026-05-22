# Some commands that are convenient for developers to use; you can adjust or use them if you know their purpose.
## for serve_gpu1_t1.sh l40
cd src
export VALUE_MODEL_PATH=/root/autodl-pub/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
export POLICY_MODEL_1_PATH=/root/autodl-pub/policy_models/Qwen3-0.6B
export POLICY_MODEL_2_PATH=/root/autodl-pub/policy_models/Qwen3-1.7B && LOGDIR="${PWD}/logs_vllm"
export HOST_ADDR=0.0.0.0 && export CONTROLLER_PORT=21001 && export WORKER_BASE_PORT=10081

## for serve_gpu1_t1.sh
bash scripts/serve_gpu1_t1.sh 2 $POLICY_MODEL_1_PATH $POLICY_MODEL_2_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT

## for serve_gpu2_t1.sh 
## for 5 lm + 1 rm 0.6b + 1.7b + 4b + 8b + 14b + skywork1.5b
## 5090
export VALUE_MODEL_PATH=/home/xingyu/pub/skywork-prm-1.5b
export POLICY_MODEL_1_PATH=/home/xingyu/pub/policy_models/Qwen3-0.6B
export POLICY_MODEL_2_PATH=/home/xingyu/pub/policy_models/Qwen3-1.7B
export POLICY_MODEL_3_PATH=/home/xingyu/pub/policy_models/Qwen3-4B
export POLICY_MODEL_4_PATH=/home/xingyu/pub/policy_models/Qwen3-8B
export POLICY_MODEL_5_PATH=/home/xingyu/pub/policy_models/Qwen3-14B
export LOGDIR="${PWD}/logs_vllm"
export HOST_ADDR=0.0.0.0
export CONTROLLER_PORT=21001
export WORKER_BASE_PORT=10081

# 7B PRM
export VALUE_MODEL_PATH=/home/xingyu/pub/Skywork-o1-Open-PRM-Qwen-2.5-7B
export POLICY_MODEL_1_PATH=/home/xingyu/pub/policy_models/Qwen3-0.6B
export POLICY_MODEL_2_PATH=/home/xingyu/pub/policy_models/Qwen3-1.7B
export POLICY_MODEL_3_PATH=/home/xingyu/pub/policy_models/Qwen3-4B
export POLICY_MODEL_4_PATH=/home/xingyu/pub/policy_models/Qwen3-8B
export POLICY_MODEL_5_PATH=/home/xingyu/pub/policy_models/Qwen3-14B
export LOGDIR="${PWD}/logs_vllm"
export HOST_ADDR=0.0.0.0
export CONTROLLER_PORT=21001
export WORKER_BASE_PORT=10081

bash scripts/serve_gpu2_t1.sh 5 $POLICY_MODEL_1_PATH $POLICY_MODEL_2_PATH $POLICY_MODEL_3_PATH $POLICY_MODEL_4_PATH $POLICY_MODEL_5_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT


## for a100x2_t1.sh
export VALUE_MODEL_PATH=/wangxingyu/pub/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
export POLICY_MODEL_1_PATH=/wangxingyu/pub/policy_models/Qwen3-0.6B
export POLICY_MODEL_2_PATH=/wangxingyu/pub/policy_models/Qwen3-1.7B
export POLICY_MODEL_3_PATH=/wangxingyu/pub/policy_models/Qwen3-4B
export POLICY_MODEL_4_PATH=/wangxingyu/pub/policy_models/Qwen3-8B
export POLICY_MODEL_5_PATH=/wangxingyu/pub/policy_models/Qwen3-14B
export POLICY_MODEL_6_PATH=/wangxingyu/pub/policy_models/Qwen3-32B
export LOGDIR="${PWD}/logs_vllm"
export HOST_ADDR=0.0.0.0
export CONTROLLER_PORT=21001
export WORKER_BASE_PORT=10081   
bash scripts/serve_a100x2.sh 6 $POLICY_MODEL_1_PATH $POLICY_MODEL_2_PATH $POLICY_MODEL_3_PATH $POLICY_MODEL_4_PATH $POLICY_MODEL_5_PATH $POLICY_MODEL_6_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT

## for serve_l40_rm32b.sh
## for 1 rm + 1 lm (32B) on dual L40
export VALUE_MODEL_PATH=/root/autodl-pub/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
export POLICY_MODEL_1_PATH=/root/autodl-pub/policy_models/Qwen3-32B
export LOGDIR="${PWD}/logs_vllm"
export HOST_ADDR=0.0.0.0
export CONTROLLER_PORT=21001
export WORKER_BASE_PORT=10081

bash scripts/serve_l40_rm32b.sh 1 $POLICY_MODEL_1_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT



## Start the TTS router service
python -m service.serve

### or with nohup
nohup python -m service.serve

## Start the TTS evaluation
## for serve_gpu1_t1.sh l40
bash scripts/run_t1.sh --method beam_search --RM $VALUE_MODEL_PATH --task_name AMC23_t1 

## 5090
bash scripts/run_t1.sh --method beam_search --LM "/home/xingyu/pub/policy_models/Qwen3-0.6B,/home/xingyu/pub/policy_models/Qwen3-1.7B,/home/xingyu/pub/policy_models/Qwen3-4B,/home/xingyu/pub/policy_models/Qwen3-8B,/home/xingyu/pub/policy_models/Qwen3-14B" --RM $VALUE_MODEL_PATH --task_name AMC23_t1 

## for serve_l40_rm32b.sh
bash scripts/run_t1.sh --method beam_search --LM "/root/autodl-pub/policy_models/Qwen3-32B" --RM $VALUE_MODEL_PATH --task_name AMC23_t1

## 单问题
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

# 17777
curl -X POST http://localhost:17777/tts-router-json \
  -H "Content-Type: application/json" \
  -d '{
    "problems": {
      "problem": "Cities $A$ and $B$ are $45$ miles apart. Alicia lives in $A$ and Beth lives in $B$. Alicia bikes towards $B$ at 18 miles per hour. Leaving at the same time, Beth bikes toward $A$ at 12 miles per hour. How many miles from City $A$ will they be when they meet?",
      "solution": "27.0",
      "lm": "Qwen3-32B",
      "beam": { "QP": 2.0, "CP": 8.0, "BS": 4 }
    },
    "eval_config": {
      "method": "beam_search"
    }
  }'


## 多问题
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

## 文件类
curl -X POST http://localhost:7777/tts-router \
  -F "file=@./test.jsonl;type=application/jsonl" \
  -F "eval_config=@-;type=application/json" <<'JSON'
{"method":"beam_search"}
JSON