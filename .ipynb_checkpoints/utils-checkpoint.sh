# 可行版
# for serve_gup1.sh
cd src
export VALUE_MODEL_PATH=/root/autodl-pub/Qwen2.5-Math-PRM-7B 
export POLICY_MODEL_PATH=/root/autodl-pub/DeepSeek-R1-Distill-Qwen-1.5B && export LOGDIR="$PWD/logs"
export HOST_ADDR=0.0.0.0 && export CONTROLLER_PORT=10014 && export WORKER_BASE_PORT=10081

# 1 gpu
bash scripts/serve_gpu1.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT

# 2 gpus (32B policy model + 1.5B-8B PRM)
bash scripts/serve_gpu2.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT

# 3 gpus (72B policy model + 1.5B-8B PRM)
bash scripts/serve_gpu3_1-2.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT

# 3 gpus (0.5B-32B policy model + 72B PRM)
bash scripts/serve_gpu3_2-1.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT

# 4 gpus (72B policy model + 72B PRM)
bash scripts/serve_gpu4.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT


# run

bash scripts/run.sh --method beam_search --LM $POLICY_MODEL_PATH --RM $VALUE_MODEL_PATH --width 4 --num_seq 1


#以下为模型测试部分

## skywork 通过
export VALUE_MODEL_PATH=/root/autodl-pub/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
export POLICY_MODEL_PATH=/root/autodl-pub/DeepSeek-R1-Distill-Qwen-1.5B && export LOGDIR="$PWD/logs"
export HOST_ADDR=0.0.0.0 && export CONTROLLER_PORT=10014 && export WORKER_BASE_PORT=10081

## Qwen3 通过，需解决后续部分不对齐问题
export VALUE_MODEL_PATH=/root/autodl-pub/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
export POLICY_MODEL_PATH=/root/autodl-pub/policy_models/Qwen3-0.6B && export LOGDIR="$PWD/logs"
export HOST_ADDR=0.0.0.0 && export CONTROLLER_PORT=10014 && export WORKER_BASE_PORT=10081



# 以下为阶段测试部分
## 测试一 2 proposer+1verifier 未通过(逻辑不好解决)



# for serve_gpu1_add.sh
cd src
export VALUE_MODEL_PATH=/root/autodl-pub/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
export POLICY_MODEL_1_PATH=/root/autodl-pub/policy_models/Qwen3-0.6B
export POLICY_MODEL_2_PATH=/root/autodl-pub/policy_models/Qwen3-1.7B && export LOGDIR=path/to/logdir
export HOST_ADDR=0.0.0.0 && export CONTROLLER_PORT=10014 && export WORKER_BASE_PORT=10081


bash scripts/serve_gpu1_t1.sh $POLICY_MODEL_1_PATH $POLICY_MODEL_2_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT

#--width 4 --num_seq 1改为程序内输入而非shell
bash scripts/run_t1.sh 



## 测试二



## 测试三 dataset
对于数学问题，可以通过改/MATH/data.py, src/envs/__init__.py等解决
思考后建议先把所有问题拼成一个jsonl，再整体评估，（但是如何解决QP.CP,BS不一致的问题？把同样的值分一类？）


## 测试四
