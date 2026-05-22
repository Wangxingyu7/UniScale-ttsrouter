#!/bin/bash

# Parse arguments: NUM_LM LM1_PATH [LM2_PATH ...] RM_PATH HOST_ADDR CONTROLLER_PORT WORKER_BASE_PORT
# Usage: script.sh 2 model1 model2 rm_model host port worker_base_port

NUM_LM_WORKER=$1
shift

# Parse LM models
POLICY_MODELS=()
for ((i=0; i<NUM_LM_WORKER; i++)); do
    POLICY_MODELS+=("$1")
    shift
done

# Parse remaining arguments
VALUE_MODEL_PATH=$1
HOST_ADDR=$2
CONTROLLER_PORT=$3
WORKER_BASE_PORT=$4

NUM_RM_WORKER=1

echo "Number of LM workers: $NUM_LM_WORKER"
echo "LM models: ${POLICY_MODELS[@]}"
echo "RM model: $VALUE_MODEL_PATH"

# Check if model paths exist
for model in "${POLICY_MODELS[@]}"; do
    if [[ ! -d "$model" ]]; then
        echo "Warning: LM model path does not exist: $model"
    else
        echo "LM model path exists: $model"
    fi
done

if [[ ! -d "$VALUE_MODEL_PATH" ]]; then
    echo "Warning: RM model path does not exist: $VALUE_MODEL_PATH"
else
    echo "RM model path exists: $VALUE_MODEL_PATH"
fi

export CUDA_VISIBLE_DEVICES=0

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
n_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "n_gpus: $n_gpus"

# Create GPU list based on number of workers
total_workers=$((NUM_LM_WORKER + NUM_RM_WORKER))
GPU_LIST=()
for ((i=0; i<total_workers; i++)); do
    GPU_LIST+=($((i % n_gpus)))
done

echo "GPU_LIST:"
echo "${GPU_LIST[@]}"

export PYTHONPATH=$(pwd)
PYTHON_EXECUTABLE=$(which python)

LOGDIR=${PYTHONPATH}/logs_vllm
export LOGDIR=$LOGDIR
echo "PYTHONPATH: $PYTHONPATH"
echo "LOGDIR: $LOGDIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
session_name=tts
if tmux has-session -t $session_name 2>/dev/null; then
    echo "Session $session_name already exists. Killing it."
    tmux kill-session -t $session_name
fi
tmux start-server
tmux new-session -s $session_name -n controller -d
    # Start controller inside a shell that loads conda and redirects output to LOGDIR
    tmux send-keys "bash -lc 'source ~/.bashrc && conda activate tts >/dev/null 2>&1 || true && export PYTHONPATH=\"${PYTHONPATH}\" && export LOGDIR=\"${LOGDIR}\" && mkdir -p \"${LOGDIR}\" && cd \"${PYTHONPATH}\" && echo Controller PYTHONPATH: \"${PYTHONPATH}\" && echo Controller LOGDIR: \"${LOGDIR}\" && ${PYTHON_EXECUTABLE} -m reason.llm_service.controller --port ${CONTROLLER_PORT} --host ${HOST_ADDR} > \"${LOGDIR}/controller.out\" 2>&1'" Enter

sleep 5


for i in $(seq 0 $((NUM_RM_WORKER-1)))
do
    WORKER_PORT=$((WORKER_BASE_PORT+i))
    tmux new-window -n reward_$i
    
    worker_cmd="CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} ${PYTHON_EXECUTABLE} -m reason.llm_service.workers.vllm_rm_worker \
    --model-path ${VALUE_MODEL_PATH} \
    --gpu-memory-utilization 0.2 \
    --controller-address http://${HOST_ADDR}:${CONTROLLER_PORT} \
    --host ${HOST_ADDR} --port ${WORKER_PORT} \
    --worker-address http://${HOST_ADDR}:${WORKER_PORT} \
    --no-enable-chunked-prefill" 
    
    tmux send-keys "bash -lc 'source ~/.bashrc && conda activate tts >/dev/null 2>&1 || true && export PYTHONPATH=\"${PYTHONPATH}\" && export LOGDIR=\"${LOGDIR}\" && mkdir -p \"${LOGDIR}\" && cd \"${PYTHONPATH}\" && echo RM Worker PYTHONPATH: \"${PYTHONPATH}\" && echo RM Worker LOGDIR: \"${LOGDIR}\" && ${worker_cmd} > \"${LOGDIR}/rm_worker_${i}.out\" 2>&1'" Enter
    echo "Reward worker $i started on GPU ${GPU_LIST[$i]} with port $WORKER_PORT, model: $VALUE_MODEL_PATH"
done

sleep 20

for i in $(seq $((NUM_RM_WORKER)) $((NUM_LM_WORKER+NUM_RM_WORKER-1)))
do
    WORKER_PORT=$((WORKER_BASE_PORT+i))
    tmux new-window -n policy_$i
    # Start LM worker in tmux window; wrap with bash -lc to ensure conda and env vars
    tmux send-keys "bash -lc 'source ~/.bashrc && conda activate tts >/dev/null 2>&1 || true && export PYTHONPATH=\"${PYTHONPATH}\" && export LOGDIR=\"${LOGDIR}\" && mkdir -p \"${LOGDIR}\" && cd \"${PYTHONPATH}\" && echo LM Worker PYTHONPATH: \"${PYTHONPATH}\" && echo LM Worker LOGDIR: \"${LOGDIR}\"'" Enter

    policy_index=$((i - NUM_RM_WORKER))
    POLICY_MODEL_PATH="${POLICY_MODELS[$policy_index]}"

    max_model_length=4096
    max_num_sequences=4
    enforce_eager=false
    cpu_offload_gb=0

    gpu_memory_utilization=0.2 # 2 proposer + 1 verifier

    command="CUDA_VISIBLE_DEVICES=${GPU_LIST[$policy_index]} ${PYTHON_EXECUTABLE} -m reason.llm_service.workers.vllm_worker \
    --max_model_length ${max_model_length} \
    --gpu_memory_utilization ${gpu_memory_utilization} \
    --swap_space 16 --model-path $POLICY_MODEL_PATH \
    --controller-address http://$HOST_ADDR:$CONTROLLER_PORT \
    --host $HOST_ADDR --port $WORKER_PORT \
    --worker-address http://$HOST_ADDR:$WORKER_PORT"

    if [[ $max_num_sequences -gt 0 ]]; then
        command="$command --max_num_sequences $max_num_sequences"
    fi
    if [[ $enforce_eager == true ]]; then
        command="$command --enforce-eager"
    fi
    if [[ $cpu_offload_gb -gt 0 ]]; then
        command="$command --cpu-offload-gb $cpu_offload_gb"
    fi
    if [[ "$POLICY_MODEL_PATH" =~ "Qwen2.5-Math-1.5B" ]] || [[ "$POLICY_MODEL_PATH" =~ "Qwen2.5-Math-7B" ]]; then
        command="VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 $command"
    fi

    # Send the actual worker command and capture its stdout/stderr into LOGDIR
    tmux send-keys "bash -lc '${command} > \"${LOGDIR}/policy_worker_${policy_index}.out\" 2>&1'" Enter
    echo "Policy worker $i started on GPU ${GPU_LIST[$i]} with port $WORKER_PORT, model: $POLICY_MODEL_PATH"

sleep 20

done

echo "Wait 5 seconds ..."

sleep 

echo "Starting workers"