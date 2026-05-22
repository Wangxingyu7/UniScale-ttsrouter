#!/bin/bash

# Default arguments
LM=/root/autodl-pub/policy_models/Qwen3-0.6B,/root/autodl-pub/policy_models/Qwen3-1.7B
RM=/root/autodl-pub/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
task_name=AMC23_t1 # {MATH, AMC23, AMC23_t1, AMIE24}
method=beam_search
temperature=0.6
top_p=0.95
top_k=20
max_new_tokens=4096
tree_max_depth=1000 
tree_max_width=4   #CP
num_sequence=2     #BS
question_parallel_num=2  #QP
batch_size=10000 # you can adjust it to a specific task limit.
max_time=3
n_gpus=1
double_line_break=1
local=0
num_worker=4





# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --LM)
        LM="$2"
        shift 2
        ;;
    --RM)
        RM="$2"
        shift 2
        ;;
    --task_name)
        task_name="$2"
        shift 2
        ;;
    --method)
        method="$2"
        shift 2
        ;;
    --temperature)
        temperature="$2"
        shift 2
        ;;
    --max_new_tokens)
        max_new_tokens="$2"
        shift 2
        ;;
    --tree_max_depth)
        tree_max_depth="$2"
        shift 2
        ;;
    --width)
        tree_max_width="$2"
        shift 2
        ;;
    --num_seq)
        num_sequence="$2"
        shift 2
        ;;
    --num_q)
        question_parallel_num="$2"
        shift 2
        ;;
    --bs)
        batch_size="$2"
        shift 2
        ;;
    --mt)
        max_time="$2"
        shift 2
        ;;
    --n_gpus)
        n_gpus="$2"
        shift 2
        ;;
    --num_worker)
        num_worker="$2"
        shift 2
        ;;
    --double_line_break)
        double_line_break="$2"
        shift 2
        ;;
    --local)
        local="$2"
        shift 2
        ;;
    *)
        echo "Unknown parameter: $1"
        exit 1
        ;;
    esac
done
echo "LM: $LM, RM: $RM, task: $task_name, tree_max_width: $tree_max_width, num_sequence: $num_sequence, question_parallel_num: $question_parallel_num"
echo "batch_size: $batch_size, max_time: $max_time, n_gpus: $n_gpus, num_worker: $num_worker, double_line_break: $double_line_break"

if [ "$method" == "beam_search" ]; then
    if [ -z "$temperature" ]; then temperature=0.7; fi
    if [ -z "$max_new_tokens" ]; then max_new_tokens=2048; fi
    if [ -z "$tree_max_depth" ]; then tree_max_depth=40; fi
elif [ "$method" == "best_of_n" ]; then
    if [ -z "$temperature" ]; then temperature=0.7; fi
    if [ -z "$max_new_tokens" ]; then max_new_tokens=8192; fi
    if [ -z "$tree_max_depth" ]; then tree_max_depth=1; fi
elif [ "$method" == "cot" ]; then
    if [ -z "$temperature" ]; then temperature=0.0; fi
    if [ -z "$max_new_tokens" ]; then max_new_tokens=8192; fi
    if [ -z "$tree_max_depth" ]; then tree_max_depth=1; fi
else
    echo "Invalid method: $method"
    exit
fi

POLICY_MODEL_PATH=${LM}
VALUE_MODEL_PATH=${RM}

export PYTHONPATH=$(pwd)
cd ${PYTHONPATH}

export CUDA_VISIBLE_DEVICES=0
GPU_LIST=($(seq 0 $((n_gpus-1))))
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES, n_gpus: $n_gpus"
echo "GPU_LIST:"
echo "${GPU_LIST[@]}"

save_dir=${PYTHONPATH}/output

if [ -n "$TTS_SAVE_DIR" ]; then
    save_dir="$TTS_SAVE_DIR"
fi
LOGDIR=${PYTHONPATH}/logs_vllm
export LOGDIR=$LOGDIR
controller_addr=http://$HOST_ADDR:$CONTROLLER_PORT

echo "Running $method evaluation ..."

python reason/evaluation/evaluate.py \
    --LM $POLICY_MODEL_PATH \
    --RM $VALUE_MODEL_PATH \
    --task_name $task_name \
    --temperature $temperature \
    --max_new_tokens $max_new_tokens \
    --num_sequence $num_sequence \
    --tree_max_width $tree_max_width \
    --tree_max_depth $tree_max_depth \
    --save_dir $save_dir \
    --method $method \
    --num_worker $num_worker \
    --controller_addr $controller_addr \
    --add_step_prompt \
    --question_parallel_num $question_parallel_num \
    --double_line_break $double_line_break \
    --batch_size $batch_size \
    --max_time $max_time \
    --local $local \
    --top_p $top_p \
    --top_k $top_k