#!/bin/bash

set -euo pipefail

if [[ $# -lt 6 ]]; then
	echo "Usage: $0 <NUM_LM_WORKER> <LM1_PATH> [LM2_PATH ...] <RM_PATH> <HOST_ADDR> <CONTROLLER_PORT> <WORKER_BASE_PORT>"
	exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
	export CUDA_VISIBLE_DEVICES=0,1
fi

NUM_LM_WORKER=$1
shift

POLICY_MODELS=()
for ((i=0; i<NUM_LM_WORKER; i++)); do
	POLICY_MODELS+=("$1")
	shift
done

VALUE_MODEL_PATH=$1
HOST_ADDR=$2
CONTROLLER_PORT=$3
WORKER_BASE_PORT=$4

NUM_RM_WORKER=1

IFS=',' read -ra AVAILABLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
n_gpus=${#AVAILABLE_GPU_IDS[@]}

if (( n_gpus < 2 )); then
	echo "Error: serve_a100x2_t1.sh expects exactly 2 visible GPUs (found $n_gpus)."
	exit 1
fi

validate_devices() {
	local devices="$1"
	local IFS=','
	read -ra dev_list <<< "$devices"
	for dev in "${dev_list[@]}"; do
		if [[ -z "$dev" || ! "$dev" =~ ^[0-9]+$ ]]; then
			echo "Error: invalid GPU id '$dev' in assignment '$devices'."
			exit 1
		fi
		local found=0
		for avail in "${AVAILABLE_GPU_IDS[@]}"; do
			if [[ "$avail" == "$dev" ]]; then
				found=1
				break
			fi
		done
		if (( ! found )); then
			echo "Error: GPU id $dev not in CUDA_VISIBLE_DEVICES ($CUDA_VISIBLE_DEVICES)."
			exit 1
		fi
		if (( dev >= n_gpus )); then
			echo "Error: GPU id $dev exceeds visible GPU count ($n_gpus)."
			exit 1
		fi
	done
}

tensor_parallel_size() {
	local devices="$1"
	local IFS=','
	read -ra dev_list <<< "$devices"
	echo "${#dev_list[@]}"
}

gpu_memory_target() {
	local model_path="$1"
	case "$model_path" in
		*Qwen3-32B* ) echo "0.42" ;;
		*Qwen3-14B* ) echo "0.37" ;;
		*Qwen3-8B* ) echo "0.22" ;;
		*Qwen3-4B* ) echo "0.12" ;;
		*Qwen3-1.7B* ) echo "0.07" ;;
		*Qwen3-0.6B* ) echo "0.05" ;;
		*Skywork*1.5B* | *skywork_1.5b_prm* ) echo "0.07" ;;
		* ) echo "0.30" ;;
	esac
}

assign_gpu_for_policy() {
	local model_path="$1"
	case "$model_path" in
		*Qwen3-32B* ) echo "0,1" ;;
		*Qwen3-14B* ) echo "0" ;;
		*Qwen3-8B* ) echo "1" ;;
		*Qwen3-4B* ) echo "1" ;;
		*Qwen3-1.7B* ) echo "0" ;;
		*Qwen3-0.6B* ) echo "1" ;;
		*Skywork*1.5B* | *skywork_1.5b_prm* ) echo "1" ;;
		* ) echo "0" ;;
	esac
}

export PYTHONPATH=$(pwd)
PYTHON_EXECUTABLE=$(which python)

LOGDIR=${PYTHONPATH}/logs_vllm
export LOGDIR=$LOGDIR
mkdir -p "$LOGDIR"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Detected policy models: ${POLICY_MODELS[*]}"
echo "Reward model: $VALUE_MODEL_PATH"

for model in "${POLICY_MODELS[@]}"; do
	if [[ ! -d "$model" ]]; then
		echo "Warning: LM model path does not exist: $model"
	fi
done

if [[ ! -d "$VALUE_MODEL_PATH" ]]; then
	echo "Warning: RM model path does not exist: $VALUE_MODEL_PATH"
fi

session_name=tts_a100x2
if tmux has-session -t $session_name 2>/dev/null; then
	echo "Session $session_name already exists. Killing it."
	tmux kill-session -t $session_name
fi

tmux start-server
tmux new-session -s $session_name -n controller -d
controller_log="${LOGDIR}/controller_a100x2.out"
controller_cmd="source ~/.bashrc && conda activate tts >/dev/null 2>&1 || true && export PYTHONPATH=${PYTHONPATH} && export LOGDIR=${LOGDIR} && cd ${PYTHONPATH} && ${PYTHON_EXECUTABLE} -m reason.llm_service.controller --port ${CONTROLLER_PORT} --host ${HOST_ADDR} > ${controller_log} 2>&1"
tmux send-keys "bash -lc '${controller_cmd}'" Enter

echo "Controller started (log: ${controller_log})."

sleep 5

rm_gpu="1"
rm_port=$WORKER_BASE_PORT
rm_log="${LOGDIR}/rm_worker_a100x2.out"
validate_devices "$rm_gpu"
rm_mem_target=$(gpu_memory_target "$VALUE_MODEL_PATH")
rm_cmd="CUDA_VISIBLE_DEVICES=${rm_gpu} ${PYTHON_EXECUTABLE} -m reason.llm_service.workers.vllm_rm_worker \\
	--model-path ${VALUE_MODEL_PATH} \\
	--gpu-memory-utilization ${rm_mem_target} \\
	--controller-address http://${HOST_ADDR}:${CONTROLLER_PORT} \\
	--host ${HOST_ADDR} --port ${rm_port} \\
	--worker-address http://${HOST_ADDR}:${rm_port} \\
	--no-enable-chunked-prefill > ${rm_log} 2>&1"

tmux new-window -t ${session_name}:1 -n rm_worker
tmux send-keys "bash -lc 'source ~/.bashrc && conda activate tts >/dev/null 2>&1 || true && export PYTHONPATH=${PYTHONPATH} && export LOGDIR=${LOGDIR} && cd ${PYTHONPATH} && ${rm_cmd}'" Enter

echo "Reward model worker started on GPU ${rm_gpu} (port ${rm_port})."

sleep 40

POLICY_BASE_INDEX=$NUM_RM_WORKER

declare -a MULTI_GPU_POLICIES=()

for i in $(seq $POLICY_BASE_INDEX $((NUM_LM_WORKER+POLICY_BASE_INDEX-1))); do
	policy_index=$((i - POLICY_BASE_INDEX))
	policy_path="${POLICY_MODELS[$policy_index]}"
	policy_gpu=$(assign_gpu_for_policy "$policy_path")
	if [[ "$policy_gpu" == *,* ]]; then
		MULTI_GPU_POLICIES+=("${i}:${policy_index}:${policy_path}:${policy_gpu}")
		continue
	fi

	WORKER_PORT=$((WORKER_BASE_PORT+i))
	policy_log="${LOGDIR}/policy_worker_${policy_index}.out"
	validate_devices "$policy_gpu"
	gpu_mem=$(gpu_memory_target "$policy_path")
	command="CUDA_VISIBLE_DEVICES=${policy_gpu} ${PYTHON_EXECUTABLE} -m reason.llm_service.workers.vllm_worker \\
		--max_model_length 4096 \\
		--gpu_memory_utilization ${gpu_mem} \\
		--swap_space 16 --model-path ${policy_path} \\
		--controller-address http://${HOST_ADDR}:${CONTROLLER_PORT} \\
		--host ${HOST_ADDR} --port ${WORKER_PORT} \\
		--worker-address http://${HOST_ADDR}:${WORKER_PORT} > ${policy_log} 2>&1"

	tmux new-window -t ${session_name}:$((i+1)) -n policy_${policy_index}
	tmux send-keys "bash -lc 'source ~/.bashrc && conda activate tts >/dev/null 2>&1 || true && export PYTHONPATH=${PYTHONPATH} && export LOGDIR=${LOGDIR} && cd ${PYTHONPATH} && ${command}'" Enter
	echo "Policy worker ${policy_index} (${policy_path##*/}) started on GPU ${policy_gpu} (port ${WORKER_PORT})."
	sleep 60
done

sleep 20

for entry in "${MULTI_GPU_POLICIES[@]}"; do
	IFS=':' read -r worker_idx policy_index policy_path policy_gpu <<< "$entry"
	WORKER_PORT=$((WORKER_BASE_PORT+worker_idx))
	policy_log="${LOGDIR}/policy_worker_${policy_index}.out"
	validate_devices "$policy_gpu"
	tp_size=$(tensor_parallel_size "$policy_gpu")
	gpu_mem=$(gpu_memory_target "$policy_path")
	command="CUDA_VISIBLE_DEVICES=${policy_gpu} ${PYTHON_EXECUTABLE} -m reason.llm_service.workers.vllm_worker \\
		--max_model_length 4096 \\
		--gpu_memory_utilization ${gpu_mem} \\
		--tensor-parallel-size ${tp_size} \\
		--swap_space 16 --model-path ${policy_path} \\
		--controller-address http://${HOST_ADDR}:${CONTROLLER_PORT} \\
		--host ${HOST_ADDR} --port ${WORKER_PORT} \\
		--worker-address http://${HOST_ADDR}:${WORKER_PORT} > ${policy_log} 2>&1"

	tmux new-window -t ${session_name}:$((worker_idx+1)) -n policy_${policy_index}
	tmux send-keys "bash -lc 'source ~/.bashrc && conda activate tts >/dev/null 2>&1 || true && export PYTHONPATH=${PYTHONPATH} && export LOGDIR=${LOGDIR} && cd ${PYTHONPATH} && ${command}'" Enter
	echo "Policy worker ${policy_index} (${policy_path##*/}) started on GPUs ${policy_gpu} (port ${WORKER_PORT})."
	sleep 40
done

echo "Controller, reward model, and all policy workers are up."
