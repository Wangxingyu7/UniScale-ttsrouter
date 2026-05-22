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

if (( NUM_LM_WORKER != 1 )); then
	echo "Error: serve_l40_rm32b.sh supports exactly 1 LM worker; received ${NUM_LM_WORKER}."
	exit 1
fi

POLICY_MODEL_PATH=${POLICY_MODELS[0]}

POLICY_PORT=$((WORKER_BASE_PORT+1))
RM_PORT=$WORKER_BASE_PORT

IFS=',' read -ra AVAILABLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
n_gpus=${#AVAILABLE_GPU_IDS[@]}

if (( n_gpus < 2 )); then
	echo "Error: serve_l40_rm32b.sh expects at least 2 visible GPUs (found $n_gpus)."
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
		*32B* ) echo "0.73" ;;
		*Skywork*1.5B* ) echo "0.10" ;;
		* ) echo "0.75" ;;
	esac
}

export PYTHONPATH=$(pwd)
PYTHON_EXECUTABLE=$(which python)

LOGDIR=${PYTHONPATH}/logs_vllm
export LOGDIR=$LOGDIR
mkdir -p "$LOGDIR"

if [[ ! -d "$POLICY_MODEL_PATH" ]]; then
	echo "Warning: policy model path does not exist: $POLICY_MODEL_PATH"
fi

if [[ ! -d "$VALUE_MODEL_PATH" ]]; then
	echo "Warning: RM model path does not exist: $VALUE_MODEL_PATH"
fi

tmux_session=tts_l40
if tmux has-session -t $tmux_session 2>/dev/null; then
	echo "Session $tmux_session already exists. Killing it."
	tmux kill-session -t $tmux_session
fi

controller_log="${LOGDIR}/controller_l40.out"
tmux start-server
tmux new-session -s $tmux_session -n controller -d

controller_cmd="source ~/.bashrc && conda activate tts >/dev/null 2>&1 || true && export PYTHONPATH=${PYTHONPATH} && export LOGDIR=${LOGDIR} && cd ${PYTHONPATH} && ${PYTHON_EXECUTABLE} -m reason.llm_service.controller --port ${CONTROLLER_PORT} --host ${HOST_ADDR} > ${controller_log} 2>&1"

tmux send-keys "bash -lc '${controller_cmd}'" Enter
sleep 5

rm_log="${LOGDIR}/rm_worker_l40.out"
rm_gpu="0"
rm_mem_target=$(gpu_memory_target "$VALUE_MODEL_PATH")
validate_devices "$rm_gpu"
rm_cmd="CUDA_VISIBLE_DEVICES=${rm_gpu} ${PYTHON_EXECUTABLE} -m reason.llm_service.workers.vllm_rm_worker \
	--model-path ${VALUE_MODEL_PATH} \
	--gpu-memory-utilization ${rm_mem_target} \
	--controller-address http://${HOST_ADDR}:${CONTROLLER_PORT} \
	--host ${HOST_ADDR} --port ${RM_PORT} \
	--worker-address http://${HOST_ADDR}:${RM_PORT} \
	--no-enable-chunked-prefill > ${rm_log} 2>&1"

tmux new-window -t ${tmux_session}:1 -n rm_worker
tmux send-keys "bash -lc 'source ~/.bashrc && conda activate tts >/dev/null 2>&1 || true && export PYTHONPATH=${PYTHONPATH} && export LOGDIR=${LOGDIR} && cd ${PYTHONPATH} && ${rm_cmd}'" Enter

echo "Reward model worker started on GPU ${rm_gpu} (port ${RM_PORT})."

sleep 20

policy_log="${LOGDIR}/policy_worker_32b.out"
policy_gpus="0,1"
validate_devices "$policy_gpus"
tp_size=$(tensor_parallel_size "$policy_gpus")
policy_mem_target=$(gpu_memory_target "$POLICY_MODEL_PATH")
policy_cmd="CUDA_VISIBLE_DEVICES=${policy_gpus} ${PYTHON_EXECUTABLE} -m reason.llm_service.workers.vllm_worker \
	--max_model_length 4096 \
	--gpu_memory_utilization ${policy_mem_target} \
	--tensor-parallel-size ${tp_size} \
	--swap_space 16 --model-path ${POLICY_MODEL_PATH} \
	--controller-address http://${HOST_ADDR}:${CONTROLLER_PORT} \
	--host ${HOST_ADDR} --port ${POLICY_PORT} \
	--worker-address http://${HOST_ADDR}:${POLICY_PORT} > ${policy_log} 2>&1"

tmux new-window -t ${tmux_session}:2 -n policy_worker

tmux send-keys "bash -lc 'source ~/.bashrc && conda activate tts >/dev/null 2>&1 || true && export PYTHONPATH=${PYTHONPATH} && export LOGDIR=${LOGDIR} && cd ${PYTHONPATH} && ${policy_cmd}'" Enter

echo "32B policy worker started on GPUs ${policy_gpus} (port ${POLICY_PORT})."

sleep 20

echo "Controller, RM worker, and 32B policy worker are up."
