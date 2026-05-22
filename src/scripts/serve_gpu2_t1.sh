#!/bin/bash
# bash scripts/serve_gpu2_t1.sh 2 $POLICY_MODEL_1_PATH $POLICY_MODEL_2_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT
set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
	export CUDA_VISIBLE_DEVICES=0,1,2,3
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

echo "Number of LM workers: $NUM_LM_WORKER"
echo "LM models: ${POLICY_MODELS[@]}"
echo "RM model: $VALUE_MODEL_PATH"

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

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
n_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if (( n_gpus < 1 )); then
	echo "Error: serve_gpu2_t1.sh expects at least 1 visible GPU (found $n_gpus)."
	exit 1
fi
echo "n_gpus: $n_gpus"

IFS=',' read -ra AVAILABLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"

GPU_LIST=()
declare -a MULTI_GPU_POLICIES=()
MULTI_GPU_DEVICES="2,3"


assign_gpu_for_policy() {
	local model_path="$1"
	if [[ "$model_path" =~ Qwen3-8B ]]; then
		echo "1"
	elif [[ "$model_path" =~ Qwen3-1.7B ]]; then
		echo "1"
	elif [[ "$model_path" =~ Qwen3-4B ]]; then
		echo "1"
	else
		echo "0"
	fi
}

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

infer_gpu_utilization() {
	local model_path="$1"
	case "$model_path" in
		*Qwen3-0.6B* ) echo "0.18" ;;
		*Qwen3-1.7B* ) echo "0.15" ;;
		*Qwen3-4B*   ) echo "0.35" ;;
		*Qwen3-8B*   ) echo "0.55" ;;
		*Qwen3-14B*  ) echo "0.50" ;;
		#*Skywork*1.5B* ) echo "0.18" ;;
		*Skywork*1.5B* ) echo "0.48" ;;
		* ) echo "0.48" ;;
	esac
}

export PYTHONPATH=$(pwd)
PYTHON_EXECUTABLE=$(which python)

LOGDIR=${PYTHONPATH}/logs_vllm
export LOGDIR=$LOGDIR
echo "PYTHONPATH: $PYTHONPATH"
echo "LOGDIR: $LOGDIR"

session_name=tts
if tmux has-session -t $session_name 2>/dev/null; then
	echo "Session $session_name already exists. Killing it."
	tmux kill-session -t $session_name
fi

tmux start-server
tmux new-session -s $session_name -n controller -d
tmux send-keys "bash -lc 'source ~/.bashrc && conda activate tts >/dev/null 2>&1 || true && export PYTHONPATH="${PYTHONPATH}" && export LOGDIR="${LOGDIR}" && mkdir -p "${LOGDIR}" && cd "${PYTHONPATH}" && ${PYTHON_EXECUTABLE} -m reason.llm_service.controller --port ${CONTROLLER_PORT} --host ${HOST_ADDR} > "${LOGDIR}/controller.out" 2>&1'" Enter

sleep 5

for i in $(seq 0 $((NUM_RM_WORKER-1))); do
	WORKER_PORT=$((WORKER_BASE_PORT+i))
	tmux new-window -n reward_$i
	rm_gpu_utilization=$(infer_gpu_utilization "$VALUE_MODEL_PATH")
	TARGET_GPU=3
	GPU_LIST[$i]=$TARGET_GPU
	worker_cmd="CUDA_VISIBLE_DEVICES=${TARGET_GPU} ${PYTHON_EXECUTABLE} -m reason.llm_service.workers.vllm_rm_worker \
		--model-path ${VALUE_MODEL_PATH} \\
		--gpu-memory-utilization ${rm_gpu_utilization} \\
		--controller-address http://${HOST_ADDR}:${CONTROLLER_PORT} \\
		--host ${HOST_ADDR} --port ${WORKER_PORT} \\
		--worker-address http://${HOST_ADDR}:${WORKER_PORT} \\
		--no-enable-chunked-prefill"

	tmux send-keys "bash -lc 'source ~/.bashrc && conda activate tts >/dev/null 2>&1 || true && export PYTHONPATH="${PYTHONPATH}" && export LOGDIR="${LOGDIR}" && mkdir -p "${LOGDIR}" && cd "${PYTHONPATH}" && ${worker_cmd} > "${LOGDIR}/rm_worker_${i}.out" 2>&1'" Enter
	echo "Reward worker $i started on GPU ${GPU_LIST[$i]} (port $WORKER_PORT)"
done

sleep 20

for i in $(seq $((NUM_RM_WORKER)) $((NUM_LM_WORKER+NUM_RM_WORKER-1))); do
	policy_index=$((i - NUM_RM_WORKER))
	POLICY_MODEL_PATH="${POLICY_MODELS[$policy_index]}"

	if [[ "$POLICY_MODEL_PATH" =~ Qwen3-14B ]]; then
		MULTI_GPU_POLICIES+=("${i}:${policy_index}:${POLICY_MODEL_PATH}")
		continue
	fi

	WORKER_PORT=$((WORKER_BASE_PORT+i))
	tmux new-window -n policy_$i
	tmux send-keys "bash -lc 'source ~/.bashrc && conda activate tts >/dev/null 2>&1 || true && export PYTHONPATH="${PYTHONPATH}" && export LOGDIR="${LOGDIR}" && mkdir -p "${LOGDIR}" && cd "${PYTHONPATH}"'" Enter

	gpu_memory_utilization=$(infer_gpu_utilization "$POLICY_MODEL_PATH")
	policy_gpu=$(assign_gpu_for_policy "$POLICY_MODEL_PATH")
	validate_devices "$policy_gpu"
	GPU_LIST[$i]=$policy_gpu
	command="CUDA_VISIBLE_DEVICES=${policy_gpu} ${PYTHON_EXECUTABLE} -m reason.llm_service.workers.vllm_worker \
		--max_model_length 4096 \
		--gpu_memory_utilization ${gpu_memory_utilization} \
		--swap_space 16 --model-path $POLICY_MODEL_PATH \
		--controller-address http://${HOST_ADDR}:${CONTROLLER_PORT} \
		--host ${HOST_ADDR} --port ${WORKER_PORT} \
		--worker-address http://${HOST_ADDR}:${WORKER_PORT}"

	tmux send-keys "bash -lc '${command} > "${LOGDIR}/policy_worker_${policy_index}.out" 2>&1'" Enter
	echo "Policy worker $policy_index started on GPU ${GPU_LIST[$i]} (port $WORKER_PORT)"
	sleep 20

done

for entry in "${MULTI_GPU_POLICIES[@]}"; do
	IFS=':' read -r worker_idx policy_index policy_path <<< "$entry"
	WORKER_PORT=$((WORKER_BASE_PORT+worker_idx))
	tmux new-window -n policy_$worker_idx
	tmux send-keys "bash -lc 'source ~/.bashrc && conda activate tts >/dev/null 2>&1 || true && export PYTHONPATH="${PYTHONPATH}" && export LOGDIR="${LOGDIR}" && mkdir -p "${LOGDIR}" && cd "${PYTHONPATH}"'" Enter

	gpu_memory_utilization=$(infer_gpu_utilization "$policy_path")
	validate_devices "$MULTI_GPU_DEVICES"
	tp_size=$(tensor_parallel_size "$MULTI_GPU_DEVICES")
	GPU_LIST[$worker_idx]=$MULTI_GPU_DEVICES
	command="CUDA_VISIBLE_DEVICES=${MULTI_GPU_DEVICES} ${PYTHON_EXECUTABLE} -m reason.llm_service.workers.vllm_worker \
		--max_model_length 4096 \
		--gpu_memory_utilization ${gpu_memory_utilization} \
		--tensor-parallel-size ${tp_size} \
		--swap_space 16 --model-path $policy_path \
		--controller-address http://${HOST_ADDR}:${CONTROLLER_PORT} \
		--host ${HOST_ADDR} --port ${WORKER_PORT} \
		--worker-address http://${HOST_ADDR}:${WORKER_PORT}"

	tmux send-keys "bash -lc '${command} > "${LOGDIR}/policy_worker_${policy_index}.out" 2>&1'" Enter
	echo "Policy worker $policy_index started on GPUs ${GPU_LIST[$worker_idx]} (port $WORKER_PORT)"
	sleep 20
done

echo "Controller and workers are up."
