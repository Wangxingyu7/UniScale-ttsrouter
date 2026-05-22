#!/bin/bash
set -euo pipefail
# 0. 依赖检查
for cmd in yq tmux; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "[ERROR] 请先安装 $cmd"; exit 1; }
done

# 1. 入参解析
CONFIG_FILE="${1:-}"
[[ -f $CONFIG_FILE ]] || { echo "Usage: $0 <config.yaml> [--stop|--status]"; exit 1; }
ACTION=${2:-start}
CONF=$(realpath "$CONFIG_FILE")
SCRIPT_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(realpath "$SCRIPT_DIR/..")          # 去掉多余的 -
LOGDIR=${ROOT_DIR}/logs_vllm
mkdir -p "$LOGDIR"

# 2. YAML 读取
read_conf() { yq -r "$1" "$CONF"; }
CTRL_HOST=$(read_conf '.controller.host')
CTRL_PORT=$(read_conf '.controller.port')
BASE_PORT=$(read_conf '.controller.worker_base_port // 10081')
PY=$(read_conf '.defaults.python')
SESS_PFX=$(read_conf '.defaults.session_prefix // "tts"')   # 默认 tts
MAX_LEN=$(read_conf '.defaults.max_model_length')
SWAP=$(read_conf '.defaults.swap_space')

# 3. 工具函数
infer_mem() {
  case "$1" in
    *0.6B*) echo 0.18 ;;
    *1.7B*) echo 0.18 ;;
    *4B*)   echo 0.35 ;;
    *8B*)   echo 0.55 ;;
    *14B*)  echo 0.50 ;;
    *32B*)  echo 0.43 ;;
    *)      echo 0.30 ;;
  esac
}
next_port() {
  local p=$1
  while netstat -ln 2>/dev/null | grep -q ":$p "; do ((p++)); done
  echo $p
}
validate_gpu() {
  local gpu=$1
  if ! nvidia-smi -L 2>/dev/null | grep -q "GPU $gpu"; then
    echo "[ERROR] GPU $gpu 不存在"; exit 1
  fi
}

# 统一会话：不存在就新建，存在就新建窗口
tmux_send() {
  local win=$1 cmd=$2
  if ! tmux has-session -t "$SESS_PFX" 2>/dev/null; then
    tmux new-session -s "$SESS_PFX" -n "$win" -d
  else
    tmux new-window -t "$SESS_PFX" -n "$win"
  fi
  tmux send-keys -t "$SESS_PFX:$win" "cd \"$ROOT_DIR\" && export PYTHONPATH=\"$ROOT_DIR\" LOGDIR=\"$LOGDIR\" && $cmd" Enter
}

# 4. 停止 / 状态
if [[ $ACTION == "--stop" ]]; then
  tmux kill-session -t "$SESS_PFX" 2>/dev/null || true
  exit 0
fi
if [[ $ACTION == "--status" ]]; then
  tmux list-sessions 2>/dev/null | grep "^$SESS_PFX" || echo "未运行"
  exit 0
fi

# 5. 启动 controller
echo "[INFO] 日志目录: $LOGDIR"
tmux_send "ctrl" \
  "$PY -m reason.llm_service.controller --host $CTRL_HOST --port $CTRL_PORT > $LOGDIR/controller.log 2>&1"
sleep 3

# 6. 启动 reward workers
REW_NUM=$(read_conf '.reward_models | length')
for idx in $(seq 0 $((REW_NUM-1))); do
  path=$(read_conf ".reward_models[$idx].path")
  gpu=$(read_conf ".reward_models[$idx].gpu")
  mem=$(read_conf ".reward_models[$idx].gpu_memory_utilization")
  [[ $mem == "auto" ]] && mem=$(infer_mem "$path")
  port=$(next_port $((BASE_PORT+idx)))
  validate_gpu "$gpu"
  cmd="CUDA_VISIBLE_DEVICES=$gpu $PY -m reason.llm_service.workers.vllm_rm_worker \
      --model-path $path \
      --gpu-memory-utilization $mem \
      --controller-address http://${CTRL_HOST}:${CTRL_PORT} \
      --host $CTRL_HOST \
      --port $port --worker-address http://${CTRL_HOST}:$port \
      --no-enable-chunked-prefill > $LOGDIR/rm_worker_${idx}.out 2>&1"
  tmux_send "rw_$idx" "$cmd"
  echo "[INFO] Reward $idx  GPU $gpu  Port $port"
  sleep 20
done
sleep 50
# 7. 启动 policy workers
POL_NUM=$(read_conf '.policy_models | length')
for idx in $(seq 0 $((POL_NUM-1))); do
  path=$(read_conf ".policy_models[$idx].path")
  gpu=$(read_conf ".policy_models[$idx].gpu")
  mem=$(read_conf ".policy_models[$idx].gpu_memory_utilization")
  [[ $mem == "auto" ]] && mem=$(infer_mem "$path")
  len=$(read_conf ".policy_models[$idx].max_model_length")
  sw=$(read_conf ".policy_models[$idx].swap_space")
  port=$(next_port $((BASE_PORT+REW_NUM+idx)))

  if [[ $gpu == \[* ]]; then
    raw_list=$(read_conf ".policy_models[$idx].gpu[]")
    gpu_list=$(echo "$raw_list" | tr -s '[:space:]' ',' | sed 's/^,//; s/,$//')
    tp_size=$(read_conf ".policy_models[$idx].gpu | length")
    cuda_devices="$gpu_list"
    tp_args="--tensor-parallel-size $tp_size"
  else
    cuda_devices=$(echo "$gpu" | tr -d '[:space:]')
    tp_size=1
    tp_args=""
  fi
  validate_gpu "${cuda_devices%%,*}"
  cmd="CUDA_VISIBLE_DEVICES=$cuda_devices $PY -m reason.llm_service.workers.vllm_worker \
    --max_model_length $len \
    --gpu_memory_utilization $mem \
    --swap_space $sw \
    --model-path $path \
    --controller-address http://${CTRL_HOST}:${CTRL_PORT} \
    --host $CTRL_HOST \
    --port $port --worker-address http://${CTRL_HOST}:$port \
    $tp_args > $LOGDIR/policy_worker_${idx}.out 2>&1"
  tmux_send "pol_$idx" "$cmd"
  echo "[INFO] Policy $idx  GPUs {$cuda_devices}  Port $port  TP=$tp_size"
  sleep 50
done

echo "[OK] 全部启动完毕！日志在 $LOGDIR"