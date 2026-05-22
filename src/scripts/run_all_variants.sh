#!/usr/bin/env bash
# 改进版：按文件顺序遍历 test150_*.jsonl 并用单条命令模式重复调用 run_t1.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/envs/MATH/dataset"
RUN_SCRIPT="$ROOT_DIR/scripts/run_t1.sh"

collect_policy_models() {
  local idx=1
  local models=()
  while true; do
    local var="POLICY_MODEL_${idx}_PATH"
    local value="${!var:-}"
    if [[ -z "$value" ]]; then
      if (( idx == 1 )); then
        # no env-defined models
        break
      fi
      # 停止在第一个缺失的连续索引位置
      break
    fi
    models+=("$value")
    idx=$((idx + 1))
  done

  if [[ ${#models[@]} -gt 0 ]]; then
    (IFS=,; echo "${models[*]}")
  fi
}

files=($DATA_DIR/aime_Qwen3*.jsonl)
# 默认 LM 列表（可被 --lm 覆盖），优先读取环境变量 POLICY_MODEL_*_PATH
FALLBACK_LM_LIST="/home/xingyu/pub/policy_models/Qwen3-0.6B,/home/xingyu/pub/policy_models/Qwen3-1.7B,/home/xingyu/pub/policy_models/Qwen3-4B,/home/xingyu/pub/policy_models/Qwen3-8B,/home/xingyu/pub/policy_models/Qwen3-14B"
# 默认 RM 路径（可被 --rm 覆盖）  
FALLBACK_RM="/root/autodl-pub/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

ENV_LM_LIST="$(collect_policy_models)"
if [[ -n "$ENV_LM_LIST" ]]; then
  DEFAULT_LM_LIST="$ENV_LM_LIST"
else
  DEFAULT_LM_LIST="$FALLBACK_LM_LIST"
fi

# 默认 RM（可被 --rm 覆盖），优先读取环境变量 VALUE_MODEL_PATH
DEFAULT_RM="${VALUE_MODEL_PATH:-$FALLBACK_RM}"

LM_LIST="$DEFAULT_LM_LIST"
RM="$DEFAULT_RM"

if [[ -z "$ENV_LM_LIST" ]]; then
  echo "[run_all_test150_variants] 警告: 未检测到 POLICY_MODEL_*_PATH 环境变量，使用内置默认 LM 列表。" >&2
fi

if [[ -z "${VALUE_MODEL_PATH:-}" ]]; then
  echo "[run_all_test150_variants] 警告: VALUE_MODEL_PATH 未设置，使用默认 RM: $DEFAULT_RM" >&2
fi

usage() {
  echo "Usage: $0 [--lm <LM_COMMA_SEP>] [--rm <RM_PATH>]"
  echo "  --lm   LM 列表，逗号分隔，传给 --LM 参数给 run_t1.sh" 
  echo "  --rm   RM 模型路径，传给 --RM 参数给 run_t1.sh" 
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --lm)
      LM_LIST="$2"
      shift 2
      ;;
    --rm)
      RM="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown arg: $1" >&2; usage
      ;;
  esac
done

if [ ! -d "$DATA_DIR" ]; then
  echo "数据目录不存在: $DATA_DIR" >&2
  exit 1
fi

shopt -s nullglob
if [ ${#files[@]} -eq 0 ]; then
  echo "未找到任何 test150_*.jsonl 文件于 $DATA_DIR" >&2
  exit 1
fi

echo "发现 ${#files[@]} 个文件，按文件名排序并依次运行 (LM=$LM_LIST, RM=$RM)..."
IFS=$'\n' sorted=($(printf "%s\n" "${files[@]}" | sort))

LOG_DIR="$ROOT_DIR/output/logs_run_all_variants"
mkdir -p "$LOG_DIR"

ERROR_PATTERN="Response status: 500"
TOTAL_RETRIES=0

for f in "${sorted[@]}"; do
  fname="$(basename "$f")"
  stem="${fname%.jsonl}"
  echo "---- 开始：$fname -> task_name=$stem ----"

  LOG_FILE="$LOG_DIR/${stem}.log"
  > "$LOG_FILE"

  while true; do
    tmp_log=$(mktemp "${stem}_retry_XXXX.log")

    # 使用单条命令模式：把 LM 列表和 RM 传入 run_t1.sh
    set +e
    bash "$RUN_SCRIPT" --method beam_search --LM "$LM_LIST" --RM "$RM" --task_name "$stem" \
      2>&1 | tee "$tmp_log"
    exit_code=${PIPESTATUS[0]}
    set -e

    cat "$tmp_log" >> "$LOG_FILE"

    if grep -q "$ERROR_PATTERN" "$tmp_log"; then
      TOTAL_RETRIES=$((TOTAL_RETRIES + 1))
      echo "检测到 500 错误，准备重跑 (task=$stem, 重试总数=$TOTAL_RETRIES)" | tee -a "$LOG_FILE"
      LOCK_ROOT="$ROOT_DIR/output/${stem}_beam_search"
      if [[ -d "$LOCK_ROOT" ]]; then
        while IFS= read -r -d '' lock_path; do
          echo "清理锁目录: $lock_path" | tee -a "$LOG_FILE"
          rm -rf "$lock_path"
        done < <(find "$LOCK_ROOT" -type d -name "lock_dir" -print0)
      fi
      rm -f "$tmp_log"
      sleep 2
      continue
    fi

    rm -f "$tmp_log"

    if [[ $exit_code -ne 0 ]]; then
      echo "任务 $stem 失败 (exit=$exit_code)，退出脚本。" | tee -a "$LOG_FILE"
      exit $exit_code
    fi

    break
  done

  echo "---- 完成：$fname (日志: $LOG_FILE) ----"
  sleep 1
done

echo "全部完成。"
echo "总重试次数：$TOTAL_RETRIES"