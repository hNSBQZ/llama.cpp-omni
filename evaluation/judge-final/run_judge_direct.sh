#!/usr/bin/env bash
# MiniCPM-o Duplex Judge
#
# 服务由本脚本间接拉起：run_judge_direct.py 会用 --llamacpp-root 下的
# llama-omni-server 起一个子进程，跑完自动收掉。不需要另开终端手动起服务。
#
# 常用覆盖（都可以用环境变量传）：
#   MODEL=/path/to/xxx.gguf   ./run_judge_direct.sh
#   NPU=3                     ./run_judge_direct.sh      # 指定卡；不设则自动挑空闲的
#   OMNI_T2M_DEVICE=cpu       ./run_judge_direct.sh      # flow_matching 回退到 CPU
#   ./run_judge_direct.sh --verbose --plot                # 额外参数原样透传
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# ---- Ascend 运行环境 ----
# 不 source 的话 llama-omni-server 起不来（找不到 libascendcl 等）
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f "$ASCEND_ENV" ]]; then
  # set_env.sh 里有未定义变量展开，先关掉 -u
  set +u
  # shellcheck disable=SC1090
  source "$ASCEND_ENV"
  set -u
fi

# ---- token2wav 设备 ----
# CANN 构建下 flow_matching 默认已走 NPU；这两个变量留作对照实验的开关。
export OMNI_T2W_DEVICE="${OMNI_T2W_DEVICE:-gpu}"
export OMNI_T2M_DEVICE="${OMNI_T2M_DEVICE:-gpu:0}"
export OMNI_VOC_DEVICE="${OMNI_VOC_DEVICE:-gpu:0}"
# 采样种子固定，保证不同选手跑出同一条 token 轨迹，RTF 才可比
export OMNI_SAMPLER_SEED="${OMNI_SAMPLER_SEED:-42}"

# ---- 默认参数 ----
# 往上找第一个带 build*/bin/llama-omni-server 的祖先目录当仓库根
if [[ -z "${LLAMACPP_ROOT:-}" ]]; then
  probe="$ROOT"
  for _ in 1 2 3; do
    probe="$(cd "$probe/.." && pwd)"
    if compgen -G "$probe/build*/bin/llama-omni-server" > /dev/null; then
      break
    fi
  done
  LLAMACPP_ROOT="$probe"
fi
# LLM 权重没有合理的默认值，用 MODEL= 或 --model 给
MODEL="${MODEL:-}"
MODEL_ARG=()
if [[ -n "$MODEL" ]]; then
  MODEL_ARG=(--model "$MODEL")
fi
# 多个视频用空格分隔
VIDEO="${VIDEO:-$ROOT/assets/video/omni_duplex1.mp4}"
read -r -a VIDEO_ARR <<< "$VIDEO"

PY="${JUDGE_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
  else
    PY="${PYTHON:-python3}"
  fi
fi

GPU_ARG=()
if [[ -n "${NPU:-}" ]]; then
  GPU_ARG=(--gpu "$NPU")
fi

echo "[judge] server bin : $(ls "$LLAMACPP_ROOT"/build*/bin/llama-omni-server 2>/dev/null | head -1)"
echo "[judge] model      : ${MODEL:-由 --model 指定}"
echo "[judge] video      : $VIDEO"
echo "[judge] t2m/voc    : $OMNI_T2M_DEVICE / $OMNI_VOC_DEVICE"

exec "$PY" "$ROOT/run_judge_direct.py" \
  "${MODEL_ARG[@]}" \
  --video "${VIDEO_ARR[@]}" \
  --llamacpp-root "$LLAMACPP_ROOT" \
  "${GPU_ARG[@]}" \
  "$@"
