#!/usr/bin/env bash
# MiniCPM-o Duplex Judge：自动拉起 llama-omni-server，跑完后退出
#
#   MODEL=/path/to/xxx.gguf ./run_judge_direct.sh
#   NPU=3 ./run_judge_direct.sh
#   OMNI_T2M_DEVICE=cpu ./run_judge_direct.sh
#   ./run_judge_direct.sh --verbose --plot
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f "$ASCEND_ENV" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ASCEND_ENV"
  set -u
fi

export OMNI_T2W_DEVICE="${OMNI_T2W_DEVICE:-gpu}"
export OMNI_T2M_DEVICE="${OMNI_T2M_DEVICE:-gpu:0}"
export OMNI_VOC_DEVICE="${OMNI_VOC_DEVICE:-gpu:0}"
export OMNI_SAMPLER_SEED="${OMNI_SAMPLER_SEED:-42}"

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

MODEL="${MODEL:-}"
MODEL_ARG=()
if [[ -n "$MODEL" ]]; then
  MODEL_ARG=(--model "$MODEL")
fi

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
