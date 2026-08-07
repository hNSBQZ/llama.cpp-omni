#!/usr/bin/env bash
# MiniCPM-o 评测套件统一入口。
#
# 干三件事：source config.env、source Ascend 的 set_env.sh、把参数转交给
# run_eval.py。配置改 config.env，命令行参数见 ./run_eval.sh --help。
set -euo pipefail

EVAL_SUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_SUITE_ROOT

CONFIG="${EVAL_CONFIG:-$EVAL_SUITE_ROOT/config.env}"
if [[ ! -f "$CONFIG" ]]; then
  echo "找不到配置文件: $CONFIG" >&2
  exit 2
fi

# config.env 里是 KEY=VALUE，allexport 让它们直接进子进程环境
set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

# CANN 运行环境：不 source 的话 llama-omni-server / eval CLI 找不到 libascendcl
ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f "$ASCEND_ENV" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ASCEND_ENV"
  set -u
else
  echo "[warn] 没找到 Ascend set_env.sh: $ASCEND_ENV" >&2
fi

# eval CLI 运行时要链 libomni.so / libllama.so
if [[ -n "${EVAL_BIN_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="${EVAL_BIN_DIR}:${LD_LIBRARY_PATH:-}"
fi

exec "${RUNNER_PYTHON:-python3}" "$EVAL_SUITE_ROOT/run_eval.py" "$@"
