#!/usr/bin/env bash
# 评测入口：加载 config.env 与 Ascend 环境后转交 run_eval.py
# 配置改 config.env；参数见 ./run_eval.sh --help
set -euo pipefail

EVAL_SUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_SUITE_ROOT

CONFIG="${EVAL_CONFIG:-$EVAL_SUITE_ROOT/config.env}"
if [[ ! -f "$CONFIG" ]]; then
  echo "找不到配置文件: $CONFIG" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f "$ASCEND_ENV" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ASCEND_ENV"
  set -u
else
  echo "[warn] 没找到 Ascend set_env.sh: $ASCEND_ENV" >&2
fi

if [[ -n "${EVAL_BIN_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="${EVAL_BIN_DIR}:${LD_LIBRARY_PATH:-}"
fi

exec "${RUNNER_PYTHON:-python3}" "$EVAL_SUITE_ROOT/run_eval.py" "$@"
