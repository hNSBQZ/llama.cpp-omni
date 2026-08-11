#!/usr/bin/env bash
# 一次编译并串行跑完指定任务。用法见 ./run_all.sh --help；细节见 README.md
#
#   ./run_all.sh
#   ./run_all.sh --smoke 5
#   ./run_all.sh --full
#   ./run_all.sh --tasks videomme,rts
#   ./run_all.sh --devices 0,1,2,3
#   ./run_all.sh --no-build
set -uo pipefail

SUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_SUITE_ROOT="$SUITE_ROOT"

TASKS="videomme,daily-omni,tts,rts"
DO_BUILD=1
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)       TASKS="$2"; shift 2 ;;
    --tasks=*)     TASKS="${1#*=}"; shift ;;
    --no-build)    DO_BUILD=0; shift ;;
    -h|--help)
      cat <<'EOF'
一次编译并串行跑完指定任务。细节见 README.md

  ./run_all.sh
  ./run_all.sh --smoke 5
  ./run_all.sh --full
  ./run_all.sh --tasks videomme,rts
  ./run_all.sh --devices 0,1,2,3
  ./run_all.sh --no-build

其余参数透传给 run_eval.sh，见 ./run_eval.sh --help
EOF
      exit 0 ;;
    *)             PASS_ARGS+=("$1"); shift ;;
  esac
done

CONFIG="${EVAL_CONFIG:-$SUITE_ROOT/config.env}"
if [[ ! -f "$CONFIG" ]]; then
  echo "找不到配置文件: $CONFIG" >&2
  exit 2
fi
set -a; source "$CONFIG"; set +a

ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f "$ASCEND_ENV" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$ASCEND_ENV"
  set -u
else
  echo "[warn] 没找到 Ascend set_env.sh: $ASCEND_ENV，编译大概会失败" >&2
fi

REPO="${LLAMACPP_ROOT:-$(cd "$SUITE_ROOT/.." && pwd)}"
BUILD_DIR="$(basename "${EVAL_BIN_DIR%/bin}")"
CMAKE_FLAGS=(-DGGML_CANN=ON -DSOC_TYPE=Ascend910 -DCMAKE_BUILD_TYPE=Release)
JOBS="${BUILD_JOBS:-$(( $(nproc) < 64 ? $(nproc) : 64 ))}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$SUITE_ROOT/output/$STAMP"
mkdir -p "$RUN_DIR"
BUILD_LOG="$RUN_DIR/build.log"

declare -A TASK_TARGET=(
  [videomme]="llama-omni-eval-cli"
  [daily-omni]="llama-omni-eval-daily-cli"
  [tts]="llama-omni-tts-eval"
  [rts]="llama-omni-server"
)

log()  { printf '\n\033[1;36m[run_all]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[run_all]\033[0m %s\n' "$*" >&2; }

cat <<EOF

$(printf '=%.0s' {1..78})
  MiniCPM-o 评测套件 —— 全链路
$(printf '=%.0s' {1..78})
  仓库        : $REPO
  构建目录    : $REPO/$BUILD_DIR
  任务        : $TASKS
  产物目录    : $RUN_DIR
  编译日志    : $BUILD_LOG
$(printf '=%.0s' {1..78})
EOF

declare -a DONE_TASKS=()
declare -a FAILED_TASKS=()
declare -a RUN_TASKS=()

IFS=',' read -r -a TASK_ARR <<< "$TASKS"
for task in "${TASK_ARR[@]}"; do
  task="$(echo "$task" | xargs)"
  [[ -z "$task" ]] && continue
  if [[ -z "${TASK_TARGET[$task]:-}" ]]; then
    warn "未知任务 $task，跳过"
    continue
  fi
  RUN_TASKS+=("$task")
done

if [[ ${#RUN_TASKS[@]} -eq 0 ]]; then
  warn "没有可跑的任务"
  exit 2
fi

if [[ $DO_BUILD -eq 1 ]]; then
  declare -a BUILD_TARGETS=()
  for task in "${RUN_TASKS[@]}"; do
    BUILD_TARGETS+=("${TASK_TARGET[$task]}")
  done
  log "编译 ${BUILD_TARGETS[*]} (-j $JOBS)"
  {
    echo "===== $(date '+%F %T') build ${BUILD_TARGETS[*]} ====="
    cmake -B "$REPO/$BUILD_DIR" -S "$REPO" "${CMAKE_FLAGS[@]}" &&
    cmake --build "$REPO/$BUILD_DIR" --target "${BUILD_TARGETS[@]}" -j "$JOBS"
  } >> "$BUILD_LOG" 2>&1
  if [[ $? -ne 0 ]]; then
    warn "编译失败，见 $BUILD_LOG"
    tail -20 "$BUILD_LOG" >&2
    exit 3
  fi
  log "编译完成"
fi

for task in "${RUN_TASKS[@]}"; do
  log "===== $task ====="
  "$SUITE_ROOT/run_eval.sh" "$task" --run-dir "$RUN_DIR" --keep-going "${PASS_ARGS[@]}"
  rc=$?
  if [[ $rc -eq 0 ]]; then
    DONE_TASKS+=("$task")
  else
    FAILED_TASKS+=("$task(rc=$rc)")
  fi
done

log "全部任务结束，汇总数值"
"${RUNNER_PYTHON:-python3}" "$SUITE_ROOT/run_eval.py" --summarize --run-dir "$RUN_DIR"

echo "  成功: ${DONE_TASKS[*]:-无}"
[[ ${#FAILED_TASKS[@]} -gt 0 ]] && echo "  失败: ${FAILED_TASKS[*]}"
echo "  编译日志: $BUILD_LOG"
echo

[[ ${#FAILED_TASKS[@]} -gt 0 ]] && exit 1
exit 0
