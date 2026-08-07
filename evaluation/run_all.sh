#!/usr/bin/env bash
# MiniCPM-o 评测套件 —— 一条命令跑完四个任务。
#
# 三个精度测试各自依赖一个不在主干里的 CLI target，由同目录下的 patch 引入，
# 而三个 patch 都改 tools/omni/CMakeLists.txt 的同一处，不能同时打。所以本脚本
# 对每个任务走一遍：
#
#     打补丁 → 编译该 target → 跑测试 → 撤补丁
#
# RTS 用主干的 llama-omni-server，不需要补丁，只在最后确保它是最新的。
# 四个任务的产物收进同一个 output/<时间戳>/，最后统一打印数值汇总。
#
# 用法:
#   ./run_all.sh                       # 按 config.env 的 SMOKE_* 跑
#   ./run_all.sh --smoke 5             # 三个精度测试各只跑 5 条
#   ./run_all.sh --full                # 精度测试跑全量
#   ./run_all.sh --tasks videomme,rts  # 只跑指定任务
#   ./run_all.sh --devices 0,1,2,3     # 指定用哪几张卡
#   ./run_all.sh --no-build            # 补丁已打好且编译过，跳过编译
#   ./run_all.sh --keep-patch          # 跑完不撤补丁（调试用）
set -uo pipefail

SUITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_SUITE_ROOT="$SUITE_ROOT"

# ---------------------------------------------------------------- 参数
TASKS="videomme,daily-omni,tts,rts"
DO_BUILD=1
KEEP_PATCH=0
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)       TASKS="$2"; shift 2 ;;
    --tasks=*)     TASKS="${1#*=}"; shift ;;
    --no-build)    DO_BUILD=0; shift ;;
    --keep-patch)  KEEP_PATCH=1; shift ;;
    -h|--help)
      sed -n '/^# MiniCPM-o/,/^# *--keep-patch/p' "${BASH_SOURCE[0]}" \
        | sed 's/^#\{1,\} \{0,1\}//'
      echo
      echo "其余参数原样透传给 run_eval.sh，见 ./run_eval.sh --help"
      exit 0 ;;
    *)             PASS_ARGS+=("$1"); shift ;;
  esac
done

# ---------------------------------------------------------------- 配置
CONFIG="${EVAL_CONFIG:-$SUITE_ROOT/config.env}"
if [[ ! -f "$CONFIG" ]]; then
  echo "找不到配置文件: $CONFIG" >&2
  exit 2
fi
set -a; source "$CONFIG"; set +a

# CANN 环境：cmake 要靠它找 CANN_INSTALL_DIR，不 source 直接 configure 就报
# "Can't find CANN_INSTALL_DIR"。run_eval.sh 里也会 source 一次，重复无害。
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
BUILD_DIR="$(basename "${EVAL_BIN_DIR%/bin}")"   # 由 config.env 的 EVAL_BIN_DIR 推出
CMAKE_FLAGS=(-DGGML_CANN=ON -DSOC_TYPE=Ascend910 -DCMAKE_BUILD_TYPE=Release)
JOBS="${BUILD_JOBS:-$(( $(nproc) < 64 ? $(nproc) : 64 ))}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$SUITE_ROOT/output/$STAMP"
mkdir -p "$RUN_DIR"
BUILD_LOG="$RUN_DIR/build.log"

# task -> "patch 相对路径:cmake target"
declare -A TASK_PATCH=(
  [videomme]="videomme/llama-omni-eval-cli.patch:llama-omni-eval-cli"
  [daily-omni]="daily-omni/llama-omni-eval-daily-cli.patch:llama-omni-eval-daily-cli"
  [tts]="tts_seed/llama-omni-tts-eval.patch:llama-omni-tts-eval"
  [rts]=":llama-omni-server"
)

log()  { printf '\n\033[1;36m[run_all]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[run_all]\033[0m %s\n' "$*" >&2; }

# 当前生效的补丁，异常退出时用来兜底撤回
CURRENT_PATCH=""
CURRENT_NEWFILE=""

revert_patch() {
  local patch="$1" newfile="$2"
  [[ -z "$patch" ]] && return 0
  if git -C "$REPO" apply -R "$patch" 2>/dev/null; then
    [[ -n "$newfile" ]] && rm -f "$REPO/$newfile"
    log "已撤回补丁 $(basename "$patch")"
  else
    warn "撤回补丁失败：$patch —— 手动检查 git -C $REPO status tools/"
  fi
  CURRENT_PATCH=""; CURRENT_NEWFILE=""
}

cleanup() {
  local rc=$?
  [[ $KEEP_PATCH -eq 0 ]] && revert_patch "$CURRENT_PATCH" "$CURRENT_NEWFILE"
  exit $rc
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------- 开场
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

if ! git -C "$REPO" diff --quiet -- tools/omni 2>/dev/null; then
  warn "tools/omni 已有未提交改动，可能是上次跑完没撤干净的补丁："
  git -C "$REPO" status --short -- tools/omni >&2
  warn "先处理掉再跑，否则打补丁会失败。"
  exit 2
fi

# ---------------------------------------------------------------- 主循环
declare -a DONE_TASKS=()
declare -a FAILED_TASKS=()

IFS=',' read -r -a TASK_ARR <<< "$TASKS"
for task in "${TASK_ARR[@]}"; do
  task="$(echo "$task" | xargs)"
  [[ -z "$task" ]] && continue
  if [[ -z "${TASK_PATCH[$task]:-}" && "$task" != "rts" ]]; then
    warn "未知任务 $task，跳过"
    continue
  fi

  spec="${TASK_PATCH[$task]}"
  patch_rel="${spec%%:*}"
  target="${spec##*:}"
  patch_abs=""
  newfile=""
  [[ -n "$patch_rel" ]] && patch_abs="$SUITE_ROOT/$patch_rel"

  log "===== $task ====="

  # --- 1) 打补丁 ---
  if [[ -n "$patch_abs" ]]; then
    if [[ ! -f "$patch_abs" ]]; then
      warn "$task: 找不到补丁 $patch_abs，跳过"
      FAILED_TASKS+=("$task(缺补丁)")
      continue
    fi
    # 补丁新建的源文件，撤回时要一并删掉（git apply -R 不删 untracked 文件）
    newfile="$(grep -m1 '^+++ b/tools/omni/.*\.cpp$' "$patch_abs" | sed 's|^+++ b/||')"
    log "$task: 打补丁 $(basename "$patch_abs")"
    if ! git -C "$REPO" apply "$patch_abs" 2>&1 | tee -a "$BUILD_LOG"; then
      warn "$task: 打补丁失败，跳过"
      FAILED_TASKS+=("$task(打补丁失败)")
      continue
    fi
    CURRENT_PATCH="$patch_abs"; CURRENT_NEWFILE="$newfile"
  fi

  # --- 2) 编译 ---
  if [[ $DO_BUILD -eq 1 ]]; then
    log "$task: 编译 $target (-j $JOBS)"
    {
      echo "===== $(date '+%F %T') build $target ====="
      cmake -B "$REPO/$BUILD_DIR" -S "$REPO" "${CMAKE_FLAGS[@]}"
      cmake --build "$REPO/$BUILD_DIR" --target "$target" -j "$JOBS"
    } >> "$BUILD_LOG" 2>&1
    if [[ $? -ne 0 ]]; then
      warn "$task: 编译失败，见 $BUILD_LOG"
      tail -20 "$BUILD_LOG" >&2
      revert_patch "$CURRENT_PATCH" "$CURRENT_NEWFILE"
      FAILED_TASKS+=("$task(编译失败)")
      continue
    fi
    log "$task: $target 编译完成"
  fi

  # --- 3) 跑测试 ---
  log "$task: 开跑"
  "$SUITE_ROOT/run_eval.sh" "$task" --run-dir "$RUN_DIR" --keep-going "${PASS_ARGS[@]}"
  rc=$?
  if [[ $rc -eq 0 ]]; then
    DONE_TASKS+=("$task")
  else
    FAILED_TASKS+=("$task(rc=$rc)")
  fi

  # --- 4) 撤补丁 ---
  if [[ $KEEP_PATCH -eq 0 ]]; then
    revert_patch "$CURRENT_PATCH" "$CURRENT_NEWFILE"
  else
    log "$task: --keep-patch，补丁保留"
    CURRENT_PATCH=""; CURRENT_NEWFILE=""
  fi
done

# ---------------------------------------------------------------- 汇总
log "全部任务结束，汇总数值"
"${RUNNER_PYTHON:-python3}" "$SUITE_ROOT/run_eval.py" --summarize --run-dir "$RUN_DIR"

echo "  成功: ${DONE_TASKS[*]:-无}"
[[ ${#FAILED_TASKS[@]} -gt 0 ]] && echo "  失败: ${FAILED_TASKS[*]}"
echo "  编译日志: $BUILD_LOG"
echo

[[ ${#FAILED_TASKS[@]} -gt 0 ]] && exit 1
exit 0
