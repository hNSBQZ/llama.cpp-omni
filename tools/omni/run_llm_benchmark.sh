#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

usage() {
  cat <<'EOF'
Usage:
  tools/omni/run_llm_benchmark.sh [extra llama-bench args...]

Environment:
  BENCH_BIN                 Path to llama-bench. Default: $ROOT_DIR/build/bin/llama-bench
  AUTO_BUILD                Build llama-bench if BENCH_BIN is missing. Default: 1
  LLM_BENCH_MODEL_MANIFEST Manifest file. Default: tools/omni/llm_bench_models.txt
  LLM_BENCH_MODEL_DIRS     Colon-separated model dirs used when manifest is absent.
                            Default: ~/o45-gguf:~/o45-gguf-gptq-int8:~/o45-awq-gguf
  MAX_MODELS_PER_DIR       Auto-discovery limit per dir. 0 means all. Default: 1
  LLM_BENCH_GPU            Default single GPU id. Default: first CUDA_VISIBLE_DEVICES entry or 0
  LLM_BENCH_GPUS           Comma-separated GPU ids for round-robin assignment, e.g. 0,1,2
  LLM_BENCH_PARALLEL       1 to run one serial worker per GPU. Default: 1 when >1 distinct GPUs are assigned
  GPU_MIN_FREE_MB          Minimum free GPU memory before launching. Default: 16000
  GPU_WAIT_FOR_FREE        1 to wait for enough free GPU memory, 0 to fail fast. Default: 1
  GPU_WAIT_INTERVAL_SEC    Seconds between free-memory checks. Default: 30
  GPU_WAIT_TIMEOUT_SEC     Max seconds to wait; 0 means no timeout. Default: 0
  REPETITIONS              llama-bench repetitions. Default: 5
  PP_SIZES                 Prefill sizes. Default: 128,512,2048
  TG_SIZES                 Decode sizes. Default: 32,128,256
  GPU_LAYERS               llama-bench -ngl. Default: 99
  BATCH_SIZE               llama-bench -b. Default: 2048
  UBATCH_SIZE              llama-bench -ub. Default: 512
  FLASH_ATTN               Optional llama-bench -fa value, e.g. 0 or 1
  LLAMA_BENCH_EXTRA_ARGS   Extra whitespace-separated llama-bench args
  OUT_DIR                  Output run directory. Default: tools/omni/output/llm_bench/<timestamp>

Manifest examples:
  o45-gguf=~/o45-gguf/MiniCPM-o-4_5-F16.gguf,gpu=0
  o45-gptq-int8=~/o45-gguf-gptq-int8/model.gguf,gpu=1
  o45-awq=~/o45-awq-gguf/model.gguf,gpu=2
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

shopt -s nullglob

BENCH_BIN="${BENCH_BIN:-${ROOT_DIR}/build/bin/llama-bench}"
AUTO_BUILD="${AUTO_BUILD:-1}"
MODEL_MANIFEST="${LLM_BENCH_MODEL_MANIFEST:-${SCRIPT_DIR}/llm_bench_models.txt}"
MODEL_DIRS="${LLM_BENCH_MODEL_DIRS:-${HOME}/o45-gguf:${HOME}/o45-gguf-gptq-int8:${HOME}/o45-awq-gguf}"
MAX_MODELS_PER_DIR="${MAX_MODELS_PER_DIR:-1}"

REPETITIONS="${REPETITIONS:-5}"
PP_SIZES="${PP_SIZES:-128,512,2048}"
TG_SIZES="${TG_SIZES:-32,128,256}"
GPU_LAYERS="${GPU_LAYERS:-99}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
FLASH_ATTN="${FLASH_ATTN:-}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-16000}"
GPU_WAIT_FOR_FREE="${GPU_WAIT_FOR_FREE:-1}"
GPU_WAIT_INTERVAL_SEC="${GPU_WAIT_INTERVAL_SEC:-30}"
GPU_WAIT_TIMEOUT_SEC="${GPU_WAIT_TIMEOUT_SEC:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_BASE="${OUT_BASE:-${SCRIPT_DIR}/output/llm_bench}"
OUT_DIR="${OUT_DIR:-${OUT_BASE}/${RUN_ID}}"
RAW_DIR="${OUT_DIR}/raw"

declare -a MODEL_NAMES=()
declare -a MODEL_PATHS=()
declare -a MODEL_GPUS=()
declare -a MODEL_SAFE_NAMES=()

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

expand_path() {
  local path="$1"
  local home_prefix="${HOME}/~/"
  case "${path}" in
    "~") printf '%s' "${HOME}" ;;
    "~/"*) printf '%s/%s' "${HOME}" "${path#~/}" ;;
    "${home_prefix}"*) printf '%s/%s' "${HOME}" "${path#"${home_prefix}"}" ;;
    *) printf '%s' "${path}" ;;
  esac
}

safe_name() {
  local name="$1"
  printf '%s' "${name}" | tr -c '[:alnum:]_.-' '_'
}

unique_safe_name() {
  local base="$1"
  local candidate="${base}"
  local suffix=2
  local existing found

  while true; do
    found=0
    for existing in "${MODEL_SAFE_NAMES[@]}"; do
      if [[ "${existing}" == "${candidate}" ]]; then
        found=1
        break
      fi
    done
    if [[ "${found}" == "0" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
    candidate="${base}_${suffix}"
    suffix=$((suffix + 1))
  done
}

add_model() {
  local name="$1"
  local path="$2"
  local gpu="${3:-}"

  path="$(expand_path "${path}")"
  if [[ ! -f "${path}" ]]; then
    printf '[skip] model not found: %s\n' "${path}" >&2
    return 0
  fi

  local safe
  safe="$(unique_safe_name "$(safe_name "${name}")")"
  MODEL_NAMES+=("${name}")
  MODEL_PATHS+=("${path}")
  MODEL_GPUS+=("${gpu}")
  MODEL_SAFE_NAMES+=("${safe}")
}

load_manifest() {
  local manifest="$1"
  local raw line spec name path gpu

  while IFS= read -r raw || [[ -n "${raw}" ]]; do
    line="$(trim "${raw%%#*}")"
    [[ -z "${line}" ]] && continue

    gpu=""
    spec="${line}"
    if [[ "${spec}" == *,gpu=* ]]; then
      gpu="${spec##*,gpu=}"
      spec="${spec%,gpu=*}"
      gpu="$(trim "${gpu}")"
    fi

    if [[ "${spec}" == *"="* ]]; then
      name="$(trim "${spec%%=*}")"
      path="$(trim "${spec#*=}")"
    else
      path="$(trim "${spec}")"
      name="$(basename "${path}")"
      name="${name%.gguf}"
    fi

    if [[ -z "${name}" || -z "${path}" ]]; then
      printf '[skip] invalid manifest line: %s\n' "${raw}" >&2
      continue
    fi
    add_model "${name}" "${path}" "${gpu}"
  done < "${manifest}"
}

discover_models() {
  local dirs="$1"
  local dir expanded base_dir model_path base_model label count max

  IFS=':' read -r -a dir_array <<< "${dirs}"
  for dir in "${dir_array[@]}"; do
    expanded="$(expand_path "$(trim "${dir}")")"
    [[ -z "${expanded}" ]] && continue
    if [[ ! -d "${expanded}" ]]; then
      printf '[skip] model dir not found: %s\n' "${expanded}" >&2
      continue
    fi

    local models=("${expanded}"/*.gguf)
    [[ "${#models[@]}" -eq 0 ]] && {
      printf '[skip] no gguf files in: %s\n' "${expanded}" >&2
      continue
    }

    base_dir="$(basename "${expanded}")"
    count=0
    max="${MAX_MODELS_PER_DIR}"
    for model_path in "${models[@]}"; do
      base_model="$(basename "${model_path}")"
      base_model="${base_model%.gguf}"
      if [[ "${max}" == "1" && "${#models[@]}" -gt 1 ]]; then
        label="${base_dir}"
      elif [[ "${#models[@]}" -eq 1 ]]; then
        label="${base_dir}"
      else
        label="${base_dir}-${base_model}"
      fi

      add_model "${label}" "${model_path}" ""
      count=$((count + 1))
      if [[ "${max}" != "0" && "${count}" -ge "${max}" ]]; then
        break
      fi
    done
  done
}

first_cuda_device() {
  local value="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -n "${value}" ]]; then
    printf '%s' "${value%%,*}"
  else
    printf '0'
  fi
}

prepare_gpu_list() {
  local raw="${LLM_BENCH_GPUS:-}"
  if [[ -z "${raw}" && "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
    raw="${CUDA_VISIBLE_DEVICES}"
  fi
  if [[ -z "${raw}" ]]; then
    raw="${LLM_BENCH_GPU:-$(first_cuda_device)}"
  fi

  IFS=',' read -r -a GPU_LIST <<< "${raw}"
  for i in "${!GPU_LIST[@]}"; do
    GPU_LIST[$i]="$(trim "${GPU_LIST[$i]}")"
  done
}

assign_gpus() {
  local default_gpu="${LLM_BENCH_GPU:-$(first_cuda_device)}"
  local gpu_count="${#GPU_LIST[@]}"
  local idx assigned

  for idx in "${!MODEL_GPUS[@]}"; do
    assigned="$(trim "${MODEL_GPUS[$idx]}")"
    if [[ -z "${assigned}" ]]; then
      if [[ "${gpu_count}" -gt 0 ]]; then
        assigned="${GPU_LIST[$((idx % gpu_count))]}"
      else
        assigned="${default_gpu}"
      fi
    fi
    MODEL_GPUS[$idx]="${assigned}"
  done
}

distinct_gpu_count() {
  local seen="" count=0 gpu
  for gpu in "${MODEL_GPUS[@]}"; do
    if [[ " ${seen} " != *" ${gpu} "* ]]; then
      seen="${seen} ${gpu}"
      count=$((count + 1))
    fi
  done
  printf '%s' "${count}"
}

distinct_gpus() {
  local seen="" gpu
  for gpu in "${MODEL_GPUS[@]}"; do
    if [[ " ${seen} " != *" ${gpu} "* ]]; then
      seen="${seen} ${gpu}"
      printf '%s\n' "${gpu}"
    fi
  done
}

gpu_free_mb() {
  local gpu="$1"
  local free

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf ''
    return 0
  fi

  free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  printf '%s' "${free}"
}

wait_for_gpu() {
  local gpu="$1"
  local name="$2"
  local min="${GPU_MIN_FREE_MB}"
  local interval="${GPU_WAIT_INTERVAL_SEC}"
  local timeout="${GPU_WAIT_TIMEOUT_SEC}"
  local start now elapsed free

  if [[ "${min}" == "0" ]]; then
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '[warn] nvidia-smi not found; cannot check free memory for GPU %s\n' "${gpu}" >&2
    return 0
  fi

  start="$(date +%s)"
  while true; do
    free="$(gpu_free_mb "${gpu}")"
    if [[ "${free}" =~ ^[0-9]+$ ]] && (( free >= min )); then
      printf '[gpu] %s: GPU %s free %s MiB >= %s MiB\n' "${name}" "${gpu}" "${free}" "${min}"
      return 0
    fi

    if [[ "${GPU_WAIT_FOR_FREE}" != "1" ]]; then
      printf '[error] %s: GPU %s free %s MiB < %s MiB\n' "${name}" "${gpu}" "${free:-unknown}" "${min}" >&2
      return 1
    fi

    now="$(date +%s)"
    elapsed=$((now - start))
    if [[ "${timeout}" != "0" && "${elapsed}" -ge "${timeout}" ]]; then
      printf '[error] %s: timed out waiting for GPU %s free memory >= %s MiB\n' "${name}" "${gpu}" "${min}" >&2
      return 1
    fi

    printf '[wait] %s: GPU %s free %s MiB < %s MiB; retry in %ss\n' \
      "${name}" "${gpu}" "${free:-unknown}" "${min}" "${interval}"
    sleep "${interval}"
  done
}

print_env() {
  {
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'timestamp=%s\n' "$(date -Is)"
    printf 'root_dir=%s\n' "${ROOT_DIR}"
    printf 'bench_bin=%s\n' "${BENCH_BIN}"
    printf 'pp_sizes=%s\n' "${PP_SIZES}"
    printf 'tg_sizes=%s\n' "${TG_SIZES}"
    printf 'repetitions=%s\n' "${REPETITIONS}"
    printf 'gpu_layers=%s\n' "${GPU_LAYERS}"
    printf 'batch_size=%s\n' "${BATCH_SIZE}"
    printf 'ubatch_size=%s\n' "${UBATCH_SIZE}"
    printf 'flash_attn=%s\n' "${FLASH_ATTN:-default}"
    printf 'gpu_min_free_mb=%s\n' "${GPU_MIN_FREE_MB}"
    printf 'gpu_wait_for_free=%s\n' "${GPU_WAIT_FOR_FREE}"
    printf 'gpu_wait_interval_sec=%s\n' "${GPU_WAIT_INTERVAL_SEC}"
    printf 'gpu_wait_timeout_sec=%s\n' "${GPU_WAIT_TIMEOUT_SEC}"
    printf 'git_commit='
    git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || true
    printf '\n[nvidia-smi]\n'
    nvidia-smi 2>/dev/null || true
  } > "${OUT_DIR}/env.txt"
}

write_models_file() {
  {
    printf 'name\tgpu\tpath\tsafe_name\n'
    for idx in "${!MODEL_NAMES[@]}"; do
      printf '%s\t%s\t%s\t%s\n' \
        "${MODEL_NAMES[$idx]}" \
        "${MODEL_GPUS[$idx]}" \
        "${MODEL_PATHS[$idx]}" \
        "${MODEL_SAFE_NAMES[$idx]}"
    done
  } > "${OUT_DIR}/models.txt"
}

write_command() {
  local gpu="$1"
  shift
  {
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
    printf '%q ' "$@"
    printf '\n'
  } >> "${OUT_DIR}/commands.txt"
}

run_one() {
  local idx="$1"
  local name="${MODEL_NAMES[$idx]}"
  local path="${MODEL_PATHS[$idx]}"
  local gpu="${MODEL_GPUS[$idx]}"
  local safe="${MODEL_SAFE_NAMES[$idx]}"
  local md_out="${RAW_DIR}/${safe}.md"
  local jsonl_out="${RAW_DIR}/${safe}.jsonl"

  local -a cmd=(
    "${BENCH_BIN}"
    -m "${path}"
    -p "${PP_SIZES}"
    -n "${TG_SIZES}"
    -ngl "${GPU_LAYERS}"
    -b "${BATCH_SIZE}"
    -ub "${UBATCH_SIZE}"
    -r "${REPETITIONS}"
    -o md
    -oe jsonl
  )

  if [[ -n "${FLASH_ATTN}" ]]; then
    cmd+=(-fa "${FLASH_ATTN}")
  fi

  if [[ -n "${LLAMA_BENCH_EXTRA_ARGS:-}" ]]; then
    local -a env_extra=()
    read -r -a env_extra <<< "${LLAMA_BENCH_EXTRA_ARGS}"
    cmd+=("${env_extra[@]}")
  fi

  if [[ "$#" -gt 1 ]]; then
    shift
    cmd+=("$@")
  fi

  write_command "${gpu}" "${cmd[@]}"
  wait_for_gpu "${gpu}" "${name}"
  printf '[run] %s on GPU %s\n' "${name}" "${gpu}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${md_out}" 2> "${jsonl_out}"
  printf '[done] %s -> %s\n' "${name}" "${md_out}"
}

run_gpu_group() {
  local target_gpu="$1"
  shift

  local idx
  printf '[worker] GPU %s serial group started\n' "${target_gpu}"
  for idx in "${!MODEL_NAMES[@]}"; do
    if [[ "${MODEL_GPUS[$idx]}" == "${target_gpu}" ]]; then
      run_one "${idx}" "$@"
    fi
  done
  printf '[worker] GPU %s serial group finished\n' "${target_gpu}"
}

BENCH_BIN="$(expand_path "${BENCH_BIN}")"
MODEL_MANIFEST="$(expand_path "${MODEL_MANIFEST}")"
OUT_DIR="$(expand_path "${OUT_DIR}")"
RAW_DIR="${OUT_DIR}/raw"

if [[ ! -x "${BENCH_BIN}" ]]; then
  if [[ "${AUTO_BUILD}" == "1" ]]; then
    printf '[build] %s is missing, building llama-bench...\n' "${BENCH_BIN}"
    cmake --build "${ROOT_DIR}/build" --target llama-bench -j"$(nproc)"
  fi
fi

if [[ ! -x "${BENCH_BIN}" ]]; then
  printf '[error] llama-bench not found or not executable: %s\n' "${BENCH_BIN}" >&2
  printf '        Build it with: cmake --build build --target llama-bench -j"$(nproc)"\n' >&2
  exit 1
fi

if [[ -f "${MODEL_MANIFEST}" ]]; then
  printf '[models] loading manifest: %s\n' "${MODEL_MANIFEST}"
  load_manifest "${MODEL_MANIFEST}"
else
  printf '[models] manifest not found, auto-discovering: %s\n' "${MODEL_MANIFEST}"
  discover_models "${MODEL_DIRS}"
fi

if [[ "${#MODEL_NAMES[@]}" -eq 0 ]]; then
  printf '[error] no models to benchmark\n' >&2
  exit 1
fi

prepare_gpu_list
assign_gpus

mkdir -p "${RAW_DIR}"
: > "${OUT_DIR}/commands.txt"
print_env
write_models_file

parallel="${LLM_BENCH_PARALLEL:-}"
if [[ -z "${parallel}" ]]; then
  if [[ "$(distinct_gpu_count)" -gt 1 ]]; then
    parallel="1"
  else
    parallel="0"
  fi
fi

printf '[output] %s\n' "${OUT_DIR}"
printf '[mode] %s\n' "$([[ "${parallel}" == "1" ]] && printf 'parallel by GPU, serial within GPU' || printf 'serial')"

if [[ "${parallel}" == "1" ]]; then
  declare -a pids=()
  while IFS= read -r gpu; do
    [[ -z "${gpu}" ]] && continue
    run_gpu_group "${gpu}" "$@" &
    pids+=("$!")
  done < <(distinct_gpus)

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    printf '[error] at least one benchmark failed\n' >&2
    exit 1
  fi
else
  for idx in "${!MODEL_NAMES[@]}"; do
    run_one "${idx}" "$@"
  done
fi

if [[ "${SKIP_SUMMARY:-0}" != "1" ]]; then
  "${PYTHON:-python3}" "${SCRIPT_DIR}/summarize_llm_benchmark.py" "${OUT_DIR}"
fi

printf '[summary] %s\n' "${OUT_DIR}/summary.md"
