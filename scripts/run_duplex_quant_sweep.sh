#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/cache/hanqingzhe/llama.cpp-omni}"
MODEL_DIR="${MODEL_DIR:-/cache/hanqingzhe/o45-gguf}"
TEST_PREFIX="${TEST_PREFIX:-${ROOT_DIR}/tools/omni/assets/test_case/duplex_omni_test_case/duplex_omni_test_case_}"
REF_AUDIO="${REF_AUDIO:-${ROOT_DIR}/tools/omni/assets/default_ref_audio/default_ref_audio.wav}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-7}"
CTX_SIZE="${CTX_SIZE:-4096}"
GPU_LAYERS="${GPU_LAYERS:-99}"
GPU_PROF_INTERVAL_MS="${GPU_PROF_INTERVAL_MS:-20}"

QUANTS=(
  "F16:MiniCPM-o-4_5-F16.gguf"
  "Q4_0:MiniCPM-o-4_5-Q4_0.gguf"
  "Q4_1:MiniCPM-o-4_5-Q4_1.gguf"
  "Q4_K_M:MiniCPM-o-4_5-Q4_K_M.gguf"
  "Q4_K_S:MiniCPM-o-4_5-Q4_K_S.gguf"
  "Q5_0:MiniCPM-o-4_5-Q5_0.gguf"
  "Q5_1:MiniCPM-o-4_5-Q5_1.gguf"
  "Q5_K_M:MiniCPM-o-4_5-Q5_K_M.gguf"
  "Q5_K_S:MiniCPM-o-4_5-Q5_K_S.gguf"
  "Q6_K:MiniCPM-o-4_5-Q6_K.gguf"
  "Q8_0:MiniCPM-o-4_5-Q8_0.gguf"
)

cd "${ROOT_DIR}"

for item in "${QUANTS[@]}"; do
  name="${item%%:*}"
  model_file="${item#*:}"
  tag="$(echo "${name}" | tr '[:upper:]' '[:lower:]')"
  model_path="${MODEL_DIR}/${model_file}"
  log_path="${ROOT_DIR}/duplex_${tag}_full.log"
  gpu_log_path="${ROOT_DIR}/duplex_${tag}_full.gpu.log"
  output_dir="${ROOT_DIR}/tools/omni/output/duplex_${tag}_full"
  report_dir="${ROOT_DIR}/docs/development/perf-duplex-${tag}-full"

  if [[ ! -f "${model_path}" ]]; then
    echo "[skip] model not found: ${model_path}" >&2
    continue
  fi

  echo
  echo "=== Running ${name}: ${model_path} ==="
  rm -f "${log_path}" "${gpu_log_path}"

  LD_PRELOAD="${NVIDIA_PRELOAD:-}" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
  OMNI_GPU_PROF=1 \
  OMNI_GPU_PROF_INTERVAL_MS="${GPU_PROF_INTERVAL_MS}" \
  OMNI_GPU_PROF_FILE="${gpu_log_path}" \
    ./build/bin/llama-omni-test-duplex \
      -m "${model_path}" \
      --omni \
      --test "${TEST_PREFIX}" 36 \
      --ref-audio "${REF_AUDIO}" \
      -ngl "${GPU_LAYERS}" \
      -c "${CTX_SIZE}" \
      -o "${output_dir}" \
      2>&1 | tee "${log_path}"

  python3 scripts/analyze_duplex_perf.py \
    --log "${log_path}" \
    --gpu-log "${gpu_log_path}" \
    --out-dir "${report_dir}"
done
