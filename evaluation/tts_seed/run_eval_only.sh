#!/bin/bash
# 仅跑 WER + SIM（跳过推理）。用法: bash run_eval_only.sh /path/to/save_dir
set -e

source "$(cd "$(dirname "$0")"; pwd)/pipeline.env"
source "$(cd "$(dirname "$0")"; pwd)/metrics_stages.sh"

echo "Python: $(command -v python3)"

SAVE_DIR="${1:?用法: bash run_eval_only.sh <save_dir>}"
if [ ! -d "$SAVE_DIR" ]; then
    echo "ERROR: SAVE_DIR not found: $SAVE_DIR"
    exit 1
fi

LANG="zh"
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
EVAL_META_PATH=${EVAL_META_PATH:-"/path/to/seedtts_testset_zh/zh/meta.lst"}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-"/path/to/seedtts_testset_zh/zh"}
SPEAKER_CKPT=${SPEAKER_CKPT:-"/path/to/wavlm_large_finetune.pth"}
WORKDIR=$(cd "$(dirname "$0")"; pwd)
EVAL_SCRIPT_DIR=${EVAL_SCRIPT_DIR:-"${WORKDIR}/eval_tools"}
SPEAKER_VERIF_DIR=${SPEAKER_VERIF_DIR:-"${EVAL_SCRIPT_DIR}/speaker_verification"}
PARAFORMER_MODEL=${PARAFORMER_MODEL:-"/path/to/paraformer-zh"}
S3PRL_REPO=${S3PRL_REPO:-"${EVAL_SCRIPT_DIR}/s3prl-main"}
export PARAFORMER_MODEL
export S3PRL_REPO
export WER_DEVICE

MODEL_PATH=${MODEL_PATH:-"/path/to/MiniCPM-o-4_5-gguf"}
CPP_BIN=${CPP_BIN:-"/path/to/llama.cpp-omni/build/bin/llama-omni-tts-eval"}
SEED=${SEED:-42}
TIME_STR=$(date +%Y%m%d_%H%M%S)

LOG_BASE="${WORKDIR}/logs"
mkdir -p "$LOG_BASE"

echo "============================================"
echo "Eval-only mode"
echo "SAVE_DIR: ${SAVE_DIR}"
echo "EVAL_SCRIPT_DIR: ${EVAL_SCRIPT_DIR}"
echo "S3PRL_REPO: ${S3PRL_REPO}"
echo "PARAFORMER: ${PARAFORMER_MODEL}"
echo "============================================"

wav_count=$(ls "$SAVE_DIR"/*.wav 2>/dev/null | wc -l)
echo "Found ${wav_count} wav files in SAVE_DIR"

echo "=== Step 1: WER Calculation ==="
WER_LOG="${LOG_BASE}/wer_evalonly_${TIME_STR}.log"
echo "  Log: ${WER_LOG}"
wer_stage "$SAVE_DIR" "$EVAL_META_PATH" "$LANG" "$WER_LOG"
echo "=== WER Calculation Done ==="

echo "=== Step 2: Speaker Similarity ==="
SIM_LOG="${LOG_BASE}/sim_evalonly_${TIME_STR}.log"
echo "  Log: ${SIM_LOG}"
sim_stage "$SAVE_DIR" "$EVAL_META_PATH" "$SIM_LOG"

RESULT_FILE="${WORKDIR}/run_cpp_eval_results.txt"
echo "==============================" >> "$RESULT_FILE"
echo "EVAL DONE: $(date)" >> "$RESULT_FILE"
cat "$SAVE_DIR"/wav_res_ref_text.wer >> "$RESULT_FILE" 2>/dev/null || true
cat "$SAVE_DIR"/wav_res_ref_text.sim >> "$RESULT_FILE" 2>/dev/null || true
echo " MODEL_PATH:     ${MODEL_PATH}" >> "$RESULT_FILE"
echo " CPP_BIN:        ${CPP_BIN}" >> "$RESULT_FILE"
echo " SAVE_DIR:       ${SAVE_DIR}" >> "$RESULT_FILE"
echo " EVAL_META_PATH: ${EVAL_META_PATH}" >> "$RESULT_FILE"
echo " EVAL_DATA_PATH: ${EVAL_DATA_PATH}" >> "$RESULT_FILE"
echo " SEED:           ${SEED}" >> "$RESULT_FILE"
echo "==============================" >> "$RESULT_FILE"

echo "=== Eval-only Done ==="
echo "Results in: ${SAVE_DIR}"
echo "Logs in:    ${LOG_BASE}/*_evalonly_${TIME_STR}.log"
echo "Summary:    ${RESULT_FILE}"
