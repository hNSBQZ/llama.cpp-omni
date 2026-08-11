#!/bin/bash
# MiniCPM-o TTS 中文评测（seed-zh）。路径与设备见 pipeline.env / 环境变量。
set -e

source "$(cd "$(dirname "$0")"; pwd)/pipeline.env"
source "$(cd "$(dirname "$0")"; pwd)/metrics_stages.sh"

echo "Python: $(command -v python3)"

export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}:${CPP_BUILD_LIB}:${LD_LIBRARY_PATH}"

TIME_STR=$(date +%Y%m%d_%H%M%S)
SEED=${SEED:-42}

MODEL_PATH=${MODEL_PATH:-"/path/to/MiniCPM-o-4_5-gguf"}
TTS_MODEL_PATH=${TTS_MODEL_PATH:-"$MODEL_PATH/tts/MiniCPM-o-4_5-tts-F16.gguf"}
CPP_BIN=${CPP_BIN:-"/path/to/llama.cpp-omni/build/bin/llama-omni-tts-eval"}
ONNX_MODEL_DIR=${ONNX_MODEL_DIR:-"/path/to/Step-Audio-2-mini/token2wav"}

SAVE_DIR=${SAVE_DIR:-"${WORKDIR:-$(cd "$(dirname "$0")"; pwd)}/eval_results/cpp-zh-${TIME_STR}-${SEED}"}

LANG="zh"
EVAL_META_PATH=${EVAL_META_PATH:-"/path/to/seedtts_testset_zh/zh/meta.lst"}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-"/path/to/seedtts_testset_zh/zh"}
SPEAKER_CKPT=${SPEAKER_CKPT:-"/path/to/wavlm_large_finetune.pth"}

echo "============================================"
echo "CPP TTS Evaluation Pipeline"
echo "============================================"
echo "MODEL_PATH:      ${MODEL_PATH}"
echo "TTS_MODEL_PATH:  ${TTS_MODEL_PATH:-(auto from MODEL_PATH)}"
echo "CPP_BIN:         ${CPP_BIN}"
echo "ONNX_MODEL_DIR:  ${ONNX_MODEL_DIR}"
echo "SAVE_DIR:        ${SAVE_DIR}"
echo "EVAL_META_PATH:  ${EVAL_META_PATH}"
echo "EVAL_DATA_PATH:  ${EVAL_DATA_PATH}"
echo "LANGUAGE:        ${LANG}"
echo "SEED:            ${SEED}"
echo "============================================"

mkdir -p "$SAVE_DIR"

WORKDIR=$(cd "$(dirname "$0")"; pwd)
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NUM_SAMPLES=${NUM_SAMPLES:-10000000}
EVAL_SCRIPT_DIR=${EVAL_SCRIPT_DIR:-"${WORKDIR}/eval_tools"}
SPEAKER_VERIF_DIR=${SPEAKER_VERIF_DIR:-"${EVAL_SCRIPT_DIR}/speaker_verification"}
export PARAFORMER_MODEL
export S3PRL_REPO
export WER_DEVICE

echo "EVAL_SCRIPT_DIR: ${EVAL_SCRIPT_DIR}"
echo "S3PRL_REPO:      ${S3PRL_REPO}"
echo "PARAFORMER:      ${PARAFORMER_MODEL}"

LOG_BASE="${SAVE_DIR}/logs"
mkdir -p "$LOG_BASE"

TTS_ARG=""
if [ -n "$TTS_MODEL_PATH" ]; then
    TTS_ARG="--tts-model-path ${TTS_MODEL_PATH}"
fi

# --- 1. 预提取 prompt_bundle（各 rank 共享）---
echo "=== Step 1: Pre-extract prompt_bundles ==="
BUNDLE_DIR="${SAVE_DIR}/_prompt_bundles"
mkdir -p "$BUNDLE_DIR"

EXTRACT_LOG="${LOG_BASE}/extract_bundle_${TIME_STR}.log"

python3 -c "
import os, hashlib
meta_path = '${EVAL_META_PATH}'
data_path = '${EVAL_DATA_PATH}'
bundle_dir = '${BUNDLE_DIR}'
num_samples = ${NUM_SAMPLES}
world = ${GPUS_PER_NODE}
cap = num_samples * world if num_samples < 10000000 else None
seen = set()
tasks = []
with open(meta_path) as f:
    for i, line in enumerate(f):
        if cap is not None and i >= cap:
            break
        parts = line.strip().split('|')
        wav_rel = None
        if len(parts) == 5:
            wav_rel = parts[2]
        elif len(parts) == 4:
            wav_rel = parts[2]
        elif len(parts) == 3:
            wav_rel = parts[2]
        if wav_rel and wav_rel not in seen:
            seen.add(wav_rel)
            wav_full = os.path.join(data_path, wav_rel)
            h = hashlib.md5(wav_rel.encode()).hexdigest()[:12]
            out = os.path.join(bundle_dir, h)
            if not os.path.exists(os.path.join(out, 'spk_f32.bin')):
                tasks.append(f'{wav_full}\t{out}')
with open(os.path.join(bundle_dir, '_batch_list.tsv'), 'w') as f:
    f.write('\n'.join(tasks) + '\n')
print(f'Total unique wavs: {len(seen)}, to extract: {len(tasks)}')
"

BATCH_LIST="${BUNDLE_DIR}/_batch_list.tsv"
N_EXTRACT=$(wc -l < "$BATCH_LIST")
if [ "$N_EXTRACT" -gt 0 ]; then
    echo "  Log: ${EXTRACT_LOG}"
    env "${DEVICE_ENV_VAR}=$(dev_of 0)" python3 "${WORKDIR}/extract_prompt_bundle.py" \
        --batch-list "$BATCH_LIST" \
        --model-dir "$ONNX_MODEL_DIR" \
        --device "${TTS_DEVICE:-cuda}" \
        --skip-existing \
        > "${EXTRACT_LOG}" 2>&1
    echo "=== Prompt bundle extraction done ==="
else
    echo "=== All prompt_bundles already cached ==="
fi

# --- 2. TTS 推理（每卡一个进程）---
echo "=== Step 2: C++ TTS Inference (${GPUS_PER_NODE} GPUs) ==="
echo "  Per-GPU logs: ${LOG_BASE}/cpp_gpu*_${TIME_STR}.log"
for i in $(seq 0 $((GPUS_PER_NODE - 1)))
do
    env "${DEVICE_ENV_VAR}=$(dev_of "$i")" python3 "${WORKDIR}/generate_cpp.py" \
        --cpp-bin "${CPP_BIN}" \
        --model-path "${MODEL_PATH}" \
        ${TTS_ARG} \
        --save-dir "${SAVE_DIR}" \
        --eval-meta-path "${EVAL_META_PATH}" \
        --eval-data-path "${EVAL_DATA_PATH}" \
        --onnx-model-dir "${ONNX_MODEL_DIR}" \
        --language "${LANG}" \
        --seed "${SEED}" \
        --temperature 0.3 \
        --teacher-forcing \
        --device "${TTS_DEVICE:-cuda}" \
        --n-gpu-layers "${CPP_NGL:-99}" \
        --t2w-device "${T2W_DEVICE:-gpu:0}" \
        --num-samples "${NUM_SAMPLES}" \
        --rank $i \
        --world-size ${GPUS_PER_NODE} \
        > "${LOG_BASE}/cpp_gpu${i}_${TIME_STR}.log" 2>&1 &
done
wait
echo "=== TTS Inference Done ==="

# --- 3. WER ---
echo "=== Step 3: WER Calculation ==="
WER_LOG="${LOG_BASE}/wer_${TIME_STR}.log"
echo "  Log: ${WER_LOG}"
wer_stage "$SAVE_DIR" "$EVAL_META_PATH" "$LANG" "$WER_LOG"
echo "=== WER Calculation Done ==="

# --- 4. SIM ---
echo "=== Step 4: Speaker Similarity ==="
SIM_LOG="${LOG_BASE}/sim_${TIME_STR}.log"
echo "  Log: ${SIM_LOG}"
sim_stage "$SAVE_DIR" "$EVAL_META_PATH" "$SIM_LOG"

# --- 5. 汇总 ---
RESULT_FILE="${WORKDIR}/run_cpp_eval_results.txt"
echo "==============================" >> "$RESULT_FILE"
echo "EVAL DONE: $(date)" >> "$RESULT_FILE"
cat "$SAVE_DIR"/wav_res_ref_text.wer >> "$RESULT_FILE" 2>/dev/null || true
cat "$SAVE_DIR"/wav_res_ref_text.sim >> "$RESULT_FILE" 2>/dev/null || true
echo " MODEL_PATH:      ${MODEL_PATH}" >> "$RESULT_FILE"
echo " TTS_MODEL_PATH:  ${TTS_MODEL_PATH:-(auto)}" >> "$RESULT_FILE"
echo " CPP_BIN:         ${CPP_BIN}" >> "$RESULT_FILE"
echo " SAVE_DIR:        ${SAVE_DIR}" >> "$RESULT_FILE"
echo " EVAL_META_PATH:  ${EVAL_META_PATH}" >> "$RESULT_FILE"
echo " EVAL_DATA_PATH:  ${EVAL_DATA_PATH}" >> "$RESULT_FILE"
echo " SEED:            ${SEED}" >> "$RESULT_FILE"
echo "==============================" >> "$RESULT_FILE"

echo "=== All Evaluation Done ==="
echo "Results saved to: ${SAVE_DIR}"
echo "Logs saved to:    ${LOG_BASE}/*_${TIME_STR}.log"
echo "Summary appended to: ${RESULT_FILE}"
