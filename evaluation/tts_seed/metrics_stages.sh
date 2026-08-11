#!/usr/bin/env bash
# WER / SIM 打分实现，供 run_tts_eval_cpp_zh.sh 与 run_eval_only.sh source。
# 分片数按 CPU 核数估算，与 GPU 卡数解耦。

WER_THREADS=${WER_THREADS:-16}
SIM_THREADS=${SIM_THREADS:-16}
SHARD_CAP=${SHARD_CAP:-32}

DEVICE_ENV_VAR=${DEVICE_ENV_VAR:-CUDA_VISIBLE_DEVICES}
IFS=',' read -r -a DEVICE_ID_ARR <<< "${DEVICE_IDS:-0}"
dev_of() { echo "${DEVICE_ID_ARR[$(( $1 % ${#DEVICE_ID_ARR[@]} ))]}"; }

# $1=样本数 $2=每片线程数 -> 分片数
_shard_count() {
    local num=$1 threads=$2 jobs
    jobs=$(( $(nproc) / threads ))
    [ "$jobs" -gt "$SHARD_CAP" ] && jobs=$SHARD_CAP
    [ "$jobs" -lt 1 ] && jobs=1
    [ "$jobs" -gt "$num" ] && jobs=$num
    echo "$jobs"
}

_wait_shards() {
    local name=$1 pid failed=0
    shift
    for pid in "$@"; do
        wait "$pid" || failed=1
    done
    [ "$failed" -eq 0 ] && return 0
    echo "ERROR: ${name} 有分片异常退出" >&2
    return 1
}

_assert_no_loss() {
    local want=$1 merged=$2 name=$3 log=$4 got
    got=$(wc -l < "$merged")
    if [ "$got" -ne "$want" ]; then
        echo "ERROR: ${name} 只算出 ${got} 条，输入 ${want} 条，缺 $((want - got)) 条；详见 ${log}" >&2
        return 1
    fi
    echo "  ${name}: ${got}/${want} 条"
}

# wer_stage <save_dir> <meta_lst> <lang> <log_file>
wer_stage() {
    local save_dir=$1 meta=$2 lang=$3 log=$4
    local pair_list="$save_dir/wav_res_ref_text"
    local score_file="$save_dir/wav_res_ref_text.wer"

    python3 "${EVAL_SCRIPT_DIR}/get_wav_res_ref_text.py" "$meta" "$save_dir" "$pair_list"

    local shard_dir="$save_dir/thread_metas_wer_$(date +%s)"
    local out_dir="$shard_dir/results"
    mkdir -p "$out_dir"

    local num jobs
    num=$(wc -l < "$pair_list")
    jobs=$(_shard_count "$num" "$WER_THREADS")
    echo "  WER: ${num} 条拆成 ${jobs} 片，每片 ${WER_THREADS} 线程"
    split -l $(( num / jobs + 1 )) --additional-suffix=.lst -d "$pair_list" "$shard_dir/thread-"

    local pids=() rank=0 lst
    for lst in "$shard_dir"/thread-*.lst; do
        OMP_NUM_THREADS=$WER_THREADS \
        env "${DEVICE_ENV_VAR}=$(dev_of "$rank")" \
            python3 "${EVAL_SCRIPT_DIR}/run_wer.py" \
            "$lst" "$out_dir/$(basename "$lst" .lst).wer.out" "$lang" \
            >> "$log" 2>&1 &
        pids+=($!)
        rank=$((rank + 1))
    done
    _wait_shards WER "${pids[@]}"

    local merged="$out_dir/merge.out"
    cat "$out_dir"/thread-*.wer.out > "$merged"
    _assert_no_loss "$num" "$merged" WER "$log"
    python3 "${EVAL_SCRIPT_DIR}/average_wer.py" "$merged" "$score_file"
    rm -f "$pair_list"
}

# sim_stage <save_dir> <meta_lst> <log_file>
sim_stage() {
    local save_dir=$1 meta=$2 log=$3
    if [ ! -f "${SPEAKER_CKPT}" ]; then
        echo "WARNING: Speaker checkpoint not found at ${SPEAKER_CKPT}, skipping SIM."
        return 0
    fi
    local pair_list="$save_dir/wav_res_ref_text"
    local score_file="$save_dir/wav_res_ref_text.sim"

    python3 "${EVAL_SCRIPT_DIR}/get_wav_res_ref_text.py" "$meta" "$save_dir" "$pair_list"

    local shard_dir="$save_dir/thread_metas_sim_$(date +%s)"
    local out_dir="$shard_dir/results"
    mkdir -p "$out_dir"

    local num jobs
    num=$(wc -l < "$pair_list")
    jobs=$(_shard_count "$num" "$SIM_THREADS")
    echo "  SIM: ${num} 对拆成 ${jobs} 片，每片 ${SIM_THREADS} 线程"
    split -l $(( num / jobs + 1 )) --additional-suffix=.lst -d "$pair_list" "$shard_dir/pair-"

    local pids=() rank=0 lst
    for lst in "$shard_dir"/pair-*.lst; do
        OMP_NUM_THREADS=$SIM_THREADS \
        env "${DEVICE_ENV_VAR}=$(dev_of "$rank")" \
            python3 "${SPEAKER_VERIF_DIR}/verification_pair_list_v2.py" \
            "$lst" \
            --model_name wavlm_large \
            --checkpoint "$SPEAKER_CKPT" \
            --scores "$out_dir/$(basename "$lst" .lst).sim.out" \
            --wav1_start_sr 0 \
            --wav2_start_sr 0 \
            --wav1_end_sr -1 \
            --wav2_end_sr -1 \
            --device "${SIM_DEVICE:-cuda:0}" \
            >> "$log" 2>&1 &
        pids+=($!)
        rank=$((rank + 1))
    done
    _wait_shards SIM "${pids[@]}"

    local merged="$save_dir/merge.out"
    : > "$merged"
    # 分片末行是无换行的 "avg score: ..."，需逐文件过滤后再合并
    local f
    for f in "$out_dir"/pair-*.sim.out; do
        grep -v "avg score" "$f" >> "$merged" || true
    done
    _assert_no_loss "$num" "$merged" SIM "$log"
    python3 "${SPEAKER_VERIF_DIR}/average.py" "$merged" "$score_file"
    rm -f "$pair_list"
    echo "=== SIM Calculation Done ==="
}
