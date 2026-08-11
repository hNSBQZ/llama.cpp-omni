#!/usr/bin/env bash
# WER 与 SIM 两段的实现，run_tts_eval_cpp_zh.sh 与 run_eval_only.sh 共用一份。
#
# 这两段都是纯 CPU：paraformer 与 WavLM 在昇腾上都没有 NPU 后端。上游按卡数给 WER
# 分片、SIM 干脆单进程，13 卡跑全量 2020 条时分别要 2 小时 11 分和 4 小时 11 分，而生成
# 段只用 12 分钟。卡数是生成段的拓扑，套到 CPU 阶段上只是白闲着核，所以分片数与卡数
# 解耦，由核数除以每片线程数决定。
#
# 用 source 而不是两边各抄一份：这两段本来就是逐行重复，分片、合并、校验的细节抄两遍
# 必然走形——WER 那处补零就是这么写错的。

# 每片给多少线程。torch 默认按物理核数开线程，多进程时会互相抢核，必须显式钉住。
WER_THREADS=${WER_THREADS:-16}
SIM_THREADS=${SIM_THREADS:-16}
# 分片上限。再往上每片样本太少，加载模型的固定开销（WER 十几秒、SIM 二十几秒）开始占主导。
SHARD_CAP=${SHARD_CAP:-32}

# 设备变量名随后端变（CUDA_VISIBLE_DEVICES / ASCEND_RT_VISIBLE_DEVICES），DEVICE_IDS 是
# 可用物理卡列表。分片数与卡数无关且通常更多，超出的片按取模落回实际存在的卡上；
# WER_DEVICE / SIM_DEVICE 是 cpu 时这个变量只是空转。
DEVICE_ENV_VAR=${DEVICE_ENV_VAR:-CUDA_VISIBLE_DEVICES}
IFS=',' read -r -a DEVICE_ID_ARR <<< "${DEVICE_IDS:-0}"
dev_of() { echo "${DEVICE_ID_ARR[$(( $1 % ${#DEVICE_ID_ARR[@]} ))]}"; }

# $1=样本数 $2=每片线程数 -> 实际分片数
_shard_count() {
    local num=$1 threads=$2 jobs
    jobs=$(( $(nproc) / threads ))
    [ "$jobs" -gt "$SHARD_CAP" ] && jobs=$SHARD_CAP
    [ "$jobs" -lt 1 ] && jobs=1
    # 分片不能多于样本数，否则 split 少产出文件。
    [ "$jobs" -gt "$num" ] && jobs=$num
    echo "$jobs"
}

# 等一组分片跑完，任一失败就返回非零。不带参数的 `wait` 恒返回 0，13 片那轮有三片
# 起手就 FileNotFoundError 退出，就是被它吞掉的。
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

# 合并结果的行数必须与输入一致。分片阶段最难发现的错是静默少算：某片没跑起来、或某条
# 样本被跳过，汇总脚本照样算得出一个漂亮的分数。宁可让评测失败，也不能报一个少算了
# 样本的成绩。
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

    # 遍历 split 真正产出的文件，不按序号拼名字去读：`split -d` 的后缀是定长两位，序号
    # 到 10 就和 `thread-0$rank` 那种拼法错开，对不上的片会静默丢掉（13 片那轮丢了三片、
    # 460 条，WER 只统计了 1560 条）。
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

    # average.py 拿 merge.out 的路径推 eval_result.out 的位置，所以它得留在 save_dir 下。
    local merged="$save_dir/merge.out"
    : > "$merged"
    # 每片最后一行是 "avg score: ..."，而且写的时候没带换行。所以要逐个文件 grep，不能
    # 先 cat 再滤：那会把上一片的 avg 行和下一片的第一对粘成一行，连带丢掉一对。
    local f
    for f in "$out_dir"/pair-*.sim.out; do
        grep -v "avg score" "$f" >> "$merged" || true
    done
    _assert_no_loss "$num" "$merged" SIM "$log"
    python3 "${SPEAKER_VERIF_DIR}/average.py" "$merged" "$score_file"
    rm -f "$pair_list"
    # 这句放在函数里而不是调用点：没有 SPEAKER_CKPT 时上面就 return 了，那种情况不该报 Done
    echo "=== SIM Calculation Done ==="
}
