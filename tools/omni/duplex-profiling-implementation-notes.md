# Duplex Profiling 实现要点与 TTS 疑问记录

## 背景

本轮改造目标是让 `tools/omni/test/test-duplex.cpp` 的双工测试能够从用户阶段判断是否满足双工：一个输入 frame 从底层开始处理，到 SPEAK 音频最终写成 wav，整体不超过 1 秒。

核心口径调整为：profiling 不看 Session 层 API 包络，不统计队列等待时间，只统计真实数据处理阶段。

## 实现要点

### 1. C++ 打点下沉到底层流水线

在 `tools/omni/omni.cpp` 中补充了 duplex 底层阶段打点，只保留用户要求的处理阶段：

- `duplex.encode`
- `duplex.llm.prefill`
- `duplex.llm.decode`
- 既有 `tts.infer`
- 既有 `t2w.infer`
- 既有 `t2w.write`

计时边界：

- `duplex.encode`：encoder 线程从 `encoder_queue` 取到 `DuplexEncodeReq` 后开始，到 `DuplexPrefillPacket` 生成完成结束。
- `duplex.llm.prefill`：LLM 线程拿到对应 `DuplexPrefillPacket` 后开始，到 fused/fallback prefill 写入 KV 完成结束。
- `duplex.llm.decode`：LLM decode 实际开始采样时开始，到本 frame decode 结束。
- `tts.infer`：TTS 线程拿到 `LLMOut` 后做实际 TTS 推理的时间。
- `t2w.infer` / `t2w.write`：Token2Wav 实际窗口推理和 wav 写盘时间。

### 2. 修复 frame 归因链路

duplex 路径中 `duplex_do_decode()` 推送 `LLMOut` 时补齐：

```cpp
llm_out->perf_chunk_index = perf_chunk_index;
```

这样 frame id 可以沿链路传递：

```text
DuplexEncodeReq.index
  -> DuplexPrefillPacket.index
  -> duplex_do_decode round_idx
  -> LLMOut.perf_chunk_index
  -> T2WOut.perf_chunk_index
  -> tts.infer / t2w.infer / t2w.write
```

### 3. 分析脚本收窄到 6 个阶段

`tools/omni/test/analyze_duplex_perf.py` 重写后只围绕以下阶段生成统计和图：

```text
duplex.encode
duplex.llm.prefill
duplex.llm.decode
tts.infer
t2w.infer
t2w.write
```

脚本不再展示 API、queue、wait、旧 `vision.encode`、旧 `audio.encode`、旧 `llm.prefill/decode` 等阶段。

输出重点：

- `omni-duplex-frame-summary.csv`
- `omni-duplex-sla.md`
- `figures/omni-duplex-pipeline.svg`
- `figures/omni-duplex-stage-latency.svg`
- `figures/omni-duplex-overlap-timeline.svg`
- `figures/omni-duplex-gpu-utilization.svg`

SLA 口径：

- 只统计 `SPEAK` frame。
- frame e2e 起点取该 frame 的 `duplex.encode.start`。
- frame e2e 终点取该 frame 最后一次 `t2w.write.end`。
- `LISTEN` frame 不要求 `tts/t2w` 阶段。
- `stream_prefill(index=0)` 的 session/ref audio 初始化不纳入 SLA。

## GPU 构建与验证

正确 GPU 构建方式需要完整激活 conda 环境：

```bash
cd /cache/hanqingzhe/llama.cpp-omni

source /cache/caitianchi/install/miniconda3/etc/profile.d/conda.sh
conda activate /cache/hanqingzhe/.conda/envs/cuda_132_clean

cmake -S . -B build-cuda132-gpu-fixed -G Ninja \
  -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
  -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-cuda132-gpu-fixed --target llama-omni-test-duplex --parallel "$(nproc)"
```

如果运行时找不到 CUDA/cuBLAS/OpenMP 库，再补：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH"
```

GPU 测试示例：

```bash
OMNI_GPU_PROF=1 \
OMNI_GPU_PROF_INTERVAL_MS=20 \
OMNI_GPU_PROF_DEVICES=0 \
OMNI_GPU_PROF_FILE=duplex_q4_k_m_1s.gpu.log \
./bin/llama-omni-test-duplex \
  -m /cache/hanqingzhe/o45-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --omni \
  --test /cache/hanqingzhe/llama.cpp-omni/tools/omni/assets/test_case/duplex_omni_test_case/duplex_omni_test_case_ 36 \
  --stream-interval 1000 \
  --ref-audio /cache/hanqingzhe/llama.cpp-omni/tools/omni/assets/default_ref_audio/default_ref_audio.wav \
  -ngl 99 \
  -c 4096 \
  -o /home/modelbest/llama.cpp-omni/tools/omni/output/duplex_q4_k_m_1s \
  2>&1 | tee duplex_q4_k_m_1s.log
```

GPU sampler 现在默认不会遍历服务器上的所有 GPU：

- 如果设置了 `OMNI_GPU_PROF_DEVICES`，按这个环境变量采样，值是 NVML 物理 GPU index，例如 `0` 或 `0,2`。
- 如果没设置 `OMNI_GPU_PROF_DEVICES`，但设置了 `CUDA_VISIBLE_DEVICES`，默认采样其中第一个可见 GPU。
- 如果两者都没设置，默认只采样 `device 0`。

这样可以避免同一台服务器上其他用户占用其他 GPU 时，被误统计进本次 duplex profiling。

分析命令：

```bash
python3 /cache/hanqingzhe/llama.cpp-omni/tools/omni/test/analyze_duplex_perf.py \
  --log /cache/hanqingzhe/llama.cpp-omni/build-cuda132-gpu-fixed/duplex_q4_k_m_full.log \
  --gpu-log /cache/hanqingzhe/llama.cpp-omni/build-cuda132-gpu-fixed/duplex_q4_k_m_full.gpu.log \
  --out-dir /cache/hanqingzhe/llama.cpp-omni/build-cuda132-gpu-fixed/duplex_q4_k_m_full_report
```

## 当前报告观察

基于 `build-cuda132-gpu-fixed/duplex_q4_k_m_full_report`：

- 总 frame：36
- SPEAK：9
- LISTEN：27
- 完整 SPEAK：8
- 不完整 SPEAK：1
- 完整 SPEAK e2e：
  - avg：654.9 ms
  - p50：657.9 ms
  - p95：831.7 ms
  - max：898.9 ms

异常点是 `frame 33`：

```text
frame_id=33
decision=speak
tts_ms=19.844
t2w_infer_ms=0
t2w_write_ms=0
missing=t2w_infer,t2w_write
```

`omni-duplex-tts-token-stats.csv` 中对应：

```text
chunk=33
tts_chunk=2
llm_tokens=0
filtered_llm_tokens=0
condition_tokens=2
compute_tokens=2
audio_tokens_generated=1
is_end_of_turn=1
flush_only=1
```

原始日志中对应事件：

```text
chunk=33 tts.infer start:
llm_tokens=0, filtered_llm_tokens=0, condition_tokens=0,
compute_tokens=0, audio_tokens_generated=0,
is_end_of_turn=1, flush_only=1

chunk=33 queue.t2w.enqueue:
tokens=1,is_final=1,is_chunk_end=0

chunk=33 tts.infer end:
condition_tokens=2, compute_tokens=2,
audio_tokens_generated=1,
is_end_of_turn=1, flush_only=1
```

而后续 T2W 实际事件归属为 `chunk=32`：

```text
chunk=32 t2w.infer window_tokens=28,is_last=0 -> wav_35000.wav
chunk=32 t2w.infer window_tokens=6,is_last=1 -> wav_36000.wav
```

## TTS 相关疑问

### 疑问 1：LLM 不生成东西，TTS 不就不该调度吗？

普通生成 chunk 口径下，是的。若 LLM 没有生成可用于 TTS 的 token，理论上不应该跑完整 TTS 正文推理。

但当前代码还有一个例外：turn 结束 flush。

当前逻辑中，如果 TTS 线程收到：

```text
token_ids.size = 0
llm_text.len = 0
is_end_of_turn = 1
```

它会走空 final chunk flush 分支：

```text
LLM 结束本轮
  -> TTS 收到空 final chunk
  -> 调一次 TTS flush
  -> T2W 收到 final/尾包信号
  -> T2W flush 滑窗缓存
```

所以 `frame 33` 不是普通文本转语音推理，而是尾包 flush。

### 疑问 2：是不是有一段 TTS 没生成东西但 T2W 还在操作？

从图和日志看，表象上是的，但更精确地说：

- `frame 33` 的 TTS 是 `flush_only=1` 的尾包 flush。
- 它的 LLM token 为 0。
- 它最终采样出了 1 个 audio token，并 enqueue 给 T2W。
- 但 T2W 线程实际处理窗口时，把这段 final/尾包操作归到了 `chunk=32`。

这说明当前 T2W 归因是按一次 drain 中的“第一个有效 `perf_chunk_index`”来标记的。当多个 `T2WOut` 被合并进同一个 token buffer/window 时，后来的 `frame 33` final token 可能被合并到 `chunk=32` 的 T2W window 里。

## 初步判断

当前问题不是 T2W 凭空生成，而是两个问题叠加：

1. `flush_only` 的空 TTS chunk 被当成独立 SPEAK frame 记录。
2. T2W 端流式 buffer/window 跨 frame 合并，导致 final flush 的归因落在前一个 chunk 上。

因此 `frame 33` 在 report 中显示为 incomplete：

```text
missing=t2w_infer,t2w_write
```

但真实语义更像是：`frame 33` 是上一段 SPEAK utterance 的尾包 flush，而不是一个新的独立语音 frame。

## 后续建议

### 1. 引入 utterance_id

仅靠 `frame_id/chunk` 很难表达 TTS/T2W 的流式尾包。建议增加独立的 `utterance_id`：

```text
SPEAK turn 开始 -> utterance_id++
同一段语音的多个 LLMOut/TTS/T2WOut 共享 utterance_id
flush_only final chunk 归属到当前 utterance_id
LISTEN 或 turn end 后关闭当前 utterance
```

这样 frame 图和 utterance 图可以分开：

- frame 图：输入 frame 的 encode/LLM 处理。
- utterance 图：输出语音的 TTS/T2W 处理。

当前分析脚本已先做了轻量版 `utterance_id`：连续 SPEAK frame 会被分到同一个 `utterance_id`，遇到 LISTEN 后结束当前 utterance。这个 id 只用于报告归组和聚焦 pipeline 图，不改变 C++ 运行逻辑。

```text
LISTEN LISTEN SPEAK SPEAK SPEAK LISTEN
              └──── utterance 1 ────┘
```

聚焦版 pipeline 图会选择一个连续 SPEAK utterance，并额外带上前后最多两个 LISTEN frame，避免把 36 个 frame 全塞进图里导致看不清。

### 2. 分析脚本临时修正

在没有 `utterance_id` 前，分析脚本可以先做保守处理：

- `flush_only=1 && llm_tokens=0` 不作为新的 SPEAK frame 主体。
- 这类 TTS flush 可以归并到最近一个已有 TTS/T2W 的 SPEAK frame。
- SPEAK frame 完整性不要因为尾包 flush 的 chunk id 不一致而误判。

### 3. T2W 打点细化

当前 `t2w.infer` 只记录一个 `perf_chunk_index`。如果一次 T2W window 包含多个 `T2WOut` 来源，建议 detail 中增加：

```text
source_chunks=31,32,33
source_token_counts=25,25,1
is_final_source=33
```

这样分析脚本可以判断：

- 哪个 frame 贡献了 token。
- 哪个 frame 触发了 final flush。
- wav 文件应该归属到哪个 utterance，而不是单个 frame。

## 结论

当前实现已经能测量用户要求的 6 个处理阶段，但这次报告暴露了一个 TTS/T2W 流式归因问题：

`frame_id` 足够描述输入侧 encode/LLM，但不够描述输出侧 TTS/T2W 的跨 frame 流式缓存和尾包 flush。

下一步如果要继续精确分析 TTS/T2W，应引入 `utterance_id` 或至少在 T2W detail 中记录多来源 chunk 信息。这样才能避免把 `flush_only` 尾包误判成独立 SPEAK frame，或把 T2W 尾包错误归到前一个 frame。

## 报告精简策略

面向用户的默认报告已收窄为三类输出：

```text
omni-duplex-perf-report.md
figures/omni-duplex-pipeline.svg
figures/omni-duplex-gpu-utilization.svg
```

主报告开头直接给出是否支持双工：

- 是否满足 `SPEAK e2e <= 1s`
- 完整 SPEAK 数量
- avg / p95 / max e2e
- 判定原因

随后只保留：

1. 聚焦一个连续 SPEAK utterance 的 pipeline 图。
2. 6 个核心阶段的平均耗时和速度。
3. GPU 采样摘要和 GPU 利用率图。

默认不再生成或展示长 CSV、多余 SVG、SLA 细表、stage timing markdown 等面向调试的材料。需要深挖时再临时打开相关导出。
