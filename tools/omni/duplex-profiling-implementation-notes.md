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
- `tts.condition`
- `tts.prefill`
- `tts.decode`
- 既有 `t2w.infer`
- 既有 `t2w.write`

计时边界：

- `duplex.encode`：encoder 线程从 `encoder_queue` 取到 `DuplexEncodeReq` 后开始，到 `DuplexPrefillPacket` 生成完成结束。
- `duplex.llm.prefill`：LLM 线程拿到对应 `DuplexPrefillPacket` 后开始，到 fused/fallback prefill 写入 KV 完成结束。
- `duplex.llm.decode`：LLM decode 实际开始采样时开始，到本 frame decode 结束。
- `tts.condition`：TTS 线程拿到 `LLMOut` 后，过滤 token、构造 text embedding、projector、normalize、merge condition embedding 的时间。
- `tts.prefill`：condition embedding 写入 TTS KV 的时间。
- `tts.decode`：TTS 自回归采样 audio token 的时间。
- `t2w.infer` / `t2w.write`：Token2Wav 实际窗口推理和 wav 写盘时间。

现有打点逻辑：

- C++ 当前只输出上述 6 个 `[DUPLEX_PERF]` stage；API 包络、queue wait、queue enqueue、旧同步 `vision.encode` / `audio.encode` / `llm.prefill` / `llm.decode` 等冗余打点已删除。
- `duplex.encode` 从 encoder 线程取出 `DuplexEncodeReq` 后开始，到 VPM/APM 生成 `DuplexPrefillPacket` 完成结束；提交请求到 `encoder_queue` 和推送 `prefill_queue` 不单独计时。
- `duplex.llm.prefill` 从 LLM 线程取到 `DuplexPrefillPacket` 后开始，到 fused/fallback prefill 写入 KV 完成结束；等待 packet 到达不计时。
- `duplex.llm.decode` 覆盖本 frame 的 decode 函数主体，包括采样前少量状态准备、force-listen 分支、采样循环、文本/TTS 输出入队等尾部工作。
- `tts.condition` / `tts.prefill` / `tts.decode` 分别对应 TTS condition 构造、TTS KV prefill、TTS 自回归 decode。TTS decode 是流式生产：自回归采样每得到一个 audio token 就先放入 `stream_buffer`，首包攒到 28 tokens 后推一次 `T2WOut`，后续每 25 tokens 推一次；如果是 `is_end_of_turn` 的 final chunk，则中途不推，最后把剩余 tokens 作为 final/flush 包一次性送给 T2W。
- `t2w.infer` / `t2w.write` 分别覆盖 token2wav window 推理和 wav 文件写盘；T2W 线程等待 queue 数据、drain queue、组 window 的控制开销不单独计时。

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
  -> tts.condition / tts.prefill / tts.decode / t2w.infer / t2w.write
```

### 3. 分析脚本收窄到 6 个阶段

`tools/omni/test/analyze_duplex_perf.py` 重写后只围绕以下阶段生成统计和图：

```text
duplex.encode
duplex.llm.prefill
duplex.llm.decode
tts.condition
tts.prefill
tts.decode
t2w.infer
t2w.write
```

脚本不再展示 API、queue、wait、旧 `vision.encode`、旧 `audio.encode`、旧 `llm.prefill/decode` 等阶段。

默认输出重点：

- `omni-duplex-perf-report.md`
- `figures/omni-duplex-pipeline.svg`
- `figures/omni-duplex-gpu-utilization.svg`

旧的 CSV、SLA 细表、stage timing markdown、多余 SVG 会被清理，不再作为默认报告产物。主报告的“阶段概览”会展示核心 stage 的耗时，其中 `duplex.llm.prefill`、`duplex.llm.decode`、`tts.prefill`、`tts.decode` 会额外展示 SPEAK frame 的 token 数和 token 推理速度。

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

cmake -S . -B build -G Ninja \
  -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
  -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --target llama-omni-test-duplex --parallel "$(nproc)"
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
  --log /cache/hanqingzhe/llama.cpp-omni/build/duplex_q4_k_m_1s.log \
  --gpu-log /cache/hanqingzhe/llama.cpp-omni/build/duplex_q4_k_m_1s.gpu.log \
  --out-dir /cache/hanqingzhe/llama.cpp-omni/build/duplex_q4_k_m_1s_report
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

## TTS/T2W 内部分层补充

严格说，`hidden_state`、LLM token embedding merge、投影、归一化、自回归采样都属于 TTS 侧，不属于 `t2w.infer`。当前打点已把原来的大包 `tts.infer` 拆成 `tts.condition`、`tts.prefill`、`tts.decode`；`t2w.infer` 只从 audio token window 开始，负责 token2wav 推理。

当前 duplex TTS 路径可以细分为：

```text
LLMOut(token_ids, hidden_states)
  -> filter_special_tokens
  -> emb_text(token_ids) 得到 TTS text embedding
  -> projector_semantic(hidden_states) 把 LLM hidden 投到 TTS embedding 维度
  -> normalize_l2_per_token(projected_hidden)
  -> merged_embedding = emb_text + normalized_projected_hidden
  -> 拼接 condition token，例如 audio_bos / final text_eos
  -> prefill_with_emb_tts(condition)
  -> 自回归采样 audio tokens
  -> enqueue T2WOut(audio_tokens)
```

其中 `projector_semantic` 当前优先走 `projector_forward()` 的 ggml 后端实现，fallback 才走旧的 CPU float 权重实现。`normalize_l2_per_token()` 仍在调用者侧单独执行；所以“投影和归一化一次算子图完成”不是当前代码的准确描述。更准确的说法是：投影可以走一次 projector graph，归一化和 merge 目前仍在 `tts_thread_func_duplex()` 里完成。

TTS 生成 audio token 是自回归的。`generate_audio_tokens_local()` 先用 condition embedding 做一次 `prefill_with_emb_tts()`，随后在最多 `max_audio_tokens` 次循环里调用 `sample_tts_token()`。每采样出一个 audio token，就用 `emb_code` 查 embedding，再通过 `prefill_with_emb_tts(..., n_pos=1)` 把这个 token 喂回 TTS KV cache，继续下一步采样。duplex 模式下当前 `max_audio_tokens=26`，非 final chunk 会强制满足最小 token 数，final/end-of-turn chunk 允许更早 EOS/flush。

T2W 路径则在 TTS 产出 audio tokens 之后：

```text
T2WOut(audio_tokens)
  -> T2W 线程 drain queue
  -> token_buffer / sliding window 组 window
  -> token2wav_session->feed_window(window, is_last_window)
  -> 如果产生 waveform，则写 wav 文件
```

当前 TTS/T2W 细分打点为：

- `tts.condition`：`emb_text`、`projector_semantic`、`normalize`、`merge`。
- `tts.prefill`：condition embedding 写入 TTS KV。
- `tts.decode`：自回归 audio token 循环。
- `t2w.window`：T2W queue drain、token buffer/window 组装。
- `t2w.infer`：`feed_window()`。
- `t2w.write`：wav 写盘。

当前代码还有一个需要后续确认的细节：duplex 外层在构造 `merged_embeddings` 时已经追加了一次 `audio_bos_embed`，而 `generate_audio_tokens_local()` 内部也会按 Python streaming 逻辑追加 `text_eos_embed`（final 时）和 `audio_bos_embed`。这可能只是历史兼容/统计口径问题，也可能导致 condition token 计数和实际 condition 内容需要重新核对。

## 初步判断

当前问题不是 T2W 凭空生成，而是两个问题叠加：

1. `flush_only` 的空 TTS chunk 被当成独立 SPEAK frame 记录。
2. T2W 端流式 buffer/window 跨 frame 合并，导致 final flush 的归因落在前一个 chunk 上。

因此 `frame 33` 在 report 中显示为 incomplete：

```text
missing=t2w_infer,t2w_write
```

但真实语义更像是：`frame 33` 是上一段 SPEAK utterance 的尾包 flush，而不是一个新的独立语音 frame。

## 当前完成情况

### 1. 核心打点收敛

C++ perf 日志现在只保留核心处理 stage：

```text
duplex.encode
duplex.llm.prefill
duplex.llm.decode
tts.condition
tts.prefill
tts.decode
t2w.infer
t2w.write
```

API 包络、queue wait、queue enqueue、旧同步 `vision.encode` / `audio.encode` / `llm.prefill` / `llm.decode` 等冗余 perf 调用已从代码中删除。当前报告口径就是“真实处理阶段”，不再统计队列等待。

### 2. frame/utterance 报告归组

分析脚本已经实现轻量版 `utterance_id`：连续 SPEAK frame 会被分到同一个 `utterance_id`，遇到 LISTEN 后结束当前 utterance。这个 id 只用于报告归组和聚焦 pipeline 图，不改变 C++ 运行逻辑。

```text
LISTEN LISTEN SPEAK SPEAK SPEAK LISTEN
              └──── utterance 1 ────┘
```

聚焦版 pipeline 图会选择一个连续 SPEAK utterance，并额外带上前后最多两个 LISTEN frame，避免把 36 个 frame 全塞进图里导致看不清。

### 3. 模型推理速度统计

需要的 token 速度统计没有删。当前有两层来源：

- C++ `[DUPLEX_PERF]` end 事件：`duplex.llm.prefill`、`duplex.llm.decode`、`tts.prefill`、`tts.decode` 的 detail 中保留真实计算 token 数，并带有 `tokens_per_s` / `ms_per_token`。
- 分析脚本：`TOKEN_SPEED_STAGES = ["duplex.llm.prefill", "duplex.llm.decode", "tts.prefill", "tts.decode"]`，会在主报告“阶段概览”里输出 SPEAK frame 的 token 数和速度列。

速度口径：

- `duplex.llm.prefill`：prefill 写入 KV 的 token 数 / prefill 阶段耗时。
- `duplex.llm.decode`：decode 阶段新增 KV token 数 / decode 阶段耗时。
- `tts.prefill`：condition embedding 写入 TTS KV 的实际 token 数 / TTS prefill 阶段耗时。
- `tts.decode`：TTS 自回归采样的 `compute_tokens` / TTS decode 阶段耗时。
- `t2w.infer` / `t2w.write` 当前只展示耗时，不展示 token/s。

### 4. 默认报告产物

面向用户的默认报告已收窄为三类输出：

```text
omni-duplex-perf-report.md
figures/omni-duplex-pipeline.svg
figures/omni-duplex-gpu-utilization.svg
```

随后只保留：

1. 聚焦一个连续 SPEAK utterance 的 pipeline 图。
2. 核心阶段的平均耗时，以及 LLM prefill / LLM decode / TTS prefill / TTS decode 的 SPEAK token 数和 token 速度。
3. GPU 采样摘要和 GPU 利用率图。

默认不再生成或展示长 CSV、多余 SVG、SLA 细表、stage timing markdown 等面向调试的材料。需要深挖时再临时打开相关导出。

## 剩余注意点

当前 C++ 仍没有真实运行时 `utterance_id`，T2W window 也没有记录 `source_chunks/source_token_counts`。因此当多个 frame 的 `T2WOut` 被合并进同一个 T2W window 时，`t2w.infer` / `t2w.write` 的归因仍然使用第一个有效 `perf_chunk_index`。

这不影响当前 1s SLA 主报告，但如果后续要精确分析 TTS/T2W 跨 frame 流式缓存和 final flush 归属，仍需要在 C++ detail 中补充多来源 chunk 信息。
