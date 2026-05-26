# 双工测试线程流水线说明

## 结论

`tools/omni/test/test-duplex.cpp` 的双工测试表面上是每个输入 chunk 串行执行：

```text
stream_prefill(chunk N) -> stream_decode(chunk N) -> stream_prefill(chunk N+1) -> stream_decode(chunk N+1)
```

但 `ctx_omni->async = true` 后，真正运行时不是单线程顺序链路，而是由测试主线程、LLM prefill 线程、TTS 线程和 T2W 线程组成的流水线。

需要特别注意一点：当前代码里 **LLM 采样/解码不在 `llm_thread_func()` 线程里**。`llm_thread_func()` 主要负责异步 LLM prefill，也就是把 audio/vision embedding 写入 LLM KV cache。真正的 LLM decode token loop 在调用 `stream_decode()` 的测试主线程里执行。

## 线程和队列

### 测试主线程

入口是 `duplex_test_case()`。它对每个测试音频 chunk 做：

1. 设置 `perf_current_chunk_index = il`。
2. 调 `stream_prefill(ctx_omni, aud_fname, img_fname, il)`。
3. 调 `stream_decode(ctx_omni, "./")`。
4. 从 `text_queue` 里读本次 decode 的文本结果。

这个主线程负责同步执行：

- 读取当前 chunk 的音频/图片路径。
- 调用 audio encoder / vision encoder 生成 embedding。
- 把 embedding 放入 LLM 队列。
- 等待 LLM prefill 完成。
- 执行 LLM decode token loop。
- 每积累一段 LLM 输出，就把 `LLMOut` 放入 TTS 队列。

### LLM prefill 线程

线程函数是 `llm_thread_func()`，输入队列是：

```text
LLMThreadInfo::queue: omni_embeds*
```

它的作用是处理 `stream_prefill()` 送来的 audio/vision embedding：

1. 等待 `llm_thread_info->queue` 非空，或等待 `need_speek=true`。
2. 如果队列非空，取出所有 `omni_embeds`。
3. 对每个 item 执行 `<unit>`、`<image>` 等 marker token eval，以及 audio/vision embedding prefill。
4. 写入 LLM KV cache 后，等待 `stream_decode()` 设置 `need_speek=true`。
5. 当队列空且 `need_speek=true` 时，设置 `prefill_done=true`，通知 `stream_decode()` 可以开始采样。

所以这个线程的核心用途是：让 `stream_prefill()` 不直接阻塞在 LLM KV prefill 上，而是先把 embedding 入队，再由后台线程写 KV。

### TTS 线程

双工测试使用 `tts_thread_func_duplex()`。输入队列是：

```text
TTSThreadInfo::queue: LLMOut*
```

`stream_decode()` 每生成一段 LLM token，就构造 `LLMOut`：

- `text`：本段文本。
- `token_ids`：用于 TTS 的 LLM token。
- `hidden_states`：对应 token 的 hidden state。
- `llm_finish`：本次 decode 是否结束。
- `is_end_of_turn`：是否需要 flush TTS/T2W 状态。
- `perf_chunk_index`：性能日志归属的输入 chunk。

TTS 线程收到后做：

1. 等待 `tts_thread_info->queue` 有数据。
2. 从队列中取出 `LLMOut`，当前实现会尽量累计队列中已有的数据，遇到 end-of-turn 边界停止。
3. 过滤特殊 token。
4. 做 TTS condition 构造：`tts_emb_text()`、`tts_projector_semantic()`、L2 normalize、embedding merge。
5. 调 `generate_audio_tokens_local()` 生成 audio tokens。
6. 把 audio tokens 通过 `T2WOut` 推给 T2W 队列。

### T2W 线程

入口是 `t2w_thread_func()`，根据配置选择：

- `t2w_thread_func_cpp()`：C++ Token2Wav。
- `t2w_thread_func_python()`：Python Token2Wav 服务。

输入队列是：

```text
T2WThreadInfo::queue: T2WOut*
```

它负责把 TTS audio tokens 变成音频：

1. 等待 T2W 队列有数据。
2. 把收到的 audio tokens 累积到 `token_buffer`。
3. 按 Token2Wav 滑窗处理，C++ 路径窗口大小是 `25 + 3`，也就是 25 个主 token 加 3 个 lookahead。
4. 调 Token2Wav 推理。
5. 写出 WAV/PCM 文件。

## 双工流水线顺序

对一个普通输入 chunk，逻辑顺序可以理解为：

```text
测试主线程:
  audio/vision encode
  -> enqueue LLM prefill
  -> stream_decode 等待 prefill_done
  -> LLM decode token loop
  -> 每 10 个有效 TTS token enqueue TTS

LLM prefill 线程:
  wait LLM queue
  -> LLM KV prefill
  -> 通知 prefill_done

TTS 线程:
  wait TTS queue
  -> LLM token/hidden -> TTS condition
  -> TTS audio token decode
  -> enqueue T2W

T2W 线程:
  wait T2W queue
  -> Token2Wav infer
  -> write wav
```

## 谁和谁能并行

### 不能并行的部分

同一个输入 chunk 内，下面这些关系是串行的：

- `audio.encode` / `vision.encode` 必须先完成，之后才能把 embedding 入 LLM 队列。
- `llm.prefill` 必须先完成，`stream_decode()` 才会开始 LLM 采样。
- 同一个 `llama_context` 上的 prefill 和 decode 不能真正同时跑；代码用等待和 `llama_mtx` 保证顺序。

所以同一个 chunk 的主链路不是：

```text
encode || llm.prefill || llm.decode
```

而是：

```text
encode -> llm.prefill -> llm.decode
```

### 能并行的部分

真正的流水线并行发生在生成侧和下一轮输入侧：

- `stream_decode()` 的 LLM token loop 可以一边继续采样后续 token，一边让 TTS 线程处理已经入队的前一段 token。
- TTS 线程生成 audio tokens 后，T2W 线程可以一边把前一批 audio tokens 转音频，一边等待/接收 TTS 的后续 audio tokens。
- `stream_decode()` 返回后，测试主线程会进入下一个输入 chunk 的 `stream_prefill()`；此时上一个 chunk 的 TTS/T2W 可能还没完全处理完，所以“下一秒音频的 encode/prefill/decode”和“上一段回复的 TTS/T2W”可以并行。
- 如果 TTS/T2W 队列还有积压，测试结束后还会等待 `omni_tts_queues_empty()` 连续空闲一段时间，再停线程。

### 会限制并行的地方

TTS 队列容量当前是 `TTSThreadInfo(1)`。这意味着 `stream_decode()` 往 TTS 队列推 `LLMOut` 时，如果 TTS 线程还没取走上一段，`tts.enqueue_from_llm` 会阻塞。

因此 LLM decode 和 TTS 的并行不是无限制的。它更像一个带背压的小流水线：

```text
LLM decode chunklet -> TTS queue(size=1) -> TTS infer -> T2W queue -> T2W infer
```

如果 TTS 比 LLM decode 慢，LLM decode 会在入队 TTS 时被卡住。

## LLM 固定数量 token 交给 TTS 的逻辑

`stream_decode()` 里有：

```text
step_size = 10
```

内层 token loop 的条件是收集够 `step_size` 个“有效 TTS token”，或者遇到结束条件：

- 每次调用 `llama_loop_with_hidden_and_token()` 采样一个 LLM token，并拿到 hidden state。
- 只有 `is_valid_tts_token(sampled_token)` 为 true 的 token 才进入 `chunk_token_ids` 和 `chunk_hidden_states`。
- `jl` 只统计有效 TTS token，不统计 `<think>`、`<|speak|>`、`<|listen|>`、`<|chunk_eos|>` 等特殊 token。
- 收集到 10 个有效 TTS token 后，构造一个 `LLMOut` 推给 TTS 线程。
- 如果遇到 `chunk_eos`、`listen`、`turn_eos`、`max_new_speak_tokens_per_chunk` 等结束/切分条件，也会提前结束当前段并推给 TTS。

这个 10-token 小段不是输入音频 chunk，而是 LLM 输出内部的一个 TTS chunklet。它的作用是降低首响延迟：不必等整段回答全部生成完，TTS 可以先处理前 10 个有效 token。

## 阶段归属建议

按照“不要拆太碎，但排除等待”的口径，可以这样归类：

| 代码行为 | 建议阶段 | 类型 |
| --- | --- | --- |
| 测试入口调用 `stream_prefill()` | `api.chunk_prefill` | API 包络 |
| audio/vision encoder 生成 embedding | `audio.encode` / `vision.encode` | Compute |
| 等 LLM queue 空间并 push `omni_embeds` | `queue.llm.enqueue` | Queue/Wait |
| LLM 线程执行 marker eval + embedding prefill | `llm.prefill` | Compute |
| `stream_decode()` 等 `prefill_done` | `wait.llm_prefill_done` | Wait |
| LLM token loop，含每步采样和写 KV | `llm.decode` | Compute |
| 达到 chunk limit 后注入 `chunk_eos`、每段后注入 `</unit>` | 可并入 `llm.decode` | Compute |
| 构造 `LLMOut` 前的文本清洗、特殊 token 处理 | `llm.decode.postprocess` 或并入 `llm.decode` detail | CPU/Postprocess |
| 等 TTS queue 空间并 push `LLMOut` | `queue.tts.enqueue` | Queue/Wait |
| TTS 线程等 `LLMOut` | `queue.tts.wait_data` | Wait |
| token/hidden 到 merged TTS condition，再到 audio tokens | `tts.infer` | Compute |
| 等 T2W queue 空间并 push `T2WOut` | `queue.t2w.enqueue` | Queue/Wait |
| T2W 线程等 audio tokens | `queue.t2w.wait_data` | Wait |
| Token2Wav 模型推理 | `t2w.infer` | Compute |
| WAV/PCM 落盘 | `t2w.write` | IO |

当前 profiling 已按这个口径改成：

- API 包络使用 `api.chunk_prefill`、`api.chunk_decode`、`api.stream_prefill`、`api.stream_decode`。
- LLM compute 使用 `llm.prefill` 和 `llm.decode`；完整 decode loop 包络保留为 `api.llm_decode_loop`。
- LLM 到 TTS 的背压拆成 `queue.tts.wait_space` 和 `queue.tts.enqueue`。
- TTS compute 使用 `tts.infer`，T2W compute 使用 `t2w.infer`，C++ WAV 落盘使用 `t2w.write`。

## 性能分析时的读法

建议把一轮双工测试拆成三条时间线看：

1. 输入侧主链路：`audio/vision encode -> queue.llm -> llm.prefill -> wait.llm_prefill_done -> llm.decode`。
2. 输出侧语音链路：`queue.tts -> tts.infer -> queue.t2w -> t2w.infer -> t2w.write`。
3. 跨 chunk 并行：chunk N 的 TTS/T2W 可能和 chunk N+1 的 encode/prefill/decode 同时发生。

如果只问“一个完整周期”，可以把它定义成：

```text
输入 1 秒音频
-> audio/vision encode
-> LLM prefill
-> LLM decode 生成 token
-> 每 10 个有效 token 交给 TTS
-> TTS 生成 audio tokens
-> T2W 生成 wav/pcm
```

但做性能统计时不要把这些阶段简单相加，因为其中一部分会并行，另一部分是父子包络关系。正确做法是用时间线看重叠，用 compute/wait/queue 分类看瓶颈。

运行命令
 用这个完整命令即可测 Q4_K_M + 全部 36 个 duplex omni chunk：

  cd /cache/hanqingzhe/llama.cpp-omni
  ./build/bin/llama-omni-test-duplex \
    -m /cache/hanqingzhe/o45-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
    --omni \
    --test /cache/hanqingzhe/llama.cpp-omni/tools/omni/assets/test_case/duplex_omni_test_case/duplex_omni_test_case_ 36 \
    --ref-audio /cache/hanqingzhe/llama.cpp-omni/tools/omni/assets/default_ref_audio/default_ref_audio.wav \
    -ngl 99 \
    -c 4096 \
    -o /cache/hanqingzhe/llama.cpp-omni/tools/omni/output/duplex_q4km_full \
    2>&1 | tee duplex_q4km_full.log

  跑完分析：

  python scripts/analyze_duplex_perf.py \
    --log duplex_q4km_full.log \
    --out-dir docs/development/perf-duplex-q4km-full
