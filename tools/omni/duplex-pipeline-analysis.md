# Duplex 推理流水线分析

本文基于 `feat/duplex-profiling-clean` 分支，从 `tools/omni/test/test-duplex.cpp` 的测试入口出发，梳理一次 Duplex Omni 推理从输入帧到文本决策、TTS、Token2Wav 落盘的阶段、线程、数据传递方式和流水线边界。

这份文档特别关注几个容易读错的点：

- `stream_prefill(index > 0)` 在 duplex async 路径下只是把 frame 提交到底层 encode pipeline，并不代表 vision/audio encode 完成，也不代表 LLM prefill 已写入 KV。
- Session 层的 `decode_pending` 只是“可以发起 decode 请求”的帧队列，不是“prefill 已完成”的队列。
- `stream_decode()` / `duplex_decode()` 阻塞到 LLM 阶段完成，但不等待 TTS/T2W 完成。
- TTS 和 T2W 是后续异步阶段，分别消费 `LLMOut` 和 `T2WOut`，最终由 T2W 写 wav 文件。
- `force_listen` 是控制层强制返回 LISTEN 状态，不会把 `<|listen|>` token 写进 LLM KV cache。

## 1. 测试入口看到的外部 API

`test-duplex.cpp` 本身不直接调 VPM/APM/LLM/TTS。它只做三件事：

1. 解析模型路径、初始化 `omni_context`，并用 `duplex_mode=true` 调 `omni_init()`。
2. 在 `duplex_test_case()` 里调用高层 Duplex Session API：
   - `omni_duplex_session_begin()`
   - `omni_duplex_push_frame()`
   - `omni_duplex_wait_next_frame()`
   - `omni_duplex_session_end()`
   - `omni_duplex_drain_tts_audio()`
3. 打印每帧的 `prefill_submit / decode / e2e` 时间、`speak/listen` 决策和文本。

测试侧有两个线程：

| 线程 | 代码位置 | 职责 |
| --- | --- | --- |
| 测试主线程 | `duplex_test_case()` 的 for-loop | 顺序调用 `wait_next_frame()` 取结果并统计 |
| 测试 producer 线程 | `std::thread producer` | 按 `stream_interval_ms` 调 `push_frame()`，模拟流式输入节奏 |

测试侧和推理内部的边界很清晰：`push_frame()` 只提交 `OmniDuplexFrame`，`wait_next_frame()` 只取 `OmniDuplexFrameResult`。测试不需要知道底层 `stream_prefill()` / `stream_decode()` 的 index 语义，也不直接管理任何推理线程。

## 2. 总体阶段

按一次会话来看，当前 Duplex 推理过程可以分成 9 个阶段：

| 阶段 | 入口/线程 | 主要输入 | 主要输出 | 是否异步 |
| --- | --- | --- | --- | --- |
| 0. 初始化 | `main()` / `omni_init()` | 模型路径、参数 | `omni_context`、模型、TTS/T2W 队列 | 同步 |
| 1. 会话初始化 | `omni_duplex_session_begin()` | ref audio / debug dir | system prompt 写入 KV，启动常驻线程 | 同步启动，后续异步 |
| 2. 帧提交 | 测试 producer -> `push_frame()` | `OmniDuplexFrame` | `pending_frames` 队列项 | 异步，队列满时阻塞 |
| 3. 帧级 prefill submit | `DuplexSession::prefill_worker` | `pending_frames` | `decode_pending`、`encoder_queue` | 异步 |
| 4. VPM/APM 编码 | `DuplexPipeline::encoder_thread` | `DuplexEncodeReq` | `DuplexPrefillPacket` | 异步；帧间单线程，帧内 VPM/APM 可并行 |
| 5. LLM prefill | `DuplexPipeline::llm_thread` | `DuplexPrefillPacket` | LLM KV cache 更新 | 同一 LLM 线程内串行 |
| 6. LLM decode | `DuplexPipeline::llm_thread` | `DuplexDecodeReq`、KV cache | `text_queue`、`LLMOut`、listen/speak 状态 | 同一 LLM 线程内串行 |
| 7. TTS semantic | `tts_thread_func_duplex` | `LLMOut` | `T2WOut` audio tokens | 异步，受 TTS 队列背压 |
| 8. Token2Wav | `t2w_thread_func` | `T2WOut` audio tokens | `tts_wav/wav_*.wav` | 异步 |
| 9. 结果/收尾 | `wait_next_frame()` / `session_end()` / `drain_tts_audio()` | `done_results`、TTS/T2W 队列 | 帧结果、音频落盘完成 | 阻塞等待 |

核心路径可以简化成：

```text
test producer
  -> omni_duplex_push_frame()
  -> DuplexSession.pending_frames
  -> prefill_worker
  -> stream_prefill(index=frame_id)
  -> DuplexPipeline.encoder_queue
  -> encoder_thread: VPM/APM encode
  -> DuplexPipeline.prefill_queue
  -> decode_worker
  -> stream_decode(round_idx=frame_id)
  -> duplex_decode()
  -> llm_thread: consume 1 prefill packet + decode
  -> text_queue / tts_thread_info.queue
  -> done_results / tts_thread -> t2w_thread -> wav files
```

如果按更理想、语义更干净的模型看，它其实应该是五段：

```text
raw frame queue
  -> encode stage: image/audio -> embedding
  -> llm stage: one encoded frame at a time, prefill + decode
  -> tts stage: LLM hidden/text -> audio tokens
  -> t2w stage: audio tokens -> wav/audio stream
```

当前代码已经有这五段的影子，但外面又套了一层 `DuplexSession` 的 `prefill_worker/decode_worker`。这层名字容易误导：它不是在 session 层真正做 LLM prefill/decode，而是在做“提交 encode 请求”和“请求 LLM 线程处理下一帧”。

## 3. 线程模型

当前 Duplex 路径实际涉及这些线程/执行体：

| 线程/执行体 | 创建位置 | 生命周期 | 访问的关键状态 |
| --- | --- | --- | --- |
| 测试主线程 | `test-duplex.cpp` | 进程主生命周期 | `wait_next_frame()`、统计结果 |
| 测试 producer | `test-duplex.cpp` | 单次 `duplex_test_case()` | `push_frame()` |
| `prefill_worker` | `omni_duplex_session_begin()` | 单次 Duplex session | `pending_frames` -> `decode_pending` |
| `decode_worker` | `omni_duplex_session_begin()` | 单次 Duplex session | `decode_pending` -> `stream_decode()` -> `done_results` |
| `encoder_thread` | `stream_prefill(index=0)` 后的 `duplex_start_threads()` | `omni_context` 生命周期 | `encoder_queue`、VPM/APM context |
| `encoder_thread` 内的 `std::async` APM 任务 | 每个同时含图像和音频的帧 | 单帧编码期间 | `ctx_audio`，生成 audio embed |
| `llm_thread` | `duplex_start_threads()` | `omni_context` 生命周期 | `ctx_llama`、`n_past`、KV cache、`text_queue`、TTS queue |
| `tts_thread` | `stream_prefill(index=0)` 或兜底 `stream_decode()` | `omni_context` 生命周期 | `tts_thread_info.queue`、`ctx_tts_llama`、TTS 状态 |
| `t2w_thread` | `stream_prefill(index=0)` 或兜底 `stream_decode()` | `omni_context` 生命周期 | `t2w_thread_info.queue`、Token2Wav session |
| Python T2W 子进程（可选） | `omni_init()` 中 Python T2W 初始化 | `omni_context` 生命周期 | 通过 stdin/stdout 服务协议处理 token2wav |

有两层流水线需要区分：

1. **高层 Session 流水线**：`push_frame()` -> `prefill_worker` -> `decode_worker` -> `wait_next_frame()`。它面向外部调用方，保证帧结果 FIFO。
2. **底层 Duplex Pipeline**：`encoder_thread` -> `llm_thread` -> `tts_thread` -> `t2w_thread`。它面向内部推理，拆开编码、LLM、语音合成。

## 4. 数据结构与传递路径

### 4.1 外部帧输入

测试构造 `OmniDuplexFrame`：

- `aud_fname`：当前 chunk 的 wav 路径。
- `img_fname`：当前 chunk 的 jpg 路径，可为空。
- `max_slice_nums`：视觉 slice 配置。
- `user_seq`：测试侧序号，原样回传。

`omni_duplex_push_frame()` 把它包装成 `OmniDuplexPendingFrame`：

- 增加内部递增 `frame_id`。
- 记录 `t_push`。
- 入队 `DuplexSession::pending_frames`，队列上限 `PENDING_MAX = 64`。

### 4.2 Session 内部调度

`prefill_worker` 从 `pending_frames` 出队后调用：

```text
stream_prefill(ctx, aud_fname, img_fname, index=frame_id, max_slice_nums)
```

在当前 Duplex 条件下，`stream_prefill(index > 0)` 会路由到 `duplex_prefill()`，并不会在 `prefill_worker` 里真正完成编码和 LLM prefill。它只是把 `DuplexEncodeReq` 入队到 `DuplexPipeline::encoder_queue`，随后 `prefill_worker` 立刻把 `OmniDuplexInflightFrame` 放入 `decode_pending`：

- `frame_id`
- `user_seq`
- `t_push`
- `t_prefilled`：更准确地说是“prefill 已提交到底层 pipeline 的时间”，不是 LLM KV 已经写完。
- `prefill_failed`

因此这几个名字按真实语义更接近：

| 当前名字 | 更准确的语义 |
| --- | --- |
| `prefill_worker` | `submit_encode_worker` |
| `t_prefilled` | `t_prefill_submitted` / `t_encode_submitted` |
| `decode_pending` | 已提交 encode、等待发起 LLM decode 请求的 frame |
| `decode_worker` | 请求 LLM 处理并同步等待结果的 worker |

`decode_worker` 从 `decode_pending` 出队后调用：

```text
stream_decode(ctx, debug_dir, round_idx=frame_id)
```

在 Duplex 条件下，`stream_decode()` 路由到 `duplex_decode()`，阻塞等待底层 `llm_thread` 完成“对应帧的 prefill + decode”。

### 4.3 编码阶段

`duplex_prefill()` 入队的数据是 `DuplexEncodeReq`：

- `aud_fname`
- `img_fname`
- `index`
- `max_slice_nums`

`encoder_thread` 消费后生成 `DuplexPrefillPacket`：

- `vision_embed`：二维向量，`[0]` 是 overview，后续是 slices。
- `audio_embed`：音频 embedding。
- `index`

如果同时有图像和音频，当前实现中 VPM 在 `encoder_thread` 当前线程跑，APM 通过 `std::async(std::launch::async, ...)` 跑，二者并行；如果只有一种模态则只跑对应编码。

编码完成后，`DuplexPrefillPacket` 入队到 `DuplexPipeline::prefill_queue`，队列上限 `PREFILL_QUEUE_CAP = 32`。

### 4.4 LLM prefill/decode 阶段

`duplex_decode()` 提交 `DuplexDecodeReq`：

- `debug_dir`
- `round_idx`
- `done`
- `ok`

它把请求设置到 `DuplexPipeline::pending_decode`，然后等待 `decode_done_cv`。

`pending_decode` 不是 queue，而是单槽指针：

```text
DuplexDecodeReq * pending_decode
```

单槽的意义是强制底层 LLM 线程同一时间只处理一个 decode 请求。`duplex_decode()` 会先等这个槽为空，再把本轮 `req` 放进去；`llm_thread` 看到后把它取走并清空槽位，随后完成对应的 prefill+decode。break 时如果槽里还有请求，则把它标记失败并唤醒等待方。

`llm_thread` 的关键约束是 **严格 1:1 prefill/decode**：

- 如果 `in_flight_prefill > 0`，先等 `prefill_queue` 非空并消费一个 `DuplexPrefillPacket`。
- 把该 packet 写入 `ctx_llama` KV cache。
- 立刻执行本轮 decode。
- 完成后标记 `DuplexDecodeReq.done = true`。

这是当前正确性的核心。`ctx_llama`、`n_past` 和 KV cache 都由 `llm_thread` 独占串行访问，避免后续 chunk 的 prefill 污染当前 chunk 的 decode 上下文。

LLM prefill 有两条路径：

- 优先 `duplex_do_prefill_one_fused()`：把固定 special token embedding、vision embedding、audio embedding 拼成一块 fused buffer，一次 `prefill_with_emb()` 写入 KV，减少多次小 `llama_decode` 的开销。
- 如果 special token embedding cache 初始化失败，则回退到 `duplex_do_prefill_one()`：按 `<unit><image>`、vision、`</image>`、slice、audio 等多段写入。

LLM decode 期间：

- 采样 token，并按 `<|speak|>` / `<|listen|>` / `<|chunk_eos|>` / `<|turn_eos|>` 等特殊 token 更新状态。
- 文本片段进入 `ctx_omni->text_queue`，供 server/SSE 或高层 session 读取。
- 可用于 TTS 的 token id 和 hidden states 组成 `LLMOut`，进入 `tts_thread_info->queue`。

`duplex_do_decode()` 内部有两层采样循环：

```text
外层 for:
  形成一个 LLM chunk
  把 chunk 推给 text_queue / TTS queue

内层 while:
  每次调用 sample_with_hidden_and_token 采 1 个 token
  直到收集 step_size 个有效 TTS token，或遇到 end token / break / chunk limit
```

`llama_loop_with_hidden_and_token()` 这个名字也容易误导。它自己没有 loop，只是调用一次 `sample_with_hidden_and_token()`，真正的循环在外层 `while`。

当采到 EOS / LISTEN / TURN_EOS 这类结束 token 时，内层会停，外层不会再进入下一轮采样；外层剩余部分只是做收尾：清理 response、推 text_queue、推最后一个 `LLMOut`，然后 `llm_finish` 触发 break。

### 4.5 帧结果返回

`decode_worker` 在 `stream_decode()` 返回后组装 `OmniDuplexFrameResult`：

- `user_seq`
- `frame_id`
- `ok`
- `is_speak = !ended_with_listen`
- `text`：从 `text_queue` 取出并过滤 `__IS_LISTEN__` / `__END_OF_TURN__`
- `n_past_after`
- `ms_prefill_submit`
- `ms_decode`
- `ms_total`

然后结果入队 `DuplexSession::done_results`，测试主线程通过 `omni_duplex_wait_next_frame()` FIFO 取出。

注意：这里的帧结果表示 LLM 决策和文本已经完成，不表示对应音频文件已经全部落盘。测试在所有帧结束后还会调用 `omni_duplex_drain_tts_audio()` 等 TTS/T2W 队列空闲。

换句话说：

```text
decode_req.done = true
```

只代表 `duplex_do_decode()` 返回，也就是 LLM 阶段完成，且需要的话已经把 `LLMOut` 提交给 TTS。它不代表 TTS 已生成完 audio tokens，也不代表 T2W 已写完 wav。

### 4.6 TTS 与 Token2Wav

LLM decode 推给 TTS 的数据结构是 `LLMOut`：

- `text`
- `n_past`
- `llm_finish`
- `debug_dir`
- `token_ids`
- `hidden_states`
- `n_embd`
- `is_end_of_turn`
- `perf_chunk_index`

`tts_thread_func_duplex` 消费 `LLMOut` 后：

1. 累积同一 TTS chunk 内的 token ids、hidden states 和文本。
2. 过滤特殊 token。
3. 通过 TTS 的 `emb_text` 和 `projector_semantic` 把 LLM hidden states 映射到 TTS embedding 空间。
4. 调 `generate_audio_tokens_local()` 生成 audio tokens。
5. 把 audio tokens 以及 `is_final` / `is_chunk_end` / `round_idx` / `perf_chunk_index` 封装成 `T2WOut`，推给 `t2w_thread_info->queue`。

这里的资源所有权是裸指针转移：

```text
LLM thread:
  new LLMOut
  push to tts_thread_info->queue

TTS thread:
  pop LLMOut
  拷贝/累积 text、token_ids、hidden_states
  delete LLMOut
  new T2WOut
  push to t2w_thread_info->queue

T2W thread:
  pop T2WOut
  拷贝/合并 audio_tokens
  delete T2WOut
  token2wav -> wav file
```

所以 TTS 不负责写 wav。TTS 负责把 LLM 的文本/hidden state 转成 audio tokens；T2W 负责把 audio tokens 变成 waveform 并落盘。

`t2w_thread_func()` 根据配置选择 Python 或 C++ 实现：

- Python：`t2w_thread_func_python()`
- C++：`t2w_thread_func_cpp()`

两者都使用 25 个主 token + 3 个 lookahead token 的滑窗策略。双工模式下通常只在 `is_final` 时 flush/reset，`is_chunk_end` 更多用于保持跨 chunk 的连续性，避免频繁切断 Token2Wav 状态产生播放 gap。

双工模式下，T2W 的输出目录通常是：

```text
{base_output_dir}/tts_wav/wav_N.wav
```

最后一段完成时会写：

```text
{base_output_dir}/tts_wav/generation_done.flag
```

测试或上层如果要等语音真正落盘，必须等 TTS/T2W 队列，而不是只等 `wait_next_frame()`。

## 5. 流水线边界

当前代码里比较明确的边界如下：

| 边界 | 数据结构 | 同步方式 | 背压 |
| --- | --- | --- | --- |
| 测试 producer -> Session | `OmniDuplexPendingFrame` | `pending_mtx` + `pending_cv` | `PENDING_MAX = 64` |
| Session prefill -> Session decode | `OmniDuplexInflightFrame` | `decode_mtx` + `decode_cv` | 无显式 cap，按 prefill_worker FIFO 推进 |
| Session -> 编码 pipeline | `DuplexEncodeReq` | `encoder_mtx` + `encoder_cv` | `ENCODER_QUEUE_CAP = 16` |
| 编码 -> LLM prefill | `DuplexPrefillPacket` | `llm_mtx` + `llm_cv` | `PREFILL_QUEUE_CAP = 32` |
| decode_worker -> LLM thread | `DuplexDecodeReq* pending_decode` | `llm_mtx` + `llm_cv` + `decode_done_cv` | 单槽，只允许一个 pending decode |
| LLM -> 文本消费者 | `text_queue` | `text_mtx` + `text_cv` | 无显式 cap |
| LLM -> TTS | `LLMOut` | `TTSThreadInfo::mtx/cv` | `MAX_QUEUE_SIZE = 1` |
| TTS -> Token2Wav | `T2WOut` | `T2WThreadInfo::mtx/cv` | 配置了 `MAX_QUEUE_SIZE = 25`，但多处入队没有统一执行等待，实际不能完全视为硬上限 |
| Token2Wav -> 文件系统 | wav 文件 | 文件写入 | 文件系统/后端速度 |

边界上的语义：

- `pending_frames` 是外部业务输入边界。
- `encoder_queue` 是“帧路径 -> 编码模块”的边界。
- `prefill_queue` 是“编码模块 -> LLM KV 写入”的边界。
- `pending_decode` 是“帧级 decode 请求 -> LLM 独占线程”的同步点，单槽保证不会并发 decode。
- `LLMOut` 是“文本模型 -> 语音模型”的边界，携带 token id、hidden state 和 `is_end_of_turn`，避免 TTS 线程依赖易竞态的全局 turn 状态。
- `T2WOut` 是“语音 token -> waveform”的边界，显式携带 `round_idx` / `perf_chunk_index`，降低输出目录和性能归属的竞态。

## 6. 控制状态：LISTEN、break 和滑动窗口

### 6.1 `force_listen`

会话开局有一段强制 LISTEN 逻辑：

```text
if force_listen_used < force_listen_count:
  force_listen_used++
  slide_last_was_listen = true
  ended_with_listen = true
  text_queue.push("__IS_LISTEN__")
  return true
```

这段不会调用 `eval_tokens(... special_token_listen ...)`，所以 `<|listen|>` 并没有被写进 LLM KV cache。它只是控制层告诉上层：“这一帧按 LISTEN 处理，不要说话”。这也意味着模型内部并不知道自己刚输出过 `<|listen|>`，属于硬规则，不是模型真实采样结果。

这段还会提前 return，因此正常 decode 尾部的一些控制 token 追加逻辑也不会执行。它的目的主要是避免会话开头几帧因为浏览器音频轨瞬态噪声导致模型抢答。

### 6.2 正常 LISTEN

正常采样到 `<|listen|>` 时，语义不同：

- 该 token 是模型真实采出来的，已经随着采样过程进入 KV。
- 代码会设置 `ended_with_listen = true`。
- 如果之前处于 SPEAK 状态，`is_end_of_turn` 会传给 TTS，让 TTS/T2W flush 尾音。
- `text_queue` 会收到 `__IS_LISTEN__`，供上层识别“模型切回听”。

### 6.3 `break_event`

`break_event` 主要由 server 的 break/interrupt API 设置，用于“打断当前生成但保持会话活跃”。

触发后：

- `llm_thread` 会丢弃当前 `prefill_queue` 和 `pending_decode`。
- decode loop 会停止采样。
- TTS 线程会清空 `tts_thread_info->queue`，重置 TTS KV/cache 状态。
- T2W 线程会清空 `t2w_thread_info->queue`，重置 token buffer，并在 Python 路径下 reset T2W cache。

本地 `test-duplex.cpp` 的 Ctrl-C 只是设置测试自己的 `g_is_interrupted`，不是 `ctx_omni->break_event`。

### 6.4 智能滑动窗口

双工对话会不断向 LLM KV cache 追加：

```text
system prompt
frame 1 prefill
model response/listen
frame 2 prefill
model response/listen
...
```

当接近 `n_ctx` 时，不能简单清空，也不能从中间随机删除。当前代码有两套相关机制：

1. `round_start_positions`：在 `ended_with_listen` 时记录当前 `n_past`，作为一轮结束/下一轮开始的边界。
2. `unit_history/current_turn_id`：记录每个输入 unit 属于哪个 turn，`mode=turn` 时优先按完整 turn 删除。

滑窗触发时会尽量：

- 保护 system prompt / `n_keep`。
- 不在 generating、TTS busy、mid speak 时滑，除非快撞上 `n_ctx` 硬限制。
- 优先丢最早的完整 round/turn。
- 如果没有完整 turn 可丢，才退化为按 unit 或 tail window 删除。
- 删除后用 `llama_memory_seq_add(..., -n_discard)` 把后续 KV position 前移，保证后续 decode 的 position 连续。

所以这里的“智能”不是模型层的智能，而是 KV cache 管理策略：尽量保留系统提示和最近完整上下文，避免把一轮对话从中间砍断。

## 7. 是否做到各阶段解耦

结论：**外部调用、帧级调度、编码、TTS、Token2Wav 已经做到较好的阶段化解耦；LLM prefill 和 decode 由于共享同一个 `ctx_llama` KV cache，当前是有意保持串行强耦合；LLM 到 TTS 虽然通过队列解耦，但受小队列和共享状态影响，属于部分解耦。**

### 已经解耦较好的部分

1. **测试/业务层与推理内部解耦**

   测试只依赖 `omni_duplex_*` API，不关心内部线程、队列和 `stream_prefill()` / `stream_decode()` 细节。同一套 API 可以复用到 server/cli。

2. **帧提交与实际推理解耦**

   `push_frame()` 只把 frame 放入 `pending_frames`，除了队列满以外不会等待 LLM 完成。测试里的 producer 可以按真实音视频节奏继续提交。

3. **编码与 LLM 解耦**

   `encoder_thread` 和 `llm_thread` 之间通过 `DuplexPrefillPacket` 队列隔开。编码可以提前处理后续帧，尽量把 VPM/APM 时间隐藏在 LLM 工作流后面。

4. **VPM 与 APM 帧内并行**

   同一帧同时有 image/audio 时，APM 通过 `std::async` 和 VPM 并行。二者分别使用 `ctx_audio` / `ctx_vision`，数据在 `DuplexPrefillPacket` 汇合。

5. **LLM 决策与音频落盘解耦**

   `wait_next_frame()` 返回的是 LLM 文本/决策结果；TTS 和 T2W 继续在后台生成音频。测试最后单独 `drain_tts_audio()`，说明 LLM 帧结果和音频落盘不是同一阻塞边界。

### 仍然耦合或半耦合的部分

1. **LLM prefill 与 decode 强耦合**

   这是设计上必要的。`ctx_llama`、`n_past` 和 KV cache 是单实例顺序状态。当前 `llm_thread` 强制“消费最多一个 prefill packet -> decode 一次”，保证每帧 decode 看到的上下文正好对应当前帧，避免后续帧提前写入 KV。

2. **`pending_decode` 是单槽同步，不是完全流水化 decode**

   `decode_worker` 可以提前提交 decode 请求，但底层 `pending_decode` 只有一个槽，`duplex_decode()` 会等 `llm_thread` 完成。这保证正确性，但也意味着 LLM 阶段本身没有多帧并发。

3. **LLM -> TTS 有队列，但背压很强**

   `TTSThreadInfo` 的 `MAX_QUEUE_SIZE = 1`。如果 TTS 慢，`llm_thread` 在推 `LLMOut` 时可能阻塞。因此 LLM 与 TTS 在慢 TTS 场景下会重新耦合，吞吐上不完全独立。

4. **仍有共享全局状态跨线程读取/写入**

   例如 `ended_with_listen`、`speek_done`、`break_event`、`tts_n_past_accumulated`、`tts_condition_saved`、`wav_turn_base` 等状态在多个阶段之间传递含义。代码已经把 `is_end_of_turn`、`round_idx`、`perf_chunk_index` 放进队列消息里，降低竞态，但还不是完全消息化。

5. **TTS/T2W 与文件系统耦合**

   T2W 的最终边界是 wav 文件落盘。输出目录、文件编号和 flush/reset 语义仍依赖 `ctx_omni` 中的状态。对测试来说这没问题，但如果未来需要实时音频流，文件边界还需要再抽象。

## 8. 当前设计的主要语义问题

从代码阅读角度看，当前最大问题不是 pipeline 完全不可用，而是几个名字和真实阶段不匹配：

| 代码名字/指标 | 容易误解成 | 实际含义 |
| --- | --- | --- |
| `prefill_worker` | 做完 LLM prefill 的 worker | 只是把 frame 提交到底层 encode queue |
| `t_prefilled` | LLM KV 已写入时间 | prefill/encode 请求提交时间 |
| `decode_pending` | prefill 完成后待 decode | 已提交 encode，待触发底层 LLM 处理 |
| `decode_req.done` | 端到端语音完成 | LLM `duplex_do_decode()` 返回 |
| `wait_next_frame()` | 等整帧音频完成 | 等 LLM 文本/决策完成 |
| `force_listen` | 模型采样出 `<|listen|>` | 控制层直接返回 LISTEN，不写 KV |

如果未来重构，建议把外层 Session 命名改成更贴近真实职责：

```text
pending_frames
  -> submit_worker
     stream_prefill(...)  // submit encode request only
  -> submitted_frames
     stream_decode(...)   // request llm stage and wait
  -> done_results
```

更彻底的方向是让内部阶段直接表达为：

```text
raw_frame_queue
  -> encoded_frame_queue
  -> llm_worker(single owner of ctx_llama)
  -> tts_queue
  -> t2w_queue
```

这样读者不会以为 session 层真的完成了 prefill/decode。

## 9. Profiling 视角

这个分支的 profiling 主要覆盖以下点：

- 测试打印每帧：
  - `ms_prefill_submit`：`push_frame()` 到 `prefill_worker` 完成提交的时间。注意它不是编码完成时间，也不是 LLM prefill 完成时间。
  - `ms_decode`：`decode_worker` 内部调用 `stream_decode()` 的阻塞时间，包含底层等待对应编码 packet、LLM prefill、LLM decode。
  - `ms_total`：`push_frame()` 到本帧 result 出队的端到端时间。
- `encoder_thread` 打印：
  - `VPM`
  - `APM`
  - `wall`
  - `parallel_savings`
- `llm_thread` 打印：
  - `llm prefill` / `llm prefill (fused)`
  - `llm decode`
- perf mark 覆盖：
  - `api.duplex.frame_total`
  - `queue.tts.wait_data`
  - `tts.infer`
  - `queue.t2w.wait_data`
  - `t2w.infer`
  - `t2w.write`

因此分析性能时建议把 `ms_prefill_submit` 理解为 API 调度延迟，把 `[prof] encoder` 看作实际编码耗时，把 `ms_decode` / `[prof] llm prefill/decode` 看作 LLM 阶段耗时，把 TTS/T2W 的 perf mark 看作语音后台延迟。不要把 `ms_prefill_submit` 当作完整 prefill 成本。

## 10. 一句话总结

当前 Duplex pipeline 已经从“外部一帧一调”拆成了 **Session 调度层 + Encoder/LLM pipeline + TTS/T2W 后台语音层**。它在 API、编码、TTS、T2W 上已经有明确队列边界；LLM prefill/decode 仍然串行，是为了维护单个 KV cache 的正确性；LLM 到 TTS 的解耦存在但受小队列和共享状态限制，属于可继续演进的半解耦边界。

真正的端到端完成语义要分三层看：

```text
wait_next_frame() 返回
  = LLM 阶段完成，文本/决策可用

tts_thread 处理完成
  = audio tokens 已生成并交给 T2W

t2w_thread 处理完成
  = wav 文件已落盘
```

当前最需要警惕的是命名误导：session 层的 `prefill/decode` 不是模型真实阶段边界，真实的 LLM prefill 和 decode 都发生在底层 `llm_thread` 中。
