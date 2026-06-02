# Duplex 性能阶段划分复核报告

## 结论

当前分支的 `[DUPLEX_PERF]` 打点已经能把双工链路的主要时间线串起来，但它还不是严格的“计算阶段”划分。现在的 stage 里混有三类东西：

- API 包络：例如 `stream_prefill`、`stream_decode`、`test.chunk_*_api`，适合看端到端调用耗时，不适合当成单一计算阶段。
- 等待/队列：例如 `decode.wait_llm_prefill`、`prefill.enqueue_llm`、`tts.enqueue_from_llm`，这些时间主要是同步、排队、等待队列空间。
- 真实计算：例如 `llama_decode`、`prefill_with_emb`、`prefill_with_emb_tts`、`vision_image_encode`、`audition_audio_encode`、`Token2WavSession::feed_window`，这些才是 GPU/CPU 推理或编码计算。

最需要修的是：TTS/T2W 后台线程目前用全局 `perf_current_chunk_index` 取 chunk id。主线程进入下一轮 prefill 后，这个全局值会被更新，而 TTS/T2W 可能还在处理上一轮已经入队的数据，所以 TTS/T2W 日志的 `chunk=` 可能会错归属。建议把 `perf_chunk_index` 放进 `LLMOut`、`T2WOut`，随队列数据一起传递。

## 代码路径口径

本次复核以当前分支 `perf/duplex-benchmarking` 的改动为准，重点看：

- `tools/omni/test/test-duplex.cpp`：测试入口，设置 `ctx_omni->async = true`，每个 chunk 调 `stream_prefill()` 后调 `stream_decode()`。
- `tools/omni/omni.cpp`：性能打点、LLM/TTS/T2W 线程、prefill/decode 主体。
- `scripts/analyze_duplex_perf.py`：后处理脚本对 stage 的解释方式。

注意：当前双工测试默认是 async 路径，因此 `prefill.audio_kv_prefill` 和 `prefill.vision_kv_prefill` 主要覆盖同步路径；测试主路径里的真实 LLM KV 写入发生在 LLM 线程的 `llm.kv_prefill`。

## 阶段逐项审计

| stage | 当前覆盖内容 | 性质 | 问题 | 建议 |
| --- | --- | --- | --- | --- |
| `test.chunk_prefill_api` | 测试入口外层 `stream_prefill()` 调用 | API 包络 | 包含 encoder、入队、index=0 session 初始化等，不是单一阶段 | 保留为外层 API 指标，统计时不要和内部 stage 相加 |
| `stream_prefill` | `stream_prefill()` 整体 | API 包络 + 少量等待 | 非双工/首轮路径里会等上一轮 TTS；async 路径里只到 LLM 入队，不包含真实 LLM KV prefill | 命名为 `api.stream_prefill` 或在报告里明确“包络” |
| `session.ref_audio_encode` | 参考音频 APM encode | 计算 | index=0 初始化，不是普通用户 chunk | 归到 `session.init.*`，和普通 chunk 分开统计 |
| `session.ref_audio_kv_prefill` | 参考音频写入 LLM KV cache | 计算 | 只表示会话初始化 ref audio，不代表用户音频 prefill | 归到 `session.init.*` |
| `prefill.vision_encode` | vision encoder 生成 embedding | 计算 | 语义基本正确 | 可保留 |
| `prefill.audio_encode` | audio encoder 生成 embedding | 计算 | 语义基本正确 | 可保留 |
| `prefill.vision_kv_prefill` | 同步路径里 vision embedding 写入 LLM KV | 计算 | async 双工测试主路径不会走这里 | async 路径需要在 `llm_thread_func()` 内补更细 stage |
| `prefill.audio_kv_prefill` | 同步路径里 audio embedding 写入 LLM KV | 计算 | async 双工测试主路径不会走这里 | async 路径需要在 `llm_thread_func()` 内补更细 stage |
| `prefill.enqueue_llm` | 等 LLM 队列有空间，然后 push | 等待/队列 | 不是计算；可能包含 `cv.wait` 等待队列空间 | 拆成 `queue.llm.wait_space` 和 `queue.llm.enqueue`，或至少标注 queue stage |
| `llm.kv_prefill` | LLM 线程取队列后写 KV cache | 计算为主 | 把 audio、vision、marker token 的 `eval_string()` 和 `prefill_with_emb()` 混在一起；一次可 drain 多个队列项，`chunk` 只取第一个 item | 拆成 `llm.prefill.marker_eval`、`llm.prefill.vision_kv`、`llm.prefill.audio_kv`，并按每个 `omni_embeds->index` 单独打点 |
| `test.chunk_decode_api` | 测试入口外层 `stream_decode()` 调用 | API 包络 | 包含等待 prefill、force listen、LLM sampling、TTS 入队和清理 | 保留为外层 API 指标，不能和内部 stage 相加 |
| `stream_decode` | `stream_decode()` 整体 | API 包络 | 包含等待、采样、prompt eval、队列操作、滑窗记账；不是纯 decode | 命名为 `api.stream_decode` 或报告里明确“包络” |
| `decode.wait_llm_prefill` | 等 LLM 线程发 `g_decode_cv` | 等待 | 这个阶段常常和后台 `llm.kv_prefill` 同时发生，不能算作计算 | 保留，但归类为 wait；不要放到“LLM 计算耗时”里 |
| `decode.force_listen` | 开局强制 listen，跳过采样 | 控制流 | 不是模型计算 | 单独归类为 policy/control |
| `decode.llm_sample` | 从开始 LLM token loop 到整个 loop 结束 | 混合 | 里面包含真实 `llama_loop_with_hidden_and_token()`/`llama_decode`，也包含文本后处理、`tts.enqueue_from_llm` 队列等待、chunk_eos/unit_end 注入 | 拆成 `llm.decode.token_loop`、`llm.decode.inject_special_tokens`、`decode.postprocess`；TTS 入队移出该 scope |
| `decode.first_token` | 从 `decode.llm_sample` 开始到首 token 完成 | 计算 + 首 token 延迟 | 当前 marker 发生在 `llama_loop_with_hidden_and_token()` 后，包含首 token sample 和首 token eval 写 KV | 语义可接受，但报告里说明它是“首 token 完成”，不是纯 sampler 时间 |
| `tts.enqueue_from_llm` | 等 TTS 队列空间，然后 push `LLMOut` | 等待/队列 | 被嵌在 `decode.llm_sample` scope 内，统计相加会双算 | 保留为 queue stage，但从 `decode.llm_sample` scope 中剥离 |
| `tts.llm_to_audio_tokens` | `generate_audio_tokens_local()` 整体 | 混合 | stage 名叫“LLM token/hidden -> audio token”，但真正的 emb_text/projector/normalize/merge 在调用前已完成；scope 内反而包含 TTS condition prefill、audio token decode、T2W 入队和文件写 token | 拆成 `tts.condition_merge`、`tts.condition_prefill`、`tts.audio_token_decode`、`tts.enqueue_t2w`、`tts.write_debug_tokens` |
| `t2w.tokens_to_wav` | T2W window 转 wav | 计算为主/后端相关 | C++ 路径基本包 `feed_window()`，不含后续 PCM/WAV 写文件；Python 路径包 IPC 发送、等待 Python 服务、服务端推理和可能的文件输出 | 拆 `t2w.queue_wait`、`t2w.feed_window`、`t2w.write_wav`；Python 路径 detail 里继续保留 `infer_ms` |

## 主要混杂点

### 1. Async prefill 的真实计算没有按音频/视觉拆开

在 async 双工测试中，`stream_prefill()` 只负责把 audio/vision embedding 放进 `LLMThreadInfo::queue`。真正调用 `eval_string()`、`prefill_with_emb()`、最终触发 `llama_decode()` 的地方在 `llm_thread_func()`。

当前 `llm.kv_prefill` 从队列 drain 后开始，到所有 item 处理完结束。这个阶段是计算为主，但粒度太粗：

- vision marker、overview、slice、audio embedding 都在同一个 stage。
- 一次可能处理多个 `omni_embeds`，日志 `chunk` 只使用 `llm_embeds.front()->index`，后续 item 的时间会被归到第一个 chunk。
- 无法从日志直接回答“audio KV prefill 花了多少、vision KV prefill 花了多少”。

建议在 LLM 线程内部围绕实际调用拆分：

- `llm.prefill.unit_marker_eval`
- `llm.prefill.vision_overview_kv`
- `llm.prefill.vision_slice_kv`
- `llm.prefill.audio_kv`
- `llm.prefill.end_marker_eval`

这些 stage 应该用当前 `embeds->index`，不要用批量队列的第一个 index。

### 2. `decode.llm_sample` 不是纯 LLM 采样

`decode.llm_sample` 的 start 在 LLM token loop 前，end 在整个 loop 结束后。loop 中确实有真实计算：`llama_loop_with_hidden_and_token()` 会采样 token，并通过 `eval_id_with_hidden()` 写回 KV，最终进入 `llama_decode()`。

但同一个 scope 里还包括：

- special token 判断、字符串清洗、text_queue 推送；
- 达到 chunk 限制时注入 `<|chunk_eos|>`，会再调用 `eval_tokens()`；
- 每个 chunk 后注入 `</unit>`，也会调用 `eval_tokens()`；
- `tts.enqueue_from_llm`，包括等待 TTS 队列空间和 push。

所以 `decode.llm_sample` 现在是“LLM token 生成主循环 + 后处理 + TTS 入队”的混合段。它可以作为 decode 主循环耗时，但不能直接解释为纯 `llama_decode` 时间。

建议改成：

- `llm.decode.token_step` 或 `llm.decode.token_loop`：只包 `llama_loop_with_hidden_and_token()` 的循环。
- `llm.decode.inject_chunk_eos`：包 chunk limit 后的 `eval_tokens()`。
- `llm.decode.inject_unit_end`：包 `</unit>` 的 `eval_tokens()`。
- `queue.tts.enqueue_from_llm`：放在 LLM compute scope 外。

### 3. TTS stage 名字和实际范围不一致

`tts.llm_to_audio_tokens` 目前包的是 `generate_audio_tokens_local()`。这个函数里确实会生成 audio token，但它不包含完整的“LLM token/hidden state -> TTS condition”过程。

在双工 TTS 线程里，真正的 condition merge 在调用 `generate_audio_tokens_local()` 之前已经完成：

- `tts_emb_text()`：LLM token id -> TTS text embedding。
- `tts_projector_semantic()`：LLM hidden state -> TTS hidden dim。
- `normalize_l2_per_token()`。
- `merged_embeddings = llm_embeds + projected_hidden`。

而 `tts.llm_to_audio_tokens` scope 内包含的是：

- 添加 `text_eos_embed`、`audio_bos`；
- `prefill_with_emb_tts()`，内部调用 TTS 模型 `llama_decode()`；
- 多次 `sample_tts_token()`，其中又包括 head_code logits 计算、采样、audio token embedding 写回 TTS KV；
- 推送 `T2WOut` 到 T2W 队列；
- 写 token debug 文件。

因此这个 stage 应改名或拆分。推荐拆成：

- `tts.condition_merge`：LLM token/hidden -> merged embedding。
- `tts.condition_prefill`：merged embedding -> TTS KV。
- `tts.audio_token_decode`：TTS audio token autoregressive decode。
- `queue.t2w.enqueue`：TTS -> T2W 入队。

### 4. TTS/T2W chunk id 可能错归属

后台线程里的 TTS/T2W 打点大多读取 `ctx_omni->perf_current_chunk_index.load()`。这个字段由主线程在每个 `stream_prefill(index)` 开始时更新。

双工里 TTS/T2W 和下一轮 prefill/decode 会并行。当主线程已经进入 chunk N+1 时，TTS/T2W 可能仍在处理 chunk N 的 LLM 输出。此时日志会把 TTS/T2W 事件标成 N+1。

建议：

- `LLMOut` 增加 `int perf_chunk_index`，在 `stream_decode()` 入队 TTS 时写入当前 `perf_chunk_index`。
- `T2WOut` 增加 `int perf_chunk_index`，由 TTS 线程转发给 T2W。
- TTS/T2W 打点都用队列数据携带的 id，不读全局 `perf_current_chunk_index`。
- 如果后续要分析连续说话的一轮，还应再加 `turn_id` / `utterance_id` / `tts_chunk_id`，避免“用户音频 chunk”和“TTS audio chunk”混在一个 `chunk` 字段里。

### 5. `stream_*` 和内部 stage 不应该相加

`stream_prefill` 包含 `prefill.audio_encode`、`prefill.vision_encode`、`prefill.enqueue_llm` 等内部阶段。`stream_decode` 包含 `decode.wait_llm_prefill`、`decode.llm_sample`、`tts.enqueue_from_llm` 等内部阶段。

后处理脚本如果同时画 `stream_decode` 和 `decode.llm_sample`，它们是父子关系，不是并列关系。看时间线可以保留父子嵌套，但算总耗时时必须只选一种口径：

- API 口径：只用 `test.chunk_prefill_api` / `test.chunk_decode_api` 或 `stream_prefill` / `stream_decode`。
- 内部分解口径：只用内部 stage，并明确哪些是 wait，哪些是 compute。

### 6. 后处理脚本对 stage 的分类需要调整

`scripts/analyze_duplex_perf.py` 目前把 `decode.wait_llm_prefill` 放在 LLM lane，并在 `STAGE_ROWS` 里和 `decode.llm_sample`、`stream_decode` 一起统计。这样图上容易被理解成同类计算耗时。

建议调整：

- 增加 stage 类型字段：`api`、`wait`、`queue`、`compute`、`control`。
- `decode.wait_llm_prefill` 放入 wait lane。
- `prefill.enqueue_llm`、`tts.enqueue_from_llm`、后续 `queue.t2w.enqueue` 放入 queue lane。
- `stream_prefill`、`stream_decode` 只在 API 图出现，不进入“阶段耗时均值”的 compute 表。
- `PERF_RE` 目前只接受数字 GPU 字段；如果无 CUDA 时日志是 `gpu_used_mb=NA`，解析会丢事件。建议允许 `NA`。

## 建议的新阶段表

| 类别 | 建议 stage | 说明 |
| --- | --- | --- |
| API | `api.chunk_prefill` | 测试入口外层 prefill |
| API | `api.chunk_decode` | 测试入口外层 decode |
| Session init | `session.ref_audio_encode` | ref audio APM encode |
| Session init | `session.ref_audio_kv_prefill` | ref audio 写 LLM KV |
| Compute | `encode.vision` | vision encoder |
| Compute | `encode.audio` | audio encoder |
| Queue | `queue.llm.wait_space` | 等 LLM 队列空间 |
| Queue | `queue.llm.enqueue` | 入 LLM 队列 |
| Compute | `llm.prefill.audio_kv` | 用户音频 embedding 写 LLM KV |
| Compute | `llm.prefill.vision_kv` | 图像 embedding 写 LLM KV |
| Compute | `llm.prefill.marker_eval` | `<unit>`、`<image>`、`</image>` 等 marker token eval |
| Wait | `wait.llm_prefill_done` | decode 等异步 prefill 完成 |
| Control | `decode.force_listen` | 强制 listen 策略 |
| Compute | `llm.decode.token_loop` | LLM 自回归 token 生成，含每步 `llama_decode` |
| Compute | `llm.decode.inject_special_token` | chunk eos、unit end 等手动注入 |
| Queue | `queue.tts.wait_space` | 等 TTS 队列空间 |
| Queue | `queue.tts.enqueue` | LLMOut 入 TTS 队列 |
| Compute | `tts.condition_merge` | LLM token/hidden -> TTS merged embedding |
| Compute | `tts.condition_prefill` | TTS condition 写 TTS KV |
| Compute | `tts.audio_token_decode` | TTS audio token 自回归生成 |
| Queue | `queue.t2w.enqueue` | audio token 入 T2W 队列 |
| Wait | `queue.t2w.wait_data` | T2W 线程等数据 |
| Compute | `t2w.feed_window` | Token2Wav window 推理 |
| IO | `t2w.write_wav` | PCM/WAV 文件写出 |

## 优先级

1. P0：给 `LLMOut` 和 `T2WOut` 增加稳定 `perf_chunk_index`，修正 TTS/T2W 的 chunk 归属。
2. P0：把 queue/wait stage 从 compute scope 中剥离，尤其是 `tts.enqueue_from_llm` 不要嵌在 `decode.llm_sample` 里。
3. P1：在 `llm_thread_func()` 内按 `embeds->index` 拆 `llm.prefill.audio_kv` / `llm.prefill.vision_kv` / marker eval。
4. P1：把 `tts.llm_to_audio_tokens` 拆成 condition merge、condition prefill、audio token decode、T2W enqueue。
5. P2：调整分析脚本的 stage 分类，避免父子 stage 相加，并兼容 `gpu_used_mb=NA`。

## 当前可用口径

在修正前，建议这样读现有日志：

- 看端到端 chunk 延迟：用 `test.chunk_prefill_api` 和 `test.chunk_decode_api`。
- 看 encoder：用 `prefill.audio_encode`、`prefill.vision_encode`。
- 看 async LLM prefill：用 `llm.kv_prefill`，但只作为合并口径，不能拆 audio/vision。
- 看等待：单独看 `decode.wait_llm_prefill`，不要算进 compute。
- 看 LLM decode 主循环：可暂用 `decode.llm_sample`，但要知道里面混了 TTS enqueue 和 special token eval。
- 看 TTS：`tts.llm_to_audio_tokens` 只能暂作“TTS 线程处理一次 LLMOut 的总耗时”，不能解释成完整“LLM hidden -> audio token”。
- 看 T2W：`t2w.tokens_to_wav` 可暂作 window 处理耗时，但 Python 后端包含 IPC/服务等待，C++ 后端不含 WAV 写文件。
