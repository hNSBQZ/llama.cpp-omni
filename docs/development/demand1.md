# 双工性能阶段划分需求

## 核心目标

当前要解决的问题不是把代码重构得很细，而是把双工链路的性能日志划分成几个稳定、可解释、可对比的大阶段。阶段划分需要能回答：

- 每个输入 chunk 从进入系统到产生输出，主要耗时花在哪几段。
- 哪些时间是真正在做模型/算子计算，哪些时间只是线程同步、队列等待或 API 包络。
- 各阶段在异步双工流水线中是否并行，以及并行时各自归属到哪个 chunk。

## 划分原则

1. 计算阶段保持中等粒度，不拆到每个小函数或每个算子。
2. 队列等待、线程等待、锁等待必须从计算阶段中剥离出来，单独作为 queue/wait 类 stage。
3. 连续发生、语义上属于同一模型阶段的 CPU 和 GPU 计算可以放在同一个 compute stage 里。
4. 不必把 prefill 里的 prefix token eval 和后续 embedding prefill 过度拆开；只要它们共同服务于同一个 LLM prefill 阶段，可以归到同一段。
5. TTS 进入核心 LLM 架构 prefill 前的线性变换、归一化、embedding merge 等算子，可以算入 TTS 计算阶段，不需要单独拆成很细的 profiling 项。
6. API 外层阶段可以保留，但只能用于端到端包络分析，不能和内部 compute stage 相加。

## 建议的大阶段

| 类别 | 阶段 | 说明 |
| --- | --- | --- |
| Compute | `vision.encode` | 图像输入经过 vision encoder 得到 vision embedding。 |
| Compute | `audio.encode` | 音频输入经过 audio encoder 得到 audio embedding。 |
| Queue/Wait | `queue.llm.*` | `stream_prefill()` 将 embedding 交给 LLM 线程时的等待和入队。 |
| Compute | `llm.prefill` | LLM 线程把当前 chunk 的 prefix/marker token 和 audio/vision embedding 写入 KV cache。 |
| Wait | `wait.llm_prefill_done` | `stream_decode()` 等待异步 LLM prefill 完成。 |
| Compute | `llm.decode` | LLM 自回归生成阶段，即循环采样并把新 token 写回 KV cache。chunk eos、unit end 这类紧随 decode 的特殊 token 注入也可以放在这一大段里。 |
| Queue/Wait | `queue.tts.*` | LLM 输出交给 TTS 线程时的等待和入队。 |
| Compute | `tts.infer` | LLM token + hidden state 到 TTS audio token 的整体计算，包括 projector、normalize、condition merge、TTS prefill 和 audio token decode。 |
| Queue/Wait | `queue.t2w.*` | TTS audio token 交给 T2W 线程时的等待和入队。 |
| Compute | `t2w.infer` | TTS audio token 到 mel/PCM/WAV 数据的模型推理部分。 |
| IO | `t2w.write` | 如果需要单独看文件写出，可以把 WAV/PCM 落盘从 T2W 推理中剥离出来。 |

## 不建议过度拆分的地方

- `llm.prefill` 不必拆成 `prefix_eval`、`marker_eval`、`embedding_prefill` 多个很细阶段；除非后续确实要比较 audio KV 和 vision KV 的耗时差异。
- `tts.infer` 不必把 projector、L2 normalize、embedding merge、TTS condition prefill、audio token decode 全部拆成独立主阶段；这些可以作为同一个 TTS 计算阶段的内部细节。
- `llm.decode` 不必把每个 token step 单独作为主阶段；可以统计整个 token loop 的总耗时，同时在 detail 里记录生成 token 数、有效 TTS token 数和 `n_past` 增量。

## 必须剥离的时间

以下内容不能算进 compute stage：

- 等 LLM 队列有空间。
- 等 TTS 队列有空间。
- TTS/T2W 后台线程等待队列数据。
- `stream_decode()` 等待异步 prefill 完成。
- 纯 API 包络时间，例如 `stream_prefill`、`stream_decode`、测试入口的 chunk API 时间。

这些时间应该单独标成 queue/wait/api，这样后续画时间线时能看出流水线并行和阻塞点。

## 日志字段要求

每个 stage 至少需要输出：

- `stage`：阶段名。
- `event`：`start` / `end` / `mark`。
- `chunk`：稳定的输入 chunk id，异步线程里不能再读全局 current chunk，必须随队列数据传递。
- `t_rel_ms`：相对测试开始时间。
- `duration_ms`：阶段结束时的耗时。
- `rss_mb`：进程内存。
- `gpu_used_mb`：GPU 显存。
- `n_past`：LLM/TTS KV cache 相关阶段需要记录。
- `detail`：记录 token 数、window size、backend、是否 final 等补充信息。

## 统计口径

后续分析时建议分三种口径：

- 端到端口径：只看测试入口或 API 包络阶段，例如 `api.chunk_prefill`、`api.chunk_decode`。
- 计算口径：只汇总 `vision.encode`、`audio.encode`、`llm.prefill`、`llm.decode`、`tts.infer`、`t2w.infer`。
- 阻塞口径：单独看 queue/wait 阶段，用来判断是否被后台线程、队列容量或同步点卡住。

这三类口径不要混加，否则会把父子阶段或并行阶段重复统计。