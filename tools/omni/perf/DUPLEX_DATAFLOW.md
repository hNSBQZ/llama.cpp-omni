# 全双工数据流与 frame ↔ wav 对齐分析

本文从 `tools/omni/perf/perf-duplex.cpp` 的入口出发，逐层梳理 duplex 模式下一帧
（图像 + 音频）从提交到落盘为 wav 的完整链路，并用 `tools/omni/output/perf_run_f16.log`
（F16 全量 36 帧、`--stream-interval 1000`）的实测数据回答四个对齐性问题，最后评估
`judge-final` 服务化评测链路能否在此基础上算出「每包 RTF」。

所有行号基于当前工作区的 `tools/omni/omni.cpp`（11843 行）。

---

## 1. 线程与队列拓扑

duplex 一共有 5 条常驻线程，串成一条 5 级流水线。前两级由 `DuplexSession` 调度，
后三级由 `DuplexPipeline` + TTS/T2W 线程承担。

```
                      perf-duplex.cpp (主线程)
  push_frame ─┐                                        ┌─ wait_next_frame
              ▼                                        │
   ┌──────────────────────┐                 ┌──────────┴────────────┐
   │ pending_frames (64)  │                 │ done_results (无上限)  │
   └──────────┬───────────┘                 └──────────▲────────────┘
              ▼                                        │
        prefill_worker ──► decode_pending (无上限) ──► decode_worker
              │                                        │
              ▼ stream_prefill                         ▼ stream_decode (阻塞)
   ┌──────────────────────┐                            │
   │ encoder_queue (16)   │                            │
   └──────────┬───────────┘                            │
              ▼                                        │
        encoder_thread  (VPM ∥ APM, std::async)        │
              │                                        │
              ▼                                        │
   ┌──────────────────────┐                            │
   │ prefill_queue (32)   │                            │
   └──────────┬───────────┘                            │
              ▼                                        │
         llm_thread ◄────── pending_decode (仅 1 槽) ◄──┘
              │  duplex_do_prefill_one_fused → duplex_do_decode
              ▼
   ┌──────────────────────┐
   │ tts_queue (MAX = 1)  │  ← LLM 在此阻塞，是全局背压点
   └──────────┬───────────┘
              ▼
        tts_thread (tts_thread_func_duplex)
              │  generate_audio_tokens_local → 每 25/28 个 audio token 推一次
              ▼
   ┌──────────────────────┐
   │ t2w_queue (duplex 下 │  ← duplex 路径不检查容量，可无界增长
   │  未 enforce 上限)     │
   └──────────┬───────────┘
              ▼
        t2w_thread (t2w_thread_func_cpp)
              │  滑窗 WINDOW=28 / CHUNK=25 → feed_window
              ├─► audio_output_cb(samples, n, sr, is_final)   ← perf-duplex 挂钩点
              └─► 写 {out}/tts_wav/wav_{wav_turn_base + wav_idx}.wav
```

关键结构体：

- `DuplexSession` — `omni.cpp:11543-11568`，`PENDING_MAX = 64`
- `DuplexPipeline` — `omni.cpp:3933-3963`，`ENCODER_QUEUE_CAP = 16`、`PREFILL_QUEUE_CAP = 32`
- `T2WOut` / `T2WThreadInfo` — `omni.h:75-92`
- TTS 队列容量 `MAX_QUEUE_SIZE = 1` — `omni.cpp:4294`

### 1:1 顺序是怎么保证的

三层串行约束叠加，保证 `push_frame` 的第 N 帧一定对应 `wait_next_frame` 的第 N 个结果：

1. `prefill_worker` / `decode_worker` 各只有一条，队列全是 FIFO（`omni.cpp:11570-11656`）
2. `duplex_decode` 同时只允许一个 `pending_decode`（`omni.cpp:10405-10407`）
3. `duplex_llm_thread_func` 每次 decode 最多消费一个 prefill packet（`omni.cpp:10189-10202`）

**这个 1:1 只覆盖 frame → LLM 判定结果，不覆盖 frame → wav。** 后者是本文的重点。

---

## 2. 一帧的完整旅程

| 阶段 | 代码位置 | 说明 |
|---|---|---|
| ① push | `omni_duplex_push_frame` `omni.cpp:11695-11716` | 入 `pending_frames`，分配自增 `frame_id`（`session_begin` 占用 id=0），立即返回 |
| ② encode | `duplex_encoder_thread_func` `omni.cpp:9456-9568` | VPM 与 APM 用 `std::async` 并行；实测 VPM≈100ms、APM≈15ms、wall≈100ms |
| ③ prefill | `duplex_do_prefill_one_fused` `omni.cpp:9758-9790` | 拼 `<unit><image>…</image>\n` + audio_emb，一次 `prefill_with_emb` 写 KV；实测 77 token / 12-18ms |
| ④ decode | `duplex_do_decode` `omni.cpp:9906-10184` | 采样循环，判定 listen/speak，产出 text + hidden states |
| ⑤ TTS | `tts_thread_func_duplex` `omni.cpp:6289+` → `generate_audio_tokens_local` `omni.cpp:5768+` | duplex 下 `max_audio_tokens = 26`；每 28（首包）/25 个 audio token 推一次 T2W |
| ⑥ T2W | `t2w_thread_func_cpp` `omni.cpp:8990+` | 滑窗取 28 token 出一段 wav；`audio_output_cb` 在 `omni.cpp:9170-9172`；写盘在 `omni.cpp:9176` |

### ④ 的采样状态机（问题 2、4 的根源）

`duplex_do_decode` 每帧开头把 `ended_with_listen` 置 false（`omni.cpp:9927`），然后：

- **force_listen**：会话前 `force_listen_count`（默认 3）帧直接判 LISTEN，不采样（`omni.cpp:9949-9968`）
- 内层循环每次采一个 token，最多 `step_size = 10` 个推一批给 TTS，单帧上限
  `max_new_speak_tokens_per_chunk = 26`（`omni.h:298`）
- token 分类见 `get_token_type`，特殊 token 在 `omni.cpp:4607-4621` 运行时查表

三类终止：

| 采到的 token | `llm_finish` | `ended_with_listen` | `current_turn_ended` | 上报 |
|---|---|---|---|---|
| `<\|listen\|>` | ✅ | ✅ | — | LISTEN |
| `<\|chunk_eos\|>` / `<\|chunk_tts_eos\|>` | ✅ | ❌ | — | SPEAK |
| `<\|turn_eos\|>` / `<\|tts_eos\|>` / EOS | ✅ | ❌ | ✅ | **SPEAK** |

`decode_worker` 的判定只有一行（`omni.cpp:11631`）：

```11631:11631:tools/omni/omni.cpp
            r.is_speak = !ctx_omni->ended_with_listen.load();
```

每批结束后无条件 eval 一个 `</unit>` 写进 KV（`omni.cpp:10077-10081`），并在
`!ended_with_listen` 时向 `text_queue` 推 `__END_OF_TURN__`（`omni.cpp:10141-10143`）。

---

## 3. F16 全量日志实测（36 帧 → 20 个 wav）

用 `perf_run_f16.log` + `perf_report_f16.json` + `stage_timing.jsonl` 交叉对齐后的完整时间线。
`wav_N` 的编号规则是 `round_idx * 1000 + wav_idx`，其中 `wav_turn_base = round_idx * 1000`
在 decode 开始时赋值（`omni.cpp:9915`），`wav_idx` 在 duplex 下是全局单调计数器、永不重置。
所以文件名前缀直接就是**写盘那一刻正在 decode 的帧号**。

| frame | 判定 | text | TTS chunk | wav | 音频时长 | t2w 耗时 | RTF |
|---|---|---|---|---|---|---|---|
| 1-3 | LISTEN (force) | — | — | — | — | — | — |
| 4 | LISTEN | — | — | — | — | — | — |
| 5 | SPEAK | 没问题，我 | c0 | `wav_5000` | 0.84s | 300.9ms | 0.36 |
| 6 | SPEAK | 会帮你 | c1 | `wav_6001` | 1.00s | 237.5ms | 0.24 |
| 7 | SPEAK | 留意的。 | c2 | `wav_7002` | 1.00s | 222.8ms | 0.22 |
| 8 | SPEAK | 现 | c3 | `wav_8003` | 1.00s | 221.8ms | 0.22 |
| 9 | SPEAK | 在二十三 | c4 | `wav_9004` | 1.00s | 230.8ms | 0.23 |
| 10 | SPEAK | 楼， | c5 | `wav_10005` | 1.00s | 221.6ms | 0.22 |
| 11 | SPEAK | 马上 | c6 | `wav_11006` | 1.00s | 233.0ms | 0.23 |
| 12 | SPEAK | 到 | c7 | `wav_12007` | 1.00s | 247.2ms | 0.25 |
| 13 | SPEAK | 二四楼了。 | c8 | `wav_13008` | 1.00s | 224.5ms | 0.22 |
| **14** | SPEAK | *(空 — 但 token_ids=[40820]，见 §3.2)* | c9 | `wav_14009` | 1.00s | 221.9ms | 0.22 |
| | | | | **`wav_14010`** *(is_final 尾包)* | 0.60s | 212.2ms | 0.35 |
| 15 | SPEAK | 四楼已 | c10 | `wav_15011` | 1.00s | 223.4ms | 0.22 |
| **16** | SPEAK | 到达。*(turn_eos)* | c11 | `wav_16012` | 1.00s | 222.7ms | 0.22 |
| | | | | **`wav_16013`** *(is_final 尾包)* | 0.36s | 186.2ms | 0.52 |
| **17-28** | **SPEAK** | *(空, turn_eos already flushed)* | — | **无** | — | — | — |
| 29 | SPEAK | 开 | c12 | `wav_29014` | 1.00s | 230.0ms | 0.23 |
| **30** | SPEAK | *(空, flush)* | c13 | **`wav_30015`** *(is_final 尾包)* | 0.40s | 189.6ms | 0.47 |
| 31 | SPEAK | 开 | c13 | `wav_31016` | 1.00s | 232.7ms | 0.23 |
| 32 | SPEAK | 了两扇 | c14 | `wav_32017` | 1.00s | 228.6ms | 0.23 |
| 33 | SPEAK | 门。 | c15 | `wav_33018` | 1.00s | 221.4ms | 0.22 |
| **34** | SPEAK | *(空, flush)* | c16 | **`wav_34019`** *(is_final 尾包)* | 0.76s | 212.0ms | 0.28 |
| **35-36** | **SPEAK** | *(空, turn_eos already flushed)* | — | **无** | — | — | — |

统计：

- 36 帧 = 4 LISTEN + 32 SPEAK，其中 **14 个 SPEAK 帧完全没有音频产出**
- 20 个 wav，4 个声学轮次（`is_final` 出现在 `wav_14010 / wav_16013 / wav_30015 / wav_34019`）
- **模型在第 4 帧之后再没有采样出 `<|listen|>`**，所以 `perf-duplex` 的 `speak_turn_id`
  从帧 5 到帧 36 一直是 0 —— 4 个声学轮次被折叠成 1 个逻辑 turn
- 阶段耗时：VPM≈100ms、APM≈15ms、prefill≈13ms、LLM decode 170-300ms、TTS≈260ms、T2W≈225ms
- 端到端（decode 完成 → wav 落盘）650-920ms，T2W 队列等待恒为 0.02ms（**当前完全没有堆积**）

### 3.1 用 n_past 增量反推每帧的 token 序列

每采一个 token `n_past` 加 1，每批末尾还会无条件 eval 一个 `</unit>`（`omni.cpp:10077-10081`）。
拿日志里的 `[prof] llm decode n_past=A->B tokens=N` 和 `token_ids.size` 相减，就能把
每帧实际采到了什么反推出来：

| 帧 | n_past 增量 | 有效 TTS token | 反推出的采样序列 |
|---|---|---|---|
| 13 | 8 | 5 | `<\|speak\|>` + 二/四/楼/了/。 + `<\|chunk_eos\|>` + `</unit>` |
| 14 | 5 | 1 | `<\|speak\|>` + `<\|turn_eos\|>` + 二 + `<\|chunk_eos\|>` + `</unit>` |
| 17-28 | 4 | 0 | `<\|speak\|>` + `<\|turn_eos\|>` + `<\|chunk_eos\|>` + `</unit>` |

两个结论：`<|speak|>` 是每个 chunk 开头都会采到的（所以 `omni.cpp:10095-10099`
才专门写了个 while 循环把它从 response 里剔干净）；帧 17-28 的序列和肉眼观察完全一致。

### 3.2 空包在 TTS 侧的去向：是控制消息，不是数据

推 TTS 的条件是 `(!response.empty() || llm_finish)`（`omni.cpp:10110-10111`）——
`llm_finish` 一为真就推，不管 response 是否为空。所以帧 17-28 推过去的 `LLMOut` 是：

```
text = "", token_ids = [], hidden_states = [], llm_finish = true, is_end_of_turn = true
```

TTS 线程 `has_llm_data = false`（`omni.cpp:6588`），落到
`else if (duplex_mode && accumulated_is_end_of_turn && llm_finish)`（`omni.cpp:6839`），
`turn_eos_flushed` 已为 true → 打印 "skipping" 直接跳过。
**空内容从来没进过 TTS 模型，也没进过 T2W 队列**，所以不会崩。

但这不是免费的，每个这样的帧仍然要付：

1. LLM 真的采样了 3 个 token + 1 次 `</unit>` eval，实测 190-215ms。14 帧 ≈ **2.8 秒纯浪费**
2. `merge_wav_files(tts_wav_output_dir, chunk_idx + 1)` 被无条件调一遍（`omni.cpp:6877`），
   而且 `chunk_idx` 只增不减，扫描列表越来越长 —— 日志里那一大片
   `TTS: chunk file ... does not exist or is empty` / `no valid WAV files to merge` 就是它
3. TTS 的 KV cache 被反复 `llama_memory_seq_rm` 清空，`tts_condition_saved = false`
   （`omni.cpp:6878-6884`），下一次真 SPEAK 时 voice-clone condition 必须重新 prefill

### 3.3 [BUG] response 截断会吞掉已经发音的字

对照 `llm_debug/llm_token_ids.txt` 和 `llm_debug/llm_text.txt` 发现一处文本与音频不一致：

```
[chunk_8] 40820 63703 99432 34187 1773     → llm_text: "二四楼了。"   (帧 13)
[chunk_9] 40820                            → llm_text: (无对应行)      (帧 14)
```

帧 14 的 `token_ids.size=1` 但 `llm_text.len=0`。成因是采样循环里两条路径不同步：

- `<|turn_eos|>` **不在** `is_end_token` 里（duplex 下只认 LISTEN / CHUNK_EOS / CHUNK_TTS_EOS，
  见 `omni.cpp:257-265`），所以采到它**不 break**，继续往下采
- 于是 `response` 累积成 `"<|speak|>" + "<|turn_eos|>" + "二"`
- 收尾清理 `response = response.substr(0, p)`（`omni.cpp:10091-10094`）在**第一个**
  `<|turn_eos|>` 处截断，把后面的 "二" 一起砍掉 → `response = ""`
- 而 `chunk_token_ids` 是在类型判断**之前**收集的（`omni.cpp:10008-10018`），不受截断影响，
  token 40820 照样进了 TTS

结果：**文本通道报"这帧没说话"，音频通道实际合成并播放了这个字**
（`wav_14009` 1.00s + `wav_14010` 0.60s 就是它的产物）。

这条对评测的杀伤力很直接：`judge-final` 的 `_is_empty_speak_text()` 会把
「空文本 SPEAK」整类排除出主指标，理由是"LLM 几乎不 decode，延迟偏低"。
但帧 14 这种**空文本却真有音频**的样本恰恰是尾包场景，是最该统计的，
现在被一起误排掉了。

> 修法上要注意：判断"本帧是否真的产出了语音"必须看 `chunk_token_ids.empty()`，
> **不能看 `response.empty()`**。

---

## 4. 四个问题的判定

### 问题 1：frame 与 TTS 是否 1 对 1？

**在当前 decode 长度下，"有文本的 SPEAK 帧" 与 "TTS chunk" 确实是 1:1，但这是巧合而非机制保证。**

机制上单帧可以推**多批**给 TTS：`duplex_do_decode` 外层 `for (il = 0; il < max_tgt_len; )`
每凑够 `step_size = 10` 个有效 token 就推一个 `LLMOut`（`omni.cpp:10110-10133`），只有
`llm_finish` 才 `break`。之所以实测恒为 1 批，是因为模型每帧只吐 1-5 个文本 token 就撞上
`chunk_eos` / `turn_eos`（日志里 `tokens=4..8`，其中还包含 `</unit>` 和结束 token），远没到 10。

一旦模型某帧多说几个字（>10 个有效 token），同一帧就会推 2 个 TTS chunk，1:1 立刻失效。
真正的硬上限是 `max_new_speak_tokens_per_chunk = 26`，也就是最多 3 批。

> 结论：这个 1:1 由「模型每帧只说 1-5 字」这一数据分布决定，不是机制保证。
>
> **但对本次算子加速评测是够用的**：评测样本固定、system prompt 固定、
> 采样种子固定（`OMNI_SAMPLER_SEED`，默认 42），所有选手跑的是同一条 token 轨迹，
> 每帧的 chunk 批数因此是确定的、可复现的。所以算 RTF 时可以放心按帧聚合。
>
> 代价是这层保证依赖"样本不变"。真要换样本或放开采样，RTF 聚合必须改回按
> `src_cnt` 分组求和（见 §6 的实现，它本来就是按 `src_cnt` 聚合的，
> 一帧多包会自动累加，不依赖 1:1）。

### 问题 2：`turn_eos + chunk_eos + </unit>` 不该算 SPEAK

**确认成立，且这是当前最严重的语义 bug。**

日志中帧 17-28、35-36 共 14 帧，全部是：

```
Duplex decode: turn_eos (type=5), wait for chunk_eos
TTS Duplex: after queue - speek_done=0, llm_finish=1, token_ids.size=0, is_end_of_turn=1, llm_text.len=0
TTS Duplex: empty final chunk but is_end_of_turn=true, will call TTS to flush buffer
TTS Duplex: turn_eos already flushed, skipping TTS generation
--- Chunk 17/36 --- ... | <|speak|>
```

即：轮次已经通过 `turn_eos` 结束、TTS 已 flush 过（`turn_eos_flushed` 见 `omni.cpp:6331/6841`）、
本帧零文本零音频（`token_ids.size=0`，注意与 §3.3 的帧 14 区分——那种是"空文本但有 token"），
但仍然上报 SPEAK。原因是 `is_speak` 只看 `ended_with_listen`
（`omni.cpp:11631`），而 `turn_eos` 分支从不置该标志（`omni.cpp:10027-10033` 只置
`current_turn_ended`）。

后果有三层：

1. `perf-duplex` 的 `speak_turn_id` 永远是 0，首响 / 每轮 e2e 全部算错
2. `judge-final` 侧的 `e2e_timing._speak_turn_id` 用同一个 `is_listen` 信号，
   `stitch_speak_turn_wavs` 会把 20 个 wav 拼成 1 个 `turn_01.wav`
   （`judge-final/sessions/.../speak_turns/` 里确实只有一个文件）
3. 主指标 `e2e_speak_recv_to_wav_poll_ms` 的分母被 14 个"假 SPEAK"污染。
   评测脚本靠 `_is_empty_speak_text()` 事后排空文本来补救，但那只是补丁：
   帧 14/16/30/34 是「空文本但真有 wav 产出」，会被一起误排除。

**正确的状态划分应该是三态**，而不是二态：

| 状态 | 条件 | 含义 |
|---|---|---|
| LISTEN | 采到 `<\|listen\|>` 或 force_listen | 在听 |
| SPEAK | `!chunk_token_ids.empty()` | 在说 |
| IDLE (turn ended) | `current_turn_ended && chunk_token_ids.empty()` | 说完了，等用户 —— 语义上等价于 LISTEN |

判据必须用 `chunk_token_ids.empty()`，**不能用 `response.empty()`** —— §3.3 的帧 14
就是 `response` 为空但 `chunk_token_ids` 非空、且真的合成了音频。

### 问题 3：frame → wav 能否保证 1:1？

**不能。实测 32 个 SPEAK 帧对应 20 个 wav，分布是 0/1/2 三种。**

三个独立的破坏源：

**(a) 零产出帧（0 个 wav）** —— 问题 2 的那 14 帧。

**(b) 尾包帧（≥2 个 wav）** —— T2W 的滑窗是 `WINDOW_SIZE = 28`、`CHUNK_SIZE = 25`
（`omni.cpp:8967-8969`），每处理一窗滑掉 25 个 token（`omni.cpp:9262-9264`），
而 TTS 每帧产 ~26 个 audio token（`max_audio_tokens = 26`）。26 ≠ 25，缓冲区余量逐帧累积；
当 `is_final` 到来时 `need_flush = true`（duplex 只认 `is_final`，`omni.cpp:9144-9146`），
`while` 循环（`omni.cpp:9151`）会一直吐到 buffer 空，最后一窗
`is_last_window = is_final && size <= 28` 清空 buffer（`omni.cpp:9259-9261`）。
于是同一帧内出现「一个满窗 1.00s + 一个残窗 0.36~0.76s」两个 wav ——
帧 14 和帧 16 就是这种情况。**buffer 积得越多，flush 时一帧内吐出的 wav 就越多。**

**(c) 归属靠写盘时刻，不是因果** —— 这是最危险的一条。文件名用的是
`ctx_omni->wav_turn_base + wav_idx`（`omni.cpp:9176`），`wav_turn_base` 是 T2W 线程
**写盘那一刻**去读的全局变量，而它由 LLM 线程在每帧 decode 开始时改写（`omni.cpp:9915`）。
T2WOut 里其实已经带了入队时的 `round_idx`（`omni.h:79`，TTS 线程在 `omni.cpp:6816` 赋值），
但 duplex 分支只在单工模式下用它（`omni.cpp:9082-9107`），**duplex 下这个字段被浪费了**。

当前实测 `t2w_queue_wait_ms ≈ 0.02ms`，decode→wav 延迟 650-920ms < 1000ms 帧间隔，
所以归属恰好没错。但余量只有 ~10%：

- `wav_14010` 在 decode 结束后 894ms 落盘
- `wav_16013` 在 919ms 落盘

**选手把算子改慢 10% 就会越过 1000ms，wav 被贴上下一帧的编号。** 这正是你担心的
「堆积后多个 TTS 结果一块 flush」——不是合成一个 wav（T2W 一次 drain 整个队列合并
tokens，但仍按 25 token 一段切分输出，见 `omni.cpp:9055-9071` 和 `9151`），
而是**多个 wav 在同一帧窗口内连续落盘、共享同一个 `round_idx` 前缀**，无法再拆回原帧。

### 问题 4：speak/listen 切换时的尾音 wav 归属

**确认成立。**

`is_final` 触发的尾包（`wav_14010 / wav_16013 / wav_30015 / wav_34019`）是把 T2W buffer
里跨帧累积的残余 token 吐出来的，它的音频内容属于**上一帧或更早**的文本。

两种表现：

- 帧 14 / 16：尾包和本帧的正常包挤在同一帧 → 该帧 2 个 wav
- 帧 30 / 34：本帧零文本，只有 flush 尾包 → 该帧唯一的 wav 其实属于帧 29 / 帧 33

如果按问题 2 的定义把帧 30/34 改判为 IDLE/LISTEN，`wav_30015` 和 `wav_34019` 就成了
**孤儿 wav**：归属帧被判为"不在说话"，但确实产出了音频。所以修状态机的同时必须
一起修归属规则，否则会从「多算」变成「漏算」。

---

## 5. 问题 6：judge-final 服务化评测能否算每包 RTF

> 注：`judge-final` 下的 `.py` 源文件目前已被删除，只剩 `__pycache__/*.pyc`。
> 以下结论来自对字节码的结构反汇编（函数名 / 常量 / 字符串表），语义可信，
> 但具体分支细节以恢复源码后为准。

### 5.1 服务化链路

```
eval_duplex_e2e_latency.py  (scripts/)
  └─ runner/duplex_eval_runner.run_direct_eval
       ├─ omni_client.DuplexSession.duplex_prepare   → POST /v1/stream/prefill (cnt=0)
       ├─ 1Hz 生产者 → asyncio.Queue → _worker → _process_one
       │    └─ _duplex_step():
       │         POST /v1/stream/prefill  (audio_path_prefix + img_path_prefix, cnt=N)
       │         POST /v1/stream/decode   (SSE, 阻塞到本帧判定完)
       │              ← {content, is_listen, end_of_turn} + metrics{vpm_ms, apm_ms,
       │                 llm_prefill_ms, cost_llm_ms, cost_tts_ms, cost_token2wav_ms}
       │         → e2e_timing.log_chunk(chunk_idx, is_listen, end_of_turn, text, …)
       ├─ _wav_poll_loop 线程（100ms 轮询）
       │    └─ wav_poller.collect_wav_output_nowait({out}/tts_wav, sent_wav_files)
       │         → e2e_timing.log_wav(files, mtimes, t_poll)
       └─ finalize() → e2e_timing.jsonl / e2e_timing_summary.json
            + 复制 C++ 的 stage_timing.jsonl 进 session 目录（env CPP_STAGE_TIMING）
```

服务端对应 `tools/server/server-omni.cpp:242`（prefill）和 `:283`（decode SSE）。
这条链路和 `perf-duplex.cpp` 走的是同一套 `stream_prefill` / `stream_decode`，
唯一区别是驱动方式（HTTP + 文件轮询 vs 进程内回调），**所以第 4 节的全部结论原样适用**。

### 5.2 现在的对齐方式，以及它为什么不够

`e2e_timing.log_wav` 给每个 wav 打两个标签：

- `speak_turn_id` —— 当前 speak 轮次
- `last_speak_chunk_idx` —— **poll 到这个文件时，最近一个 SPEAK chunk 的序号**

`eval_duplex_e2e_latency._analyze_e2e` 就按 `last_speak_chunk_idx` 做一对一配对，
它自己的 note 也写明了「wav 与 chunk 非 1:1，按 speak_turn_id 对齐」。

这套口径有四个问题：

1. **归属是 poll 时刻，不是因果**。轮询周期 100ms，再叠加 C++ 侧 650-920ms 的
   decode→落盘延迟，误差窗口比帧间隔（1000ms）只小一点点。
2. **一帧多 wav 时会丢**。`last_speak_chunk_idx` 相同的多个 wav，一对一配对只能留一个。
3. **`speak_turn_id` 因为问题 2 恒为 0**，`e2e_first_wav_ms` 只有一个值，
   `stitch_speak_turn_wavs` 会把整场拼成一个 turn。
4. **没有记录音频时长**。`log_wav` 只写 `file` / `t_poll_ms` / `t_file_mtime_rel_ms`。

### 5.3 每包 RTF：现在算不出来，但差的东西很少

RTF 需要「单包计算耗时 ÷ 单包音频时长」。两个分量的现状：

| 分量 | 现状 |
|---|---|
| 计算耗时 | ✅ 已有。`stage_timing.jsonl` 的 `{"event":"t2w","wav":"wav_5000.wav","token2wav_ms":300.933,…}`（`omni.cpp:9231-9235`）按 wav 文件名逐条记录；`{"event":"tts","chunk_idx":N,"tts_ms":…}` 按 chunk 记录 |
| 音频时长 | ❌ 缺失。`e2e_timing.jsonl` 不记；`stage_timing.jsonl` 也不记。只有 C++ 的 stdout 日志里有（`T2W线程: wav_5000.wav \| 0.84s audio \| … \| RTF=0.36`，`omni.cpp:9224`），但那是给人看的，评测脚本不解析 |

而且 `_analyze_e2e` 读 `stage_timing.jsonl` 时只把 `tts_ms` / `token2wav_ms` 求了均值，
没有按 `wav` 字段和 `e2e_timing.jsonl` 做 join。

**结论：现在的脚本算不出每包 RTF，也不能保证 wav ↔ frame 一一对应。**

### 5.4 需要的改动

C++ 侧 4 项、judge 侧 3 项。(0) 是纯 bug，(1)(2) 决定评测在选手把算子改慢之后还准不准，
(3) 决定轮次划分对不对。

#### C++ 侧

**(0) 修 §3.3 的 response 截断**（`omni.cpp:10091-10094`）

`response` 和 `chunk_token_ids` 现在会不同步，导致"发了音但文本为空"。两个改法：

- **推荐**：采到 `<|turn_eos|>` 时就不要再把它的 piece 拼进 `response`
  （和 `<|speak|>` 一样在拼接前跳过），这样收尾的 `substr` 截断就永远不会误伤后续文本
- 或者：把收尾清理从"截断到第一个结束 token"改成"删除所有结束 token 子串"，
  和现在处理 `<|speak|>` 的 while 循环保持一致

改完之后 `response` 与 `chunk_token_ids` 一致，(3) 的三态判据和 judge 的
`_is_empty_speak_text()` 才能同时成立。

**(1) 给 `stage_timing.jsonl` 的 t2w 事件补上溯源和时长**（`omni.cpp:9231-9235`）

现在：

```json
{"event":"t2w","wav":"wav_5000.wav","token2wav_ms":300.933,"t2w_queue_wait_ms":0.018,"speak_t2w_acc_ms":300.933}
```

补成：

```json
{"event":"t2w","wav":"wav_5000.wav","src_cnt":5,"n_samples":20160,"sample_rate":24000,
 "duration_ms":840.0,"is_final":false,"token2wav_ms":300.933,"t2w_queue_wait_ms":0.018}
```

`n_samples` 就是 `chunk_wav.size()`，`is_final` 就是 `is_last_window`，都在手边。
有了这三个字段，per-package RTF 直接可算，评测脚本连 wav 文件都不用读。

**(2) `src_cnt` 必须来自入队时刻，不能读全局变量**

`T2WOut.round_idx` 已经在 TTS 线程按入队时的 `simplex_round_idx` 填好了（`omni.cpp:6816`），
T2W 线程也已经解出了 `effective_round_idx`（`omni.cpp:9082`），只是 duplex 分支没用它。
把 `omni.cpp:9176` 的命名和上面的 `src_cnt` 都改成基于 `effective_round_idx`：

```cpp
// 现在（duplex 下 wav_turn_base 是写盘时刻读的全局值，会漂）
std::string wav_path = tts_wav_output_dir + "/wav_" +
                       std::to_string(ctx_omni->wav_turn_base + wav_idx) + ".wav";

// 改为按入队时捕获的轮次编号
const int src_round = (received_round_idx >= 0) ? received_round_idx : ctx_omni->simplex_round_idx;
std::string wav_path = tts_wav_output_dir + "/wav_" +
                       std::to_string(src_round * 1000 + wav_idx) + ".wav";
```

更彻底的做法是把源帧号从 `LLMOut` 一路透传到 `T2WOut`（`LLMOut` 已有 `n_past`，
再加一个 `src_cnt` 即可），这样 TTS 攒批、T2W 合并 drain 都不会丢失溯源。

**(3) 修 `is_speak` 三态**（问题 2）

在 `duplex_do_decode` 里，当 `current_turn_ended && chunk_token_ids.empty()` 时
按 LISTEN 语义上报（判据见问题 2 的表，**不能用 `response.empty()`**）。
最小改法是让这种帧也走 `__IS_LISTEN__` 分支并置 `ended_with_listen`，
这样 `omni.cpp:11631` 和 SSE 的 `is_listen` 两边一起变正确，
judge 侧的 `speak_turn_id` 也就自然分出 4 个轮次。

这里有个协议兼容性的取舍待定：

- 复用 `is_listen`：语义最干净，但改变了 SSE 行为，是破坏性变更
- 新增 `turn_idle` 字段、`is_listen` 保持原样：judge 旧脚本零改动仍能跑，
  新脚本可以区分"真 listen"和"说完了"

注意：**改完必须同时处理孤儿尾包**（问题 4）。帧 30/34 会从 SPEAK 变成 LISTEN，
但它们各自带着一个 `is_final` 尾包 wav。用 (2) 的 `src_cnt` 透传就能自洽 ——
尾包的 `src_cnt` 应该指向真正产出这段音频的那一帧，而不是触发 flush 的那一帧。

**(4) 顺带：跳过 IDLE 帧的无用功**

确认是 IDLE 帧后，`tts_thread_func_duplex` 的
`merge_wav_files()` + TTS KV 清空（`omni.cpp:6876-6891`）可以整段跳过。
实测这 14 帧每帧还额外烧了 190-215ms 的 LLM 采样，共 ~2.8s（见 §3.2）。

#### judge-final 侧

> 注意：`judge-final/` 下的 `.py` 源文件目前已被删除（只剩 `__pycache__/*.pyc`），
> 这三项要等源码恢复后才能动。

**(5) `wav_poller` / `e2e_timing.log_wav` 记录时长**

`collect_wav_output_nowait` 本来就用 `soundfile.read` 读了每个 wav 的 samples，
只是没把 per-file 的 `len(data)` 返回出来。顺手带上即可。

**(6) `_analyze_e2e` 改成三步走**

- 对齐键从 `last_speak_chunk_idx` 换成 **从文件名解析** `cnt = int(wav_id) // 1000`
  （有了 (2) 之后这就是可信的因果溯源），或直接读 `stage_timing.jsonl` 的 `src_cnt`
- 允许一个 chunk 对多个 wav（1:N，N ∈ {0,1,2,…}），不要强行一对一丢数据
- 按 `wav` 字段 join `stage_timing.jsonl`，输出
  `rtf_t2w = token2wav_ms / duration_ms`、
  `rtf_pipeline = (cost_llm_ms + tts_ms + token2wav_ms) / duration_ms`，
  以及分位数

**(7) 加一条硬校验**：`sum(wav duration)` 对 `sum(stage_timing t2w duration_ms)`，
以及 `set(wav files)` 对 `set(stage_timing wav)`。任何一边缺项都说明轮询漏采或
归属漂移，应当直接判该次评测无效，而不是悄悄算出一个偏低的 RTF。

### 5.5 当前算力余量

按 F16 实测，稳态瓶颈是 LLM 线程（prefill 13ms + decode 170-300ms ≈ 0.43s/帧），
远低于 1s 帧间隔；TTS 0.26s、T2W 0.23s 也各自宽裕。真正紧的是**串行延迟**：

```
decode 完成 → TTS(0.26s) → T2W(0.23s) → 落盘   ≈ 0.49s
push → 落盘（含 encode + prefill + decode）    ≈ 0.65~0.92s
```

也就是说 wav 归属正确性的余量只有 **~10%**。选手只要把 TTS 或 T2W 算子改慢一点，
`wav_turn_base` 漂移就会发生。**这正是必须先做 (2) 的原因** ——
它把归属从「靠延迟够小」变成「结构上正确」，评测才敢在慢实现上跑。

---

## 6. 已落地的改动

评测口径以 `~/judge-final` 为准（选手起服务，脚本发请求）；`perf-duplex` 保留作交叉对账。

### 6.1 C++ 侧

| # | 位置 | 改动 |
|---|---|---|
| (0) | `omni.cpp` `duplex_do_decode` 收尾清理 | 控制 token 的清理从「截断到第一个」改成「逐个 erase」。修 §3.3 的吞字：`<\|turn_eos\|>` 之后采到的正文不再被连带砍掉，`response` 与 `chunk_token_ids` 恢复同步 |
| (1) | `omni.cpp` `t2w_thread_func_cpp` | 新增 `wav_base`：duplex 下用 `effective_round_idx * 1000`（**入队时**捕获的帧号），不再读写盘时刻的全局 `ctx_omni->wav_turn_base`。wav 文件名、日志、`generation_done.flag` 全部改用它 |
| (2) | `omni.cpp` t2w 的 `stage_timing.jsonl` | 补 `src_cnt` / `n_samples` / `sample_rate` / `duration_ms` / `is_final` |
| (3) | `omni.cpp` `OmniTtsStageTimer` | tts 事件补 `src_cnt`，让 TTS 耗时能归到帧上 |
| (4) | `omni.h` + `omni.cpp` + `server-omni.cpp` | 新增 IDLE 三态：`omni_context::duplex_frame_idle`、`OmniDuplexFrameResult::is_idle`、`__TURN_IDLE__` marker、SSE 的 `turn_idle` 字段 |
| (5) | `server.cpp` / `ws_handler.cpp` | 三处 SSE/WS handler 补 `__TURN_IDLE__` 分支，否则这个 marker 会被当正文文本发给客户端 |
| (6) | `perf/perf-duplex.cpp` | 统计与 JSON 输出区分 IDLE：`really_speaking = is_speak && !is_idle`，`speak_turn_id` 按此划分，控制台打 `<\|idle\|>` |

IDLE 判据（`omni.cpp` `duplex_do_decode` 收尾）：

```cpp
const bool frame_idle = !ctx_omni->ended_with_listen.load()
                        && !produced_tts_tokens          // 注意：不是 response.empty()
                        && ctx_omni->current_turn_ended;
```

`produced_tts_tokens` 在采样循环里跟着 `chunk_token_ids.push_back` 一起置位，跨本帧所有
chunk 批次累计。**不能用 `response.empty()`** —— §3.3 的帧 14 就是 `response` 为空但
确实产出了 token 并合成了音频。

SSE 的兼容性取舍：`__TURN_IDLE__` 仍然发 `is_listen=false` / `end_of_turn=true`
（和 `__END_OF_TURN__` 一致），只额外带 `turn_idle: true`。老客户端忽略未知字段即可，
零破坏。

改完后 t2w 事件长这样：

```json
{"event":"t2w","wav":"wav_5000.wav","src_cnt":5,"n_samples":20160,"sample_rate":24000,
 "duration_ms":840.000,"is_final":false,"token2wav_ms":300.933,
 "t2w_queue_wait_ms":0.018,"speak_t2w_acc_ms":300.933}
{"event":"tts","chunk_idx":0,"src_cnt":5,"tts_ms":335.596,"speak_tts_acc_ms":335.596}
```

### 6.2 judge-final 侧（`~/judge-final`，RTF 评测口径以此为准）

| # | 文件 | 改动 |
|---|---|---|
| (7) | `omni_client/wav_poller.py` | `collect_wav_output_nowait` 多返回一个 `wav_info`（每个 wav 的 `n_samples`/`sample_rate`/`duration_ms`）。反正这里已经把 wav 读进内存了，顺手带出来，下游不必再读一遍 |
| (8) | `e2e_timing.py` | 新增 `wav_src_cnt()`（从 `wav_{id}.wav` 解析 `id // 1000` 得源帧号）；`log_wav` 记录 `src_cnt` + 时长；`log_chunk` 新增 `turn_idle` 参数，IDLE 帧按 LISTEN 归类（不开新 speak turn、不更新 `last_speak_chunk_idx`），`mode` 上仍单独标 `IDLE` |
| (9) | `omni_client/duplex.py` | `DuplexGenerateResult` 新增 `turn_idle`，从 SSE 解析 |
| (10) | `runner/duplex_eval_runner.py` | 两处 wav 轮询传 `wav_info`；`log_chunk` 传 `turn_idle`；控制台 mode 显示 IDLE |
| (11) | `scripts/eval_duplex_e2e_latency.py` | 新增 `_analyze_rtf()` 算每包 RTF；`report["rtf"]`；对齐键从 `last_speak_chunk_idx` 换成 `src_cnt`（拿不到才退回）；新增 `wav_integrity` 硬校验；`_print_rtf()` 打 RTF；`n_idle` 计数 |
| (12) | `judge_support.py` | `print_latency_summary` 把 RTF 放在最前面打；`_average_rtf()` 支持多视频汇总 |

#### RTF 的两级口径

```
per_package      以单个 wav 为单位。只有 token2wav 能切到这个粒度
                 rtf_t2w = token2wav_ms / duration_ms

per_speak_frame  以 frame 为单位，同一 src_cnt 的所有 wav 合并。
                 TTS 和 LLM 只能算到帧上，所以完整 RTF 只有这一级
                 rtf_t2w      = Σt2w / Σaudio
                 rtf_tts_t2w  = (Σtts + Σt2w) / Σaudio          ← 声码链路
                 rtf_full     = (max(VPM,APM)+prefill+decode + Σtts + Σt2w) / Σaudio

rtf_aggregate_tts_t2w = 总计算耗时 / 总音频时长（不受短尾包权重影响）
```

一帧多包会自动累加，**不依赖 frame↔wav 的 1:1**。

主指标：`report["metric_rtf"] = "rtf.per_speak_frame.rtf_tts_t2w.mean"`。

#### 输出长这样

```
    ★ RTF  speak包=20  speak帧=18  音频17.96s
    每包 token2wav         平均 0.270  (中位 0.230, p95 0.474, min 0.221, max 0.517, n=20)
    ── 按 speak 帧聚合（同帧多包已合并）──
    token2wav              平均 0.259  (中位 0.232, p95 0.358, ..., n=18)
    tts+token2wav          平均 0.505  (中位 0.494, p95 0.581, ..., n=18)
    全链路(含LLM)          平均 0.863  (中位 0.825, p95 1.154, ..., n=18)
    总计算耗时/总音频时长 (tts+t2w)  0.499
```

`run_judge_direct.py` 收尾调的 `print_latency_summary` 也会打同一份 RTF，
多视频时按视频等权平均，aggregate 用总音频重新汇总。

#### 硬校验

`report["wav_integrity"]` 对比「轮询到的 wav 文件名集合」与
「C++ `stage_timing.jsonl` 里 t2w 事件记录的集合」。任一边缺项就
`ok: false` 并在总结里打警告 —— 说明轮询漏采或归属漂移，此时的 RTF 不可信。

### 6.3 构建 / 环境

链接会失败在 Ascend CANN 的一堆 undefined symbol（`rtFree` / `mmGetTid` /
`ascend_private::protobuf::*`）——**这不是代码问题，是没 source 环境**：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd build-kunpeng
cmake --build . --target llama-omni-server llama-omni-perf-duplex -j"$(nproc)"
```

judge-final 的 Python 依赖（本机 pypi 直连不通，用清华源）：

```bash
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    websockets httpx soundfile numpy pillow
```

### 6.4 还没做的

- **未跑真机验证**。上面的数字来自用 F16 实测值构造的离线 smoke test，
  验证的是「链路通、口径对、渲染正常」，不是真实 RTF。
  需要起服务跑一遍 `run_judge_direct.sh`，再和 `perf-duplex` 交叉对账。
- `omni.cpp` 的 IDLE 帧仍然会走一遍 `merge_wav_files()` + 清 TTS KV
  （§5.4 的 (4)），每帧约 190-215ms 的无效 LLM 采样也还在。
  这属于性能优化，不影响 RTF 正确性，暂时没动。
