# Omni Duplex GPU 利用率 Profiling 规划

## 目的

当前 `[DUPLEX_PERF]` 已经能回答“某个阶段耗时多久”和“阶段结束时显存占用多少”，但还不能回答“这段时间 GPU 是否真的忙”。`gpu_used_mb` 是显存占用，不是 SM 利用率；显存高只能说明模型、KV cache、workspace 等占着显存，不能说明 `tts.infer` / `llm.decode` / `t2w.infer` 正在打满 GPU。

这份规划的目标是给 duplex profiling 增加 GPU 使用率维度，帮助判断：

- `tts.infer` 长是否因为 GPU compute 忙，还是 CPU / IO / 同步等待拖慢。
- `llm.prefill`、`llm.decode`、`tts.infer`、`t2w.infer` 是否互相重叠并竞争 GPU。
- 当前 pipeline 是否存在 GPU 空闲但主线程/队列/IO 阻塞的情况。
- 未来优化后，阶段耗时下降是否对应 GPU 利用率变化。

## 设计原则

1. GPU 利用率不要写进阶段计时路径。NVML 查询、文件写入和锁都不能放进 `tts.infer` / `llm.decode` 的热路径里，否则 profiling 会改变被测对象。
2. 显存采样和 GPU 利用率采样要区分。`gpu_used_mb` 可以继续在 `[DUPLEX_PERF]` 中保留，`sm_util_pct` / `mem_util_pct` 通过独立采样流记录。
3. GPU 利用率按时间线离线 join。NVML utilization 是采样窗口统计，不是函数瞬时值；把采样点按 `t_ms` 对齐到阶段 interval 比在 start/end 读一次可靠。
4. 多线程日志必须完整成行。GPU samples、stage start/end 都依赖解析，结构化日志被普通日志插断后，离线重排也恢复不了半行。
5. 阶段间会重叠，GPU 统计不能简单相加。`tts.infer` 和下一轮 `vision.encode` / `llm.prefill` 可能同时运行，同一段 GPU utilization 可能同时落入多个 stage interval。

## 当前缺口

### 已有内容

- `[DUPLEX_PERF]` 记录 `stage`、`event`、`chunk`、`t_ms`、`dur_ms`、`rss_mb`、`gpu_used_mb`、`gpu_total_mb`、`n_past`、`detail`。
- `scripts/analyze_duplex_perf.py` 能解析阶段耗时、生成 CSV、SVG 和报告。
- `tts.infer` 已经独立于 `queue.tts.wait_data`，但内部仍包含 condition 构造、TTS audio token 生成、T2W enqueue，以及可能的 debug dump。

### 缺少内容

- 没有 GPU SM utilization / memory controller utilization / power draw。
- 没有按阶段聚合 GPU busy 情况。
- 没有办法判断 `tts.infer` 长是“GPU 忙”还是“GPU 空闲但 CPU/IO 忙”。
- 结构化日志当前仍可能被多线程输出交织影响。

## 采样方案

### 采样来源

首选 NVML：

- `nvmlDeviceGetUtilizationRates()` 获取 `gpu` 和 `memory` utilization。
- `nvmlDeviceGetMemoryInfo()` 获取 used/free/total memory。
- `nvmlDeviceGetPowerUsage()` 获取功耗，单位换算为 W。
- `nvmlDeviceGetTemperature()` 获取温度。
- `nvmlDeviceGetClockInfo()` 获取 graphics/memory clock。

不要把 `nvidia-smi` 子进程作为默认实现。它可以作为 baseline 对照，但不适合作为程序内 profiling：

- 子进程开销大。
- 输出解析脆弱。
- 刷新粒度和真实采样窗口不稳定。
- 很难和 `omni_perf_now_ms()` 使用同一时钟。

### 采样线程

新增一个后台采样器，例如 `OmniGpuPerfSampler`：

- 在测试开始前启动。
- 使用 `omni_perf_now_ms()` 记录时间戳，和 `[DUPLEX_PERF]` 使用同一个相对时间基准。
- 默认采样间隔建议 `20 ms`，需要更细时可调到 `10 ms`。
- 采样线程只做 NVML 查询和结构化写出，不访问 LLM/TTS context。
- 测试结束、TTS/T2W drain 完成后停止采样线程。

建议环境变量：

- `OMNI_GPU_PROF=1`：启用 GPU utilization profiling。
- `OMNI_GPU_PROF_INTERVAL_MS=20`：设置采样间隔。
- `OMNI_GPU_PROF_FILE=/path/to/gpu.log`：可选，写独立文件；默认跟随 perf 输出。

### 日志格式

建议新增 `[DUPLEX_GPU]` 行，每个 device 每次采样一行：

```text
[DUPLEX_GPU] sample_id=123 t_ms=1024.512 device=0 sm_util_pct=87 mem_util_pct=42 gpu_used_mb=10640 gpu_total_mb=24564 power_w=188.2 temp_c=63 graphics_clock_mhz=1830 mem_clock_mhz=9501
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `sample_id` | 单调递增采样序号，便于稳定排序和排障 |
| `t_ms` | 与 `[DUPLEX_PERF]` 一致的相对时间戳 |
| `device` | GPU device id |
| `sm_util_pct` | NVML `utilization.gpu` |
| `mem_util_pct` | NVML `utilization.memory` |
| `gpu_used_mb` / `gpu_total_mb` | 显存占用 |
| `power_w` | 当前功耗 |
| `temp_c` | GPU 温度 |
| `graphics_clock_mhz` / `mem_clock_mhz` | 当前时钟 |

如果 NVML 不可用，输出一次状态行即可：

```text
[DUPLEX_GPU] event=unavailable reason="nvml_not_found"
```

## 结构化日志写入

在加 GPU utilization 前，建议先修结构化日志输出。当前 `print_with_timestamp()` 先写时间戳，再 `vprintf()` 写正文，多线程下可能交织。

建议新增专用 writer：

```text
omni_perf_write_line(kind, formatted_line)
```

要求：

- 在内存里先拼出完整一行。
- 用专用 mutex 保护一次完整写入。
- 或写独立 perf 文件，使用 append 模式并保证一行一次写入。
- `[DUPLEX_PERF]` 增加 `seq`，`[DUPLEX_GPU]` 使用 `sample_id`。

这样离线分析可以按 `(t_ms, seq/sample_id)` 稳定排序。只靠文档分析时重排时间戳不够，因为重排不能恢复已经被打断的日志行。

## 阶段对齐算法

分析脚本新增 GPU sample 解析后，按以下步骤做 join：

1. 解析 `[DUPLEX_PERF]`，把 start/end 匹配成 stage interval。
2. 解析 `[DUPLEX_GPU]`，按 `device` 分组并按 `t_ms` 排序。
3. 对每个 stage interval `[begin_ms, end_ms]`，选取 `begin_ms <= sample.t_ms <= end_ms` 的 samples。
4. 每个 stage、每个 device 分别计算 GPU 统计。
5. 对没有采样点的短阶段，用最近邻样本估算，并标记 `estimated=1`。

建议输出字段：

```text
stage,device,n_intervals,n_samples,total_ms,avg_sm_util_pct,p50_sm_util_pct,p90_sm_util_pct,max_sm_util_pct,avg_mem_util_pct,max_mem_util_pct,avg_power_w,max_power_w,estimated_samples
```

可以额外计算：

```text
stage_gpu_busy_ms = duration_ms * avg_sm_util_pct / 100
```

这个值只用于观察该 stage 内 GPU 忙闲，不用于跨 stage 求和，因为阶段可能重叠。

## 报告输出

新增文件：

- `omni-duplex-gpu-samples.csv`：原始 GPU 采样点。
- `omni-duplex-stage-gpu-stats.csv`：按 stage/device 聚合的 GPU 利用率。
- `figures/omni-duplex-gpu-utilization.svg`：GPU utilization 时间线。
- 可选 `figures/omni-duplex-stage-gpu.svg`：按 stage 展示 `avg/max SM util`。

报告中新增一个小节：

```text
## GPU 利用率

- `tts.infer`: avg SM util X%, max Y%, avg power Z W。
- `llm.decode`: avg SM util X%, max Y%。
- `llm.prefill`: avg SM util X%, max Y%。
- `t2w.infer`: avg SM util X%, max Y%。
- 低 SM util + 高耗时的阶段优先检查 CPU、IO、队列和同步点。
```

图表建议：

- 上半部分画 `sm_util_pct` 时间线。
- 下半部分用半透明色块叠加 `llm.prefill`、`llm.decode`、`tts.infer`、`t2w.infer` interval。
- 多 GPU 时每张图一条 device lane，避免把不同设备混成一个均值。

## 代码落点

### `tools/omni/omni.cpp`

新增：

- `omni_gpu_prof_enabled()`
- `OmniGpuPerfSampler`
- `omni_perf_write_line()`
- `[DUPLEX_GPU]` 输出格式化逻辑

调整：

- `omni_perf_mark()` 继续记录 `gpu_used_mb`，但结构化输出改成单行原子写。
- 测试入口或 `omni_init()` 创建 sampler。
- 程序退出或测试 drain 完成后停止 sampler。

### `tools/omni/CMakeLists.txt`

两种实现可选：

1. CUDA 构建时直接链接 NVML。
2. 使用 `dlopen("libnvidia-ml.so.1")` 动态加载，避免非 NVIDIA 环境链接失败。

建议优先动态加载，原因是 profiling 是可选能力，不应该影响普通构建。

### `scripts/analyze_duplex_perf.py`

新增：

- `DUPLEX_GPU_RE`
- GPU samples CSV 输出。
- stage interval 与 samples join。
- stage GPU stats CSV 输出。
- GPU utilization SVG。
- report GPU 小结。

## 与 TTS debug dump 的关系

为了避免 debug 写盘污染 `tts.infer`，TTS debug 文件必须被显式开关控制。建议统一使用：

```bash
OMNI_TTS_DEBUG_DUMP=1
```

默认关闭时，以下内容都不应写文件：

- `llm_debug/llm_text.txt`
- `llm_debug/llm_token_ids.txt`
- `llm_debug/llm_hidden_states.*`
- `llm_debug/merged_embeddings.*`
- `audio_tokens_chunk_*.bin`
- `tts_audio_tokens_chunk_*.txt`
- `wav_timing.txt`
- `TTS_SAVE_HIDDEN_STATES_DIR`、`TTS_LOGITS_DEBUG_DIR`、`TTS_OUTPUT_DIR` 触发的 dump

实际 WAV 输出、T2W 结果、功能性 `generation_done.flag` 不属于 debug dump，不应受这个开关影响。

## 与 token 数量的关系

GPU 利用率只能说明硬件忙闲，不能单独解释 `tts.infer` 为什么长。这里需要同时看“生成了多少 token”和“每 token 成本”。

### 当前上下限控制

LLM decode 有三层限制：

- 整次 `stream_decode()` 的总上限是 `max_tgt_len`，来自 `params->n_predict`；如果 `n_predict < 0`，则使用 `n_ctx`。
- 每次送给 TTS 的小段默认收集 `step_size = 10` 个有效 TTS token。这里的“有效”不包含 `<|speak|>`、`<|listen|>`、`<|chunk_eos|>`、`<think>` 等特殊 token。
- 双工模式还有 `max_new_speak_tokens_per_chunk = 26` 的 speak chunk 上限。达到上限时会注入 `<|chunk_eos|>` 并结束当前 decode 调用，方便及时让外层进入下一轮。

LLM decode 没有硬性下限：遇到 `<|listen|>`、`<|chunk_eos|>`、EOS、`break_event`，或者开局 `force_listen_count` 触发时，都可能生成少于 10 个有效 TTS token，甚至不进入正常采样。

TTS audio token 生成也有上下限：

- 双工模式 `max_audio_tokens = 26`，与 Python 的 `max_token_per_chunk = 25 + 1` 对齐。
- 双工非 `end_of_turn` 时 `min_new_tokens = 26`，采样时会屏蔽 EOS，因此通常每个 TTS chunk 生成 26 个 audio token。
- 双工 `end_of_turn` 时 `min_new_tokens = 0`，允许提前 EOS，但仍受 `max_audio_tokens = 26` 限制。
- 单工模式 `max_audio_tokens = 500`，`min_new_tokens = 100`，用于避免 TTS 过早 EOS。
- TTS 输入 token 做了安全检查：`n_tokens < 0` 直接失败，`n_tokens == 0` 只有在 `is_end_of_turn=true` 时允许，`n_tokens > 10000` 视为异常。

T2W 侧不是“生成 token”，而是消费 TTS audio tokens。C++ T2W 使用滑窗，窗口约为 `25 + 3` 个 token；TTS 侧首批推送阈值是 28，后续是 25，并在 chunk 结束时把剩余 token 发给 T2W。

### 当前已有统计

已有一部分 token 数写进 `[DUPLEX_PERF] detail`：

- `llm.decode` end：已有 `tokens`、`valid_tts_tokens`、`n_past_delta`。这是单个 LLM decode 小段的统计。
- `api.llm_decode_loop` end：已有整次 `stream_decode()` 的 `tokens`、`valid_tts_tokens`。
- `queue.tts.wait_space` / `queue.tts.enqueue`：已有推给 TTS 的 `tokens`、`llm_finish`、`text_bytes`。
- `tts.infer` start/end：已有 `tts_chunk`、`llm_tokens`、`is_end_of_turn`，flush 路径还有 `flush_only=1`。
- `queue.t2w.enqueue`：已有推给 T2W 的 audio token `tokens`、`is_final`、`is_chunk_end`。
- `t2w.infer`：已有 `window_tokens`、`is_last`，end 里还有 `wav_samples`。

但当前统计还不完整：

- `tts.infer` detail 里的 `llm_tokens` 是输入 LLM token 数，不是 TTS 生成的 audio token 数。
- `tts.infer` 没有直接记录 `condition_tokens`，也没有直接记录 `audio_tokens_generated`。
- `queue.t2w.enqueue` 的 `tokens` 可以间接反映 TTS 输出给 T2W 的 token 数，但它和 `tts.infer` 之间缺少稳定的 `utterance_id` / `audio_chunk_id`，跨线程重叠时不能可靠归因。
- `scripts/analyze_duplex_perf.py` 目前主要聚合 `dur_ms`，没有把 detail 里的 token 字段解析成 CSV/报告统计。

### 建议补充

后续建议把 token 数规范化写入 detail，并让分析脚本解析：

- `llm.decode`: `tokens`、`valid_tts_tokens`、`n_past_delta`。
- `api.llm_decode_loop`: `tokens`、`valid_tts_tokens`。
- `tts.infer`: `llm_tokens`、`filtered_llm_tokens`、`condition_tokens`、`audio_tokens_generated`、`is_end_of_turn`、`flush_only`。
- `queue.t2w.enqueue`: `audio_tokens`、`is_final`、`is_chunk_end`、`audio_chunk_id`。
- `t2w.infer`: `window_tokens`、`wav_samples`、`is_last`。

这样可以同时看：

- 每阶段耗时。
- GPU 忙闲。
- 输入/输出 token 规模。
- 每 token 成本。

## 实施顺序

1. 修结构化日志单行原子性，确保 `[DUPLEX_PERF]` 不再被插断。
2. 用 `OMNI_TTS_DEBUG_DUMP` 关掉默认 TTS debug dump，先让 `tts.infer` 计时更干净。
3. 加 NVML 采样线程，只输出 `[DUPLEX_GPU]` 原始 samples。
4. 跑短测试，对比 `nvidia-smi dmon` 趋势，确认采样值数量级正确。
5. 在分析脚本里输出 `omni-duplex-gpu-samples.csv`。
6. 实现 stage interval join，输出 `omni-duplex-stage-gpu-stats.csv`。
7. 增加 GPU utilization SVG 和报告小结。

## 验证计划

### 单元级验证

- `OMNI_GPU_PROF` 未设置时不输出 `[DUPLEX_GPU]`。
- NVML 不存在时程序继续运行，只记录 unavailable 状态。
- 多 GPU 环境下每个 device 都有样本。
- 采样线程 start/stop 无泄漏、无 join hang。

### 集成验证

- 跑 2-3 个 duplex chunk，确认 `[DUPLEX_PERF]` 和 `[DUPLEX_GPU]` 都能完整解析。
- 对比 `nvidia-smi dmon` 或 `nvidia-smi --query-gpu=utilization.gpu,power.draw --loop-ms=100`，确认曲线趋势一致。
- 打开/关闭 `OMNI_TTS_DEBUG_DUMP` 各跑一次，确认默认关闭时不生成 TTS debug dump。
- 分别用 `10 ms` 和 `20 ms` 采样间隔跑一次，检查 stage GPU 结论是否稳定。

### 结果判读

- `tts.infer` 高耗时 + 高 SM util：主要是 TTS 模型生成成本。
- `tts.infer` 高耗时 + 低 SM util：优先查 CPU projector、采样、debug IO、锁或 T2W enqueue。
- `llm.decode` 高耗时 + 低 SM util：优先查采样/同步/队列，而不是 GPU kernel。
- `queue.*` 高耗时 + GPU 空闲：pipeline 背压或线程调度问题。

## 风险和注意事项

- NVML utilization 有采样窗口，不是 kernel 级精确 profiling；如果要看 kernel，需要 Nsight Systems/Compute。
- 阶段 interval 过短时，样本可能不足，需要 `estimated` 标记。
- 多阶段重叠时，不能把各阶段 `busy_ms` 相加当作总 GPU 时间。
- 采样频率过高会增加 CPU 开销；建议默认 `20 ms`，排障时再降到 `10 ms`。
- 结构化日志如果仍通过普通 stdout 混写，解析质量会限制 GPU 分析质量。
