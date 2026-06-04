可以参考/cache/hanqingzhe/llama.cpp-omni/docs/development/omni-duplex-performance-analysis.md这个文档是对双工的分析，必要的也结合具体代码分析
现在要从tools/omni/test/test-duplex.cpp的双工测试开始进行性能分析
我现在也要对双工的各段进行分析
1.分好阶段，现在最大的问题是大块的几个函数写的太屎山了，但我也没权限重构，各个阶段完全没分清，现在得细分一下我感觉应该有初始音频载入；whisper音频处理，clip音频处理，但这俩应该可以和在encode阶段去看他的profiling；还有llm的prefill，这个我感觉主体是llmfuncthread但还有streamdecode还有streamprefilling的各部分。TTS分为llm token+hiddenstate-tts token还有tts token-mel mel-pcm byte，这些边界都有划分出来
2.每个阶段我需要内存，gpu显存占用，开始结束时间
3.因为各阶段存在并行，所以必须明确打好时间表，后面可以写个python的分析脚本，用来画出个阶段的时序图，这个python的活先不做，先能让cpp代码各阶段输出明白

## 实现总结

本次改动从 `tools/omni/test/test-duplex.cpp` 的双工测试入口开始，新增了统一的 `[DUPLEX_PERF]` 性能日志输出。每条日志会带上阶段名、事件类型、chunk index、相对开始时间、阶段耗时、进程 RSS、CUDA 显存占用、`n_past` 和补充 detail，方便后续脚本按时间线重建各阶段并行关系。

主要代码改动：

1. 在 `tools/omni/omni.cpp` 中新增 `omni_perf_mark()` 和 `OmniPerfScope`，统一负责性能打点、耗时统计、RSS 读取和 CUDA 显存读取。
2. 在 `tools/omni/omni.h` 中新增 `perf_current_chunk_index`，用于跨 LLM/TTS/T2W 异步线程关联当前双工 chunk；同时导出 `omni_perf_mark()` 方便测试入口打点。
3. 在 `tools/omni/test/test-duplex.cpp` 中为每个 chunk 设置当前 index，并在外层 `stream_prefill()`、`stream_decode()` 调用前后输出 API 级别的开始/结束日志。
4. 在 `stream_prefill()` 内部细分 system/ref audio 初始化、vision encode、vision KV prefill、audio encode、audio KV prefill、异步 LLM queue enqueue 等阶段。
5. 在 `llm_thread_func()` 中为异步 LLM KV prefill 打点，记录实际写入 KV cache 的耗时和 `n_past` 增量，避免只看到测试外层 prefill 的入队时间。
6. 在 `stream_decode()` 中细分等待异步 prefill、force listen、LLM sampling、first token、TTS enqueue 等阶段。
7. 在 TTS 链路中区分 `llm token/hidden state -> audio token` 和 `audio token -> wav` 两段；Python Token2Wav 和 C++ Token2Wav 都会输出 `t2w.tokens_to_wav` 阶段日志。

阶段边界目前覆盖：

- `test.chunk_prefill_api`：测试入口外层 prefill API 耗时。
- `stream_prefill`：单个 chunk 的 prefill 总入口。
- `session.ref_audio_encode` / `session.ref_audio_kv_prefill`：会话初始化参考音频相关阶段。
- `prefill.vision_encode` / `prefill.vision_kv_prefill`：图像编码和视觉 embedding 写入 KV。
- `prefill.audio_encode` / `prefill.audio_kv_prefill`：音频编码和音频 embedding 写入 KV。
- `prefill.enqueue_llm`：异步模式下将 embedding 放入 LLM 线程队列。
- `llm.kv_prefill`：LLM 线程里真实执行 KV prefill 的阶段。
- `test.chunk_decode_api` / `stream_decode`：测试入口和 decode 内部总耗时。
- `decode.wait_llm_prefill`：decode 等待异步 LLM prefill 完成的时间。
- `decode.llm_sample` / `decode.first_token`：LLM 采样总耗时和首 token 延迟。
- `tts.enqueue_from_llm`：LLM 输出 token 入 TTS 队列。
- `tts.llm_to_audio_tokens`：文本 token/hidden state 转 TTS audio token。
- `t2w.tokens_to_wav`：TTS audio token 转 wav/pcm。

## 编译和运行

编译双工测试目标：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --target llama-omni-test-duplex -j"$(nproc)"
```

快速验证可以先关闭 TTS，只看 prefill/decode 主链路：

```bash
./build/bin/llama-omni-test-duplex \
  -m /path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --test /path/to/chunks/audio_test_case_ 3 \
  --ref-audio /path/to/ref_audio.wav \
  --no-tts \
  -ngl 99 \
  -c 4096 \
  2>&1 | tee duplex_perf.log
```

需要验证 TTS/T2W 阶段时去掉 `--no-tts`：

```bash
./build/bin/llama-omni-test-duplex \
  -m /path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --test /path/to/chunks/audio_test_case_ 3 \
  --ref-audio /path/to/ref_audio.wav \
  -ngl 99 \
  -c 4096 \
  -o ./tools/omni/output \
  2>&1 | tee duplex_perf.log
```

查看性能日志：

```bash
rg '\[DUPLEX_PERF\]' duplex_perf.log
```