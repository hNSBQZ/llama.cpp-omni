# LLM Token/Hidden State 到 TTS 的数据流

本文从 `tools/omni/omni.cpp` 里的 `tts_thread_func` 视角梳理：LLM 生成的 token 和 hidden state 如何被收集、打包、变换成 TTS condition embedding，并最终驱动 TTS 模型生成 audio tokens，再交给 Token2Wav 生成 WAV。

说明：

- 主路径以单工模式 `tts_thread_func` 为主。
- 双工模式 `tts_thread_func_duplex` 复用相同的核心变换思路，但调用的是 `generate_audio_tokens_local`，并额外依赖 `is_end_of_turn` 控制 chunk/turn 结束。
- 这里的 “hidden state” 实际来自 llama.cpp embeddings 输出：采样出 token 后，再把该 token 喂回 LLM forward，并通过 `llama_get_embeddings()` 取出对应输出。

## 1. LLM 端生成 token 并收集 hidden state

入口在 `stream_decode` 的生成循环中。

关键函数链：

- `stream_decode`
- `llama_loop_with_hidden_and_token`
- `sample_with_hidden_and_token`
- `eval_id_with_hidden`
- `eval_tokens_with_hidden`

处理过程：

1. `stream_decode` 设置 LLM chunk 大小 `step_size = 10`，表示每个 chunk 默认收集 10 个对 TTS 有效的 LLM token。
2. 每次循环调用 `llama_loop_with_hidden_and_token`：
   - `sample_with_hidden_and_token` 先从 LLM logits 采样出一个 `sampled_token`。
   - 它把 `sampled_token` 转成文本片段 `tmp`。
   - 然后调用 `eval_id_with_hidden`，把这个 token 再喂回 LLM。
3. `eval_tokens_with_hidden` 做真正的 hidden state 采集：
   - 调 `llama_set_embeddings(ctx_llama, true)` 打开 embeddings 输出。
   - 用 `llama_decode` forward 当前 token。
   - 用 `llama_get_embeddings(ctx_llama)` 取当前 batch 的 embeddings。
   - 拷贝到 `hidden_states`，形状是 `[N, llm_n_embd]`。
4. `stream_decode` 只把 `is_valid_tts_token(sampled_token)` 通过的 token 收集进当前 chunk：
   - `chunk_token_ids.push_back(sampled_token)`
   - `chunk_hidden_states.insert(... hidden_states ... hidden_states + llm_n_embd)`
   - `jl++`，也就是只有有效 token 才计入 `step_size`。

这里有一个对齐点很重要：

- Python 注释里提到过 `forward(T_{i-1}) -> H_{i-1} -> sample T_i` 的延迟关系。
- 当前 C++ 路径是 `sample T_i -> forward(T_i) -> collect H_i`。
- 因此 `chunk_token_ids[i]` 和 `chunk_hidden_states[i]` 是直接一一对应的，不需要再做延迟偏移。

## 2. 过滤：哪些 LLM token 会进入 TTS

过滤相关函数：

- `is_valid_tts_token`
- `filter_special_tokens`

LLM 端先过滤一次，TTS 端在生成 condition 前又过滤一次。

`is_valid_tts_token` 会排除：

- 显式特殊 token：`<think>`、`</think>`、`<|tts_eos|>`、`<|speak|>`、`<|listen|>`、`<|chunk_eos|>`、`<|chunk_tts_eos|>`、`<|turn_eos|>`、保留换行 token 等。
- `tid >= 150000` 的 token，通常是 `<|im_end|>`、`<|tts_bos|>` 等特殊 token。
- `g_known_empty_token_ids` 中确认过的空 token，目前该集合为空。

`filter_special_tokens` 会同步过滤 `token_ids` 和 `hidden_states`，保证过滤后仍满足：

- `filtered_hidden_states.size() == filtered_token_ids.size() * n_embd`

## 3. LLMOut：把一个 LLM chunk 送到 TTS 线程

相关结构和队列：

- `struct LLMOut`
- `struct TTSThreadInfo`
- `ctx_omni->tts_thread_info->queue`

`stream_decode` 在 async + use_tts 时创建 `LLMOut`：

- `text`：当前 chunk 的文本片段，主要用于 debug/UI。
- `n_past`：LLM 当前上下文位置。
- `llm_finish`：当前 LLM 生成是否结束。
- `debug_dir`：调试输出目录。
- `token_ids`：当前 chunk 的有效 LLM token IDs。
- `hidden_states`：当前 chunk 对应 hidden state，flatten 后形状 `[n_tokens, llm_n_embd]`。
- `n_embd`：LLM hidden size，一般是 4096。
- `is_end_of_turn`：双工模式传递给 TTS，用于区分 chunk 结束和 turn 结束。

然后它等待 TTS 队列有空位，并把 `LLMOut*` push 到 `tts_thread_info->queue`，通知 `tts_thread_func`。

## 4. tts_thread_func 消费 LLM chunk

`tts_thread_func` 的核心消费逻辑：

1. 等待 `tts_thread_info->queue`。
2. 每次只 pop 一个 `LLMOut`。
3. 拷贝这一条 chunk 的：
   - `llm_text`
   - `current_chunk_token_ids`
   - `current_chunk_hidden_states`
   - `current_chunk_n_embd`
4. 做 UTF-8 不完整字节处理，避免文本片段拆断多字节字符。
5. 如果 `response.empty() && llm_finish`，说明 LLM 已结束但没有新 chunk，需要 flush 剩余 `tts_token_buffer`，并发送 `is_final=true` 给 Token2Wav。

注意：TTS condition 的核心输入不是 `response` 字符串，而是 `current_chunk_token_ids + current_chunk_hidden_states`。`response` 更多用于日志、文本输出和调试文件。

## 5. 从 LLM token/hidden state 到 TTS condition embedding

这是 `tts_thread_func` 里最关键的 hidden-text-merge 流程。

相关函数：

- `filter_special_tokens`
- `tts_emb_text`
- `tts_projector_semantic`
- `normalize_l2_per_token`
- `generate_audio_tokens_local_simplex`

输入：

- `filtered_token_ids`，长度 `n_tokens_filtered`。
- `filtered_hidden_states`，形状 `[n_tokens_filtered, current_chunk_n_embd]`。
- `current_chunk_n_embd`，LLM hidden size，一般是 4096。
- `tts_n_embd`，TTS hidden size，一般是 768。

变换步骤：

1. 用 TTS 的文本 embedding 表处理 LLM token：
   - 函数：`tts_emb_text`
   - 输入：`filtered_token_ids[i]`
   - 权重：`ctx_omni->emb_text_weight`
   - 输出：`llm_embeds[i]`
   - 形状：`[n_tokens_filtered, tts_n_embd]`

2. 用 TTS 的 semantic projector 投影 LLM hidden state：
   - 函数：`tts_projector_semantic`
   - 输入：`filtered_hidden_states`
   - 输出：`projected_hidden`
   - 形状：从 `[n_tokens_filtered, llm_n_embd]` 变成 `[n_tokens_filtered, tts_n_embd]`
   - 优先走 `projector_forward(ctx_omni->projector, ...)`。
   - fallback 是两层 MLP：
     - `linear1 + bias1`
     - `ReLU`
     - `linear2 + bias2`

3. 对投影后的 hidden 做逐 token L2 归一化：
   - 函数：`normalize_l2_per_token`
   - 对齐 Python 的 `F.normalize(hidden_embeds, p=2, dim=-1)`。

4. 合并 token embedding 和 projected hidden：
   - `merged_embeddings[i] = llm_embeds[i] + projected_hidden[i]`
   - 输出形状：`[n_tokens_filtered, tts_n_embd]`
   - 这就是送入 TTS 的文本条件 embedding。

5. 保存调试文件：
   - `llm_text.txt`
   - `llm_token_ids.txt`
   - `llm_hidden_states.bin`
   - `llm_hidden_states.txt`
   - `llm_embeds_cpp.txt`
   - `projected_hidden_before_norm_cpp.txt`
   - `projected_hidden_after_norm_cpp.txt`
   - `merged_embeddings.bin`
   - `merged_embeddings.txt`

## 6. TTS condition 进入 TTS 模型

单工主路径调用：

- `generate_audio_tokens_local_simplex`

该函数接收：

- `merged_embeddings`
- `n_tokens_filtered`
- `tts_n_embd`
- `chunk_idx`
- `is_final_text_chunk`

内部处理：

1. 复制 `merged_embeddings` 到 `condition_with_bos`。
2. 用 `tts_emb_text(audio_bos_token_id)` 取 `audio_bos` embedding。
3. 把 `audio_bos` append 到 condition 尾部。
4. 保存到 `ctx_omni->tts_condition_embeddings`，用于首个 audio token 的 re-forward。
5. 如果 `chunk_idx == 0`：
   - 清空 TTS KV cache。
   - 清空 `tts_all_generated_tokens`。
   - 清空 `tts_token_buffer`。
6. 如果不是第一个 chunk：
   - 使用 `ctx_omni->tts_n_past_accumulated` 继续 TTS KV cache。
7. 调 `prefill_with_emb_tts`，把 condition 作为 embedding 输入 TTS 模型。

`prefill_with_emb_tts` 的要点：

- 构造 `llama_batch`。
- 使用 `batch.embd = embed + i * n_embd`，也就是直接喂 embedding，不走 token id。
- 设置 position，从 `text_start_pos` 开始。
- 打开 TTS embeddings 输出：`llama_set_embeddings(ctx_tts_llama, true)`。
- 调 `llama_decode(ctx_tts_llama, batch)`。
- 更新 `n_past_tts`。

到这里，LLM 的 token/hidden state 已经变成 TTS 模型 KV cache 中的一段 condition。

## 7. TTS 模型生成 audio token

相关函数：

- `sample_tts_token_simplex`
- `prefill_with_emb_tts`
- `random_sampling_tts`

`generate_audio_tokens_local_simplex` 进入 Phase 1 生成循环，最多生成 `max_audio_tokens = 500` 个 audio token。

每一步：

1. 调 `sample_tts_token_simplex`。
2. `sample_tts_token_simplex` 从 TTS 模型当前最后位置取 hidden：
   - `llama_get_embeddings_ith(ctx_tts_llama, -1)`
3. 用 `head_code_weight` 把 hidden 投影到 audio token logits：
   - 输入 hidden size：768。
   - 输出 audio vocab：6562。
4. 采样前处理：
   - temperature 固定为 0.8。
   - repetition penalty 固定为 1.05。
   - 只看最近 8 个 audio tokens。
   - `force_no_eos` 时把 EOS relative id 6561 置为 `-inf`。
5. 用 `random_sampling_tts` 采样 relative audio token id。
6. 转成 absolute audio token：
   - `absolute_id = audio_bos_token_id + relative_idx`
   - `audio_bos_token_id = 151687`
7. 如果不是 EOS：
   - relative id 加入 `output_audio_tokens`。
   - absolute id 加入 `ctx_omni->tts_all_generated_tokens`。
   - relative id 加入 `ctx_omni->tts_token_buffer`。
8. 为了让 TTS 继续生成下一个 token，使用 `emb_code_weight` 把刚采样的 audio token relative id 转为 embedding，再调用 `prefill_with_emb_tts` 喂回 TTS KV cache。

EOS 处理：

- Phase 1 采样到 EOS 时不会把 EOS 放入输出 token。
- 如果当前是最后一个 text chunk，则进入 Phase 2。
- Phase 2 会先注入 `text_eos_embed`，再注入 `audio_bos_embed`，然后继续生成最后一批 audio tokens。
- Phase 2 结束或最终 flush 时，剩余不足 25 个的 tokens 也会送到 Token2Wav。

## 8. audio token 到 Token2Wav

相关结构和函数：

- `struct T2WOut`
- `generate_audio_tokens_local_simplex`
- `t2w_thread_func`
- `t2w_thread_func_python`
- `t2w_thread_func_cpp`
- `process_python_t2w_tokens`
- `Token2WavSession::feed_window`

`T2WOut` 携带：

- `audio_tokens`：audio token relative ids。
- `is_final`：是否是 turn 结束。
- `is_chunk_end`：是否是 TTS chunk 结束但非 turn 结束。
- `round_idx`：单工模式下用于输出目录同步。

TTS 侧发送 audio token 的规则：

- `ctx_omni->tts_token_buffer` 满 25 个 token 时，创建 `T2WOut` 推给 `t2w_thread_info->queue`。
- 最后一个 text chunk 结束时，如果 buffer 里还有不足 25 个 token，也会 flush。
- `llm_finish` 时额外发送一个 `is_final=true` 的空 `T2WOut`，用于通知 Token2Wav flush/reset 并写 `generation_done.flag`。

Token2Wav 侧：

1. `t2w_thread_func` 根据配置分发到 Python 或 C++ 实现。
2. 两个实现都维护 token 滑窗：
   - 初始 buffer：`[4218, 4218, 4218]`，作为 3 个静音前缀。
   - `CHUNK_SIZE = 25`
   - `PRE_LOOKAHEAD = 3`
   - `WINDOW_SIZE = 28`
3. 收到 `T2WOut` 后把新 token append 到 `token_buffer`。
4. 当 buffer 达到窗口大小，或 final/chunk_end 需要 flush 时，处理一个 window。
5. Python 路径调用 `process_python_t2w_tokens`。
6. C++ 路径调用 `ctx_omni->token2wav_session->feed_window(window, is_last_window, chunk_wav)`。
7. 输出 WAV 到当前 round 的 `tts_wav` 目录。
8. final 时写入 `generation_done.flag`，并重置 token buffer。

## 9. 一句话总览

整体链路可以压缩成：

```text
LLM logits
  -> sample_with_hidden_and_token 采样 LLM token
  -> eval_tokens_with_hidden forward 该 token 并取 LLM hidden state
  -> stream_decode 收集有效 token_ids + hidden_states
  -> LLMOut 入 TTS 队列
  -> tts_thread_func 逐 chunk 消费
  -> filter_special_tokens 二次过滤
  -> tts_emb_text(token_ids) 得到 TTS 文本 embedding
  -> tts_projector_semantic(hidden_states) 投影到 TTS hidden size
  -> normalize_l2_per_token
  -> merged_embeddings = emb_text + normalized_projected_hidden
  -> generate_audio_tokens_local_simplex
  -> prefill_with_emb_tts 把 condition embedding 喂进 TTS
  -> sample_tts_token_simplex 用 TTS hidden + head_code 采样 audio token
  -> emb_code(audio token) 回灌 TTS，循环生成
  -> T2WOut 推给 Token2Wav
  -> Token2Wav 滑窗生成 WAV
```

## 10. 关键数据形状

- LLM chunk token IDs：`[n_tokens]`
- LLM hidden states：`[n_tokens, llm_n_embd]`，常见 `llm_n_embd = 4096`
- TTS text embeddings：`[n_tokens_filtered, tts_n_embd]`，常见 `tts_n_embd = 768`
- projected hidden：`[n_tokens_filtered, tts_n_embd]`
- merged embeddings：`[n_tokens_filtered, tts_n_embd]`
- TTS condition with audio_bos：`[n_tokens_filtered + 1, tts_n_embd]`
- TTS audio logits：`[6562]`
- TTS 输出给 Token2Wav 的 audio token：relative id，范围 `[0, 6561]`

## 11. 需要特别注意的点

- `tts_thread_func` 当前主路径不会再 tokenize `response` 来构建 TTS 输入，而是使用 LLM 原始 token id 和 hidden state。
- LLM 端已经过滤 TTS 无效 token，TTS 端又用 `filter_special_tokens` 过滤一次，用来防止特殊 token 进入 `emb_text + projector`。
- `merged_embeddings` 不包含 `audio_bos`；`audio_bos` 是在 `generate_audio_tokens_local_simplex` 里 prefill 前动态追加的。
- 单工模式第一 chunk 会清 TTS KV cache，后续 chunk 保持 TTS KV cache 连续。
- audio token 对 TTS 模型内部是 absolute id，但送 Token2Wav 时使用 relative id。
- `llm_finish` 不只表示没有更多文本，也负责触发 TTS token buffer flush、`is_final=true` 通知和 round 目录切换。

## 12. ggml 融合优化计划

目标不是把 “LLM 采样 -> TTS 采样 -> Token2Wav” 全部塞进一个静态图，而是先把 `tts_thread_func` 里最稳定、最适合图化的 condition 构建阶段融合掉：

```text
filtered_token_ids + filtered_hidden_states
  -> emb_text 查表
  -> projector_semantic
  -> L2 normalize
  -> add
  -> merged_embeddings
```

这段目前跨了 `tts_emb_text`、`tts_projector_semantic`、`normalize_l2_per_token` 和手写 merge loop。它没有采样分支，也不更新 KV cache，是最适合先合成 ggml 图的部分。

### 12.1 建议新增的数据结构

建议在 `omni.h` 里新增一个专门管理 TTS condition 图的结构，和现有 `projector_model` 类似，但多持有 `emb_text` 权重 tensor：

```cpp
struct tts_condition_graph_model {
    int32_t llm_hidden_dim = 4096;
    int32_t tts_hidden_dim = 768;
    int32_t text_vocab_size = 152064;

    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buf_type = nullptr;

    struct ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;

    struct ggml_tensor * emb_text_weight = nullptr;      // [tts_hidden_dim, text_vocab_size]
    struct ggml_tensor * linear1_weight = nullptr;       // [llm_hidden_dim, tts_hidden_dim]
    struct ggml_tensor * linear1_bias = nullptr;         // [tts_hidden_dim]
    struct ggml_tensor * linear2_weight = nullptr;       // [tts_hidden_dim, tts_hidden_dim]
    struct ggml_tensor * linear2_bias = nullptr;         // [tts_hidden_dim]

    bool initialized = false;
};
```

如果短期不想调整权重加载，也可以先不复制权重，直接复用现有的 `ctx_omni->projector` 和 `ctx_omni->emb_text_weight`，只把计算图函数独立出来。长期更干净的做法是把 `emb_text.weight` 也作为 backend tensor 常驻，避免每个 chunk 从 CPU 查表再拷贝到后端。

### 12.2 建议新增函数头

第一阶段只做 condition 构建，不碰 TTS decode/KV cache：

```cpp
bool tts_condition_graph_init(
    struct omni_context * ctx_omni,
    const char * tts_model_path,
    bool use_cuda
);

void tts_condition_graph_free(
    struct omni_context * ctx_omni
);

bool tts_condition_graph_forward(
    struct omni_context * ctx_omni,
    const llama_token * token_ids,
    const float * llm_hidden_states,
    int n_tokens,
    int llm_n_embd,
    std::vector<float> & merged_embeddings,
    int & tts_n_embd
);
```

其中 `tts_condition_graph_forward` 对外语义等价于当前这几步：

```cpp
tts_emb_text(...);
tts_projector_semantic(...);
normalize_l2_per_token(...);
merged_embeddings[i] = llm_embeds[i] + projected_hidden[i];
```

第二阶段可以把 `audio_bos` 也放进同一个输出 buffer，但建议先保留在 `generate_audio_tokens_local_simplex` 里动态追加，降低改动风险：

```cpp
bool tts_condition_graph_forward_with_bos(
    struct omni_context * ctx_omni,
    const llama_token * token_ids,
    const float * llm_hidden_states,
    int n_tokens,
    int llm_n_embd,
    llama_token audio_bos_token_id,
    std::vector<float> & condition_with_bos,
    int & tts_n_embd
);
```

第三阶段再考虑 TTS audio token 采样里的局部图化，把 `head_code` logits 和 `emb_code` 查表从 CPU loop 改成 ggml：

```cpp
bool tts_audio_logits_graph_forward(
    struct omni_context * ctx_omni,
    const float * tts_last_hidden,
    std::vector<float> & audio_logits
);

bool tts_emb_code_graph_forward(
    struct omni_context * ctx_omni,
    int relative_audio_token_id,
    std::vector<float> & audio_token_embedding
);
```

这两个函数不要和 condition graph 混在第一阶段做，因为它们处在逐 audio token 采样循环里，控制流、EOS、repetition penalty 都更动态。

### 12.3 建议的 ggml 图内部算子

`tts_condition_graph_forward` 的图可以按下面构建：

```text
token_ids_i32
  -> ggml_get_rows(emb_text_weight, token_ids_i32)
     = llm_embeds [tts_hidden_dim, n_tokens]

llm_hidden_f32 [llm_hidden_dim, n_tokens]
  -> ggml_mul_mat(linear1_weight, llm_hidden_f32)
  -> ggml_add(linear1_bias)
  -> ggml_relu
  -> ggml_mul_mat(linear2_weight)
  -> ggml_add(linear2_bias)
     = projected_hidden [tts_hidden_dim, n_tokens]

projected_hidden
  -> per-token L2 normalize
     = normalized_projected_hidden

llm_embeds + normalized_projected_hidden
  -> merged_embeddings [tts_hidden_dim, n_tokens]
```

注意点：

- 当前内存里很多地方把形状描述为 `[n_tokens, hidden]`，ggml tensor 通常会写成 `[hidden, n_tokens]`，需要统一 stride/转置理解。
- `emb_text_weight` 现在在 C++ 侧是 `float*`，布局按 `emb_text_weight[token_id * tts_n_embd + j]` 访问；如果迁移成 ggml tensor，最好按 `ne[0] = tts_hidden_dim, ne[1] = vocab_size` 存储，方便 `ggml_get_rows`。
- L2 normalize 如果没有现成完全匹配的 ggml op，可以先用 `ggml_sqr -> ggml_sum_rows -> ggml_sqrt -> ggml_repeat -> ggml_div` 拼图；如果后端支持不理想，再保留 `normalize_l2_per_token` 作为阶段性 fallback。

### 12.4 替换点

第一阶段替换 `tts_thread_func` 中这段逻辑：

```text
for token -> tts_emb_text -> llm_embeds
tts_projector_semantic -> projected_hidden
normalize_l2_per_token
for i -> merged_embeddings[i] = llm_embeds[i] + projected_hidden[i]
```

替换后主路径变成：

```cpp
std::vector<float> merged_embeddings;
int tts_n_embd = 0;

bool merged_success = tts_condition_graph_forward(
    ctx_omni,
    filtered_token_ids.data(),
    filtered_hidden_states.data(),
    n_tokens_filtered,
    current_chunk_n_embd,
    merged_embeddings,
    tts_n_embd
);
```

`merged_embeddings` 后续仍然传给现有：

```cpp
generate_audio_tokens_local_simplex(
    ctx_omni,
    params,
    merged_embeddings,
    n_tokens_filtered,
    tts_n_embd,
    current_chunk_idx,
    audio_tokens,
    tts_wav_output_dir,
    is_final_text_chunk
);
```

这样第一阶段不会影响 TTS KV cache、audio token 采样、Token2Wav，也不会改变 `LLMOut` 队列协议。

### 12.5 分阶段计划

第一阶段：只融合 condition 构建。

- 新增 `tts_condition_graph_forward`。
- 先复用现有 projector ggml 权重，`emb_text` 可以临时保留 CPU 查表作为 baseline 对照。
- 输出与旧路径逐元素比较，误差阈值建议先看 `max_abs_diff < 1e-4`。
- 通过后再把 `emb_text` 查表迁移到 ggml tensor。

第二阶段：减少中间 buffer 和 host 往返。

- 把 `emb_text.weight` 常驻 backend。
- `projected_hidden`、`normalized_projected_hidden`、`llm_embeds` 不再落 CPU，只取最终 `merged_embeddings`。
- Debug 文件需求通过开关保留旧路径或额外 readback。

第三阶段：优化 TTS audio token 循环热点。

- 把 `head_code` logits 从手写 CPU matmul 改成 ggml 图。
- 把 `emb_code` 查表改成 ggml get_rows。
- sampling、EOS、repetition penalty 仍保留 C++ 控制流。

### 12.6 不建议图化的边界

这些部分不建议塞进同一个 ggml 图：

- LLM token 采样和 `common_sampler_accept`。
- `is_valid_tts_token` / `filter_special_tokens` 带来的动态长度过滤。
- `LLMOut` 队列和 TTS 线程同步。
- TTS KV cache reset/continue。
- audio token 逐步采样、EOS、Phase 2 注入 `text_eos_embed`。
- Token2Wav 滑窗、WAV 写文件和 `generation_done.flag`。

这些是控制流和状态机，不是稳定的张量算子链。保持它们在 C++ 里，先把纯张量计算段图化，风险更低。
