# GGUF Read-To-Model Load Chain

这份文档整理了 `llama.cpp-omni` 中，从读取 GGUF 文件到把权重和元信息放入 `llama_model` 结构，再到后续构图时使用这些字段的主要调用链。

目标是帮助定位 AWQ 接入时最可能需要插手的位置。

## 1. 外部入口

通常外部工具或服务会调用：

- `llama_model_load_from_file(const char * path_model, llama_model_params params)`
- 位置：`src/llama.cpp`

它继续调用：

- `llama_model_load_from_file_impl(...)`
- `llama_model_load(...)`

其中真正的统一模型加载逻辑在：

- `static int llama_model_load(const std::string & fname, std::vector<std::string> & splits, llama_model & model, llama_model_params & params)`
- 位置：`src/llama.cpp`

## 2. 构造 GGUF 加载器

在 `llama_model_load(...)` 中，会先构造：

- `llama_model_loader ml(fname, splits, params.use_mmap, params.check_tensors, params.kv_overrides, params.tensor_buft_overrides);`
- 位置：`src/llama.cpp`

这里开始进入 GGUF 读取流程。

## 3. 真正读取 GGUF 文件

函数：

- `llama_model_loader::llama_model_loader(...)`
- 位置：`src/llama-model-loader.cpp`

在这个构造函数中，主 GGUF 文件通过下面的调用被读取：

- `gguf_init_from_file(fname.c_str(), params)`
- 位置：`src/llama-model-loader.cpp`

底层实际 GGUF 解析函数定义在：

- `gguf_init_from_file(...)`
- `gguf_init_from_file_impl(...)`
- 位置：`ggml/src/gguf.cpp`

## 4. GGUF 元信息和 tensor 索引如何存入加载器

在 `llama_model_loader::llama_model_loader(...)` 中：

### 4.1 GGUF metadata context

读取结果保存到：

- `meta`
- 类型：`gguf_context_ptr`

用途：

- 保存 GGUF key-value metadata
- 后续通过 `get_key()`、`get_arr()`、`get_arch()` 等接口读取

### 4.2 GGML tensor metadata context

通过 `gguf_init_from_file(..., {.ctx = &ctx})` 还会得到一个 `ggml_context * ctx`，随后保存到：

- `contexts`
- 类型：`std::vector<ggml_context *>`

用途：

- 保存 GGUF 中每个 tensor 的元信息
- 可通过 `ggml_get_first_tensor(ctx)` 遍历所有 tensor

### 4.3 文件句柄

模型文件保存到：

- `files`
- 类型：`std::vector<std::unique_ptr<llama_file>>`

用途：

- 后续按 offset 真正加载 tensor 数据

### 4.4 模型架构名

通过：

- `get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);`

保存到：

- `arch_name`
- 类型：`std::string`

随后再转换为：

- `llm_kv = LLM_KV(llm_arch_from_string(arch_name));`

### 4.5 tensor 名称到权重数据的映射

构造函数会遍历所有 GGUF tensor：

- `for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur))`

并把它们保存到：

- `weights_map`
- 类型：`std::map<std::string, llama_tensor_weight>`

其中 key 是 tensor 名称，value 是：

- `llama_tensor_weight(files.back().get(), 0, meta.get(), cur)`

`llama_tensor_weight` 中持有的信息本质上包括：

- 该 tensor 位于哪个文件
- 来自哪个 split
- 对应的 GGUF metadata context
- 对应的 `ggml_tensor *` 元信息对象

这一步是后续按名字找 weight 的核心索引。

### 4.6 统计信息

同时还会累计：

- `n_elements`
- `n_bytes`
- `n_kv`
- `n_tensors`
- `fver`

这些主要用于日志、检查和后续加载控制。

## 5. 从加载器进入 llama_model

在 `llama_model_load(...)` 中，构造好 `llama_model_loader` 之后，会依次调用：

- `model.load_arch(ml);`
- `model.load_hparams(ml);`
- `model.load_vocab(ml);`
- `model.load_tensors(ml);`

相关函数定义在：

- `src/llama-model.cpp`

## 6. 各阶段把数据写到 llama_model 的什么字段

### 6.1 读取架构

函数：

- `void llama_model::load_arch(llama_model_loader & ml)`

核心逻辑：

- `arch = ml.get_arch();`

写入字段：

- `llama_model::arch`

作用：

- 决定这是 `QWEN3`、`QWEN3MOE`、`LLAMA` 等哪种模型
- 后面 `build_graph()` 时会根据它选择不同的 graph builder

### 6.2 读取超参数

函数：

- `void llama_model::load_hparams(llama_model_loader & ml)`

写入字段：

- `llama_model::hparams`

包含：

- 层数 `n_layer`
- 隐层维度 `n_embd`
- 头数 `n_head`
- FFN 尺寸 `n_ff`
- rope 参数
- 以及不同架构特有参数

### 6.3 读取词表

函数：

- `void llama_model::load_vocab(llama_model_loader & ml)`

写入字段：

- `llama_model::vocab`

### 6.4 读取所有权重 tensor

函数：

- `bool llama_model::load_tensors(llama_model_loader & ml)`

这是最关键的“把 GGUF tensor 接进模型结构”的函数。

## 7. load_tensors() 中的关键调用链

### 7.1 根据 architecture 进入不同分支

`load_tensors()` 中会根据：

- `arch`

进入不同的 `case LLM_ARCH_...`

对你的 MiniCPM-o 4.5 主干，如果它以 Qwen3 LLM 进入，则会落到：

- `case LLM_ARCH_QWEN3:`
- 位置：`src/llama-model.cpp`

### 7.2 create_tensor() 是把 GGUF tensor 接入模型字段的核心函数

在 `load_tensors()` 内定义了一个局部 lambda：

- `auto create_tensor = [&](const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) -> ggml_tensor *`

作用：

1. 用名字去 `ml.get_tensor_meta(tn.str().c_str())` 找 GGUF 里的 tensor 元信息
2. 检查这个 tensor 是否存在、维度是否合理
3. 根据 tensor 的用途选择 buffer type
4. 创建一个新的 `ggml_tensor *`
5. 最终把这个新建 tensor 返回给模型字段

注意：

- 这里返回的是 `ggml_tensor *`
- 它不是简单的裸数据指针，而是后续图构建和后端执行都要使用的权重对象

### 7.3 Qwen3 分支中，tensor 被写入哪些 layer 字段

在 `case LLM_ARCH_QWEN3:` 中，每层会写入这些字段：

- `layer.attn_norm`
- `layer.wq`
- `layer.wk`
- `layer.wv`
- `layer.wo`
- `layer.ffn_norm`
- `layer.rope_freqs`
- `layer.ffn_gate`
- `layer.ffn_down`
- `layer.ffn_up`

这些字段都属于：

- `llama_model::layers[i]`

其中最关键的 Linear 权重字段是：

- `layers[i].wq`
- `layers[i].wk`
- `layers[i].wv`
- `layers[i].wo`
- `layers[i].ffn_gate`
- `layers[i].ffn_down`
- `layers[i].ffn_up`

这几个字段就是后面最可能被 AWQ 替换或扩展的地方。

## 8. 权重数据何时真正装入内存

在 `load_tensors()` 里，`create_tensor()` 主要先建立 tensor 结构与映射关系。

真正把 tensor 对应的数据区映射/加载进来，依赖后续：

- `ml.init_mappings(...)`
- `ml.load_data_for(...)`
- `ml.load_all_data(...)`

这些函数定义在：

- `src/llama-model-loader.cpp`

因此可以理解为：

- `load_tensors()` 负责把“GGUF 中有哪些 tensor”接成模型字段
- `llama_model_loader` 负责把“这些 tensor 的底层数据”映射或读入

## 9. 后续如何进入实际执行图

模型加载完成后，真正构图使用的是：

- `ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const`
- 位置：`src/llama-model.cpp`

它会根据 `arch` 分发：

- `LLM_ARCH_QWEN3 -> llm_build_qwen3`

即：

- `llm = std::make_unique<llm_build_qwen3>(*this, params);`

在 `llm_build_qwen3` 中，前面保存到 `model.layers[il]` 里的权重会被实际拿出来使用：

- `build_lora_mm(model.layers[il].wq, cur)`
- `build_lora_mm(model.layers[il].wk, cur)`
- `build_lora_mm(model.layers[il].wv, cur)`
- `build_attn(..., model.layers[il].wo, ...)`
- `build_ffn(..., model.layers[il].ffn_up, model.layers[il].ffn_gate, model.layers[il].ffn_down, ...)`

## 10. 对 AWQ 接入最关键的观察

如果要以最小侵入方式接入 AWQ，最值得关注的是下面这条链：

1. `llama_model_loader::llama_model_loader(...)`
   - 读 GGUF
   - 建立 `weights_map`

2. `llama_model::load_tensors(...)`
   - 通过 `create_tensor(...)` 把权重接到 `layers[i].wq/wk/wv/wo/ffn_*`

3. `llm_build_qwen3`
   - 通过 `build_lora_mm()` / `build_attn()` / `build_ffn()` 使用这些权重

对于 AWQ 来说，最有可能修改的点通常是：

- GGUF 中如何表示 AWQ tensor
- `load_tensors()` 如何把 AWQ tensor 挂进模型结构
- `build_lora_mm()` 或更底层 `mul_mat` 分发如何根据权重类型选择 AWQ 路径

## 11. 简版调用链总览

```text
llama_model_load_from_file
-> llama_model_load_from_file_impl
-> llama_model_load
-> llama_model_loader::llama_model_loader
-> gguf_init_from_file
-> llama_model::load_arch
-> llama_model::load_hparams
-> llama_model::load_vocab
-> llama_model::load_tensors
   -> create_tensor(...)
   -> layers[i].wq / wk / wv / wo / ffn_gate / ffn_down / ffn_up
-> llama_model::build_graph
   -> llm_build_qwen3
   -> build_lora_mm / build_attn / build_ffn
```
