# Qwen3 + AWQ 全 Linear 切到 Marlin 的改动说明（按 Git History 更新）

这份文档基于当前分支 `master` 的实际提交历史整理，重点解释为了让 `Qwen3 + AWQ` 在推理图中把主干 linear 都走 Marlin 实现，我们改了什么代码、为什么要这么改。

## 1. 目标与当前结论

目标是让 `LLM_ARCH_QWEN3 + AWQ(W4A16)` 在图构建和 CUDA 执行阶段命中 Marlin 路径，而不是继续走普通 `ggml_mul_mat`。

当前状态（已落地）：

- Qwen3 主干 7 个 linear 已支持 AWQ -> Marlin 分流：
  - `wq / wk / wv / wo / ffn_up / ffn_gate / ffn_down`
- 在图里新增了正式 op：`GGML_OP_MARLIN_W4A16`
- CUDA backend 已能执行该 op，并调到 `ggml_cuda_marlin_w4a16_gemm(...)`
- 模型加载阶段已接入 AWQ triplet（`qweight/qzeros/scales`）并做 Marlin 布局处理
- 支持两种权重来源：
  - 旧 AWQ GGUF：运行时 restore+repack+permute
  - 离线转换的 Marlin GGUF：跳过运行时变换，直接上 CUDA

当前仍保留的边界：

- `lm_head(output)` 仍是 dense 路径（未切 Marlin）
- AWQ Marlin 路径暂不支持 LoRA（代码里有 assert）

---

## 2. 关键提交时间线（里程碑）

按 commit 顺序的主线如下：

1. `ba79015 add marlin`
   - 引入 Marlin kernel、repack 相关 CUDA 代码、基础测试 `tests/test-marlin-w4a16.cpp`
2. `9eff9b2 finish tensor loading and marlin gemm`
   - 打通 AWQ 元信息读取、Qwen3 AWQ tensor 注册、模型层字段
3. `1e4f022 add build_graph for awq, add debug_for_tensor`
   - 加入 `GGML_OP_MARLIN_W4A16`
   - 加入图 helper `build_awq_marlin_mm(...)`
   - 在 `llm_build_qwen3` 把 7 个 linear 分流到 Marlin
   - CUDA backend 增加 `case GGML_OP_MARLIN_W4A16`
4. `b168d62` 到 `536a23b` 一组诊断提交
   - 围绕 qzeros/scales/group-offset 行为做 probe 和定位
5. `d8408cc successfully run awq w4a16`
   - 修正 AWQ 布局恢复和零点转换细节，保证可稳定跑通
6. `64c5b42 remove repack and trans`
   - 增加离线转换脚本，形成“离线 Marlin GGUF 优先”的加载策略
7. `5ae6a42 fix sync for prompt`
   - 后续调优与 profiling 相关，不改变上述主链路设计

---

## 3. 按模块看“改了什么 + 为什么”

## 3.1 元信息与 AWQ tensor 建模

主要文件：

- `src/llama-hparams.h`
- `src/llama-arch.h`
- `src/llama-arch.cpp`
- `src/llama-model.h`
- `src/llama-model.cpp`

改动点：

- 增加量化方法枚举与字段：
  - `LLAMA_QUANTIZATION_METHOD_AWQ`
  - `quant_method / quant_bits / quant_group_size / quant_zero_point`
- 给 Qwen3 注册 AWQ tensor 名字映射：
  - `self_attn.{q,k,v,o}_proj.{qweight,qzeros,scales}`
  - `mlp.{gate,up,down}_proj.{qweight,qzeros,scales}`
- 在 `llama_layer` 增加 7 组 AWQ triplet 指针字段
- 在 Qwen3 分支创建 AWQ tensor（按 `quant_bits` pack 因子和 group size 算形状）

为什么必须这么做：

- 图构建分流前，模型层里必须“有 AWQ triplet 可拿”
- 没有这层注册，后续 Marlin 路径没有输入源

---

## 3.2 模型加载后的 AWQ -> Marlin 布局处理

主要文件：

- `src/llama-model.cpp`
- `ggml/include/ggml-cuda.h`
- `ggml/src/ggml-cuda/awq_marlin_repack.cu`

改动点：

- 在加载完成后统一调用：
  - `llama_repack_qwen3_awq_tensors_to_marlin(*this, pimpl->bufs);`
- 处理逻辑分两类：

1) 离线 Marlin GGUF（推荐路径）
- 判断条件：`scales->type == GGML_TYPE_F16`
- 认为 triplet 已是 Marlin 布局，只做 CUDA rehome

2) 旧 AWQ GGUF（兼容路径）
- 先把 gguf/ggml 语义下的 2D 导出布局恢复到 HF 逻辑布局
- 再做：
  - `qweight` repack
  - `qzeros` 变换（含 undo interleave + permute + interleave）
  - `scales` permutation 并转 `F16`

为什么必须这么做：

- Marlin kernel 假设输入布局是特定 packed/permute 形式
- 直接吃原始 AWQ triplet，数值和性能都不对

---

## 3.3 Qwen3 图构建：7 个 linear 显式分流

主要文件：

- `src/llama-model.cpp`（`llm_build_qwen3`）
- `src/llama-graph.h`
- `src/llama-graph.cpp`

改动点：

- 在 `llm_build_qwen3` 中新增本地分流 helper：
  - `build_qwen3_linear(dense, qweight, qzeros, scales, input, il)`
- 分流条件：
  - `hparams.quant_method == AWQ`
  - 对应 triplet 非空
- 命中后走 `build_awq_marlin_mm(...)`
- 否则回退 `build_lora_mm(...)`

命中的 7 个 linear：

- Attention: `wq / wk / wv / wo`
- FFN: `ffn_up / ffn_gate / ffn_down`

为什么这样改：

- 这些层此前全走 dense helper，AWQ triplet 即使加载了也不会被用到
- 在 Qwen3 builder 层做显式分流，风险面最小，不影响其它架构

---

## 3.4 GGML 新增正式 op：`GGML_OP_MARLIN_W4A16`

主要文件：

- `ggml/include/ggml.h`
- `ggml/src/ggml.c`

改动点：

- 新增 op enum：`GGML_OP_MARLIN_W4A16`
- 新增建图 API：`ggml_marlin_w4a16(...)`
- 明确 src 输入槽位：
  - `src[0]=a`
  - `src[1]=qweight`
  - `src[2]=scales`
  - `src[3]=qzeros`
  - `src[4]=workspace`

为什么必须是“正式 op”：

- 这样 backend 调度、能力判断、batch/offload 策略才能统一接入
- 比临时 custom op 更可维护，后续扩展更稳

---

## 3.5 CUDA backend 执行链路

主要文件：

- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/marlin-op.cu`
- `ggml/src/ggml-cuda/marlin-op.cuh`

改动点：

- dispatch 增加：
  - `case GGML_OP_MARLIN_W4A16: ggml_cuda_op_marlin_w4a16(ctx, dst);`
- 能力判断加入该 op 的合法性检查（类型、连续性、cc>=Turing 等）
- 在 `ggml_cuda_op_marlin_w4a16(...)` 中：
  - 解析 5 个 src
  - 从 CUDA pool 分配实际 workspace（graph 中只保留 placeholder）
  - 调用 `ggml_cuda_marlin_w4a16_gemm(...)`

为什么 workspace 这样设计：

- 图里只保留语义占位，避免把 device 相关 workspace 大小硬编码在前端
- 后端按当前设备动态申请更稳

---

## 3.6 诊断与验证工具链

主要文件/提交：

- `tests/test-marlin-w4a16.cpp`（多次增强 probe）
- `tools/omni/inspect_awq_ffn_down.py`
- `tools/omni/convert_awq_marlin_gguf.py`
- `tools/omni/awq_marlin_converter.py`
- `tools/omni/*difftest.py`

改动点：

- 加了 qzeros/group-offset/scales 对齐相关探针和差分脚本
- 增加离线转换脚本，把 AWQ GGUF 预处理成 Marlin 可直接消费的布局

为什么要这些工具：

- AWQ -> Marlin 的核心难点在布局与索引语义，不是单一 kernel 调用
- 没有 probe 与 diff，很难定位“能跑但结果偏”的问题

---

## 4. 这套改动解决了什么关键问题

关键问题与对应改法：

1. 之前图里只有 `mul_mat`，AWQ triplet 不会生效
- 通过 `build_qwen3_linear + build_awq_marlin_mm` 显式切流

2. 之前没有可调度的 Marlin op
- 在 GGML 增加 `GGML_OP_MARLIN_W4A16`，打通图到 backend

3. 之前 backend 不知道如何执行 AWQ Marlin 节点
- 增加 CUDA dispatch + marlin-op 执行函数

4. 之前权重布局与 Marlin 输入契约不一致
- 运行时 repack/permute + 离线转换双路径支持

---

## 5. 当前实现边界与后续方向

当前边界：

- `Qwen3 + AWQ + CUDA + W4A16` 是主支持路径
- `lm_head` 仍未切 Marlin
- LoRA 与 AWQ Marlin 混用尚未支持

后续可做：

- `lm_head` AWQ/Marlin 化
- 把当前 Qwen3 专用分流抽象为更通用的 linear descriptor
- 扩展到更多架构（在不破坏现有稳定性的前提下）

---

## 6. 一句话总结

这次不是“只换 kernel”，而是把 `Qwen3 + AWQ` 的完整链路打通了：  
**元信息与 tensor 注册 -> 加载后布局处理 -> 图中 7 个 linear 分流 -> GGML 正式 op -> CUDA backend 执行与 workspace 管理 -> 诊断与离线转换工具闭环。**
