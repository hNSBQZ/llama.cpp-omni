# 纯 LLM Benchmark 规划

目标是在当前推理框架里只测试 LLM 的吞吐，不跑 vision/audio/TTS/duplex 链路，输出与 `pp128`、`pp512`、`pp2048`、`tg32`、`tg128`、`tg256` 同口径的 tokens/s 结果。

## 结论

仓库已经有现成 benchmark：`tools/llama-bench`，构建后对应 `build/bin/llama-bench`。它已经支持：

- `pp`：prompt processing/prefill，通过 `-p/--n-prompt` 控制 prompt token 数。
- `tg`：text generation/decode，通过 `-n/--n-gen` 控制 decode token 数。
- `-r`：每个测试重复次数，输出平均值和标准差。
- `-o md|csv|json|jsonl|sql`：输出格式，`md` 适合直接看，`jsonl/csv` 适合后处理。

所以第一版不建议在 `tools/omni` 里重新实现计时内核，而是在 `tools/omni` 下放一个薄封装脚本，负责枚举模型、固定 benchmark 参数、保存原始结果和汇总表。

## 测试矩阵

固定测试项：

| test | 含义 | llama-bench 参数 |
| --- | --- | --- |
| `pp128` | prefill 128 tokens | `-p 128` |
| `pp512` | prefill 512 tokens | `-p 512` |
| `pp2048` | prefill 2048 tokens | `-p 2048` |
| `tg32` | decode 32 tokens | `-n 32` |
| `tg128` | decode 128 tokens | `-n 128` |
| `tg256` | decode 256 tokens | `-n 256` |

模型目录先按这三个入口规划：

- `~/o45-gguf`
- `~/o45-gguf-gptq-int8`
- `~/o45-awq-gguf`

为了控制数量，建议第一版不要递归扫所有文件后无脑全跑，而是采用“manifest 优先，自动发现兜底”的策略：

- 如果提供 `tools/omni/llm_bench_models.txt`，每行一个模型路径或 `name=path`，脚本只跑 manifest 里列出的模型。
- 如果没有 manifest，再从三个目录里按 `*.gguf` 找模型。
- 自动发现时默认每个目录最多跑 1 个模型；需要全量时显式设置 `MAX_MODELS_PER_DIR=0`。
- 输出里记录完整模型路径、模型文件名、目录来源和实际运行参数，避免后续混淆量化来源。

这样可以先快速验证流程，再按需要扩大到全目录或指定量化集合。

## 单卡运行策略

这些是小参数模型，默认按“每个模型一张卡”跑，不做多卡 tensor split，也不把单个模型切到多张 GPU 上。

- 单个 benchmark 进程只暴露一张 GPU，例如 `CUDA_VISIBLE_DEVICES=7`。
- `-ngl 99` 仍然用于把模型层尽量放到这张可见 GPU 上。
- 不设置 `-ts/--tensor-split`，也不依赖多 GPU split。
- 如果机器上有多张空闲卡，可以并行启动多个 benchmark 进程，每个模型绑定不同 GPU。
- 输出里记录模型到 GPU 的映射，避免后面混淆不同卡的性能差异。

推荐 manifest 支持可选 GPU 字段，例如：

```text
o45-gguf=~/o45-gguf/MiniCPM-o-4_5-F16.gguf,gpu=0
o45-gptq-int8=~/o45-gguf-gptq-int8/model.gguf,gpu=1
o45-awq=~/o45-awq-gguf/model.gguf,gpu=2
```

如果不配置 GPU 字段，就使用统一的 `CUDA_VISIBLE_DEVICES` 或 `LLM_BENCH_GPU` 默认值。

## 数量控制

单个模型的基础测试数是 6 项：

```text
3 个 pp 测试 + 3 个 tg 测试 = 6
```

总耗时约等于：

```text
模型数 * 6 * repetitions * 单次运行耗时
```

推荐默认：

- `REPETITIONS=5`：与 `llama-bench` 默认一致，能得到 `mean ± stddev`。
- `REPETITIONS=10`：正式对比时使用，标准差更稳定，但耗时约翻倍。
- `CUDA_VISIBLE_DEVICES` 对每个模型固定一张卡；串行跑时可以都用同一张空闲卡，并行跑时每个模型分配不同卡。
- `-ngl 99` 固定全量 GPU offload。
- `-b 2048 -ub 512` 固定 batch/ubatch，覆盖 `pp2048`，且便于跨模型比较。
- `-fa` 是否开启 flash attention 必须固定；如果目标是贴近当前部署参数，就按部署配置；如果只是比较模型量化吞吐，建议先统一 `-fa 0` 或统一 `-fa 1`，不要混跑。

示例规模：

```text
3 个模型 * 6 个测试 * 5 次重复 = 90 次测量样本
3 个模型 * 6 个测试 * 10 次重复 = 180 次测量样本
```

如果 `~/o45-gguf` 里有多个常规 GGUF 量化版本，建议先只选一个 baseline，例如 F16 或 Q8_0，再把 GPTQ-int8 和 AWQ 各选一个对应模型。等脚本和结果表确认无误后，再扩展到全量 sweep。

## 数据构造

纯吞吐 benchmark 不需要真实文本数据集。`llama-bench` 会直接构造 token 序列：

- `pp` 测试使用随机 token 批量 prefill，首 token 视词表配置可能使用 BOS。
- `tg` 测试每次 decode 1 个 token，并用随机 token 作为下一步输入。
- 每轮正式测量前会清 KV cache，默认还有 warmup。

这类构造更适合测推理内核吞吐，因为它绕过 tokenizer、采样策略、prompt 模板、I/O 和业务数据差异。它的结果含义就是当前模型、后端、batch/ubatch、KV cache 类型和 GPU offload 设置下的 raw LLM throughput。

不建议第一版引入 ShareGPT/vLLM 之类真实请求数据集，因为用户给出的目标指标是固定长度的 `pp/tg` microbenchmark，不是端到端 serving benchmark。如果后续要测真实并发请求、TTFT、TPOT、QPS，再考虑 `tools/server/bench` 或单独的 server benchmark。

## 运行方式草案

先构建现成目标：

```bash
cmake --build build --target llama-bench -j"$(nproc)"
```

单模型命令形态：

```bash
CUDA_VISIBLE_DEVICES=7 ./build/bin/llama-bench \
  -m ~/o45-gguf/MiniCPM-o-4_5-F16.gguf \
  -p 128,512,2048 \
  -n 32,128,256 \
  -ngl 99 \
  -b 2048 \
  -ub 512 \
  -r 5 \
  -o md
```

封装脚本建议放在：

```text
tools/omni/run_llm_benchmark.sh
```

脚本职责：

- 检查 `build/bin/llama-bench` 是否存在，不存在时提示构建或自动构建。
- 读取 `LLM_BENCH_MODEL_DIRS`，默认是上述三个目录。
- 读取 `LLM_BENCH_MODEL_MANIFEST`，存在时优先按 manifest 跑。
- 按 manifest 或环境变量为每个模型选择单张 GPU，启动命令时设置 `CUDA_VISIBLE_DEVICES=<gpu_id>`。
- 固定 `-p 128,512,2048 -n 32,128,256`。
- 将每个模型的 `md` 和 `jsonl` 结果保存到 `tools/omni/output/llm_bench/<timestamp>/`。
- 记录 `nvidia-smi`、git commit、完整命令行和环境变量，方便复现。

后处理脚本建议放在：

```text
tools/omni/summarize_llm_benchmark.py
```

脚本职责：

- 读取所有 `jsonl` 原始结果。
- 抽取 `model_filename`、`model_type`、`backend`、`n_batch`、`n_ubatch`、`n_gpu_layers`、`n_prompt`、`n_gen`、`avg_ts`、`stddev_ts`。
- 生成最终 Markdown 汇总表，展示为 `tokens/s = avg_ts ± stddev_ts`。
- 按模型维度输出一张表，行是 `pp128/pp512/pp2048/tg32/tg128/tg256`。

## 输出目录建议

```text
tools/omni/output/llm_bench/
  20260602-180000/
    env.txt
    commands.txt
    models.txt
    raw/
      o45-gguf.md
      o45-gguf.jsonl
      o45-gguf-gptq-int8.md
      o45-gguf-gptq-int8.jsonl
      o45-awq-gguf.md
      o45-awq-gguf.jsonl
    summary.md
```

`summary.md` 目标格式：

| model | test | tokens/s | 含义 |
| --- | --- | --- | --- |
| `o45-gguf` | `pp128` | `... ± ...` | prefill 128 tokens |
| `o45-gguf` | `pp512` | `... ± ...` | prefill 512 tokens |
| `o45-gguf` | `pp2048` | `... ± ...` | prefill 2048 tokens |
| `o45-gguf` | `tg32` | `... ± ...` | decode 32 tokens |
| `o45-gguf` | `tg128` | `... ± ...` | decode 128 tokens |
| `o45-gguf` | `tg256` | `... ± ...` | decode 256 tokens |

## 注意事项

- 保持同一 GPU、同一驱动、同一构建产物、同一 `CUDA_VISIBLE_DEVICES`，否则不同轮结果不可直接比较。
- 运行前尽量确认 GPU 空闲；必要时记录 `nvidia-smi`。
- 每次比较 GPTQ/AWQ/GGUF 时，确保模型结构一致，否则吞吐差异不只来自量化格式。
- `pp2048` 对 `n_batch`/`n_ubatch` 更敏感；跨模型比较时不要让脚本自动调整这些参数。
- `tg` 是逐 token decode，主要体现 batch=1 的 decode 吞吐；如果要测并发 decode，需要另开 `llama-batched-bench` 或 server benchmark。
- 如果要复现类似示例里的 `3727.24 ± 129.94`，优先使用 `llama-bench` 的 Markdown 输出，因为它原生就是 `avg_ts ± stddev_ts`。

## 下一步

1. 在 `tools/omni` 下实现 `run_llm_benchmark.sh`。
2. 可选实现 `llm_bench_models.txt`，明确三类模型各自跑哪个文件。
3. 实现 `summarize_llm_benchmark.py`，把 `jsonl` 汇总成目标表格。
4. 先用 `REPETITIONS=1` 和每个目录 1 个模型 dry run，确认模型能加载。
5. 再用 `REPETITIONS=5` 或 `10` 跑正式结果。
