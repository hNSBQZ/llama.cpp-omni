# TTS Condition Graph 验证记录

验证目标：确认 `TTSCUDA=1` 下的 `tts_condition_graph_forward` 与未开启融合图时的 legacy condition 构建路径输出一致，并记录耗时差异。

## 改动

- 在 `tools/omni/omni-impl.h` 新增 `dumptensor()`，用于将 `[dim, len, batch, 1]` 连续张量按 6 位小数保存为纯文本矩阵：每行 `dim` 个数字，共 `len * batch` 行。
- 在单工 TTS condition 构建后额外保存 `merged_embeddings_tensor.txt`，用于机器对比。
- 新增 `tools/omni/compare_tensor_dump.py`，对两个 `dumptensor()` 输出文件做形状检查和绝对误差统计。
- 给 `tools/omni/test/single_test_omni.cpp` 增加 `--output-dir` 和 `--seed`，方便两次验证输出隔离且采样可复现。

## 验证命令

编译：

```bash
LD_LIBRARY_PATH=/cache/hanqingzhe/.local/cuda-12.4/lib64:${LD_LIBRARY_PATH} \
cmake --build build --target llama-omni-single-test-omni -j 8
```

legacy 路径：

```bash
LD_LIBRARY_PATH=/cache/hanqingzhe/.local/cuda-12.4/lib64:${LD_LIBRARY_PATH} \
CUDA_VISIBLE_DEVICES=0 \
/usr/bin/time -f "wall_time_sec %e" \
build/bin/llama-omni-single-test-omni \
  -m /cache/hanqingzhe/o45-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --test tools/omni/assets/test_case/omni_test_case/omni_test_case_ 1 \
  --output-dir tools/omni/output_condition_validation/cpu_final \
  --ref-audio tools/omni/assets/default_ref_audio/default_ref_audio.wav \
  -s 42 -c 4096 -ngl 99
```

融合图路径：

```bash
LD_LIBRARY_PATH=/cache/hanqingzhe/.local/cuda-12.4/lib64:${LD_LIBRARY_PATH} \
CUDA_VISIBLE_DEVICES=0 TTSCUDA=1 \
/usr/bin/time -f "wall_time_sec %e" \
build/bin/llama-omni-single-test-omni \
  -m /cache/hanqingzhe/o45-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --test tools/omni/assets/test_case/omni_test_case/omni_test_case_ 1 \
  --output-dir tools/omni/output_condition_validation/gpu_final \
  --ref-audio tools/omni/assets/default_ref_audio/default_ref_audio.wav \
  -s 42 -c 4096 -ngl 99
```

## 结果

- 对比范围：`round_000/llm_debug/chunk_*/merged_embeddings_tensor.txt`
- chunk 数：16
- 总行数：152
- 每行维度：768
- 总元素数：116736

精度：

| 指标 | 结果 |
| --- | ---: |
| 平均绝对误差 | `4.50589364034e-09` |
| 最大绝对误差 | `1.00000000014e-06` |
| 最小绝对误差 | `0` |
| 标准差 | `6.69745516061e-08` |

最大误差已经落在 6 位小数文本 dump 的量化粒度上。逐 chunk 结果见 `tools/omni/logs/condition_compare_final.log`。

耗时：

| 路径 | condition 构建次数 | condition 平均耗时 | condition 最小耗时 | condition 最大耗时 | 端到端 wall time | llama total time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 16 | `4.96669 ms` | `0.288 ms` | `8.544 ms` | `10.80 s` | `10347.22 ms` |
| TTSCUDA/ggml graph | 16 | `0.643312 ms` | `0.290 ms` | `4.252 ms` | `10.39 s` | `10000.38 ms` |

融合图首次初始化耗时：`39.544 ms`。该初始化耗时不计入日志里的单次 `TTS condition build [ggml]` forward 耗时。

## 结论

`TTSCUDA=1` 融合图路径在本次单工 omni 样例上与 legacy 路径的 `merged_embeddings` 形状完全一致，最大绝对误差为 `1e-6`，满足当前 6 位小数 dump 粒度下的精度验证。condition 构建平均耗时从约 `4.97 ms` 降到约 `0.64 ms`。

