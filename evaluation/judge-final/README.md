# MiniCPM-o Duplex Judge

双工端到端延迟评测：输入视频，自动切分、推理，输出 SPEAK→wav 等关键延迟指标。

## 环境

本目录自带 Python 虚拟环境（`.venv/`）。`./run_judge_direct.sh` 会默认使用其中的解释器，无需单独安装 Python 依赖。`requirements.txt` 仅在需要重建虚拟环境时使用。

本目录自带示例输入：`assets/video/omni_duplex1.mp4`（双工评测用短视频）。

### 依赖获取

另需自行准备 **llama.cpp-omni** 与 **MiniCPM-o-4_5-gguf**（可放在本目录同级或任意路径，通过 `--llamacpp-root` / `--model` 指定）。

**llama.cpp-omni**（`bench/pref-e2e` 分支）：

- 仓库：<https://github.com/tc-mb/llama.cpp-omni/tree/bench/pref-e2e>

```bash
git clone https://github.com/tc-mb/llama.cpp-omni.git
cd llama.cpp-omni && git checkout bench/pref-e2e
# 编译 llama-omni-server 后再评测，详见该仓库说明
```

**MiniCPM-o-4_5-gguf**（含 LLM / vision / audio / TTS 等子目录）：

- Hugging Face：<https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf>
- ModelScope：<https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf>

`--model` 指向其中的 LLM 文件，例如 `MiniCPM-o-4_5-F16.gguf` 或量化版 `MiniCPM-o-4_5-Q4_K_M.gguf`；其余模态权重需与 `--model` 同目录树（`audio/`、`vision/`、`tts/` 等）。

## 用法

```bash
./run_judge_direct.sh --gpu 0 \
  --model ../models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-F16.gguf \
  --llamacpp-root ../llama.cpp-omni \
  --video assets/video/omni_duplex1.mp4 \
  --min-free-mib 22000
```

多视频：

```bash
./run_judge_direct.sh --gpu 0 \
  --model ../models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-F16.gguf \
  --llamacpp-root ../llama.cpp-omni \
  --video assets/video/omni_duplex1.mp4 \
         assets/video/omni_duplex1.mp4
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--gpu` | GPU 编号；省略则自动选空闲卡 |
| `--model` | LLM GGUF 路径（必填） |
| `--llamacpp-root` | llama.cpp-omni 根目录（默认同级 `../llama.cpp-omni`） |
| `--video` | 输入视频，可多个 |
| `--max-chunks` | 最多处理多少个 chunk |
| `--max-duration` | 最多处理多少秒（默认 120） |
| `--verbose` / `-v` | 向控制台打印进度 |
| `--plot` | 评测结束后画 turn-position 曲线（需 `matplotlib`） |
| `--keep-alive` | 跑完后不停止 llama-server |

## 输出

评测结束后控制台会打印关键延迟摘要，最后一行给出日志目录，例如：

```
  log: tmp/runs/xx_xx/logs
```

查看完整参数：`./run_judge_direct.sh --help`

---

说明：本工具供自行评测与调优使用，非最终版本。
