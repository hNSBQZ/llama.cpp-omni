# MiniCPM-o Duplex Judge

双工端到端延迟评测：输入视频，自动切分、推理，输出 SPEAK→wav 等关键延迟指标。

## 环境

Python 依赖见 `requirements.txt`：

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

`./run_judge_direct.sh` 优先用本目录下的 `.venv/`，没有就退回系统 `python3`。

本目录自带示例输入：`assets/video/omni_duplex1.mp4`（双工评测用短视频）。

### 依赖获取

评测跑的是本仓库的 `llama-omni-server`，按 [docs/build.md](../../docs/build.md) 编译出来即可，用 `--llamacpp-root` 指向仓库根（默认是本目录上溯两级）。

另需 **MiniCPM-o-4_5-gguf**（含 LLM / vision / audio / TTS 等子目录），放在任意路径，通过 `--model` 指定：

- Hugging Face：<https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf>
- ModelScope：<https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf>

`--model` 指向其中的 LLM 文件，例如 `MiniCPM-o-4_5-F16.gguf` 或量化版 `MiniCPM-o-4_5-Q4_K_M.gguf`；其余模态权重需与 `--model` 同目录树（`audio/`、`vision/`、`tts/` 等）。

## 用法

```bash
./run_judge_direct.sh --gpu 0 \
  --model /path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-F16.gguf \
  --video assets/video/omni_duplex1.mp4 \
  --min-free-mib 22000
```

多视频：

```bash
./run_judge_direct.sh --gpu 0 \
  --model /path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-F16.gguf \
  --video assets/video/omni_duplex1.mp4 \
         assets/video/another.mp4
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--gpu` | GPU 编号；省略则自动选空闲卡 |
| `--model` | LLM GGUF 路径（必填） |
| `--llamacpp-root` | 仓库根目录；省略则从本目录往上找带 `build*/bin/llama-omni-server` 的祖先 |
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
