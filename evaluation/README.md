# MiniCPM-o 评测套件

跑通四项评测，并按规范提交代码。

| 任务 | 数据集 | 指标 | 依赖的 C++ target |
|------|--------|------|-------------------|
| `videomme` | Video-MME（900 视频 / 2700 题） | 选择题准确率 | `llama-omni-eval-cli` |
| `daily-omni` | Daily-Omni（1197 题，音视频交错） | 选择题准确率 | `llama-omni-eval-daily-cli` |
| `tts` | Seed-TTS 中文（2020 条） | WER / SIM(ASV) | `llama-omni-tts-eval` |
| `rts` | 双工短视频 | RTF / SPEAK→wav 延迟 | `llama-omni-server` |

四项任务共用 `config.env` 与入口脚本；评测 CLI / server 均在仓库主干中。

---

## 1. 环境准备

推荐：Linux aarch64 + Ascend 910。需预先安装：

- CANN Toolkit（能找到 `/usr/local/Ascend/ascend-toolkit/set_env.sh`）
- CMake、C/C++ 编译器、Git、`ffmpeg`
- Python 3.10+、pip
- 可选：`rubberband`（使用 `pyrubberband` 音频变速时需要）

创建 Python 环境：

```bash
python3 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install -U pip
python -m pip install -r evaluation/requirements.txt
```

另需安装与当前平台匹配的 `torch` / `torchaudio` / `torchvision`（Ascend 请使用平台提供的兼容 wheel）。SIM 打分还需要本地 `s3prl` 源码，路径写在 `S3PRL_REPO`。`decord` 可选；aarch64 装不上时会自动改用系统 `ffmpeg`。

---

## 2. 配置

复制并编辑 `evaluation/config.env`，至少确认：

```bash
MODEL_DIR=/path/to/weights
MODEL_LLM=/path/to/weights/MiniCPM-o-4_5-F16.gguf
TTS_MODEL_PATH=/path/to/weights/tts/MiniCPM-o-4_5-tts-F16.gguf
ASSETS_DIR=/path/to/assets
DEVICE_IDS=0,1,2,3
DEVICE_COUNT=4
EVAL_PYTHON=/absolute/path/to/.venv-eval/bin/python
RTS_PYTHON=/absolute/path/to/.venv-eval/bin/python
```

常用可选项：

| 类别 | 关键项 |
|------|--------|
| 模型 | `MODEL_DIR` `MODEL_LLM` `RTS_MODEL_LLM` `TTS_MODEL_PATH` |
| 路径 | `LLAMACPP_ROOT` `EVAL_BIN_DIR` `OMNI_SERVER_BIN` `ASCEND_ENV` |
| 设备 | `DEVICE_ENV_VAR` `DEVICE_IDS` `DEVICE_COUNT` `RTS_DEVICE_ID` |
| 样本 | `SMOKE_*`（0=全量）`RTS_MAX_DURATION` `EVAL_SEED` |
| 数据 | `ASSETS_DIR` 及各数据集路径、`RTS_VIDEO` |
| 打分 | `PARAFORMER_MODEL` `SPEAKER_CKPT` `S3PRL_REPO` `WAVLM_LARGE_PT` `ONNX_MODEL_DIR` |

优先级：命令行参数 > 环境变量 > `config.env`。

Ascend 上请保持默认（否则精度任务可能异常或崩溃）：

```bash
GGML_CANN_WEIGHT_NZ=off
GGML_CANN_ACL_GRAPH=off
```

### 数据与权重布局

数据集和打分模型不入库，统一放在 `ASSETS_DIR`（默认 `evaluation/appendix/`）。下载后软链或改路径均可：

```text
appendix/
├── videomme/test-00000-of-00001.parquet
├── videomme/data/
├── daily-omni/daily_omni.jsonl          # 同目录放音视频
├── seedtts_testset_zh/zh/meta.lst
├── paraformer-zh/                       # WER
├── Step-Audio-2-mini/token2wav/         # prompt bundle ONNX
├── s3prl/                               # SIM backbone 源码
├── wavlm_large.pt
└── wavlm_large_finetune.pth             # 缺失则跳过 SIM
```

下载说明见各子目录 README。模型权重（`MODEL_DIR`）单独配置，不要放进 `appendix/`。

---

## 3. 快速跑通

先 smoke，确认编译、数据、推理与打分链路：

```bash
cd evaluation
./run_all.sh --smoke 2
```

通过后再跑全量：

```bash
./run_all.sh --full
```

按需选择任务或跳过编译：

```bash
./run_all.sh --tasks videomme,rts --smoke 2
./run_all.sh --tasks videomme,daily-omni,tts,rts --full --no-build
./run_eval.sh tts --smoke 5
```

覆盖模型或卡号：

```bash
./run_all.sh --model /path/to/MiniCPM-o-4_5-Q4_K_M.gguf
./run_all.sh --devices 4,5,6,7
./run_all.sh --device-count 2
```

### 手动编译（可选）

`run_all.sh` 默认会按任务编译对应 target。若需手动编译：

```bash
cd ..   # 仓库根目录
cmake -B build -DGGML_CANN=ON -DSOC_TYPE=Ascend910 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j \
      --target llama-omni-eval-cli llama-omni-eval-daily-cli llama-omni-tts-eval
```

NVIDIA 将 `-DGGML_CANN=ON -DSOC_TYPE=Ascend910` 换成 `-DGGML_CUDA=ON`。构建产物目录需与 `EVAL_BIN_DIR` 一致（默认 `build/bin`）。

| 任务 | CMake target | 源文件 |
|------|--------------|--------|
| videomme | `llama-omni-eval-cli` | `tools/omni/omni-eval-cli.cpp` |
| daily-omni | `llama-omni-eval-daily-cli` | `tools/omni/omni-eval-daily-cli.cpp` |
| tts | `llama-omni-tts-eval` | `tools/omni/omni-tts-eval.cpp` |
| rts | `llama-omni-server` | `tools/server/server-omni.cpp` |

---

## 4. 结果与指标

每次运行产物在 `output/<时间戳>/`：

```text
output/<时间戳>/
├── build.log
├── videomme.log / videomme_output.json
├── daily-omni.log / daily_omni_output.json
├── tts.log / tts_seed/
├── rts.log / rts_runs/
├── metrics_<任务>.json
└── summary_<任务>.json
```

| 指标 | 读取位置 |
|------|----------|
| videomme / daily-omni 准确率 | pipeline 输出的 `Accuracy: n/m = x%` |
| 官方 Overall | 评分脚本输出（**仅全量**有） |
| WER | `tts_seed/wav_res_ref_text.wer` 末尾 `WER:` / `WER_NORMALIZED:` |
| SIM | `tts_seed/wav_res_ref_text.sim` 的 `ASV:` / `ASV-var:` |
| RTF、SPEAK→wav 延迟 | `eval_e2e_report.json`（多视频见 `batch_avg_report.json`） |

重新打印某次汇总：

```bash
./run_eval.sh --summarize --run-dir output/20260806_111206
```

### RTF 口径（速度成绩）

RTF = 稳定帧上的模型计算时间 / 对应音频时长（pooled ratio：`Σ compute / Σ audio`）。

每个语音 turn 去掉首帧（冷启动）与含最终 flush 的尾帧，再对剩余帧汇总。单帧计算时间为：

```text
compute = max(VPM, APM) + LLM_prefill + LLM_decode + TTS + token2wav
```

不含 judge 侧临时文件与 HTTP 往返；SPEAK→wav 为单独的端到端延迟。

仓库自带示例视频 `judge-final/assets/video/omni_duplex1.mp4` rtf值为1.1609，用于验证链路和参考，**不是最终测试集**，请勿针对其特化。

---

## 5. 提交规范

### 上传前自测

至少在裸机完成一次：

```bash
cd evaluation
./run_all.sh --smoke 2
```

自测通过标准：

- 四个任务均成功结束，无 CLI 超时/反复重启
- Video-MME / Daily-Omni 无明显大量空答案或纯换行
- TTS 能生成 wav 并产出 WER/SIM
- RTS 能输出 RTF 均值

提交性能成绩前，请用固定模型、数据、`EVAL_SEED` 与输入跑完整评测。

### 不可修改文件

正式评测会用基线覆盖并校验下列内容，**参赛代码不得修改**：

```text
evaluation/
tools/omni/omni-eval-cli.cpp
tools/omni/omni-eval-daily-cli.cpp
tools/omni/omni-tts-eval.cpp
tools/omni/CMakeLists.txt
```

改动这些文件不会进入最终测评，并可能触发完整性校验失败。优化应放在模型执行路径、后端算子或其他允许修改的实现中。上传前确认工作区未误改上述路径。

---

## 6. 常见问题

**Python 环境**  
精度与 TTS 打分用 `EVAL_PYTHON`，RTS 用 `RTS_PYTHON`，可指向同一 venv。PyTorch 需按平台单独安装。

**F16 权重**  
必须保持 `GGML_CANN_WEIGHT_NZ=off`，否则可能出现空串、换行复读等异常输出。

**Ascend ACL Graph**  
必须保持 `GGML_CANN_ACL_GRAPH=off`，否则 vision encode 阶段可能因非法同步拷贝直接 abort。

**SIM 打分**  
离线环境请预先配置好 `WAVLM_LARGE_PT` 与 `S3PRL_REPO`；`run_eval.py` 会把权重链入 s3prl 缓存目录。若日志提示找不到 `wavlm_large.pt`，先修正 `config.env` 再跑。SIM 为单进程 CPU；`--smoke N` 表示**每张卡**前 N 条，若只想总共 N 条请加 `--device-count 1`。

**官方评分**  
smoke / 子集模式下会跳过官方 Overall（子集无法满足 short/medium/long 各 300 视频的断言）。需要官方分请跑 `--full`。

