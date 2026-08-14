# MiniCPM-o 评测套件

本套件用于跑通下列四项评测并按规范提交代码。

| 任务 | 数据集 | 指标 | 依赖的 C++ target |
|------|--------|------|-------------------|
| `videomme` | Video-MME（900 视频 / 2700 题） | 选择题准确率 | `llama-omni-eval-cli` |
| `daily-omni` | Daily-Omni（1197 题，音视频交错） | 选择题准确率 | `llama-omni-eval-daily-cli` |
| `tts` | Seed-TTS 中文（2020 条） | WER / SIM(ASV) | `llama-omni-tts-eval` |
| `rts` | 双工短视频 | RTF / SPEAK→wav 延迟 | `llama-omni-server` |

四项任务共用 `config.env` 与入口脚本，评测 CLI / server 都在仓库主干里。

---

## 1. 环境准备

推荐 Linux aarch64 + Ascend 910。需要预先安装：

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

还需要安装与当前平台匹配的 `torch` / `torchaudio` / `torchvision`（Ascend 请用平台提供的兼容 wheel）。SIM 打分需要本地 `s3prl` 源码，路径写在 `S3PRL_REPO`。`decord` 可选，aarch64 装不上时会自动改用系统 `ffmpeg`。

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
| 路径 | `LLAMACPP_ROOT` `EVAL_BIN_DIR` `OMNI_SERVER_BIN` `ASCEND_ENV` `RTS_BASE_PORT` `RTS_VIDEO_DIR` `RTS_TEST_CASE_DIR` |
| 设备 | `DEVICE_ENV_VAR` `DEVICE_IDS` `DEVICE_COUNT` `RTS_DEVICE_ID` `RTS_DEVICE_IDS` `RTS_DEVICE_COUNT` |
| 样本 | `SMOKE_*`（0=全量）`RTS_MAX_DURATION` `RTS_VIDEO_COUNT` `RTS_MIN_CORE_FRAMES` `RTS_MAX_RETRIES` `RTS_ROTATION_ROUNDS` `EVAL_SEED` |
| 数据 | `ASSETS_DIR` 及各数据集路径、`RTS_VIDEO` `RTS_VIDEO_DIR` `RTS_ASSIGNMENT_MODE` |
| 打分 | `PARAFORMER_MODEL` `SPEAKER_CKPT` `S3PRL_REPO` `WAVLM_LARGE_PT` `ONNX_MODEL_DIR` |

优先级：命令行参数 > 环境变量 > `config.env`。

RTS 配置项：

- `RTS_TEST_CASE_DIR`：预切分输入根目录，优先级最高。直接读取各子目录中的 `*_test_case_NNNN.wav/.jpg`，不重新解码 MP4、不自行取帧。
- `RTS_VIDEO` / `RTS_VIDEO_DIR`：仅在 `RTS_TEST_CASE_DIR` 留空时生效的实时解码 MP4 路径，正式评测不走这条。
- `RTS_VIDEO_COUNT`：最多评测解析后列表中的前 N 个；默认 `0` 不截断。
- `RTS_DEVICE_IDS` / `RTS_DEVICE_COUNT`：RTS worker 卡号与数量。留空时先回退到单卡项 `RTS_DEVICE_ID`，再回退到通用 `DEVICE_IDS`。worker 数取 `min(卡数, 输入数)`。
- `RTS_BASE_PORT`：worker 0 的 server 端口，后续 worker 使用 `RTS_BASE_PORT + worker_id`。
- `RTS_ASSIGNMENT_MODE` / `RTS_ROTATION_ROUNDS`：`round_robin` 单轮交错分配；`rotating_groups` 先连续均衡分组再跨卡轮换，N 轮后每个输入在每张卡上各跑一次。轮数不能超过 worker 数，单卡只能用 `round_robin` + 1 轮。
- `RTS_SEND_INTERVAL_S`：相邻 frame 的提交间隔，即"实时"的定义。预切分输入同样按这个节奏逐帧发送，不会背靠背灌入。
- `RTS_MAX_DURATION`：单个输入最多评测多少秒。
- `RTS_PAD_BEFORE` / `RTS_PAD_AFTER`：输入前后静音 padding 秒数；预切分输入均为 `0`，MP4 路径尾部默认为 `2`。
- `RTS_MODEL_LOAD_SLEEP_S` / `RTS_READY_TIMEOUT_S`：每次启动或重启 server 前的等待时间与健康检查超时。
- `RTS_MIN_CORE_FRAMES`：整批累计的有效 core 帧下限，不足时主 RTF 不可用。模板里的 `3` 只够单个输入跑通自测，正式评测的门槛远高于此。
- `RTS_MAX_RETRIES`：单个输入因数据链路或基础设施失败后的重试次数。`0` 表示不重试——性能不达标不允许靠重跑挑更快的结果。

为保证输入之间状态隔离，**同一 worker 的相邻输入之间会停止并重启 C++ server、重新加载模型**，而不只是清空 KV cache。最后一个输入完成后直接停止，不再多重启一次。

RTS 自测输入需要先生成一次（结果在 `.gitignore` 里，不入库）：

```bash
python3 judge-final/scripts/make_test_case.py
```

它把仓库自带的样例视频按与正式评测相同的参数切成 1 秒 WAV/JPG，输出到 `judge-final/assets/test_case/`，也就是 `config.env` 里 `RTS_TEST_CASE_DIR` 的默认值。想用自己的视频就把路径传给同一个脚本。

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
python3 judge-final/scripts/make_test_case.py   # 只需跑一次，生成 RTS 输入
./run_all.sh --smoke 2
```

通过后再跑全量：

```bash
./run_all.sh --full
```

任务默认顺序为 `rts,videomme,daily-omni,tts`。rts 只要几分钟，精度三项是小时级，所以速度链路一旦有问题会立刻停下来，不再往下跑；想让它失败后继续跑精度任务加 `--keep-going`。

按需选择任务或跳过编译：

```bash
./run_all.sh --tasks videomme,rts --smoke 2
./run_all.sh --tasks rts,videomme,daily-omni,tts --full --no-build
./run_eval.sh tts --smoke 5
./run_eval.sh videomme --full --videomme-sample-ratio 0.5
```

Video-MME 的比例采样以视频为单位，按 `duration / domain / sub_category` 分层后在每层
等距取样；`0.5` 会从每个 10 视频分层中固定取 5 个，共 450 视频 / 1350 题。算法不使用
随机数，同一数据集每次选择相同。也可在 `config.env` 设置 `VIDEOMME_SAMPLE_RATIO`；
`--smoke` 优先用于链路检查，此时不会再叠加比例采样。

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

NVIDIA 平台把 `-DGGML_CANN=ON -DSOC_TYPE=Ascend910` 换成 `-DGGML_CUDA=ON`。构建产物目录需与 `EVAL_BIN_DIR` 一致（默认 `build/bin`）。

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
| RTF、SPEAK→wav 延迟 | 批次口径见 `rts_runs/<批次>/batch_pooled_report.json`，单个输入见对应 session 的 `eval_e2e_report.json` |
| RTS 有效性 | `batch_pooled_report.json` 的 `batch_validity`，不通过时不出主 RTF |

重新打印某次汇总：

```bash
./run_eval.sh --summarize --run-dir output/20260806_111206
```

### RTF 口径（速度成绩）

RTF = 稳定帧上的模型计算时间 / 对应音频时长（pooled ratio：`Σ compute / Σ audio`）。

每个语音 turn 去掉首帧（冷启动）与含最终 flush 的尾帧，剩下的叫 core 帧，只对 core 帧汇总。单帧计算时间为：

```text
compute = max(VPM, APM) + LLM_prefill + LLM_decode + TTS + token2wav
```

不含 judge 侧临时文件与 HTTP 往返；SPEAK→wav 为单独的端到端延迟。RTF 小于 1 才表示算得比实时快。

#### 正式评测怎么跑

输入是一组不公开的预切分音视频片段，评测流程与本目录的代码一致，只有三处配置不同：

- **输入集合**：多个片段，内容和数量暂不公布。
- **多卡轮换**：`RTS_ASSIGNMENT_MODE=rotating_groups`，输入分组后跨卡轮换，轮数等于卡数，每个输入在每张卡上各跑一次，抵消卡间差异。
- **core 帧门槛**：`RTS_MIN_CORE_FRAMES` 取远高于模板里那个 `3` 的值，靠多输入池化满足。

成绩是整批的 pooled 值 `Σ 所有合法 core 的 compute / Σ 对应音频`，不是先算每个输入的 RTF 再取平均——后者会让只有几帧的短输入和几十帧的长输入等权。

每个输入还要通过有效性检查才会进入汇总，任一输入不合法则整批不出成绩。检查的是归帧和因果关系是否自洽，例如：输入 chunk 有没有被丢弃、模态编码有没有静默失败、SPEAK 帧是否都产出了 WAV、WAV 是否归到了正确的源帧、是否出现负的因果延迟、非尾帧 TTS 的 token 数是否正常。这些都是异步流水线出竞态时的特征，不属于性能指标，但只有它们成立，RTF 才有意义。

#### 计时口径不得改动

RTF 的分子来自 server 上报的各阶段耗时（`vpm_ms` / `apm_ms` / `llm_prefill_ms` / `cost_llm_ms` 走 SSE metrics，`tts_ms` / `token2wav_ms` 走 `stage_timing.jsonl`）。这些字段的含义、计时起止点和上报时机都属于评测口径，**优化实现可以改，计时本身不能改**。

正式评测会对上报的耗时做核对，也会人工复核 diff。上报值与实际执行对不上的提交按不合规处理。

#### 自测数值怎么看

自测跑的是单个样例输入，用来验证链路和有效性，**不用来预测成绩**。基线 F16 在这个样例上的 core RTF 大致落在 `1.1~1.2`，单次跑之间就有这么大的跨度：这个样例只有 3 个 core 帧，样本量本来就不够，正式评测靠多输入池化把这个抖动压下去。正式测试集上的基线同样在 `1.1` 这个量级，但不同输入集合之间绝对值会差 10% 上下。所以自测数字只适合和你自己改动前的数字比，看相对变化。

样例视频 `judge-final/assets/video/omni_duplex1.mp4` 是公开的链路验证素材，**不是最终测试集**，针对它做特化没有意义。

#### 改这些变量改不了正式成绩

正式评测时，下列变量会被测评方替换成固定值，选手改本地这份不会影响线上成绩。这里逐一说明每个变量的作用，是为了让本地自测和正式评测保持同一口径——改了之后自测照样会出数字，但和正式评测对不上：

| 变量 | 作用 / 改动后果 |
|------|-------------------|
| `RTS_TEST_CASE_DIR` | 留空会退回实时解码 MP4，ffmpeg 取帧的耗时和抖动会混进速度成绩 |
| `RTS_SEND_INTERVAL_S` | 调大会放慢输入喂入节奏，模型获得富余时间，偏离实时场景 |
| `RTS_PAD_BEFORE` / `RTS_PAD_AFTER` | 给预切分输入补 padding 会多出没有真实输入的帧 |
| `RTS_MODEL_LOAD_SLEEP_S` | 相邻输入之间整体重启 server 是为了状态隔离，跳过等待会让下一个输入在未就绪状态上起跑 |
| `RTS_MIN_CORE_FRAMES` | 调低会让样本量不足的 RTF 也被当成有效成绩 |
| `RTS_MAX_RETRIES` | 调高等于允许重跑挑更快的结果 |
| `EVAL_SEED` | 改动后各队之间不再可比 |
| `GGML_CANN_WEIGHT_NZ` / `GGML_CANN_ACL_GRAPH` | F16 上默认 `off` 以避免输出异常或 abort；若后端优化需要开启，可在提交时一并上传自己的环境变量 |

---

## 5. 提交规范

### 上传前自测

至少在裸机完成一次：

```bash
cd evaluation
python3 judge-final/scripts/make_test_case.py
./run_all.sh --smoke 2
```

自测通过标准：

- 四个任务均成功结束，无 CLI 超时/反复重启
- Video-MME / Daily-Omni 无明显大量空答案或纯换行
- TTS 能生成 wav 并产出 WER/SIM
- RTS 输出 RTF，且批次报告里 `batch_validity` 的 `data_valid` 与 `realtime_eligible` 均为 `true`

RTS 这一项最值得关注：`batch_pooled_report.json` 的 `batch_validity` 会列出判定不通过的原因。它不通过说明双工链路存在归帧或竞态问题，正式评测同样不会给出成绩。线上把 rts 排在第一位跑，改动没验证过就提交，等于把几个小时的机时浪费在必然失败的一轮上。

自测只验证流程，不预测成绩，原因见上一节的 RTF 口径。

### 不可修改文件

正式评测会用基线覆盖并校验下列内容，**参赛代码不得修改**：

```text
evaluation/
tools/omni/omni-eval-cli.cpp
tools/omni/omni-eval-daily-cli.cpp
tools/omni/omni-tts-eval.cpp
tools/omni/CMakeLists.txt
```

改动这些文件不会进入最终测评，并可能触发完整性校验失败。优化应放在模型执行路径、后端算子或其他允许修改的实现中。上传前请确认工作区没有误改上述路径。

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
smoke 的 `head(N)` 模式会跳过官方 Overall，因为它可能截断单个视频。分层比例采样会放宽
完整性断言，并照常输出同口径的分类准确率和 Overall。
