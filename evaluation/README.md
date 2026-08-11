# MiniCPM-o 评测套件

三项精度 + 一项速度，共用一份配置、一个入口。

| 任务 | 数据集 | 指标 | 依赖的 C++ target |
|------|--------|------|-------------------|
| `videomme` | Video-MME（900 视频 / 2700 题） | 选择题准确率 | `llama-omni-eval-cli` |
| `daily-omni` | Daily-Omni（1197 题，音视频交错） | 选择题准确率 | `llama-omni-eval-daily-cli` |
| `tts` | Seed-TTS 中文（2020 条） | WER / SIM(ASV) | `llama-omni-tts-eval` |
| `rts` | 双工短视频 | RTF / SPEAK→wav 延迟 | `llama-omni-server`（主干自带） |

## 裸机准备与启动

评测必须直接在用户自己的裸机环境跑通；不需要 Kubernetes、k3s 或评测服务端容器。
推荐 Linux aarch64 + Ascend 910，系统需要预先安装：

- CANN Toolkit，并能找到 `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- CMake、C/C++ 编译器和 Git
- Python 3.10+、pip
- `ffmpeg`（Video-MME 长视频抽帧的默认回退后端）
- `rubberband`（使用 `pyrubberband` 音频变速时需要）

创建 Python 环境：

```bash
python3 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install -U pip
python -m pip install -r evaluation/requirements.txt
```

`torch`、`torchaudio`、`torchvision` 需要另外安装与机器架构和运行时匹配的版本；Ascend
机器应使用平台提供的兼容 wheel。SIM 还需要一份本地 `s3prl` 源码，路径通过
`S3PRL_REPO` 配置。`decord` 是可选项，aarch64 上装不上时自动改用系统 `ffmpeg`。

复制并检查 `evaluation/config.env`，至少确认模型、数据、打分模型、卡号和 Python：

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

先跑 smoke test，确认编译、数据读取、四项推理和打分链路都能走通：

```bash
cd evaluation
./run_all.sh --smoke 2
```

smoke 通过后再跑全量：

```bash
./run_all.sh --full
```

也可以只跑某几个任务，或复用已经编译好的二进制：

```bash
./run_all.sh --tasks videomme,rts --smoke 2
./run_all.sh --tasks videomme,daily-omni,tts,rts --full --no-build
./run_eval.sh tts --smoke 5
```

## 依赖的 target

四个任务各依赖一个 CMake target，全都在主干里，**不需要打补丁**：

```
videomme    llama-omni-eval-cli        tools/omni/omni-eval-cli.cpp
daily-omni  llama-omni-eval-daily-cli  tools/omni/omni-eval-daily-cli.cpp
tts         llama-omni-tts-eval        tools/omni/omni-tts-eval.cpp
rts         llama-omni-server          tools/server/server-omni.cpp
```

三个精度评测 CLI 互不依赖，可以一次编完。`run_all.sh` 开跑前就是这么做的：把本次
要跑的任务对应的 target 一起交给 cmake，编一次，然后依次跑测试。

手动编：

```bash
cd ..                                        # 到仓库根
cmake -B build -DGGML_CANN=ON -DSOC_TYPE=Ascend910 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j \
      --target llama-omni-eval-cli llama-omni-eval-daily-cli llama-omni-tts-eval
cd evaluation && ./run_eval.sh videomme --smoke 2
```

上面用的是 Ascend 的 cmake 开关，NVIDIA 换成 `-DGGML_CUDA=ON`。构建目录名要和
`config.env` 的 `EVAL_BIN_DIR` 对上（默认 `build/bin`）。

三个 CLI 都是常驻进程：模型只加载一次，Python 侧把成百上千条样本喂进同一个进程，
每张卡起一个。videomme / daily-omni 走 stdin/stdout 上的 JSONL 协议，tts 走 TSV
manifest。协议细节见各 `.cpp` 顶部的注释。

TTS 评测额外用到 `omni.cpp` 里的 `eval_tokens_with_hidden`、`tts_emb_text`、
`tts_projector_semantic`、`normalize_l2_per_token`,这几个函数在 `omni.h` 里有声明。
它跑的是 teacher-forced 合成（condition 用给定目标文本的 hidden states，而不是模型
自己生成的文本），这是 seed-tts-eval 的口径要求，所以不能复用框架的流式 TTS 路径。
用的公式跟主干 `tts-condition-graph.cpp` 那个 fused graph 一致。

## 配置

只改 `config.env`，分七块：

| 块 | 关键项 |
|----|--------|
| 模型权重 | `MODEL_DIR` `MODEL_LLM`（精度） `RTS_MODEL_LLM`（速度） `TTS_MODEL_PATH` |
| 服务/二进制 | `LLAMACPP_ROOT` `EVAL_BIN_DIR` `OMNI_SERVER_BIN` `ASCEND_ENV` |
| 算力卡 | `DEVICE_ENV_VAR` `DEVICE_IDS` `DEVICE_COUNT` `RTS_DEVICE_ID` |
| smoke 数量 | `SMOKE_VIDEOMME` `SMOKE_DAILY_OMNI` `SMOKE_TTS`（0=全量） `RTS_MAX_DURATION` |
| 数据集 | `ASSETS_DIR` 及各数据集路径、`RTS_VIDEO` |
| TTS 打分模型 | `PARAFORMER_MODEL` `SPEAKER_CKPT` `S3PRL_REPO` `ONNX_MODEL_DIR` |
| Python / 推理参数 | `EVAL_PYTHON` `RTS_PYTHON` `CTX_SIZE` `GGML_CANN_WEIGHT_NZ` `GGML_CANN_ACL_GRAPH` `EVAL_SEED` |

优先级：命令行参数 > 环境变量 > `config.env`。常用覆盖：

```bash
./run_all.sh --model /path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf
./run_all.sh --devices 4,5,6,7          # 用后四张卡
./run_all.sh --device-count 2           # 只用 DEVICE_IDS 的前两张
```

### 数据与权重

数据集和打分模型都不入库，统一挂在 `ASSETS_DIR`（默认 `evaluation/appendix/`，已被
`.gitignore` 忽略）。自己下载后软链进去即可，也可以直接改 `config.env` 里对应的路径：

```
appendix/
├── videomme/test-00000-of-00001.parquet   Video-MME 题目
├── videomme/data/                         Video-MME 视频
├── daily-omni/daily_omni.jsonl            Daily-Omni 标注（同目录放音视频）
├── seedtts_testset_zh/zh/meta.lst         Seed-TTS 中文测试集
├── paraformer-zh/                         WER 用 ASR 模型
├── Step-Audio-2-mini/token2wav/           prompt bundle 提取用的 ONNX
├── s3prl/                                 git clone 的 s3prl，SIM 的 WavLM backbone
├── wavlm_large.pt                         WavLM-large 预训练权重
└── wavlm_large_finetune.pth               SIM 的 ECAPA 微调 ckpt（缺了就跳过 SIM）
```

下载地址见各子目录的 README。模型权重（`MODEL_DIR`）不放这里，单独配。

### 采样种子

`EVAL_SEED`（默认 42）是四个任务唯一的种子旋钮，`run_eval.py` 把它翻译成各流水线
认的变量名：`SAMPLER_SEED`（videomme / daily-omni）、`SEED`（tts）、
`OMNI_SAMPLER_SEED`（rts）。

必须固定的原因是 `common_params` 的默认值 `LLAMA_DEFAULT_SEED` 会让采样器每次从
`std::random_device` 重新播种，同一份构建重复跑分数就不一样，改动带来的真实变化和
采样噪声分不开。三个精度 CLI 都接受 `--seed`，RTS 走 server 的 `--seed` 与
`omni_init` 请求里的 `seed` 字段（两处取同一个值）。

验证过的确定性：同一 seed 连跑两遍 daily-omni，逐题预测完全一致；连跑两遍 tts，
生成的 wav 字节级相同（md5 一致，TTS 要走两千步自回归采样，这个比选择题敏感得多）。

想知道分数的噪声底线就换几个 seed 各跑一遍：

```bash
for s in 42 1337 20260807; do EVAL_SEED=$s ./run_all.sh --tasks videomme; done
```

### 卡的分配

精度测试是每卡一个常驻 CLI 进程、卡间并行分数据：`DEVICE_IDS` 给出可用物理卡，
第 n 个 worker 拿列表里第 n 张。`DEVICE_ENV_VAR` 决定用哪个变量名传给子进程，
Ascend 上是 `ASCEND_RT_VISIBLE_DEVICES`，NVIDIA 机器改成 `CUDA_VISIBLE_DEVICES`。

RTS 是单卡串行测延迟，`RTS_DEVICE_ID` 留空则按空闲显存自动挑（Ascend 每颗芯片
空闲时有约 3GB 驱动基线占用，判定阈值见 `judge-final/judge_support.py`）。

## 产物

```
output/<时间戳>/
├── build.log                  各 target 的编译输出
├── videomme.log               ← 每个任务的完整 stdout
├── videomme_output.json       ← 逐题结果
├── daily-omni.log
├── daily_omni_output.json
├── tts.log
├── tts_seed/                  生成的 wav + WER/SIM 明细
├── rts.log
├── rts_runs/<时间戳>/         judge 的 run_meta / 延迟报告
├── metrics_<任务>.json        ← 抽好的指标，汇总表就读这些
└── summary_<任务>.json
```

数值从这些位置提取：

| 指标 | 来源 |
|------|------|
| videomme / daily-omni 准确率 | pipeline stdout 的 `Accuracy: n/m = x%` |
| 官方 Overall | 评分脚本输出（`Overall: x%` / `Total: x% (n/m)`），**只有全量才有** |
| WER | `tts_seed/wav_res_ref_text.wer` 末尾的 `WER:` / `WER_NORMALIZED:` |
| SIM | `tts_seed/wav_res_ref_text.sim` 的 `ASV:` / `ASV-var:` |
| RTF、SPEAK→wav 延迟 | session 的 `eval_e2e_report.json`（多视频取 `batch_avg_report.json`） |

### RTF 计算方法

RTF 衡量生成一段音频需要多少模型计算时间。评测先按 C++ `stage_timing.jsonl` 中
`t2w.is_final`（即 `turn_eos` 的最终 flush）切分语音 turn，再按 `src_cnt` 将同一输入帧
产生的 TTS、token2wav 和 wav 归到一起。

每个 turn 只取中间稳定帧：

1. 去掉首帧。首帧包含 TTS/T2W 冷启动，且首个 wav 经常不足 1 秒。
2. 去掉包含最终 flush wav 的尾帧。尾音对应的 TTS 计算已发生在前面的帧，只保留这段
   音频会额外增加分母，使结果虚低；由于该帧的 decode/TTS 无法在两个 wav 间可靠拆分，
   所以尾帧整体排除。
3. 对剩余帧汇总，不对 turn 或短 wav 做等权平均。量化改变 turn 数或每个 turn 的 wav
   数量时，统计口径仍保持一致。

第 `i` 个稳定帧的计算时间为：

```text
compute_i = max(VPM_i, APM_i)
          + LLM_prefill_i
          + LLM_decode_i
          + TTS_i
          + token2wav_i
```

最终展示的 **RTF 均值** 是 pooled ratio：

```text
均值 = Σ compute_i / Σ audio_i
```

阶段分解使用同一个音频总时长作分母，因此
`encode + llm_prefill + llm_decode + tts + token2wav = 均值`。这里不包含 judge
侧临时文件和 HTTP 往返时间；SPEAK→wav 是单独报告的端到端延迟。短 smoke 视频可能只剩
很少的稳定帧，只用于检查链路，正式比较应使用固定的完整输入。

### 示例视频结果

当前仓库自带的 `evaluation/judge-final/assets/video/omni_duplex1.mp4` 使用 F16 权重
跑出以下结果：

```text
均值                     1.1812
分解                     encode 0.2528 + llm_prefill 0.015
                         + llm_decode 0.3981 + tts 0.2763
                         + token2wav 0.239
SPEAK→wav均值(ms)         1284.0
SPEAK→wav中位数(ms)       1220.9
```

**这只是用于验证评测链路和展示结果格式的示例视频，不是最终测试视频，也不代表最终测试
集的 RTF 或延迟。** 最终成绩以评测方提供的隐藏测试视频、固定环境和正式运行结果为准；
不要针对这个示例视频做特化。

重新打印某次的汇总：

```bash
./run_eval.sh --summarize --run-dir output/20260806_111206
```

## 上传前自测与不可修改文件

上传代码前，贡献者必须在裸机上至少完成一次：

```bash
cd evaluation
./run_all.sh --smoke 2
```

检查四个任务均为成功、没有 CLI 超时/重启、Video-MME/Daily-Omni 没有大量空答案或纯换行、
TTS 能生成 wav 并得到 WER/SIM、RTS 能输出 RTF 均值。建议再单独跑与改动相关的任务；
提交性能成绩前应使用固定模型、数据、seed 和输入跑完整评测。

正式评测会用基线版本覆盖并校验以下受保护内容，参赛代码不得修改：

```text
evaluation/
tools/omni/omni-eval-cli.cpp
tools/omni/omni-eval-daily-cli.cpp
tools/omni/omni-tts-eval.cpp
tools/omni/CMakeLists.txt
```

修改这些文件不会进入最终测评，且可能触发完整性校验失败。优化应放在模型执行、后端算子
或其他允许修改的实现中。上传前请确认本地工作区没有误改上述文件，并保留足够日志以便
复现结果。

## 注意事项

**Python 依赖见 `evaluation/requirements.txt`。** 精度与 TTS 打分使用
`EVAL_PYTHON`，RTS 使用 `RTS_PYTHON`；两者可以指向同一个完整环境。PyTorch 三件套
需要按平台单独安装，`s3prl` 使用本地源码目录。

**F16 权重必须关闭 `GGML_CANN_WEIGHT_NZ`。** ggml-cann 默认会将非量化 matmul 权重
转换成 CANN NZ 格式；当前实现下，F16 会出现 logits 退化，表现为大量空串、纯换行复读
或无关 grounding token。Q4/Q8 等量化权重不走这条转换路径。为保证不同权重统一且正确，
`config.env` 和 `run_eval.py` 都默认设置：

```bash
GGML_CANN_WEIGHT_NZ=off
```

**Ascend 上必须关 `GGML_CANN_ACL_GRAPH`。** ggml-cann 默认开 ACL graph capture，而
capture 期间不允许同步 `aclrtMemcpy`，`vision_image_batch_encode` 偏偏要逐帧往
device 拷数据，于是 abort：

```
CANN error EE9999: rtMemcpy execution failed,
reason=the current capture mode does not support this operation (107030)
```

四个任务都走 vision encode，`config.env` 里已统一置 `off`。关掉之后精度测试反而快
很多（单题约 5 分钟 → 14 秒），因为 capture 每轮都 miss 重建。

```bash
GGML_CANN_ACL_GRAPH=off
```

**没有 decord 时用 ffmpeg 抽帧。** PyPI 不发 aarch64 轮子，这种平台上视频抽帧退回
ffmpeg。直接用 `-vf fps=1 -vframes 64` 只能覆盖视频前 64 秒，而 Video-MME 的
medium/long 是十几分钟到一小时的长视频，所以实现改成先算出 decord 路径会取的帧下标、
换成时间戳，再逐个用 input seek 抓单帧，见 `videomme/eval_cpp_video_prep.py`。

**SIM 打分的 WavLM 权重要预先摆进 s3prl 缓存。** s3prl 的 `wavlm_large` 入口只认
`https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt`，下载目录
硬编码成 `$HOME/.cache/s3prl/download`（`s3prl/util/download.py` 的 `_download_dir`，
没有环境变量可改），且只在文件不存在时才联网。`run_eval.py` 的 `ensure_s3prl_cache()`
会在跑 tts 前把 `WAVLM_LARGE_PT` 按 URL 的 sha256 命名软链进那个缓存目录。

离线机器上不摆好的后果很隐蔽：每一对都先等一次连接超时，异常又被
`verification_pair_list_v2.py` 的 `except: print; continue` 吞掉，`model` 永远是
`None`，跑几十分钟一个分数都落不下来。要是脚本打印「没找到本地 wavlm_large.pt」，先
去修 `config.env` 里的路径再开跑。

**SIM 是单进程的。** WER 那步按 `GPUS_PER_NODE` 切成多线程并行，SIM 一个进程跑到底，
且只能 CPU（Ascend 上没有 torch 后端）。另外 `NUM_SAMPLES` 的语义是「**每个 rank** 前
N 条」，8 卡 + `--smoke 2` 实际是 16 条，smoke 想真只跑 2 条要配 `--device-count 1`。

**smoke 模式下官方评分被跳过。** `videomme/eval_your_result.py` 硬断言 short/medium/
long 各 300 个视频，子集必然 assert 失败，所以 `--limit > 0` 时自动加 `--skip-scoring`，
只报 pipeline 自己算的准确率。要官方分就跑 `--full`。
