# MiniCPM-o 评测套件

三项精度 + 一项速度，共用一份配置、一个入口。

| 任务 | 数据集 | 指标 | 依赖的 C++ target |
|------|--------|------|-------------------|
| `videomme` | Video-MME（900 视频 / 2700 题） | 选择题准确率 | `llama-omni-eval-cli` |
| `daily-omni` | Daily-Omni（1197 题，音视频交错） | 选择题准确率 | `llama-omni-eval-daily-cli` |
| `tts` | Seed-TTS 中文（2020 条） | WER / SIM(ASV) | `llama-omni-tts-eval` |
| `rts` | 双工短视频 | RTF / SPEAK→wav 延迟 | `llama-omni-server`（主干自带） |

## 快跑

```bash
cd evaluation
./run_all.sh --smoke 2        # 三个精度测试各 2 条，冒烟
./run_all.sh                  # 用 config.env 里的 SMOKE_* 值
./run_all.sh --full           # 精度测试跑全量
```

只跑某几个任务：

```bash
./run_all.sh --tasks videomme,rts
./run_eval.sh tts --smoke 5          # 单任务（要求补丁已打好、target 已编译）
```

## 为什么要打补丁

三个精度测试各依赖一个不在主干里的常驻推理 CLI，由本目录下的 patch 引入：

```
videomme/llama-omni-eval-cli.patch          → tools/omni/omni-eval-cli.cpp
daily-omni/llama-omni-eval-daily-cli.patch  → tools/omni/omni-eval-daily-cli.cpp
tts_seed/llama-omni-tts-eval.patch          → tools/omni/omni-tts-eval.cpp
                                              （还改 omni.cpp/omni.h/token2wav）
```

三个补丁都在 `tools/omni/CMakeLists.txt` 的同一处插 target，**不能同时打**。所以
`run_all.sh` 对每个任务走一遍：

```
git apply <patch> → cmake --build --target <target> → 跑测试 → git apply -R <patch>
```

补丁新建的 `.cpp` 是 untracked，`git apply -R` 不会删，脚本会一并清掉。跑完
`git status tools/` 应该是干净的。开跑前脚本也会检查 `tools/omni` 有没有残留改动，
有的话直接退出，免得打补丁失败。

手动做同一件事：

```bash
cd ..                                        # 到仓库根
git apply evaluation/videomme/llama-omni-eval-cli.patch
cmake -B build-kunpeng -DGGML_CANN=ON -DSOC_TYPE=Ascend910 -DCMAKE_BUILD_TYPE=Release
cmake --build build-kunpeng --target llama-omni-eval-cli -j 64
cd evaluation && ./run_eval.sh videomme --smoke 2
cd .. && git apply -R evaluation/videomme/llama-omni-eval-cli.patch
rm -f tools/omni/omni-eval-cli.cpp
```

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
| Python / 推理参数 | `EVAL_PYTHON` `RTS_PYTHON` `CTX_SIZE` `GGML_CANN_ACL_GRAPH` |

优先级：命令行参数 > 环境变量 > `config.env`。常用覆盖：

```bash
./run_all.sh --model ~/o45-gguf/MiniCPM-o-4_5-Q4_K_M.gguf
./run_all.sh --devices 4,5,6,7          # 用后四张卡
./run_all.sh --device-count 2           # 只用 DEVICE_IDS 的前两张
```

### 卡的分配

精度测试是每卡一个常驻 CLI 进程、卡间并行分数据：`DEVICE_IDS` 给出可用物理卡，
第 n 个 worker 拿列表里第 n 张。`DEVICE_ENV_VAR` 决定用哪个变量名传给子进程，
本机 Ascend 是 `ASCEND_RT_VISIBLE_DEVICES`，NVIDIA 机器改成 `CUDA_VISIBLE_DEVICES`。

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

重新打印某次的汇总：

```bash
./run_eval.sh --summarize --run-dir output/20260806_111206
```

## 本机上的几个坑

**必须关 `GGML_CANN_ACL_GRAPH`。** ggml-cann 默认开 ACL graph capture，而
capture 期间不允许同步 `aclrtMemcpy`，`vision_image_batch_encode` 偏偏要逐帧往
device 拷数据，于是 abort：

```
CANN error EE9999: rtMemcpy execution failed,
reason=the current capture mode does not support this operation (107030)
```

`config.env` 里已置 `off`。顺带一提，关掉后精度测试反而快很多（单题约 5 分钟 →
14 秒），因为 capture 每轮都 miss 重建。

**没有 decord。** PyPI 不发 aarch64 轮子，视频抽帧退回 ffmpeg。原本的 ffmpeg 实现
是 `-vf fps=1 -vframes 64`，只覆盖视频前 64 秒，而 Video-MME 的 medium/long 是十几
分钟到一小时的长视频。已改成先算出 decord 路径会取的帧下标、换成时间戳，再逐个用
input seek 抓单帧，见 `videomme/eval_cpp_video_prep.py`。

**`judge-final/.venv` 用不了**，那是 x86_64 的（`_cffi_backend...x86_64-linux-gnu.so`）。
RTS 走 `RTS_PYTHON`（默认系统 `python3`），需要 `httpx requests websockets numpy soundfile`。

**精度测试的解释器**要带 `pandas pyarrow torch torchaudio funasr jiwer librosa
soundfile onnxruntime s3tokenizer zhconv scipy transformers`，配在 `EVAL_PYTHON`。

**SIM 打分的 WavLM 权重必须预先摆进 s3prl 缓存。** s3prl 的 `wavlm_large` 入口只
认 `https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt`，
下载目录硬编码成 `$HOME/.cache/s3prl/download`（`s3prl/util/download.py` 的
`_download_dir`，没有环境变量可改），且只在文件不存在时才联网。本机不通外网，不摆
好的后果很隐蔽：每一对都先等一次连接超时（实测 130→280s 递增），异常又被
`verification_pair_list_v2.py` 的 `except: print; continue` 吞掉，`model` 永远是
`None`，于是 16 对全在重试，跑几十分钟一个分数都落不下来。

`run_eval.py` 的 `ensure_s3prl_cache()` 会在跑 tts 前把 `WAVLM_LARGE_PT` 按 URL 的
sha256 命名软链进那个缓存目录。要是它打印「没找到本地 wavlm_large.pt」，先去修
`config.env` 里的路径，别急着开跑。

那个 `except` 会连着 print 一起把异常吞进 stdout，而 stdout 重定向到文件时是块缓冲
的，进程不结束就看不到。排查这类问题要么等它退出、要么用 `python3 -u` 单跑一对。

**SIM 是单进程的。** WER 那步按 `GPUS_PER_NODE` 切成多线程并行，SIM 一个进程跑到底，
只能 CPU（Ascend 上没有 torch 后端）。另外 `NUM_SAMPLES` 的语义是「**每个 rank** 前
N 条」，8 卡 + `--smoke 2` 实际是 16 条，smoke 想真只跑 2 条要配 `--device-count 1`。

进度看 `tts_seed/logs/sim_*.log`（脚本把它重定向走了，终端上看不到；终端里那个
`100%|2020/2020` 是前一步扫 `meta.lst` 生成配对清单，不是 SIM 本身）。

**smoke 模式下官方评分被跳过。** `videomme/eval_your_result.py` 硬断言 short/medium/
long 各 300 个视频，子集必然 assert 失败，所以 `--limit > 0` 时自动加 `--skip-scoring`，
只报 pipeline 自己算的准确率。要官方分就跑 `--full`。

## 已知待确认

评测 prompt 已验证生效（同一张图换 prompt，输出跟着变），但有两点没定：

- `videomme/eval_cpp_config.py` 的 `USER_PROMPT_TEMPLATE` 拼出来是
  `...correct answer.Highlight the applicable choices...`，句号后没有分隔符；
  且与 Video-MME 官方给的 `Select the best answer to the following
  multiple-choice question based on the video...` 不是一套。目前按现状保留。
- `omni_init` 无条件把非双工 omni 的 system prompt 设成音色克隆模板（含
  `<|audio_start|>`），`use_tts=false` 也不例外；eval CLI 又传相对路径的 ref_audio
  导致加载失败。结果评测时 system prompt 是「模仿音频样本的音色…`<|audio_start|>`
  `<|audio_end|>`你的任务是用这种声音模式来当一个助手…」，带一个空音频槽和语音助手
  人格。`omni.cpp` 里本来有 `has_ref_audio_slot == false → plain system message`
  的分支，纯文本评测本该走那条。尚未改动。
