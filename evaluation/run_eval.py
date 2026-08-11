#!/usr/bin/env python3
"""评测调度：把 config.env 映射到各子流水线并汇总指标。请用 run_eval.sh 启动。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SUITE_ROOT = Path(__file__).resolve().parent

TASKS = ["videomme", "daily-omni", "tts", "rts"]
ACCURACY_TASKS = ["videomme", "daily-omni", "tts"]


# ---------------------------------------------------------------- 配置读取


def cfg(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def cfg_int(key: str, default: int) -> int:
    raw = cfg(key)
    return int(raw) if raw else default


def device_ids() -> List[str]:
    ids = [x.strip() for x in cfg("DEVICE_IDS", "0").split(",") if x.strip()]
    n = cfg_int("DEVICE_COUNT", 0)
    if n > 0:
        ids = ids[:n]
    return ids or ["0"]


def require_file(path: str, what: str) -> None:
    if not path or not Path(path).is_file():
        raise SystemExit(f"[config] {what} 不存在: {path or '(空)'}")


# ---------------------------------------------------------------- 子进程执行


def run_step(
    name: str,
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
) -> Dict[str, Any]:
    """跑一个子流水线，输出同时进控制台和日志文件。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 72}\n[{name}] {' '.join(cmd)}\n[{name}] cwd={cwd}\n"
          f"[{name}] log={log_path}\n{'=' * 72}", flush=True)

    t0 = time.time()
    lines: List[str] = []
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
            lines.append(line)
        rc = proc.wait()

    elapsed = time.time() - t0
    status = "ok" if rc == 0 else f"failed(rc={rc})"
    print(f"[{name}] {status}  用时 {elapsed:.1f}s", flush=True)
    return {"rc": rc, "elapsed_s": round(elapsed, 1),
            "log": str(log_path), "stdout": "".join(lines)}


def base_env(extra: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    # EVAL_SEED → 各流水线认的变量名
    seed = cfg("EVAL_SEED", "42")
    env.setdefault("SAMPLER_SEED", seed)
    env.setdefault("SEED", seed)
    env.setdefault("OMNI_SAMPLER_SEED", seed)
    env.update({k: v for k, v in extra.items() if v is not None})
    return env


def _grep_last(text: str, pattern: str) -> Optional[str]:
    """取最后一个匹配（汇总行通常在文件末尾）。"""
    m = re.findall(pattern, text, re.MULTILINE)
    return m[-1] if m else None


# ---------------------------------------------------------------- 任务定义


def task_videomme(args, run_dir: Path) -> Dict[str, Any]:
    root = SUITE_ROOT / "videomme"
    ids = device_ids()
    cli = str(Path(cfg("EVAL_BIN_DIR")) / "llama-omni-eval-cli")
    require_file(cli, "llama-omni-eval-cli")
    require_file(cfg("MODEL_LLM"), "MODEL_LLM")
    require_file(cfg("PARQUET_PATH"), "PARQUET_PATH")

    limit = args.smoke_videomme
    out_json = run_dir / "videomme_output.json"

    cmd = [cfg("EVAL_PYTHON", "python3"), "eval_cpp_pipeline.py",
           "--num-gpus", str(len(ids)),
           "--limit", str(limit),
           "--output", str(out_json)]
    if args.skip_rerun:
        cmd.append("--skip-rerun")
    # 子集无法过官方评分断言，仅全量保留 scoring
    if limit > 0:
        cmd.append("--skip-scoring")

    env = base_env({
        "LLAMA_CLI_BIN": cli,
        "LLM_MODEL_PATH": cfg("MODEL_LLM"),
        "GGUF_MODEL_DIR": cfg("MODEL_DIR"),
        "PARQUET_PATH": cfg("PARQUET_PATH"),
        "VIDEO_DATA_DIR": cfg("VIDEO_DATA_DIR"),
        "CTX_SIZE": cfg("CTX_SIZE", "40960"),
        "NUM_GPUS": str(len(ids)),
        "DEVICE_ENV_VAR": cfg("DEVICE_ENV_VAR", "ASCEND_RT_VISIBLE_DEVICES"),
        cfg("DEVICE_ENV_VAR", "ASCEND_RT_VISIBLE_DEVICES"): ",".join(ids),
        "EXTRA_LD_LIBRARY_PATH": cfg("EVAL_BIN_DIR"),
        "GGML_CANN_WEIGHT_NZ": cfg("GGML_CANN_WEIGHT_NZ", "off"),
        "GGML_CANN_ACL_GRAPH": cfg("GGML_CANN_ACL_GRAPH", "off"),
    })

    res = run_step("videomme", cmd, root, env, run_dir / "videomme.log")
    res["metrics"] = {
        "n_samples": "全量 2700 题" if limit == 0 else f"前 {limit} 题",
        "准确率": _grep_last(res["stdout"], r"Accuracy: (\d+/\d+ = [\d.]+%)"),
        "官方Overall": _grep_last(res["stdout"], r"^Overall:\s*([\d.]+%)"),
        "output_json": str(out_json),
    }
    return res


def task_daily_omni(args, run_dir: Path) -> Dict[str, Any]:
    root = SUITE_ROOT / "daily-omni"
    ids = device_ids()
    cli = str(Path(cfg("EVAL_BIN_DIR")) / "llama-omni-eval-daily-cli")
    require_file(cli, "llama-omni-eval-daily-cli")
    require_file(cfg("MODEL_LLM"), "MODEL_LLM")
    require_file(cfg("ANNOTATION_PATH"), "ANNOTATION_PATH")

    limit = args.smoke_daily_omni
    out_json = run_dir / "daily_omni_output.json"

    cmd = [cfg("EVAL_PYTHON", "python3"), "eval_cpp_pipeline.py",
           "--num-gpus", str(len(ids)),
           "--limit", str(limit),
           "--output", str(out_json)]
    if args.skip_rerun:
        cmd.append("--skip-rerun")

    env = base_env({
        "LLAMA_CLI_BIN": cli,
        "LLM_MODEL_PATH": cfg("MODEL_LLM"),
        "GGUF_MODEL_DIR": cfg("MODEL_DIR"),
        "DATASET_DIR": cfg("DATASET_DIR"),
        "ANNOTATION_PATH": cfg("ANNOTATION_PATH"),
        "CTX_SIZE": cfg("CTX_SIZE", "40960"),
        "NUM_GPUS": str(len(ids)),
        "DEVICE_ENV_VAR": cfg("DEVICE_ENV_VAR", "ASCEND_RT_VISIBLE_DEVICES"),
        cfg("DEVICE_ENV_VAR", "ASCEND_RT_VISIBLE_DEVICES"): ",".join(ids),
        "EXTRA_LD_LIBRARY_PATH": cfg("EVAL_BIN_DIR"),
        "GGML_CANN_WEIGHT_NZ": cfg("GGML_CANN_WEIGHT_NZ", "off"),
        "GGML_CANN_ACL_GRAPH": cfg("GGML_CANN_ACL_GRAPH", "off"),
    })

    res = run_step("daily-omni", cmd, root, env, run_dir / "daily-omni.log")
    res["metrics"] = {
        "n_samples": "全量 1197 题" if limit == 0 else f"前 {limit} 题",
        "准确率": _grep_last(res["stdout"], r"Accuracy: (\d+/\d+ = [\d.]+%)"),
        "官方Overall": _grep_last(res["stdout"], r"Total:\s*([\d.]+%\s*\(\d+/\d+\))"),
        "output_json": str(out_json),
    }
    return res


# s3prl wavlm_large 缓存按该 URL 的 sha256 命名
S3PRL_WAVLM_URL = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt"


def ensure_s3prl_cache() -> Optional[str]:
    """把本地 wavlm_large.pt 链到 s3prl 缓存目录，供离线 SIM 打分。"""
    src = cfg("WAVLM_LARGE_PT")
    if not src or not Path(src).exists():
        return None

    fname = "{}.{}".format(
        hashlib.sha256(S3PRL_WAVLM_URL.encode()).hexdigest(),
        S3PRL_WAVLM_URL.rsplit("/", 1)[-1])
    dst = Path.home() / ".cache" / "s3prl" / "download" / fname
    if dst.exists():
        return str(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(Path(src).resolve())
    except OSError as e:
        print(f"[tts] 摆放 s3prl 缓存失败: {e}", flush=True)
        return None
    return str(dst)


def task_tts(args, run_dir: Path) -> Dict[str, Any]:
    root = SUITE_ROOT / "tts_seed"
    ids = device_ids()
    bin_dir = Path(cfg("EVAL_BIN_DIR"))
    cpp_bin = str(bin_dir / "llama-omni-tts-eval")
    require_file(cpp_bin, "llama-omni-tts-eval")
    require_file(cfg("EVAL_META_PATH"), "EVAL_META_PATH")

    limit = args.smoke_tts
    save_dir = run_dir / "tts_seed"

    cached = ensure_s3prl_cache()
    if cached:
        print(f"[tts] s3prl 缓存就位: {cached}", flush=True)
    else:
        print("[tts] 没找到本地 wavlm_large.pt（配 WAVLM_LARGE_PT），"
              "SIM 会尝试联网下载", flush=True)

    # 子脚本内部调用 python3，把 EVAL_PYTHON 所在目录放到 PATH 前
    py = Path(cfg("EVAL_PYTHON", "python3"))
    path = os.environ.get("PATH", "")
    if py.is_file():
        path = f"{py.parent}:{path}"

    env = base_env({
        "PATH": path,
        "CPP_BIN": cpp_bin,
        "CPP_BUILD_LIB": str(bin_dir),
        "MODEL_PATH": cfg("MODEL_DIR"),
        "TTS_MODEL_PATH": cfg("TTS_MODEL_PATH"),
        "ONNX_MODEL_DIR": cfg("ONNX_MODEL_DIR"),
        "EVAL_DATA_PATH": cfg("EVAL_DATA_PATH"),
        "EVAL_META_PATH": cfg("EVAL_META_PATH"),
        "PARAFORMER_MODEL": cfg("PARAFORMER_MODEL"),
        "SPEAKER_CKPT": cfg("SPEAKER_CKPT"),
        "S3PRL_REPO": cfg("S3PRL_REPO"),
        "TTS_DEVICE": cfg("TTS_DEVICE", "cpu"),
        "T2W_DEVICE": cfg("T2W_DEVICE", "cpu"),
        "CPP_NGL": cfg("CPP_NGL", "0"),
        "WER_DEVICE": cfg("WER_DEVICE", "cpu"),
        "SIM_DEVICE": cfg("SIM_DEVICE", "cpu"),
        "SAVE_DIR": str(save_dir),
        "GPUS_PER_NODE": str(len(ids)),
        "NUM_SAMPLES": str(limit if limit > 0 else 10_000_000),
        "DEVICE_ENV_VAR": cfg("DEVICE_ENV_VAR", "ASCEND_RT_VISIBLE_DEVICES"),
        "DEVICE_IDS": ",".join(ids),
        "GGML_CANN_WEIGHT_NZ": cfg("GGML_CANN_WEIGHT_NZ", "off"),
        "GGML_CANN_ACL_GRAPH": cfg("GGML_CANN_ACL_GRAPH", "off"),
    })

    res = run_step("tts", ["bash", "run_tts_eval_cpp_zh.sh"], root, env,
                   run_dir / "tts.log")

    metrics: Dict[str, Any] = {
        "n_samples": "全量 2020 条" if limit == 0 else f"每卡前 {limit} 条",
        "生成wav数": len(list(save_dir.glob("*.wav"))) if save_dir.is_dir() else 0,
    }
    wer_file = save_dir / "wav_res_ref_text.wer"
    if wer_file.is_file():
        txt = wer_file.read_text(encoding="utf-8", errors="replace")
        metrics["WER"] = _grep_last(txt, r"^WER:\s*([\d.]+%?)")
        metrics["WER_NORMALIZED"] = _grep_last(txt, r"^WER_NORMALIZED:\s*([\d.]+%?)")
    sim_file = save_dir / "wav_res_ref_text.sim"
    if sim_file.is_file():
        txt = sim_file.read_text(encoding="utf-8", errors="replace")
        metrics["SIM(ASV)"] = _grep_last(txt, r"^ASV:\s*([\d.]+)")
        metrics["SIM方差"] = _grep_last(txt, r"^ASV-var:\s*([\d.]+)")
    metrics["save_dir"] = str(save_dir)
    res["metrics"] = metrics
    return res


def task_rts(args, run_dir: Path) -> Dict[str, Any]:
    root = SUITE_ROOT / "judge-final"
    model = cfg("RTS_MODEL_LLM") or cfg("MODEL_LLM")
    require_file(model, "RTS_MODEL_LLM")

    videos = [v for v in cfg("RTS_VIDEO").split() if v]
    if not videos:
        raise SystemExit("[config] RTS_VIDEO 为空")
    for v in videos:
        require_file(v, "RTS_VIDEO")

    runs_dir = run_dir / "rts_runs"
    cmd = [cfg("RTS_PYTHON", "python3"), "run_judge_direct.py",
           "--model", model,
           "--llamacpp-root", cfg("LLAMACPP_ROOT", str(SUITE_ROOT.parent)),
           "--video", *videos,
           "--max-duration", cfg("RTS_MAX_DURATION", "120"),
           "--runs-dir", str(runs_dir),
           "--verbose"]
    dev = cfg("RTS_DEVICE_ID")
    if dev:
        cmd += ["--gpu", dev]

    env = base_env({
        "OMNI_SERVER_BIN": cfg("OMNI_SERVER_BIN"),
        "GGML_CANN_WEIGHT_NZ": cfg("GGML_CANN_WEIGHT_NZ", "off"),
        "GGML_CANN_ACL_GRAPH": cfg("GGML_CANN_ACL_GRAPH", "off"),
        "OMNI_T2W_DEVICE": os.environ.get("OMNI_T2W_DEVICE", "gpu"),
        "OMNI_T2M_DEVICE": os.environ.get("OMNI_T2M_DEVICE", "gpu:0"),
        "OMNI_VOC_DEVICE": os.environ.get("OMNI_VOC_DEVICE", "gpu:0"),
        "OMNI_SAMPLER_SEED": os.environ.get("OMNI_SAMPLER_SEED", "42"),
    })

    res = run_step("rts", cmd, root, env, run_dir / "rts.log")
    res["metrics"] = _collect_rts_metrics(root, runs_dir, res["stdout"])
    return res


def _collect_rts_metrics(judge_root: Path, runs_dir: Path, stdout: str) -> Dict[str, Any]:
    """从 judge 产物里取 RTF 与端到端延迟。

    单视频取 session 的 eval_e2e_report.json，多视频取跨视频平均的
    batch_avg_report.json；两者都读不到再退回解析控制台摘要。
    """
    metrics: Dict[str, Any] = {"runs_dir": str(runs_dir)}

    report: Dict[str, Any] = {}
    metas = sorted(runs_dir.glob("*/run_meta.json"))
    if metas:
        meta_path = metas[-1]
        metrics["run_meta"] = str(meta_path)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        for key in ("batch_avg_report", "session_dir"):
            rel = meta.get(key)
            if not rel:
                continue
            p = judge_root / rel
            if key == "session_dir":
                p = p / "eval_e2e_report.json"
            if p.is_file():
                try:
                    report = json.loads(p.read_text(encoding="utf-8"))
                    metrics["report"] = str(p)
                    break
                except Exception:
                    pass

    if report:
        rtf = report.get("rtf") or {}
        # 优先用 core 稳态窗口；完整对照仍在原始 report
        core = rtf.get("core") or {}
        main = core if core.get("available") else rtf
        stage = main.get("stage_rtf") or {}
        metrics["均值"] = main.get("rtf_aggregate")
        if stage:
            metrics["分解"] = " + ".join(
                f"{k} {v}" for k, v in stage.items())
        e2e = report.get("e2e_speak_recv_to_wav_poll_ms") or {}
        metrics["SPEAK→wav均值(ms)"] = e2e.get("mean_ms")
        metrics["SPEAK→wav中位数(ms)"] = e2e.get("median_ms")
    else:
        metrics["均值"] = _grep_last(
            stdout, r"RTF（掐头去尾）[\s\S]*?Σ耗时/Σ音频\s+([\d.]+)")
        metrics["SPEAK→wav均值(ms)"] = _grep_last(
            stdout, r"SPEAK→wav e2e\s+([\d.]+) ms")

    return metrics


HANDLERS = {
    "videomme": task_videomme,
    "daily-omni": task_daily_omni,
    "tts": task_tts,
    "rts": task_rts,
}


# ---------------------------------------------------------------- 汇总输出


TASK_TITLE = {
    "videomme": "Video-MME  视频问答精度",
    "daily-omni": "Daily-Omni 音视频问答精度",
    "tts": "Seed-TTS   中文语音克隆精度",
    "rts": "RTS        双工端到端速度",
}

# 每个任务在总表里最想看的那几个数
HEADLINE = {
    "videomme": ["准确率", "官方Overall"],
    "daily-omni": ["准确率", "官方Overall"],
    "tts": ["WER", "SIM(ASV)"],
    "rts": ["均值", "SPEAK→wav均值(ms)"],
}


def _pad(s: Any, width: int) -> str:
    """按终端显示宽度右侧补空格（CJK 占两列）。"""
    s = str(s)
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(width - shown, 1)


def print_summary(results: Dict[str, Dict[str, Any]], run_dir: Path) -> None:
    print(f"\n{'=' * 78}")
    print("  评测汇总")
    print(f"{'=' * 78}")
    for name, r in results.items():
        state = "OK  " if r["rc"] == 0 else "FAIL"
        print(f"\n  [{state}] {TASK_TITLE.get(name, name)}   用时 {r['elapsed_s']}s")
        rows = [(k, v) for k, v in (r.get("metrics") or {}).items()
                if v not in (None, "")]
        if r.get("log") and r["log"] != "-":
            rows.append(("log", r["log"]))
        for k, v in rows:
            print(f"        {_pad(k, 24)}{v}")

    print(f"\n{'-' * 78}")
    print("  关键指标")
    print(f"{'-' * 78}")
    for name, r in results.items():
        m = r.get("metrics") or {}
        vals = [f"{k}={m[k]}" for k in HEADLINE.get(name, [])
                if m.get(k) not in (None, "")]
        line = "  ".join(vals) if vals else ("跑失败，无数值" if r["rc"] else "无数值")
        print(f"  {_pad(TASK_TITLE.get(name, name), 34)}{line}")

    print(f"\n  产物目录: {run_dir}")
    print(f"{'=' * 78}\n")


def reextract(name: str, run_dir: Path, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """从已有产物重新抽取指标（供 --summarize）。"""
    m = dict(metrics)
    log = run_dir / f"{name}.log"
    txt = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""

    if name in ("videomme", "daily-omni"):
        m["准确率"] = _grep_last(txt, r"Accuracy: (\d+/\d+ = [\d.]+%)") or m.get("准确率")
        pat = (r"^Overall:\s*([\d.]+%)" if name == "videomme"
               else r"Total:\s*([\d.]+%\s*\(\d+/\d+\))")
        m["官方Overall"] = _grep_last(txt, pat) or m.get("官方Overall")

    elif name == "tts":
        sd = Path(m.get("save_dir") or (run_dir / "tts_seed"))
        wer = sd / "wav_res_ref_text.wer"
        if wer.is_file():
            t = wer.read_text(encoding="utf-8", errors="replace")
            m["WER"] = _grep_last(t, r"^WER:\s*([\d.]+%?)")
            m["WER_NORMALIZED"] = _grep_last(t, r"^WER_NORMALIZED:\s*([\d.]+%?)")
        sim = sd / "wav_res_ref_text.sim"
        if sim.is_file():
            t = sim.read_text(encoding="utf-8", errors="replace")
            m["SIM(ASV)"] = _grep_last(t, r"^ASV:\s*([\d.]+)")
            m["SIM方差"] = _grep_last(t, r"^ASV-var:\s*([\d.]+)")

    elif name == "rts":
        runs_dir = Path(m.get("runs_dir") or (run_dir / "rts_runs"))
        m.update(_collect_rts_metrics(SUITE_ROOT / "judge-final", runs_dir, txt))

    return m


def summarize_only(run_dir: Path) -> int:
    """读取 metrics_*.json，重抽指标并打印总表。"""
    results: Dict[str, Dict[str, Any]] = {}
    for name in TASKS:
        f = run_dir / f"metrics_{name}.json"
        if not f.is_file():
            continue
        try:
            res = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] 读不了 {f}: {e}")
            continue
        before = res.get("metrics") or {}
        after = reextract(name, run_dir, before)
        if after != before:
            res["metrics"] = after
            f.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                         encoding="utf-8")
            added = [k for k, v in after.items()
                     if v not in (None, "") and before.get(k) in (None, "")]
            if added:
                print(f"[{name}] 重新抽到: {', '.join(added)}")
        results[name] = res

    if not results:
        print(f"[warn] {run_dir} 下没有任何 metrics_*.json")
        return 1
    print_summary(results, run_dir)
    return 0


# ---------------------------------------------------------------- main


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_eval.sh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="MiniCPM-o 评测套件统一入口",
        epilog="""任务:
  videomme     Video-MME 视频问答精度
  daily-omni   Daily-Omni 音视频问答精度
  tts          Seed-TTS 中文语音克隆精度 (WER / SIM)
  rts          双工端到端速度 (RTF / SPEAK→wav 延迟)
  all          以上全部

示例:
  ./run_eval.sh all                            # 按 config.env 跑全部
  ./run_eval.sh videomme daily-omni tts --smoke 5
  ./run_eval.sh rts --devices 7
  ./run_eval.sh all --full                     # 精度测试跑全量
""")
    p.add_argument("tasks", nargs="*", default=["all"],
                   help="要跑的任务，默认 all")
    p.add_argument("--smoke", type=int, default=None,
                   help="统一覆盖三个精度测试的样本数（0=全量）")
    p.add_argument("--full", action="store_true", help="等价于 --smoke 0")
    p.add_argument("--devices", default=None, help="覆盖 DEVICE_IDS，如 0,1,2,3")
    p.add_argument("--device-count", type=int, default=None,
                   help="覆盖 DEVICE_COUNT")
    p.add_argument("--model", default=None, help="覆盖精度测试的 MODEL_LLM")
    p.add_argument("--rts-model", default=None, help="覆盖 RTS 的 RTS_MODEL_LLM")
    p.add_argument("--skip-rerun", action="store_true",
                   help="精度测试跳过失败题重跑（省一次模型加载）")
    p.add_argument("--keep-going", action="store_true",
                   help="某个任务失败后继续跑后面的")
    p.add_argument("--run-dir", default=None,
                   help="产物目录。run_all.sh 用它把分批跑的四个任务收进同一个目录")
    p.add_argument("--dry-run", action="store_true", help="只打印解析后的配置")
    p.add_argument("--summarize", action="store_true",
                   help="不跑测试，只把 --run-dir 下已有的结果汇总打印")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if args.summarize:
        if not args.run_dir:
            raise SystemExit("--summarize 需要配 --run-dir")
        return summarize_only(Path(args.run_dir).resolve())

    if args.devices:
        os.environ["DEVICE_IDS"] = args.devices
    if args.device_count is not None:
        os.environ["DEVICE_COUNT"] = str(args.device_count)
    if args.model:
        os.environ["MODEL_LLM"] = args.model
    if args.rts_model:
        os.environ["RTS_MODEL_LLM"] = args.rts_model

    smoke = 0 if args.full else args.smoke
    args.smoke_videomme = smoke if smoke is not None else cfg_int("SMOKE_VIDEOMME", 0)
    args.smoke_daily_omni = smoke if smoke is not None else cfg_int("SMOKE_DAILY_OMNI", 0)
    args.smoke_tts = smoke if smoke is not None else cfg_int("SMOKE_TTS", 0)

    wanted: List[str] = []
    for t in args.tasks:
        for one in t.split(","):
            one = one.strip()
            if not one:
                continue
            if one == "all":
                wanted += TASKS
            elif one == "accuracy":
                wanted += ACCURACY_TASKS
            elif one in HANDLERS:
                wanted.append(one)
            else:
                raise SystemExit(f"未知任务: {one}（可选: {', '.join(TASKS)}, all, accuracy）")
    wanted = list(dict.fromkeys(wanted)) or TASKS

    ids = device_ids()
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        stamp = run_dir.name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = SUITE_ROOT / "output" / stamp

    print(f"{'=' * 72}")
    print("  MiniCPM-o 评测套件")
    print(f"{'=' * 72}")
    print(f"  任务          : {', '.join(wanted)}")
    print(f"  精度权重      : {cfg('MODEL_LLM')}")
    print(f"  RTS 权重      : {cfg('RTS_MODEL_LLM')}")
    print(f"  eval CLI 目录 : {cfg('EVAL_BIN_DIR')}")
    print(f"  server 仓库   : {cfg('LLAMACPP_ROOT')}")
    print(f"  卡 ({cfg('DEVICE_ENV_VAR')}) : {','.join(ids)}  共 {len(ids)} 张")
    print(f"  RTS 卡        : {cfg('RTS_DEVICE_ID') or '自动挑空闲'}")
    print(f"  样本数        : videomme={args.smoke_videomme or '全量'}  "
          f"daily-omni={args.smoke_daily_omni or '全量'}  "
          f"tts={args.smoke_tts or '全量'}")
    print(f"  产物目录      : {run_dir}")
    print(f"{'=' * 72}")

    if args.dry_run:
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}
    failed = False

    for name in wanted:
        try:
            res = HANDLERS[name](args, run_dir)
        except SystemExit as e:
            print(f"[{name}] 跳过: {e}", flush=True)
            res = {"rc": 1, "elapsed_s": 0.0, "log": "-", "stdout": "",
                   "metrics": {"error": str(e)}}
        results[name] = res
        (run_dir / f"metrics_{name}.json").write_text(
            json.dumps({k: v for k, v in res.items() if k != "stdout"},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        if res["rc"] != 0:
            failed = True
            if not args.keep_going:
                print(f"[{name}] 失败，停止后续任务（加 --keep-going 可继续）",
                      flush=True)
                break

    summary = {
        "stamp": stamp,
        "tasks": wanted,
        "config": {
            "MODEL_LLM": cfg("MODEL_LLM"),
            "RTS_MODEL_LLM": cfg("RTS_MODEL_LLM"),
            "EVAL_BIN_DIR": cfg("EVAL_BIN_DIR"),
            "LLAMACPP_ROOT": cfg("LLAMACPP_ROOT"),
            "DEVICE_ENV_VAR": cfg("DEVICE_ENV_VAR"),
            "DEVICE_IDS": ",".join(ids),
            "smoke": {
                "videomme": args.smoke_videomme,
                "daily-omni": args.smoke_daily_omni,
                "tts": args.smoke_tts,
            },
        },
        "results": {
            k: {kk: vv for kk, vv in v.items() if kk != "stdout"}
            for k, v in results.items()
        },
    }
    suffix = "_".join(wanted) if args.run_dir else "all"
    (run_dir / f"summary_{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(results, run_dir)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
