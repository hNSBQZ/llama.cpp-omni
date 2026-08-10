"""Judge 共用工具。"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

JUDGE_ROOT = Path(__file__).resolve().parent


def display_path(path: Path | str, *, base: Path | None = None) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    anchor = (base or JUDGE_ROOT).resolve()
    return os.path.relpath(str(p.resolve()), str(anchor))


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) != 0


def _find_free_npu(max_used_mib: int, min_free_mib: int) -> int:
    """Ascend 版：解析 npu-smi info，返回 chip 的逻辑 device id。

    npu-smi 的表格里 NPU 列是板号、Chip 列是板上芯片号，而
    ASCEND_RT_VISIBLE_DEVICES 用的是跨板连续编号的 Device 号
    （表格第二行 "| NPU  Chip | Bus-Id | ... HBM-Usage |" 里的第二个数字）。

    注意 max_used_mib 的默认值 2000 是照 CUDA 卡定的，套到 Ascend 上会把所有卡
    都判成忙：910 空闲时每颗芯片就有约 2.9GB 的驱动/固件基线占用，不是用户进程。
    所以这里另设一条下限（NPU_IDLE_BASELINE_MIB，默认 6000），取两者较大值。
    """
    baseline = int(os.environ.get("NPU_IDLE_BASELINE_MIB", "6000"))
    eff_max_used = max(max_used_mib, baseline)
    if eff_max_used != max_used_mib:
        _log(
            f"  (Ascend 空闲基线约 3GB，used 门槛 {max_used_mib} → {eff_max_used} MiB)"
        )
    max_used_mib = eff_max_used

    out = subprocess.check_output(["npu-smi", "info"], text=True)
    cands: List[tuple[int, int, int]] = []
    for line in out.splitlines():
        # 形如: | 0     0                   | 0000:9D:00.0  | 0  0 / 0  24302/ 65536  |
        m = re.match(
            r"^\|\s*(\d+)\s+(\d+)\s*\|\s*[0-9A-Fa-f:.]+\s*\|.*?(\d+)\s*/\s*(\d+)\s*\|",
            line,
        )
        if not m:
            continue
        dev = int(m.group(2))
        used = int(m.group(3))
        total = int(m.group(4))
        free = total - used
        _log(f"  NPU dev{dev}: used={used}MiB free={free}MiB")
        if used <= max_used_mib and free >= min_free_mib:
            cands.append((used, -free, dev))
    if not cands:
        raise RuntimeError(
            f"无空闲 NPU（要求 used<={max_used_mib}MiB 且 free>={min_free_mib}MiB）"
        )
    cands.sort()
    chosen = cands[0][2]
    _log(f"选中空闲 NPU {chosen}")
    return chosen


def find_free_gpu(
    max_used_mib: int,
    min_free_mib: int,
    prefer: Optional[int],
) -> int:
    """扫描 nvidia-smi（没有则退回 npu-smi），返回物理 device index。"""
    if prefer is not None:
        _log(f"使用指定 GPU {prefer}")
        return prefer

    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        _log("没有 nvidia-smi，改扫 npu-smi")
        return _find_free_npu(max_used_mib, min_free_mib)
    cands: List[tuple[int, int, int, int]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx = int(parts[0])
        used = int(float(parts[1]))
        free = int(float(parts[2]))
        util = int(float(parts[4])) if len(parts) > 4 else 0
        _log(f"  GPU{idx}: used={used}MiB free={free}MiB util={util}%")
        if used <= max_used_mib and free >= min_free_mib:
            cands.append((used, -free, util, idx))
    if not cands:
        raise RuntimeError(
            f"无空闲 GPU（要求 used<={max_used_mib}MiB 且 free>={min_free_mib}MiB）"
        )
    cands.sort()
    chosen = cands[0][3]
    _log(f"选中空闲 GPU {chosen}")
    return chosen


def _pids_listening_on(port: int) -> List[int]:
    """返回占用 TCP listen port 的 PID 列表。"""
    pids: List[int] = []
    try:
        out = subprocess.check_output(
            ["ss", "-ltnp", f"sport = :{port}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        out = ""
    for m in re.findall(r"pid=(\d+)", out):
        pids.append(int(m))
    if not pids:
        try:
            out = subprocess.check_output(
                ["fuser", f"{port}/tcp"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for tok in out.split():
                if tok.isdigit():
                    pids.append(int(tok))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return sorted(set(pids))


def _kill_pid_tree(pid: int, sig: int = signal.SIGTERM) -> None:
    """终止进程及其子进程组。"""
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _resolve_plot_python(python: str) -> str:
    """返回用于绘图的 Python 解释器。"""
    plot_py = JUDGE_ROOT / ".venv-plot" / "bin" / "python"
    if plot_py.is_file():
        return str(plot_py)
    return python


def run_plot(python: str, session_dir: Path, out_dir: Path, label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_py = _resolve_plot_python(python)
    cmd = [
        plot_py,
        str(JUDGE_ROOT / "scripts" / "plot_turn_position_e2e.py"),
        "--case",
        f"{label}={session_dir}",
        "--out-dir",
        str(out_dir),
        "--title",
        f"turn-position e2e ({label})",
    ]
    _log(f"绘图 ({plot_py}): {' '.join(display_path(a) if isinstance(a, str) and os.path.isabs(a) else a for a in cmd)}")
    subprocess.check_call(cmd, cwd=str(JUDGE_ROOT))
    return out_dir


def _fmt_ms_block(stats: Optional[Dict[str, Any]]) -> str:
    if not stats or not stats.get("n"):
        return "(无样本)"
    mean = f"{stats['mean_ms']:.1f} ms"
    return (
        f"{mean:<12}\t"
        f"(中位 {stats['median_ms']:.1f}, "
        f"min {stats['min_ms']:.1f}, max {stats['max_ms']:.1f}, n={stats['n']})"
    )


def _fmt_rtf_block(stats: Optional[Dict[str, Any]]) -> str:
    if not isinstance(stats, dict) or not stats.get("n"):
        return "(无样本)"
    head = f"{stats['mean']:.3f}"
    extra = f"(中位 {stats['median']:.3f}"
    if stats.get("p95") is not None:
        extra += f", p95 {stats['p95']:.3f}"
    extra += f", max {stats['max']:.3f}, n={stats['n']})"
    return f"{head:<12}\t{extra}"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def print_latency_summary(
    report: Dict[str, Any],
    *,
    title: Optional[str] = None,
) -> None:
    """打印关键延迟结果。"""
    from console_ui import JUDGE_BAR

    primary = report.get("e2e_speak_recv_to_wav_poll_ms") or {}
    wall = report.get("speak_chunk_wall_ms") or {}
    decode = report.get("e2e_decode_end_to_wav_poll_ms") or {}
    stage = report.get("stage_ms") or {}
    rtf = report.get("rtf") or {}

    bar = JUDGE_BAR

    def line(s: str = "") -> None:
        print(s, flush=True)

    def _pad(label: str) -> str:
        left = f"    {label}"
        target = 20
        while len(left.expandtabs(8)) < target:
            left += "\t"
        return left

    def row(label: str, stats: Any) -> None:
        line(f"{_pad(label)}{_fmt_ms_block(stats if isinstance(stats, dict) else None)}")

    def rtf_row(label: str, stats: Any) -> None:
        line(f"{_pad(label)}{_fmt_rtf_block(stats if isinstance(stats, dict) else None)}")

    line()
    line(bar)
    if title:
        line(f"  {title}")
        line(bar)

    # RTF 是算子加速比赛的核心指标，放最前面
    if rtf.get("available"):
        order = ("encode", "llm_prefill", "llm_decode", "tts", "token2wav")

        def _breakdown(srtf: Any) -> Optional[str]:
            if not isinstance(srtf, dict):
                return None
            parts = [f"{s} {srtf[s]:.3f}" for s in order if srtf.get(s) is not None]
            return " + ".join(parts) if parts else None

        core = rtf.get("core") or {}
        if core.get("available"):
            head = f"    ★ RTF（掐头去尾）  turn={core.get('n_turns')}  帧数={core.get('n_frames')}"
            if core.get("audio_total_ms"):
                head += f"  音频{float(core['audio_total_ms']) / 1000:.1f}s"
            line(head)
            if core.get("rtf_aggregate") is not None:
                line(f"    Σ耗时/Σ音频         {core['rtf_aggregate']:.3f}\t← 主指标")
            rtf_row("每帧全链路", core.get("rtf"))
            parts = _breakdown(core.get("stage_rtf"))
            if parts:
                line("    分解: " + parts)

        head = ("    (对照) 全量" if core.get("available") else "    ★ RTF  全量")
        head += f"  帧数={rtf.get('n_frames')}  wav={rtf.get('n_wav')}"
        if rtf.get("audio_total_ms"):
            head += f"  音频{float(rtf['audio_total_ms']) / 1000:.1f}s"
        line(head)
        agg = rtf.get("rtf_aggregate")
        if agg is not None:
            line(f"    Σ耗时/Σ音频         {agg:.3f}")
        if not core.get("available"):
            rtf_row("每帧全链路", rtf.get("rtf"))
            parts = _breakdown(rtf.get("stage_rtf"))
            if parts:
                line("    分解: " + parts)
        for w in rtf.get("warnings") or []:
            line(f"    ! {w}")
        line("    " + "-" * 40)
    elif rtf:
        line("    ★ RTF 不可用：stage_timing.jsonl 缺 duration_ms/src_cnt（C++ 端需重编）")

    integ = report.get("wav_integrity") or {}
    if integ and integ.get("ok") is False:
        line(
            f"    ! wav 对账不一致: poll={integ.get('n_polled_wav')} "
            f"cpp={integ.get('n_cpp_wav')} "
            f"漏采={len(integ.get('missing_in_poll') or [])} "
            f"多出={len(integ.get('missing_in_cpp') or [])} —— RTF/延迟可能失真"
        )

    row("★ SPEAK→wav e2e", primary)
    row("SPEAK wall", wall)
    row("decode→wav", decode)
    if stage:
        line("    ---- stage（e2e≈wall+decode→wav）----")
        row("http prefill", stage.get("prefill_ms"))
        row("image(VPM)", stage.get("vpm_ms"))
        row("audio(APM)", stage.get("apm_ms"))
        row("llm prefill", stage.get("llm_prefill_ms"))
        row("llm decode", stage.get("cost_llm_ms"))
        row("tts", stage.get("cost_tts_ms"))
        row("t2w", stage.get("cost_token2wav_ms"))
        recon = stage.get("wall_recon_ms")
        if isinstance(recon, dict) and recon.get("n"):
            msg = (
                f"    (参考) http_prefill+max(VPM,APM)+llm Σ "
                f"{recon['mean_ms']:.1f} ms"
            )
            if wall.get("n"):
                msg += f" ≈ SPEAK wall {wall['mean_ms']:.1f} ms（公式校验）"
            line(msg)
    line(bar)
    line()


def _avg_stat_field(
    reports: List[Dict[str, Any]],
    *keys: str,
) -> Optional[Dict[str, Any]]:
    """对各 report 同名字段的 mean_ms 做无权重平均。"""
    means: List[float] = []
    medians: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []
    for rep in reports:
        cur: Any = rep
        for k in keys:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(k)
        if not isinstance(cur, dict) or not cur.get("n"):
            continue
        means.append(float(cur["mean_ms"]))
        medians.append(float(cur["median_ms"]))
        mins.append(float(cur["min_ms"]))
        maxs.append(float(cur["max_ms"]))
    if not means:
        return {"n": 0}
    return {
        "n": len(means),
        "mean_ms": round(sum(means) / len(means), 1),
        "median_ms": round(sum(medians) / len(medians), 1),
        "min_ms": round(min(mins), 1),
        "max_ms": round(max(maxs), 1),
    }


def _avg_rtf_stat(reports: List[Dict[str, Any]], *keys: str) -> Dict[str, Any]:
    """对各 report 的 RTF 统计做视频等权平均。"""
    means: List[float] = []
    medians: List[float] = []
    maxs: List[float] = []
    p95s: List[float] = []
    n_total = 0
    for rep in reports:
        cur: Any = rep
        for k in keys:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(k)
        if not isinstance(cur, dict) or not cur.get("n"):
            continue
        means.append(float(cur["mean"]))
        medians.append(float(cur["median"]))
        maxs.append(float(cur["max"]))
        if cur.get("p95") is not None:
            p95s.append(float(cur["p95"]))
        n_total += int(cur["n"])
    if not means:
        return {"n": 0}
    return {
        "n": n_total,
        "n_videos": len(means),
        "mean": round(sum(means) / len(means), 4),
        "median": round(sum(medians) / len(medians), 4),
        "max": round(max(maxs), 4),
        "p95": round(sum(p95s) / len(p95s), 4) if p95s else None,
    }


def _average_rtf(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """多视频 RTF 汇总。

    分布类指标（mean/median/p95）按视频等权平均；
    aggregate 与 stage_rtf 用总耗时/总音频重新算，不受各视频音频长短影响。
    core（掐头去尾）同样按帧汇总：只取 role=core 的中间稳态帧。
    """
    avail = [r for r in reports if isinstance(r.get("rtf"), dict) and r["rtf"].get("available")]
    if not avail:
        return {"available": False}

    stages = ("encode", "llm_prefill", "llm_decode", "tts", "token2wav")

    def _pool(core_only: bool) -> Dict[str, Any]:
        audio_total = 0.0
        compute_total = 0.0
        stage_total = {s: 0.0 for s in stages}
        n_frames = 0
        for r in avail:
            for f in r["rtf"].get("frames") or []:
                if core_only and f.get("role") != "core":
                    continue
                n_frames += 1
                audio_total += float(f.get("audio_ms") or 0.0)
                compute_total += float(f.get("compute_ms") or 0.0)
                for s in stages:
                    stage_total[s] += float(f.get(f"{s}_ms") or 0.0)
        return {
            "n_frames": n_frames,
            "audio_total_ms": round(audio_total, 1),
            "compute_total_ms": round(compute_total, 1),
            "rtf_aggregate": (
                round(compute_total / audio_total, 4) if audio_total else None
            ),
            "stage_rtf": {
                s: (round(stage_total[s] / audio_total, 4) if audio_total else None)
                for s in stages
            },
        }

    core = _pool(core_only=True)
    core["available"] = core["n_frames"] > 0
    core["rtf"] = _avg_rtf_stat(reports, "rtf", "core", "rtf")
    core["n_turns"] = sum(
        int((r["rtf"].get("core") or {}).get("n_turns") or 0) for r in avail
    )
    core["n_videos"] = sum(
        1 for r in avail if (r["rtf"].get("core") or {}).get("available")
    )

    warnings: List[str] = []
    n_no_core = len(avail) - int(core["n_videos"])
    if n_no_core:
        warnings.append(
            f"{n_no_core}/{len(avail)} 个视频掐头去尾后无中间帧（turn 太短），"
            "未计入 core"
        )

    full = _pool(core_only=False)
    return {
        "available": True,
        "n_videos": len(avail),
        "n_frames": full["n_frames"],
        "n_wav": sum(int(r["rtf"].get("n_wav") or 0) for r in avail),
        "audio_total_ms": full["audio_total_ms"],
        "compute_total_ms": full["compute_total_ms"],
        "rtf": _avg_rtf_stat(reports, "rtf", "rtf"),
        "rtf_aggregate": full["rtf_aggregate"],
        "stage_rtf": full["stage_rtf"],
        "core": core,
        "warnings": warnings,
        "avg_mode": "per-video unweighted mean; aggregate pooled by total audio",
    }


def average_latency_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """多视频关键延迟：对每个视频的 mean 再取平均（视频等权）。"""
    stage_keys = (
        "prefill_ms",
        "vpm_ms",
        "apm_ms",
        "llm_prefill_ms",
        "cost_llm_ms",
        "cost_tts_ms",
        "cost_token2wav_ms",
        "wall_recon_ms",
    )
    stage: Dict[str, Any] = {}
    for sk in stage_keys:
        stage[sk] = _avg_stat_field(reports, "stage_ms", sk)
    return {
        "e2e_speak_recv_to_wav_poll_ms": _avg_stat_field(
            reports, "e2e_speak_recv_to_wav_poll_ms"
        ),
        "speak_chunk_wall_ms": _avg_stat_field(reports, "speak_chunk_wall_ms"),
        "e2e_decode_end_to_wav_poll_ms": _avg_stat_field(
            reports, "e2e_decode_end_to_wav_poll_ms"
        ),
        "rtf": _average_rtf(reports),
        "stage_ms": stage,
        "n_videos": len(reports),
        "avg_mode": "unweighted_mean_of_per_video_means",
    }
