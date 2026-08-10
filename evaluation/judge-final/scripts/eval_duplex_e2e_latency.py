#!/usr/bin/env python3
"""Duplex 端到端延迟评测与分析。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import ssl
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from judge_support import display_path  # noqa: E402

import websockets

from replay_duplex_session import (
    _collect_send_items,
    _load_ref_audio_b64,
    _read_json,
    _resolve_session_id,
)
from e2e_timing import stitch_speak_turn_wavs

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from duplex_video_chunker import VideoChunkerConfig, build_send_items_from_video


def _log(msg: str) -> None:
    print(msg, flush=True)


def _find_latest_session(sessions_root: Path) -> Path:
    cands: List[Tuple[float, Path]] = []
    if not sessions_root.is_dir():
        raise FileNotFoundError(f"sessions 目录不存在: {sessions_root}")
    for d in sessions_root.iterdir():
        if not d.is_dir():
            continue
        rec = d / "recording.json"
        ua = d / "user_audio"
        if rec.exists() and ua.is_dir() and any(ua.glob("*.wav")):
            cands.append((rec.stat().st_mtime, d))
    if not cands:
        raise FileNotFoundError(f"在 {sessions_root} 下找不到含 recording.json+user_audio 的 session")
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def _default_ref_audio() -> Optional[str]:
    p = _ROOT / "assets" / "ref_audio" / "ref_minicpm_signature.wav"
    return str(p) if p.is_file() else None


def _is_empty_speak_text(text: Any) -> bool:
    """判断 SPEAK chunk 文本是否为空。"""
    return not str(text or "").strip()


def _pct(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return round(s[k], 4)


def _rtf_stats(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(mean(vals), 4),
        "median": round(median(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "p95": _pct(vals, 0.95),
    }


def _read_stage_timing(session_dir: Path) -> Optional[Path]:
    for cand in (
        session_dir / "stage_timing.jsonl",
        Path(os.environ.get("CPP_STAGE_TIMING", "")),
    ):
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


def _parse_stage_timing(path: Optional[Path]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"chunk": [], "tts": [], "t2w": []}
    if not path or not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = e.get("event")
        if ev in out:
            out[ev].append(e)
    return out


_RTF_STAGES = ("encode", "llm_prefill", "llm_decode", "tts", "token2wav")

# 掐头去尾后剩这么少帧就该提醒了：能进 core 的只有中间稳态帧，样本太少时
# 单帧抖动会直接写进 pooled RTF。
_RTF_CORE_MIN_FRAMES = 5


def _mark_rtf_turns(frames: List[Dict[str, Any]]) -> int:
    """按 turn_eos 的 flush 尾包切音频轮次，给每帧标 turn_idx 与 role。返回轮次数。

    轮次边界不用 e2e 的 speak_turn_id：那个只在遇到 LISTEN/IDLE 帧时才断，模型说完
    一轮紧接着又说下一轮时会被并成一个 turn（实测一个 speak_turn 里出现过 3 次
    turn_eos）。C++ 的 is_final 才是 turn_eos 的直接标记 —— 那一包 audio_tokens 是空的，
    只为让 T2W 把 buffer 吐干净。

    role:
      head  轮次首帧。TTS/T2W 缓存空，首个 wav 常不足一帧时长，与稳态不可比。
      tail  含 flush 尾包的帧。尾音的 tts 耗时早算在前面的帧上，这里只有 t2w，
            白送一段分母，RTF 系统性偏低。
      core  中间稳态帧：一帧输入 1s 音频、产出 1s 音频，口径最干净。
    """
    turn_idx = 0
    turn_head = True
    for f in frames:
        f["turn_idx"] = turn_idx
        if f["has_final_wav"]:
            f["role"] = "tail"
            turn_idx += 1
            turn_head = True
        elif turn_head:
            f["role"] = "head"
            turn_head = False
        else:
            f["role"] = "core"
    # 末段没有 flush 收尾（被 stop 截断）时不补 tail：那一帧本身就是稳态帧。
    return turn_idx if turn_head else turn_idx + 1


def _rtf_pool(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """一组帧的 pooled RTF：Σ耗时 / Σ音频。"""
    audio_total = sum(f["audio_ms"] for f in frames)
    compute_total = sum(f["compute_ms"] for f in frames)
    out: Dict[str, Any] = {
        "n_frames": len(frames),
        "audio_total_ms": round(audio_total, 1),
        "compute_total_ms": round(compute_total, 1),
        "rtf_aggregate": round(compute_total / audio_total, 4) if audio_total else None,
        "rtf": _rtf_stats([f["rtf"] for f in frames]),
        "stage_rtf": {
            s: (
                round(sum(f[f"{s}_ms"] for f in frames) / audio_total, 4)
                if audio_total
                else None
            )
            for s in _RTF_STAGES
        },
    }
    if frames:
        auds = [f["audio_ms"] for f in frames]
        out["audio_ms_min"] = round(min(auds), 1)
        out["audio_ms_max"] = round(max(auds), 1)
    return out


def _analyze_rtf(
    stage: Dict[str, List[Dict[str, Any]]],
    chunks_by_cnt: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """每帧 RTF = 该帧的整条生命周期耗时 / 该帧产出的音频时长。

    一个 frame 的生命周期从它被送进 prefill 开始：
        encode(VPM ∥ APM) → llm_prefill → llm_decode → tts → token2wav → wav

    帧号来自 C++ 的 src_cnt（在 prefill packet 上带下来的 cnt），所有属于同一帧的
    wav 会合并成一条记录 —— 绝大多数帧只产一个 wav，轮次结尾的 flush 尾包会和
    正常包并到同一帧，所以这里始终是"一帧一条 RTF"，不需要再区分包和帧。

    主指标是 core（掐头去尾）：按 turn_eos 切音频轮次，去掉每轮的首帧和含 flush 尾包
    的帧，只对中间稳态帧做 Σ耗时/Σ音频。这样分母被钉在"每帧 1s 音频"上，不同选手
    量化后 turn 数、每 turn 的 wav 数怎么变，都不影响这个比值的口径。
    全量 rtf_aggregate 仍然保留作对照。

    数据来源：
      - encode / llm_prefill / llm_decode  ← e2e_timing 的 chunk 事件（SSE metrics）
      - tts / token2wav                     ← C++ stage_timing.jsonl，按 src_cnt 归帧
    不含 judge 侧的 http_prefill（存临时文件 + HTTP 往返），那不是模型算力。
    """
    warnings: List[str] = []

    t2w_events = [e for e in stage["t2w"] if e.get("duration_ms")]
    if not t2w_events:
        if stage["t2w"]:
            warnings.append(
                "stage_timing.jsonl 的 t2w 事件缺少 duration_ms —— "
                "C++ 端是旧版本，请重新编译 llama-omni-server"
            )
        return {"available": False, "warnings": warnings}

    if any(e.get("src_cnt") is None for e in t2w_events):
        warnings.append("部分 t2w 事件缺少 src_cnt，无法归帧")

    # 按 src_cnt 归帧
    by_cnt: Dict[int, Dict[str, Any]] = {}
    for e in t2w_events:
        cnt = e.get("src_cnt")
        if cnt is None:
            continue
        agg = by_cnt.setdefault(
            int(cnt),
            {
                "audio_ms": 0.0,
                "token2wav": 0.0,
                "tts": 0.0,
                "n_wav": 0,
                "wavs": [],
                "has_final": False,
                "flush_audio_ms": 0.0,
            },
        )
        agg["audio_ms"] += float(e.get("duration_ms") or 0.0)
        agg["token2wav"] += float(e.get("token2wav_ms") or 0.0)
        agg["n_wav"] += 1
        agg["wavs"].append(e.get("wav"))
        if e.get("is_final"):
            agg["has_final"] = True
            agg["flush_audio_ms"] += float(e.get("duration_ms") or 0.0)
    for e in stage["tts"]:
        cnt = e.get("src_cnt")
        if cnt is None or int(cnt) not in by_cnt:
            continue
        by_cnt[int(cnt)]["tts"] += float(e.get("tts_ms") or 0.0)

    if len(by_cnt) == 1 and len(t2w_events) > 2:
        warnings.append(
            f"{len(t2w_events)} 个 wav 全部归到同一帧 —— src_cnt 没有正确递增，"
            "服务端二进制需要重新编译"
        )

    def _f(d: Dict[str, Any], k: str) -> float:
        try:
            v = d.get(k)
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    frames: List[Dict[str, Any]] = []
    n_missing_llm = 0
    for cnt in sorted(by_cnt):
        agg = by_cnt[cnt]
        audio_ms = agg["audio_ms"]
        if audio_ms <= 0:
            continue
        c = chunks_by_cnt.get(cnt) or {}
        if not c:
            n_missing_llm += 1
        # VPM 与 APM 并行跑，取较大者
        stages = {
            "encode": max(_f(c, "vpm_ms"), _f(c, "apm_ms")),
            "llm_prefill": _f(c, "llm_prefill_ms"),
            "llm_decode": _f(c, "cost_llm_ms"),
            "tts": agg["tts"],
            "token2wav": agg["token2wav"],
        }
        compute_ms = sum(stages.values())
        frames.append({
            "cnt": cnt,
            "mode": c.get("mode", "?"),
            "n_wav": agg["n_wav"],
            "wavs": agg["wavs"],
            "has_final_wav": bool(agg["has_final"]),
            "flush_audio_ms": round(agg["flush_audio_ms"], 1),
            "audio_ms": round(audio_ms, 1),
            **{f"{k}_ms": round(v, 1) for k, v in stages.items()},
            "compute_ms": round(compute_ms, 1),
            "rtf": round(compute_ms / audio_ms, 4),
            "has_llm": bool(c),
        })

    if n_missing_llm:
        warnings.append(
            f"{n_missing_llm}/{len(frames)} 帧在 e2e_timing 里找不到对应 cnt 的 chunk，"
            "这些帧的 RTF 只含 tts+token2wav"
        )

    n_turns = _mark_rtf_turns(frames)
    core_frames = [f for f in frames if f["role"] == "core"]
    core = _rtf_pool(core_frames)
    core["available"] = bool(core_frames)
    core["n_turns"] = n_turns
    core["n_excluded_head"] = sum(1 for f in frames if f["role"] == "head")
    core["n_excluded_tail"] = sum(1 for f in frames if f["role"] == "tail")
    core["excluded_audio_ms"] = round(
        sum(f["audio_ms"] for f in frames if f["role"] != "core"), 1
    )

    if not core_frames:
        warnings.append(
            f"掐头去尾后没有剩余帧（{n_turns} 个 turn 都不足 3 帧），"
            "core RTF 不可用，只能看全量 rtf_aggregate"
        )
    else:
        if len(core_frames) < _RTF_CORE_MIN_FRAMES:
            warnings.append(
                f"core 只剩 {len(core_frames)} 帧（{n_turns} 个 turn 大多不足 3 帧），"
                "样本太少，core RTF 波动大"
            )
        # core 帧本该一帧一 wav、一帧 1s 音频。不齐说明 turn 切分或 wav 归帧有问题，
        # 此时分母不再是恒定的，跨选手比较会失真，必须报出来。
        n_core_multi = sum(1 for f in core_frames if f["n_wav"] > 1)
        if n_core_multi:
            warnings.append(
                f"{n_core_multi}/{len(core_frames)} 个 core 帧产出多个 wav，"
                "中间段的音频时长不再恒定"
            )
        if core["audio_ms_max"] - core["audio_ms_min"] > 1.0:
            warnings.append(
                f"core 帧音频时长不齐（{core['audio_ms_min']}~{core['audio_ms_max']}ms），"
                "分母不恒定，跨选手对比需谨慎"
            )

    audio_total = sum(f["audio_ms"] for f in frames)
    compute_total = sum(f["compute_ms"] for f in frames)
    n_multi = sum(1 for f in frames if f["n_wav"] > 1)

    # 各阶段单独的 RTF（相加即为全链路 RTF），用于定位瓶颈
    stage_rtf = {}
    for s in _RTF_STAGES:
        tot = sum(f[f"{s}_ms"] for f in frames)
        stage_rtf[s] = round(tot / audio_total, 4) if audio_total else None

    return {
        "available": True,
        "n_frames": len(frames),
        "n_wav": len(t2w_events),
        "n_frames_multi_wav": n_multi,
        "n_turns": n_turns,
        "audio_total_ms": round(audio_total, 1),
        "compute_total_ms": round(compute_total, 1),
        "rtf": _rtf_stats([f["rtf"] for f in frames]),
        "rtf_aggregate": round(compute_total / audio_total, 4) if audio_total else None,
        "stage_rtf": stage_rtf,
        "core": core,
        "frames": frames,
        "warnings": warnings,
        "note": (
            "每帧 RTF = (encode + llm_prefill + llm_decode + tts + token2wav) / 该帧音频时长。"
            "encode 取 max(VPM, APM)（两者并行）。帧号来自 C++ 的 src_cnt，"
            "同一帧的多个 wav（轮次结尾的 flush 尾包）已合并。"
            "主指标 core.rtf_aggregate：按 turn_eos(is_final) 切音频轮次，剔掉每轮首帧"
            "（TTS/T2W 冷启动、首个 wav 常不足一帧）和含 flush 尾包的帧（尾音的 tts 耗时"
            "早算在前面的帧上，只剩 t2w，白送分母），剩下的中间稳态帧 Σ耗时/Σ音频。"
            "含 flush 的帧整帧剔除而不是只剔那个小 wav：该帧的 tts/decode 是为两个 wav 的"
            "token 一起花的，拆不开，只减音频会把 RTF 抬虚高。"
            "rtf_aggregate/rtf/stage_rtf 是全量口径（含头尾），作对照；"
            "stage_rtf 是各阶段单独的 RTF，相加等于 rtf_aggregate。"
            "不含 judge 侧 http_prefill（临时文件 + HTTP 往返，非模型算力）。"
        ),
    }


def _analyze_e2e(session_dir: Path) -> Dict[str, Any]:
    """从 e2e_timing.jsonl 计算 SPEAK chunk → 对应 wav 的延迟。

    主指标默认排除空 text 的 SPEAK chunk：此时 LLM 侧几乎不做 decode，
    延迟显著偏低，会拉低均值、扭曲稳定性评估。
    """
    jsonl = session_dir / "e2e_timing.jsonl"
    summary_path = session_dir / "e2e_timing_summary.json"
    if not jsonl.exists():
        raise FileNotFoundError(f"缺少 e2e_timing.jsonl: {jsonl}")

    chunks: List[Dict[str, Any]] = []
    wavs: List[Dict[str, Any]] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event") == "chunk":
            chunks.append(e)
        elif e.get("event") == "wav":
            wavs.append(e)

    speak_chunks = [c for c in chunks if c.get("mode") == "SPEAK"]
    listen_chunks = [c for c in chunks if c.get("mode") == "LISTEN"]
    # IDLE：轮次已 turn_eos 结束、本帧零产出。语义等价 LISTEN，不参与 speak 统计。
    idle_chunks = [c for c in chunks if c.get("mode") == "IDLE"]
    by_idx = {c["chunk_idx"]: c for c in speak_chunks}
    by_cnt = {int(c["cnt"]): c for c in speak_chunks if c.get("cnt") is not None}

    pairs_all: List[Dict[str, Any]] = []
    seen_idx = set()
    for w in wavs:
        # 优先用 src_cnt（C++ 在 T2WOut 入队时捕获的帧号，是真正的因果溯源）；
        # 只有拿不到时才退回 last_speak_chunk_idx 这个"poll 时刻最近的 SPEAK"的猜测。
        c = None
        src = w.get("src_cnt")
        if src is not None:
            c = by_cnt.get(int(src))
        if c is None:
            idx0 = w.get("last_speak_chunk_idx")
            if idx0 is not None:
                c = by_idx.get(idx0)
        if not c:
            continue
        idx = c["chunk_idx"]
        # 一帧可能产出多个 wav（尾包 flush）。e2e 延迟只看首包，其余交给 rtf 段落。
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        t_recv = float(c["t_recv_ms"])
        t_dec = float(c.get("t_decode_end_ms") or t_recv)
        t_poll = float(w["t_poll_ms"])
        t_mtime = w.get("t_file_mtime_rel_ms")
        text_full = c.get("text") or ""
        row = {
            "chunk_idx": idx,
            "speak_turn_id": c.get("speak_turn_id"),
            "text": text_full[:40],
            "text_empty": _is_empty_speak_text(text_full),
            "t_recv_ms": t_recv,
            "t_decode_end_ms": t_dec,
            "wav_file": w.get("file"),
            "t_wav_poll_ms": t_poll,
            "t_wav_mtime_rel_ms": t_mtime,
            "e2e_recv_to_wav_poll_ms": round(t_poll - t_recv, 1),
            "e2e_decode_to_wav_poll_ms": round(t_poll - t_dec, 1),
            "wall_clock_ms": c.get("wall_clock_ms"),
        }
        if t_mtime is not None:
            row["e2e_recv_to_wav_mtime_ms"] = round(float(t_mtime) - t_recv, 1)
        pairs_all.append(row)

    pairs = [p for p in pairs_all if not p["text_empty"]]
    pairs_empty = [p for p in pairs_all if p["text_empty"]]

    def _stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "mean_ms": round(mean(vals), 1),
            "median_ms": round(median(vals), 1),
            "min_ms": round(min(vals), 1),
            "max_ms": round(max(vals), 1),
        }

    def _lats(ps: List[Dict[str, Any]], key: str) -> List[float]:
        return [float(p[key]) for p in ps if key in p and p[key] is not None]

    recv_lats = _lats(pairs, "e2e_recv_to_wav_poll_ms")
    dec_lats = _lats(pairs, "e2e_decode_to_wav_poll_ms")
    mtime_lats = _lats(pairs, "e2e_recv_to_wav_mtime_ms")
    walls = [
        float(p["wall_clock_ms"])
        for p in pairs
        if p.get("wall_clock_ms") is not None
    ]

    # Stage timings from e2e_timing chunk events (SPEAK, non-empty text)
    stage_keys = (
        "prefill_ms",
        "vpm_ms",
        "apm_ms",
        "llm_prefill_ms",
        "cost_llm_ms",
        "cost_tts_ms",
        "cost_token2wav_ms",
    )
    speak_nonempty = [c for c in speak_chunks if not _is_empty_speak_text(c.get("text"))]
    stage_stats: Dict[str, Any] = {}
    for sk in stage_keys:
        vals = [
            float(c[sk])
            for c in speak_nonempty
            if c.get(sk) is not None
        ]
        stage_stats[sk] = _stats(vals)

    # wall ≈ prefill_ms + max(VPM,APM) + llm_prefill + llm_decode (VPM/APM parallel)
    wall_recon: List[float] = []
    for c in speak_nonempty:
        if (
            c.get("prefill_ms") is None
            or c.get("vpm_ms") is None
            or c.get("apm_ms") is None
            or c.get("llm_prefill_ms") is None
            or c.get("cost_llm_ms") is None
        ):
            continue
        wall_recon.append(
            float(c["prefill_ms"])
            + max(float(c["vpm_ms"]), float(c["apm_ms"]))
            + float(c["llm_prefill_ms"])
            + float(c["cost_llm_ms"])
        )
    stage_stats["wall_recon_ms"] = _stats(wall_recon)

    # Optional offline merge from C++ stage_timing.jsonl (tts/t2w async).
    # tts: event=tts (real codec wall); ignore legacy t2w.tts_ms (was queue_wait ≈0).
    stage = _parse_stage_timing(_read_stage_timing(session_dir))
    tts_vals = [float(e["tts_ms"]) for e in stage["tts"] if e.get("tts_ms") is not None]
    t2w_vals = [
        float(e["token2wav_ms"]) for e in stage["t2w"] if e.get("token2wav_ms") is not None
    ]
    # Prefer jsonl for async stages (SSE usually has no TTS yet at decode return).
    if tts_vals:
        stage_stats["cost_tts_ms"] = _stats(tts_vals)
    if t2w_vals:
        stage_stats["cost_token2wav_ms"] = _stats(t2w_vals)

    # 每帧 RTF：按 src_cnt 把 tts/t2w 归到 frame 上，再除以该帧产出的音频时长。
    #
    # 查找表要用**全部** chunk 而不只是 SPEAK：轮次结尾的 flush 尾音会落在紧随其后的
    # IDLE 帧上（那一帧模型没说话，但 encode/prefill/decode 照跑，也确实产出了 wav）。
    # 只放 SPEAK 的话这些帧的 RTF 会缺掉前半条链，把均值拉低。
    rtf = _analyze_rtf(
        stage, {int(c["cnt"]): c for c in chunks if c.get("cnt") is not None}
    )

    # 硬校验：poll 到的 wav 必须和 C++ 记录的 wav 一一对上。
    # 对不上说明轮询漏采或 wav 归属漂移，此时的 RTF/延迟都不可信，宁可报错也别静默出数。
    integrity: Dict[str, Any] = {}
    polled = {w.get("file") for w in wavs if w.get("file")}
    recorded = {e.get("wav") for e in stage["t2w"] if e.get("wav")}
    if recorded:
        integrity = {
            "n_polled_wav": len(polled),
            "n_cpp_wav": len(recorded),
            "missing_in_poll": sorted(recorded - polled),
            "missing_in_cpp": sorted(polled - recorded),
        }
        integrity["ok"] = not integrity["missing_in_poll"] and not integrity["missing_in_cpp"]

    turn_first: List[Dict[str, Any]] = []
    if summary_path.exists():
        summary = _read_json(summary_path)
        for t in summary.get("speak_turns", []):
            turn_first.append({
                "speak_turn_id": t.get("speak_turn_id"),
                "n_speak_chunks": t.get("n_speak_chunks"),
                "n_wavs": t.get("n_wavs"),
                "e2e_first_wav_ms": t.get("e2e_first_wav_ms"),
            })

    report = {
        "session_dir": display_path(session_dir),
        "n_chunks": len(chunks),
        "n_listen": len(listen_chunks),
        "n_idle": len(idle_chunks),
        "n_speak": len(speak_chunks),
        "n_wav_events": len(wavs),
        "n_aligned_pairs_all": len(pairs_all),
        "n_aligned_pairs": len(pairs),
        "n_excluded_empty_text": len(pairs_empty),
        "speak_chunk_wall_ms": _stats(walls),
        "e2e_speak_recv_to_wav_poll_ms": _stats(recv_lats),
        "e2e_decode_end_to_wav_poll_ms": _stats(dec_lats),
        "e2e_speak_recv_to_wav_mtime_ms": _stats(mtime_lats),
        "e2e_all_incl_empty_text_ms": _stats(_lats(pairs_all, "e2e_recv_to_wav_poll_ms")),
        "e2e_empty_text_only_ms": _stats(_lats(pairs_empty, "e2e_recv_to_wav_poll_ms")),
        "rtf": rtf,
        "wav_integrity": integrity,
        "stage_ms": stage_stats,
        "speak_turn_first_wav_ms": turn_first,
        "pairs_head": pairs[:8],
        "excluded_empty_text_pairs": [
            {
                "chunk_idx": p["chunk_idx"],
                "speak_turn_id": p["speak_turn_id"],
                "e2e_recv_to_wav_poll_ms": p["e2e_recv_to_wav_poll_ms"],
                "wall_clock_ms": p.get("wall_clock_ms"),
            }
            for p in pairs_empty
        ],
        "metric_primary": "e2e_speak_recv_to_wav_poll_ms.mean_ms",
        "metric_rtf": "rtf.core.rtf_aggregate",
        "note": (
            "主指标：SPEAK chunk 进入 worker 处理时刻(t_recv) → 对应 wav 被 poll 到(t_poll)；"
            "按 wav 的 src_cnt 归帧（拿不到才退回 last_speak_chunk_idx），一帧多包只取首包。"
            "已排除空 text 的 SPEAK chunk（LLM 几乎只 prefill、TTS/t2w 仍在跑，延迟系统性偏低）。"
            "IDLE（轮次已结束、零产出）不计入 SPEAK。"
            "stage_ms：VPM/APM/llm_prefill/cost_llm 来自 e2e chunk；tts/t2w 来自 stage_timing.jsonl。"
            "口径：e2e≈wall+decode→wav；各 stage 为流水线算力，不可直接相加得到 e2e。"
            "RTF 见 rtf 段落，主指标 rtf.core.rtf_aggregate（掐头去尾）。"
        ),
    }
    return report


def _stitch_turns(session_dir: Path, cpp_tts_dir: Optional[str] = None) -> Dict[str, Any]:
    cpp = cpp_tts_dir
    if cpp is None:
        cand = _ROOT.parent / "llama.cpp-omni" / "tools" / "omni" / "output_19060" / "tts_wav"
        if cand.is_dir():
            cpp = str(cand)
    index = stitch_speak_turn_wavs(str(session_dir), cpp_tts_dir=cpp)
    _log("")
    _log("  SPEAK turn 试听文件:")
    for t in index.get("turns") or []:
        f = t.get("file")
        if not f:
            _log(f"    turn{t.get('speak_turn_id')}: (无音频) text={t.get('text','')[:40]!r}")
            continue
        _log(
            f"    turn{t.get('speak_turn_id')}: {f}  "
            f"{t.get('duration_ms')}ms  segs={t.get('n_segments')}  "
            f"text={t.get('text','')[:40]!r}"
        )
    _log(f"  index → {display_path(session_dir / 'speak_turns' / 'index.json')}")
    return index


_STAGE_LABEL = {
    "encode": "encode(VPM|APM)",
    "llm_prefill": "llm_prefill",
    "llm_decode": "llm_decode",
    "tts": "tts",
    "token2wav": "token2wav",
}


def _stage_breakdown(stage_rtf: Dict[str, Any]) -> str:
    parts = [
        f"{_STAGE_LABEL[s]} {stage_rtf[s]:.3f}"
        for s in _RTF_STAGES
        if stage_rtf.get(s) is not None
    ]
    return " + ".join(parts)


def _print_rtf(rtf: Dict[str, Any]) -> None:
    """RTF 报告：算子加速比赛的核心指标，放在延迟前面打。"""
    if not rtf.get("available"):
        _log("  ★ RTF 不可用：stage_timing.jsonl 缺少 t2w 的 duration_ms/src_cnt")
        for w in rtf.get("warnings") or []:
            _log(f"    ! {w}")
        return

    core = rtf.get("core") or {}
    if core.get("available"):
        _log(
            f"  ★ RTF（掐头去尾）  turn={core.get('n_turns')}  "
            f"帧数={core['n_frames']}  音频={core['audio_total_ms'] / 1000:.2f}s  "
            f"算力耗时={core['compute_total_ms'] / 1000:.2f}s"
        )
        if core.get("rtf_aggregate") is not None:
            _log(f"    Σ耗时/Σ音频     {core['rtf_aggregate']:.3f}   ← 主指标")
        cst = core.get("rtf") or {}
        if cst.get("n"):
            p95 = f", p95 {cst['p95']:.3f}" if cst.get("p95") is not None else ""
            _log(
                f"    每帧全链路      平均 {cst['mean']:.3f}  "
                f"(中位 {cst['median']:.3f}{p95}, "
                f"min {cst['min']:.3f}, max {cst['max']:.3f}, n={cst['n']})"
            )
        parts = _stage_breakdown(core.get("stage_rtf") or {})
        if parts:
            _log("    分解: " + parts)
        _log(
            f"    剔除: 每轮首帧 ×{core.get('n_excluded_head')} + "
            f"含 flush 尾包的帧 ×{core.get('n_excluded_tail')}"
            f"（共 {float(core.get('excluded_audio_ms') or 0) / 1000:.2f}s 音频）"
        )

    st = rtf.get("rtf") or {}
    label = "  (对照) 全量" if core.get("available") else "  ★ RTF  全量"
    _log(
        f"{label}  帧数={rtf['n_frames']}  wav={rtf['n_wav']}  "
        f"音频={rtf['audio_total_ms'] / 1000:.2f}s  "
        f"算力耗时={rtf['compute_total_ms'] / 1000:.2f}s"
    )
    agg = rtf.get("rtf_aggregate")
    if agg is not None:
        _log(f"    Σ耗时/Σ音频     {agg:.3f}")
    if st.get("n"):
        p95 = f", p95 {st['p95']:.3f}" if st.get("p95") is not None else ""
        _log(
            f"    每帧全链路      平均 {st['mean']:.3f}  "
            f"(中位 {st['median']:.3f}{p95}, "
            f"min {st['min']:.3f}, max {st['max']:.3f}, n={st['n']})"
        )
    if not core.get("available"):
        parts = _stage_breakdown(rtf.get("stage_rtf") or {})
        if parts:
            _log("    分解: " + parts)

    n_multi = rtf.get("n_frames_multi_wav") or 0
    n_idle = sum(1 for f in (rtf.get("frames") or []) if f.get("mode") == "IDLE")
    extra = []
    if n_multi:
        extra.append(f"{n_multi} 帧产出 2 个 wav（轮次结尾 flush 的尾音）")
    if n_idle:
        extra.append(f"{n_idle} 帧是 IDLE（自己没说话，只承接了上一轮的尾音）")
    if extra:
        _log("    其中: " + "；".join(extra))
    for w in rtf.get("warnings") or []:
        _log(f"    ! {w}")
    _log("")


def _print_report(report: Dict[str, Any]) -> None:
    primary = report.get("e2e_speak_recv_to_wav_poll_ms", {})
    _log("")
    _log("=" * 60)
    _log("  Duplex E2E Latency Report")
    _log("=" * 60)
    _log(f"  session: {report['session_dir']}")
    n_excl = int(report.get("n_excluded_empty_text") or 0)
    n_all = int(report.get("n_aligned_pairs_all") or report.get("n_aligned_pairs") or 0)
    n_idle = int(report.get("n_idle") or 0)
    _log(
        f"  chunks: {report['n_chunks']} "
        f"(LISTEN={report['n_listen']}"
        + (f" IDLE={n_idle}" if n_idle else "")
        + f" SPEAK={report['n_speak']}) "
        f"wav_events={report['n_wav_events']} "
        f"aligned={report['n_aligned_pairs']}"
        + (f" (excluded empty text={n_excl}/{n_all})" if n_excl else "")
    )
    _log("")
    _print_rtf(report.get("rtf") or {})
    if primary.get("n", 0):
        _log(
            f"  ★ SPEAK输入 → wav生成(poll)  平均 {primary['mean_ms']:.1f} ms"
            f"  (中位 {primary['median_ms']:.1f}, "
            f"min {primary['min_ms']:.1f}, max {primary['max_ms']:.1f}, n={primary['n']})"
            f"  [已排除空text]"
        )
    else:
        _log("  ★ 无对齐的 SPEAK→wav 样本（可能全程 LISTEN 或未产出 wav）")

    stage = report.get("stage_ms") or {}
    if stage:
        _log("  stage (SPEAK, excl empty text):")
        for label, key in (
            ("http_prefill", "prefill_ms"),
            ("VPM", "vpm_ms"),
            ("APM", "apm_ms"),
            ("llm_prefill", "llm_prefill_ms"),
            ("llm_decode", "cost_llm_ms"),
            ("tts", "cost_tts_ms"),
            ("t2w", "cost_token2wav_ms"),
        ):
            st = stage.get(key) or {}
            if st.get("n"):
                _log(
                    f"    {label:12s} 平均 {st['mean_ms']:.1f} ms "
                    f"(中位 {st['median_ms']:.1f}, n={st['n']})"
                )
    empty_stats = report.get("e2e_empty_text_only_ms") or {}
    if empty_stats.get("n"):
        _log(
            f"    (对照) 空text SPEAK→wav   平均 {empty_stats['mean_ms']:.1f} ms"
            f"  (min {empty_stats['min_ms']:.1f}, max {empty_stats['max_ms']:.1f}, "
            f"n={empty_stats['n']})"
        )

    dec = report.get("e2e_decode_end_to_wav_poll_ms", {})
    if dec.get("n"):
        _log(
            f"    decode结束 → wav(poll)     平均 {dec['mean_ms']:.1f} ms"
            f"  (中位 {dec['median_ms']:.1f})"
        )
    mt = report.get("e2e_speak_recv_to_wav_mtime_ms", {})
    if mt.get("n"):
        _log(
            f"    SPEAK输入 → wav(mtime)     平均 {mt['mean_ms']:.1f} ms"
            f"  (中位 {mt['median_ms']:.1f})"
        )
    wall = report.get("speak_chunk_wall_ms", {})
    if wall.get("n"):
        _log(
            f"    SPEAK chunk wall           平均 {wall['mean_ms']:.1f} ms"
            f"  (中位 {wall['median_ms']:.1f})"
        )

    turns = report.get("speak_turn_first_wav_ms") or []
    if turns:
        _log("")
        _log("  各 turn 首包 e2e_first_wav_ms:")
        for t in turns:
            _log(
                f"    turn{t.get('speak_turn_id')}: "
                f"{t.get('e2e_first_wav_ms')} ms  "
                f"(speak_chunks={t.get('n_speak_chunks')} wavs={t.get('n_wavs')})"
            )
    _log("=" * 60)


async def _check_health(gateway_http: str, worker_http: str, cpp_http: str) -> None:
    import httpx

    async with httpx.AsyncClient(verify=False, timeout=5.0) as client:
        for name, url in (
            ("cpp", f"{cpp_http.rstrip('/')}/health"),
            ("worker", f"{worker_http.rstrip('/')}/health"),
            ("gateway", f"{gateway_http.rstrip('/')}/health"),
        ):
            try:
                r = await client.get(url)
                _log(f"[health] {name}: {r.status_code} {r.text[:120]}")
            except Exception as e:
                raise RuntimeError(f"{name} 健康检查失败 ({url}): {e}") from e


async def _replay_and_get_recording_session(
    session_dir: Optional[Path],
    gateway_ws_base: str,
    speed: float,
    timing_mode: str,
    send_interval_s: float,
    pre_stop_wait_s: float,
    post_stop_wait_s: float,
    insecure: bool,
    ref_audio: Optional[str],
    max_chunks: Optional[int],
    send_items: Optional[List[Dict[str, Any]]] = None,
    app_type: str = "omni_duplex",
    system_prompt: Optional[str] = None,
    max_slice_nums: int = 1,
) -> str:
    """回放 session 或预构造的 send_items，返回 recording_session_id。"""
    meta: Dict[str, Any] = {}
    if session_dir is not None and (session_dir / "meta.json").exists():
        meta = _read_json(session_dir / "meta.json")
        app_type = str(meta.get("app_type") or app_type)
        if system_prompt is None:
            system_prompt = meta.get("config", {}).get("system_prompt")
        max_slice_nums = int(meta.get("config", {}).get("max_slice_nums", max_slice_nums))

    if send_items is None:
        if session_dir is None:
            raise ValueError("既无 session_dir 也无 send_items")
        recording = _read_json(session_dir / "recording.json")
        send_items = _collect_send_items(
            session_dir=session_dir,
            recording=recording,
            app_type=app_type,
            chunk_duration_s=1.0,
        )

    if max_chunks is not None and max_chunks > 0:
        send_items = send_items[:max_chunks]
    if not send_items:
        raise RuntimeError("没有可发送的 chunk")

    session_id = _resolve_session_id(app_type=app_type, cli_session_id=None)
    ws_url = f"{gateway_ws_base.rstrip('/')}/ws/duplex/{session_id}"
    _log(f"[replay] app_type={app_type} ws={ws_url}")
    _log(f"[replay] chunks to send: {len(send_items)}")

    ref_audio_b64 = _load_ref_audio_b64(meta, session_dir or _ROOT, ref_audio)
    if ref_audio_b64 is None:
        fallback = _default_ref_audio()
        if fallback:
            _log(f"[replay] meta ref 不可用，改用默认: {fallback}")
            ref_audio_b64 = _load_ref_audio_b64(meta, session_dir or _ROOT, fallback)

    if not system_prompt:
        system_prompt = "Streaming Omni Conversation."

    async def _connect(url: str, use_insecure: bool):
        ssl_arg = None
        if url.startswith("wss://"):
            ctx = ssl.create_default_context()
            if use_insecure:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            ssl_arg = ctx
        cm = websockets.connect(url, max_size=128 * 1024 * 1024, ssl=ssl_arg)
        ws = await cm.__aenter__()
        return cm, ws

    actual = ws_url
    try:
        ws_cm, ws = await _connect(actual, insecure)
    except Exception:
        if actual.startswith("ws://"):
            actual = "wss://" + actual[len("ws://"):]
            _log(f"[replay] fallback → {actual}")
            ws_cm, ws = await _connect(actual, True)
        else:
            raise

    recording_session_id: Optional[str] = None
    sender_done = asyncio.Event()
    duplex_stats = {"result_listen": 0, "result_speak": 0, "audio_only": 0}

    try:
        prepare_msg: Dict[str, Any] = {
            "type": "prepare",
            "system_prompt": system_prompt,
            "max_slice_nums": max_slice_nums,
            "deferred_finalize": True,
            "config": {"length_penalty": 1.0},
        }
        if ref_audio_b64:
            prepare_msg["ref_audio_base64"] = ref_audio_b64
            prepare_msg["tts_ref_audio_base64"] = ref_audio_b64
        await ws.send(json.dumps(prepare_msg, ensure_ascii=False))

        while True:
            msg = json.loads(await ws.recv())
            tp = msg.get("type")
            if tp == "prepared":
                recording_session_id = msg.get("recording_session_id")
                _log(f"[replay] prepared, recording_session_id={recording_session_id}")
                break
            if tp == "error":
                raise RuntimeError(f"prepare failed: {msg}")

        start_send_wall = time.perf_counter()
        first_ts = float(send_items[0]["ts_ms"]) if send_items else 0.0
        speed = max(speed, 1e-6)

        async def sender() -> None:
            for i, item in enumerate(send_items):
                if timing_mode == "recorded":
                    target = (float(item["ts_ms"]) - first_ts) / 1000.0 / speed
                else:
                    target = (i * send_interval_s) / speed
                sleep_s = target - (time.perf_counter() - start_send_wall)
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                await ws.send(json.dumps(item["msg"], ensure_ascii=False))
                if i % 20 == 0 or i + 1 == len(send_items):
                    _log(f"[SEND] {i + 1}/{len(send_items)}")
            if pre_stop_wait_s > 0:
                await asyncio.sleep(pre_stop_wait_s)
            await ws.send(json.dumps({"type": "stop"}))
            sender_done.set()
            _log("[SEND] stop")

        async def receiver() -> None:
            stop_deadline: Optional[float] = None
            while True:
                timeout_s = 2.0
                if sender_done.is_set():
                    if stop_deadline is None:
                        stop_deadline = time.perf_counter() + post_stop_wait_s
                    remaining = stop_deadline - time.perf_counter()
                    if remaining <= 0:
                        return
                    timeout_s = min(timeout_s, max(0.1, remaining))
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    return
                msg = json.loads(raw)
                tp = msg.get("type")
                if tp == "result":
                    if msg.get("is_listen", True):
                        duplex_stats["result_listen"] += 1
                    else:
                        duplex_stats["result_speak"] += 1
                elif tp == "audio_only":
                    duplex_stats["audio_only"] += 1
                elif tp == "stopped" and sender_done.is_set():
                    return
                elif tp == "error":
                    _log(f"[WARN] server error: {msg}")

        await asyncio.gather(sender(), receiver())
    finally:
        await ws_cm.__aexit__(None, None, None)

    _log(f"[replay] duplex_stats={duplex_stats}")
    if not recording_session_id:
        raise RuntimeError("prepare 未返回 recording_session_id，无法定位 e2e_timing 输出")
    return recording_session_id


def _wait_e2e_summary(sessions_root: Path, recording_session_id: str, timeout_s: float = 30.0) -> Path:
    session_dir = sessions_root / recording_session_id
    summary = session_dir / "e2e_timing_summary.json"
    jsonl = session_dir / "e2e_timing.jsonl"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if summary.exists() or (jsonl.exists() and jsonl.stat().st_size > 0):
            if not summary.exists():
                time.sleep(0.5)
            if summary.exists() or jsonl.exists():
                return session_dir
        time.sleep(0.2)
    raise TimeoutError(
        f"等待 e2e 输出超时 ({timeout_s}s): 期望目录 {session_dir} "
        f"(summary={summary.exists()}, jsonl={jsonl.exists()})"
    )


async def _amain(args: argparse.Namespace) -> int:
    sessions_root = Path(args.sessions_root).resolve()
    if not sessions_root.is_absolute():
        sessions_root = (_ROOT / sessions_root).resolve()

    if args.offline:
        out_session = Path(args.offline).resolve()
        if not out_session.is_absolute():
            out_session = (_ROOT / out_session).resolve()
        _log(f"[offline] analyze {out_session}")
        report = _analyze_e2e(out_session)
        stitch = _stitch_turns(out_session, cpp_tts_dir=args.cpp_tts_dir)
        report["speak_turn_wavs"] = stitch
        _print_report(report)
        out_path = out_session / "eval_e2e_report.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"[offline] wrote {out_path}")
        return 0

    send_items: Optional[List[Dict[str, Any]]] = None
    in_session: Optional[Path] = None
    input_label = ""

    if args.video:
        video_path = Path(args.video).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"视频不存在: {video_path}")
        max_chunks = args.max_chunks
        max_duration = float(args.max_duration)
        if max_chunks is not None and max_chunks > 0:
            max_duration = min(max_duration, max_chunks * float(args.chunk_duration_s))
        system_prompt = args.system_prompt or (
            "扮演一个具有以上声音特征的助手。请认真、高质量地回复用户的问题。"
            "请用高自然度的方式和用户聊天。你处于双工模式，可以一边听、一边说，"
            "并且能看到用户的画面。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
        )
        cfg = VideoChunkerConfig(
            chunk_duration_s=float(args.chunk_duration_s),
            frame_offset_s=float(args.frame_offset),
            max_duration_s=max_duration,
            max_chunks=max_chunks,
            audio_mode=str(args.video_audio_mode),
            external_audio_path=args.audio,
            include_frames=not bool(args.no_video_frames),
            pad_before_s=int(args.pad_before),
            pad_after_s=int(args.pad_after),
        )
        _log(
            f"[input] video={video_path}  "
            f"audio_mode={cfg.audio_mode}  chunk={cfg.chunk_duration_s}s  "
            f"frame_offset={cfg.frame_offset_s}s  include_frames={cfg.include_frames}  "
            f"pad_before={cfg.pad_before_s}s pad_after={cfg.pad_after_s}s"
        )
        send_items = build_send_items_from_video(video_path, cfg)
        n_pad = sum(1 for it in send_items if it.get("is_pad"))
        _log(
            f"[input] extracted {len(send_items)} chunks "
            f"(main={len(send_items) - n_pad}, pad={n_pad})"
        )
        input_label = display_path(video_path)
        if args.session_dir:
            in_session = Path(args.session_dir).resolve()
            if not in_session.is_absolute():
                in_session = (_ROOT / in_session).resolve()
    else:
        system_prompt = args.system_prompt
        if args.session_dir:
            in_session = Path(args.session_dir).resolve()
            if not in_session.is_absolute():
                in_session = (_ROOT / in_session).resolve()
        else:
            in_session = _find_latest_session(sessions_root)
        _log(f"[input] session={in_session}")
        input_label = display_path(in_session)

    await _check_health(args.gateway_http, args.worker_http, args.cpp_http)

    t_run0 = time.perf_counter()
    recording_sid = await _replay_and_get_recording_session(
        session_dir=in_session,
        gateway_ws_base=args.gateway_ws_base,
        speed=args.speed,
        timing_mode=args.timing_mode,
        send_interval_s=args.send_interval_s,
        pre_stop_wait_s=args.pre_stop_wait_s,
        post_stop_wait_s=args.post_stop_wait_s,
        insecure=args.insecure,
        ref_audio=args.ref_audio,
        max_chunks=args.max_chunks if not args.video else None,  # video 已在切分时截断
        send_items=send_items,
        app_type="omni_duplex",
        system_prompt=system_prompt,
        max_slice_nums=int(args.max_slice_nums),
    )
    out_session = _wait_e2e_summary(sessions_root, recording_sid, timeout_s=args.wait_e2e_s)
    # stop 后 worker finally 会 finalize e2e；再稍等
    time.sleep(1.0)
    if not (out_session / "e2e_timing.jsonl").exists():
        raise FileNotFoundError(f"回放后仍无 e2e_timing.jsonl: {out_session}")

    report = _analyze_e2e(out_session)
    report["input"] = input_label
    if in_session is not None:
        report["input_session_dir"] = display_path(in_session)
    if args.video:
        report["input_video"] = display_path(args.video)
    report["recording_session_id"] = recording_sid
    report["replay_wall_s"] = round(time.perf_counter() - t_run0, 1)
    report["created_at"] = datetime.now().isoformat(timespec="seconds")
    stitch = _stitch_turns(out_session, cpp_tts_dir=args.cpp_tts_dir)
    report["speak_turn_wavs"] = stitch
    _print_report(report)

    out_path = out_session / "eval_e2e_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pointer = {
        "input": input_label,
        "eval_session": display_path(out_session),
        "report": display_path(out_path),
        "primary_mean_ms": (report.get("e2e_speak_recv_to_wav_poll_ms") or {}).get("mean_ms"),
    }
    if in_session is not None:
        pointer_path = in_session / "last_eval_e2e_pointer.json"
        pointer["input_session"] = display_path(in_session)
        pointer_path.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"[done] pointer → {display_path(pointer_path)}")
    _log(f"[done] report → {display_path(out_path)}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="一键评测 SPEAK→wav 端到端延迟")
    p.add_argument(
        "--session-dir",
        default=None,
        help="输入 session（含 recording.json + user_audio）。默认取 sessions 下最新一场",
    )
    p.add_argument(
        "--video",
        default=None,
        help="直接输入视频文件（mp4 等）：按 1s 切分音画后回放，无需预先录 session",
    )
    p.add_argument(
        "--video-audio-mode",
        choices=["video", "silent", "external"],
        default="video",
        help="视频音轨来源：video=视频轨 / silent=静音 / external=--audio",
    )
    p.add_argument("--audio", default=None, help="external 模式下的外部音轨 wav/mp3")
    p.add_argument("--max-duration", type=float, default=120.0, help="视频最大秒数")
    p.add_argument("--chunk-duration-s", type=float, default=1.0, help="切分 chunk 时长（秒）")
    p.add_argument(
        "--frame-offset",
        type=float,
        default=0.5,
        help="每秒帧取样偏移（秒），默认 0.5",
    )
    p.add_argument(
        "--no-video-frames",
        action="store_true",
        help="只发音频、不发视频帧（对照实验）",
    )
    p.add_argument(
        "--pad-before",
        type=int,
        default=0,
        help="视频片头 pad 秒数（静音+首帧）",
    )
    p.add_argument(
        "--pad-after",
        type=int,
        default=2,
        help="视频片尾 pad 秒数（静音+末帧），默认 2",
    )
    p.add_argument("--system-prompt", default=None, help="覆盖 prepare 的 system_prompt")
    p.add_argument("--max-slice-nums", type=int, default=1, help="vision max_slice_nums")
    p.add_argument(
        "--offline",
        default=None,
        help="只离线分析该 session 的 e2e_timing，不回放",
    )
    p.add_argument(
        "--sessions-root",
        default=str(_ROOT / "sessions"),
        help="session 根目录（默认 ./sessions）",
    )
    p.add_argument("--gateway-ws-base", default="wss://127.0.0.1:8006")
    p.add_argument("--gateway-http", default="https://127.0.0.1:8006")
    p.add_argument("--worker-http", default="http://127.0.0.1:22400")
    p.add_argument("--cpp-http", default="http://127.0.0.1:19060")
    p.add_argument("--insecure", action="store_true", default=True, help="忽略自签名证书（默认开）")
    p.add_argument("--no-insecure", action="store_false", dest="insecure")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--timing-mode", choices=["fixed", "recorded"], default="fixed")
    p.add_argument("--send-interval-s", type=float, default=1.0)
    p.add_argument("--pre-stop-wait-s", type=float, default=8.0)
    p.add_argument("--post-stop-wait-s", type=float, default=5.0)
    p.add_argument("--wait-e2e-s", type=float, default=60.0)
    p.add_argument("--ref-audio", default=None, help="覆盖 ref wav；默认用 assets/ref_audio/...")
    p.add_argument("--max-chunks", type=int, default=None, help="只回放前 N 个 chunk（冒烟用）")
    p.add_argument(
        "--cpp-tts-dir",
        default=None,
        help="可选：C++ tts_wav 目录，作 ai_audio 缺失时的回退",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        _log("interrupted")
        return 130
    except Exception as e:
        _log(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
