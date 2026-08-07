"""Duplex 端到端时间线：chunk 输入、SPEAK、wav 落盘。

输出（session 目录）：
  - e2e_timing.jsonl
  - e2e_timing_summary.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from judge_support import display_path

logger = logging.getLogger(__name__)

_WAV_ID_RE = re.compile(r"wav_(\d+)")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def wav_src_cnt(fname: str) -> Optional[int]:
    """从 wav 文件名解析产出它的 frame 号。

    C++ 侧 duplex 的命名是 `wav_{src_cnt * 1000 + wav_seq}.wav`，其中 src_cnt 是
    T2WOut **入队时**捕获的帧号（不是写盘时刻的全局值），所以这是可信的因果溯源，
    比按 poll 时刻猜 last_speak_chunk_idx 稳得多。
    """
    m = _WAV_ID_RE.search(fname or "")
    if not m:
        return None
    return int(m.group(1)) // 1000


class DuplexE2ETiming:
    def __init__(self, session_dir: str, session_start_perf: float) -> None:
        self.session_dir = session_dir
        self.session_start_perf = session_start_perf
        self.session_start_wall = time.time()
        self.jsonl_path = os.path.join(session_dir, "e2e_timing.jsonl")
        self.summary_path = os.path.join(session_dir, "e2e_timing_summary.json")
        self._lock = threading.Lock()
        self._events: List[Dict[str, Any]] = []
        self._speak_turn_id = 0
        self._in_speak = False
        self._last_speak_chunk_idx: Optional[int] = None
        self._last_speak_recv_ms: Optional[float] = None
        self._first_speak_recv_ms_by_turn: Dict[int, float] = {}
        # speak_turn -> chunk / wav events
        self._turns: Dict[int, Dict[str, Any]] = {}
        os.makedirs(session_dir, exist_ok=True)
        # truncate previous run if same dir reused
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            f.write("")
        logger.info(f"[E2ETiming] writing → {display_path(self.jsonl_path)}")

    def _rel_ms(self, perf: Optional[float] = None) -> float:
        if perf is None:
            perf = time.perf_counter()
        return round((perf - self.session_start_perf) * 1000, 1)

    def mtime_rel_ms(self, path: str) -> Optional[float]:
        try:
            return round((os.path.getmtime(path) - self.session_start_wall) * 1000, 1)
        except OSError:
            return None


    def _append_unlocked(self, event: Dict[str, Any]) -> None:
        event.setdefault("wall_iso", _now_iso())
        self._events.append(event)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _ensure_turn_unlocked(self, turn_id: int) -> Dict[str, Any]:
        if turn_id not in self._turns:
            self._turns[turn_id] = {
                "speak_turn_id": turn_id,
                "chunks": [],
                "wavs": [],
            }
        return self._turns[turn_id]

    def log_chunk(
        self,
        chunk_idx: int,
        cnt: int,
        is_listen: bool,
        end_of_turn: bool,
        text: str,
        t_recv_perf: float,
        t_prefill_end_perf: Optional[float] = None,
        t_decode_end_perf: Optional[float] = None,
        prefill_ms: Optional[float] = None,
        wall_clock_ms: Optional[float] = None,
        dropped_before: int = 0,
        turn_idle: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """记录一个用户 audio_chunk 的处理结果。返回 speak_turn_id（SPEAK 时有效）。

        turn_idle：服务端报的三态第三态 —— 轮次已经 turn_eos 结束、本帧零 TTS token
        产出。模型没主动采样 <|listen|>，所以 is_listen 仍是 false，但语义上等价于
        LISTEN。这类帧不计入 speak 轮次，否则整场会被折叠成一个 turn。
        """
        t_recv_ms = self._rel_ms(t_recv_perf)
        turn_idle = bool(turn_idle)
        model_listen = bool(is_listen)
        # IDLE 帧按 LISTEN 归类：既不开新 speak turn，也不更新 last_speak_chunk_idx。
        # mode 上仍然区分出来，方便事后核对到底是模型主动 listen 还是说完了在等。
        not_speaking = model_listen or turn_idle
        mode = "LISTEN" if model_listen else ("IDLE" if turn_idle else "SPEAK")

        with self._lock:
            if not not_speaking:
                if not self._in_speak:
                    self._speak_turn_id += 1
                    self._in_speak = True
                    self._first_speak_recv_ms_by_turn[self._speak_turn_id] = t_recv_ms
                speak_turn_id = self._speak_turn_id
                self._last_speak_chunk_idx = chunk_idx
                self._last_speak_recv_ms = t_recv_ms
            else:
                speak_turn_id = self._speak_turn_id if self._speak_turn_id else 0
                if self._in_speak:
                    self._in_speak = False

            event: Dict[str, Any] = {
                "event": "chunk",
                "chunk_idx": chunk_idx,
                "cnt": cnt,
                "mode": mode,
                "is_listen": model_listen,
                "turn_idle": turn_idle,
                "end_of_turn": bool(end_of_turn),
                "speak_turn_id": speak_turn_id if mode == "SPEAK" else None,
                "t_recv_ms": t_recv_ms,
                "text": (text or "")[:200],
                "dropped_before": dropped_before,
            }
            if t_prefill_end_perf is not None:
                event["t_prefill_end_ms"] = self._rel_ms(t_prefill_end_perf)
            if t_decode_end_perf is not None:
                event["t_decode_end_ms"] = self._rel_ms(t_decode_end_perf)
            if prefill_ms is not None:
                event["prefill_ms"] = round(prefill_ms, 1)
            if wall_clock_ms is not None:
                event["wall_clock_ms"] = round(wall_clock_ms, 1)
            if extra:
                event.update(extra)

            self._append_unlocked(event)

            if mode == "SPEAK" and speak_turn_id:
                turn = self._ensure_turn_unlocked(speak_turn_id)
                turn["chunks"].append({
                    "chunk_idx": chunk_idx,
                    "cnt": cnt,
                    "t_recv_ms": t_recv_ms,
                    "t_decode_end_ms": event.get("t_decode_end_ms"),
                    "wall_clock_ms": event.get("wall_clock_ms"),
                    "text": event["text"],
                    "end_of_turn": bool(end_of_turn),
                    "wall_iso": event["wall_iso"],
                })

        return speak_turn_id

    def log_wav(
        self,
        files: List[str],
        file_mtimes_rel_ms: Optional[Dict[str, float]] = None,
        t_poll_perf: Optional[float] = None,
        wav_info: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """记录一批新检出的 wav（poll 时刻 + 文件 mtime + 时长 + 源帧号）。

        src_cnt 从文件名解析，是 C++ 侧在 T2WOut 入队时捕获的帧号；
        last_speak_chunk_idx 仍然保留，但只作为 src_cnt 缺失时的退路。
        """
        t_poll_ms = self._rel_ms(t_poll_perf)
        with self._lock:
            speak_turn_id = self._speak_turn_id if self._speak_turn_id > 0 else None
            last_chunk = self._last_speak_chunk_idx
            first_recv = (
                self._first_speak_recv_ms_by_turn.get(speak_turn_id)
                if speak_turn_id
                else None
            )
            last_recv = self._last_speak_recv_ms

            for fname in files:
                mtime_ms = None
                if file_mtimes_rel_ms and fname in file_mtimes_rel_ms:
                    mtime_ms = file_mtimes_rel_ms[fname]
                event: Dict[str, Any] = {
                    "event": "wav",
                    "file": fname,
                    "src_cnt": wav_src_cnt(fname),
                    "t_poll_ms": t_poll_ms,
                    "speak_turn_id": speak_turn_id,
                    "last_speak_chunk_idx": last_chunk,
                }
                info = (wav_info or {}).get(fname)
                if info:
                    event["n_samples"] = info.get("n_samples")
                    event["sample_rate"] = info.get("sample_rate")
                    event["duration_ms"] = info.get("duration_ms")
                if mtime_ms is not None:
                    event["t_file_mtime_rel_ms"] = mtime_ms
                if first_recv is not None:
                    event["latency_from_first_speak_recv_ms"] = round(
                        t_poll_ms - first_recv, 1
                    )
                if last_recv is not None:
                    event["latency_from_last_speak_recv_ms"] = round(
                        t_poll_ms - last_recv, 1
                    )

                self._append_unlocked(event)

                if speak_turn_id:
                    turn = self._ensure_turn_unlocked(speak_turn_id)
                    turn["wavs"].append({
                        "file": fname,
                        "src_cnt": event["src_cnt"],
                        "duration_ms": event.get("duration_ms"),
                        "t_poll_ms": t_poll_ms,
                        "t_file_mtime_rel_ms": mtime_ms,
                        "last_speak_chunk_idx": last_chunk,
                        "latency_from_first_speak_recv_ms": event.get(
                            "latency_from_first_speak_recv_ms"
                        ),
                        "latency_from_last_speak_recv_ms": event.get(
                            "latency_from_last_speak_recv_ms"
                        ),
                        "wall_iso": event["wall_iso"],
                    })

    def finalize(self) -> Dict[str, Any]:
        """写汇总：每个 SPEAK turn 的输入 chunk 时间 vs wav 时间对照。"""
        turns_out: List[Dict[str, Any]] = []
        for turn_id in sorted(self._turns.keys()):
            turn = self._turns[turn_id]
            chunks = turn["chunks"]
            wavs = turn["wavs"]
            first_chunk_ms = chunks[0]["t_recv_ms"] if chunks else None
            first_wav_ms = wavs[0]["t_poll_ms"] if wavs else None
            last_wav_ms = wavs[-1]["t_poll_ms"] if wavs else None
            row: Dict[str, Any] = {
                "speak_turn_id": turn_id,
                "n_speak_chunks": len(chunks),
                "n_wavs": len(wavs),
                "chunks": chunks,
                "wavs": wavs,
            }
            if first_chunk_ms is not None and first_wav_ms is not None:
                row["e2e_first_wav_ms"] = round(first_wav_ms - first_chunk_ms, 1)
            if first_chunk_ms is not None and last_wav_ms is not None:
                row["e2e_last_wav_ms"] = round(last_wav_ms - first_chunk_ms, 1)
            turns_out.append(row)

        summary = {
            "session_dir": display_path(self.session_dir),
            "jsonl": os.path.basename(self.jsonl_path),
            "n_events": len(self._events),
            "n_speak_turns": len(turns_out),
            "speak_turns": turns_out,
            "note": (
                "t_*_ms 相对 session 开始；wav 与 chunk 非 1:1，"
                "按 speak_turn_id 对齐；e2e_first_wav_ms = 首个 wav poll - 该 turn 首个 SPEAK chunk 输入"
            ),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(
            f"[E2ETiming] summary → {display_path(self.summary_path)} "
            f"(speak_turns={len(turns_out)}, events={len(self._events)})"
        )
        return summary


def stitch_speak_turn_wavs(
    session_dir: str,
    cpp_tts_dir: Optional[str] = None,
    out_subdir: str = "speak_turns",
) -> Dict[str, Any]:
    """按 speak_turn 拼接 AI 语音到 speak_turns/。"""
    import numpy as np
    import soundfile as sf

    session_dir = os.path.abspath(session_dir)
    summary_path = os.path.join(session_dir, "e2e_timing_summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"缺少 e2e_timing_summary.json: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    turns = summary.get("speak_turns") or []
    if not turns:
        return {"session_dir": display_path(session_dir), "n_turns": 0, "files": []}

    timeline: List[Dict[str, Any]] = []
    rec_path = os.path.join(session_dir, "recording.json")
    if os.path.isfile(rec_path):
        with open(rec_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        timeline = list(rec.get("ai_audio_timeline") or [])
        timeline.sort(key=lambda x: (float(x.get("receive_ts_ms", 0)), int(x.get("seq", 0))))

    used_ai_seq: set = set()

    def _read_mono(path: str) -> Optional[Any]:
        if not os.path.isfile(path):
            return None
        try:
            data, sr = sf.read(path, always_2d=False)
            if getattr(data, "ndim", 1) > 1:
                data = data[:, 0]
            data = data.astype(np.float32)
            if sr != 24000 and len(data) > 1:
                old_x = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
                new_len = max(1, int(round(len(data) * 24000 / sr)))
                new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
                data = np.interp(new_x, old_x, data).astype(np.float32)
            return data
        except Exception as e:
            logger.warning(f"[stitch] read fail {path}: {e}")
            return None

    def _match_ai_audio(t_poll_ms: float) -> Optional[str]:
        """timeline 中未用过的、receive_ts 最接近 t_poll 的条目。"""
        best = None
        best_dt = None
        for item in timeline:
            seq = item.get("seq")
            if seq in used_ai_seq:
                continue
            dt = abs(float(item.get("receive_ts_ms", 0)) - float(t_poll_ms))
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = item
        if best is None or best_dt is None or best_dt > 2000:
            return None
        used_ai_seq.add(best.get("seq"))
        return os.path.join(session_dir, best["rel"])

    out_dir = os.path.join(session_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    files_out: List[Dict[str, Any]] = []

    for turn in turns:
        turn_id = int(turn.get("speak_turn_id") or 0)
        wav_metas = turn.get("wavs") or []
        pieces: List[Any] = []
        sources: List[str] = []

        for w in wav_metas:
            t_poll = float(w.get("t_poll_ms") or 0)
            src = _match_ai_audio(t_poll)
            if src is None and cpp_tts_dir and w.get("file"):
                cand = os.path.join(cpp_tts_dir, w["file"])
                if os.path.isfile(cand):
                    src = cand
            if src is None:
                continue
            pcm = _read_mono(src)
            if pcm is None or len(pcm) == 0:
                continue
            pieces.append(pcm)
            sources.append(
                os.path.relpath(src, session_dir)
                if src.startswith(session_dir)
                else display_path(src)
            )

        out_name = f"turn_{turn_id:02d}.wav"
        out_path = os.path.join(out_dir, out_name)
        duration_ms = 0.0
        if pieces:
            merged = np.concatenate(pieces).astype(np.float32)
            sf.write(out_path, merged, 24000, subtype="PCM_16")
            duration_ms = round(len(merged) / 24000 * 1000, 1)
        else:
            if os.path.exists(out_path):
                os.remove(out_path)

        texts = [c.get("text") or "" for c in (turn.get("chunks") or [])]
        text_joined = "".join(texts).strip()

        files_out.append({
            "speak_turn_id": turn_id,
            "file": f"{out_subdir}/{out_name}" if pieces else None,
            "n_segments": len(pieces),
            "n_wav_events": len(wav_metas),
            "duration_ms": duration_ms,
            "sources": sources,
            "text": text_joined[:200],
        })
        logger.info(
            f"[stitch] turn={turn_id} segments={len(pieces)} "
            f"duration={duration_ms:.0f}ms → {out_name}"
        )

    index = {
        "session_dir": display_path(session_dir),
        "out_dir": display_path(out_dir),
        "sample_rate": 24000,
        "n_turns": len(files_out),
        "turns": files_out,
        "note": "按 e2e speak_turn_id 拼接；音频优先 session/ai_audio（时间戳对齐），否则 cpp tts_wav",
    }
    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    logger.info(f"[stitch] index → {display_path(index_path)}")
    return index
