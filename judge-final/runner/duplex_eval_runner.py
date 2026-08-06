"""Duplex 评测 runner：1Hz 发送 + e2e 计时。"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

JUDGE_ROOT = Path(__file__).resolve().parents[1]
if str(JUDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(JUDGE_ROOT))
if str(JUDGE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(JUDGE_ROOT / "scripts"))

from judge_support import display_path  # noqa: E402
from e2e_timing import DuplexE2ETiming  # noqa: E402
from omni_client import DuplexSession  # noqa: E402
from omni_client.wav_poller import tts_wav_dir  # noqa: E402

logger = logging.getLogger("duplex_eval_runner")


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _default_ref_audio() -> Optional[str]:
    p = JUDGE_ROOT / "assets" / "ref_audio" / "ref_minicpm_signature.wav"
    return str(p) if p.is_file() else None


def _import_analyze():
    from eval_duplex_e2e_latency import (  # type: ignore
        _analyze_e2e,
        _print_report,
        _stitch_turns,
    )

    return _analyze_e2e, _print_report, _stitch_turns


def _decode_send_item(item: Dict[str, Any]):
    """从 video chunker / session replay 的 send_item 解码 audio + frames。"""
    msg = item.get("msg") or item
    audio_b64 = msg.get("audio_base64")
    if not audio_b64:
        raise ValueError("send_item missing audio_base64")
    audio_waveform = np.frombuffer(base64.b64decode(audio_b64), dtype=np.float32)

    frame_list = None
    frame_b64_list = msg.get("frame_base64_list")
    if frame_b64_list:
        from PIL import Image

        frame_list = []
        for fb64 in frame_b64_list:
            frame_bytes = base64.b64decode(fb64)
            frame_list.append(Image.open(io.BytesIO(frame_bytes)))
    return audio_waveform, frame_list, msg


def _save_user_media(
    session_dir: Path,
    chunk_idx: int,
    audio_waveform: np.ndarray,
    frame_list,
) -> None:
    import soundfile as sf

    ua = session_dir / "user_audio"
    ua.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(ua / f"{chunk_idx:03d}.wav"),
        np.clip(audio_waveform, -1.0, 1.0).astype(np.float32),
        16000,
        format="WAV",
        subtype="PCM_16",
    )
    if frame_list:
        uf = session_dir / "user_frames"
        uf.mkdir(parents=True, exist_ok=True)
        frame_list[0].convert("RGB").save(uf / f"{chunk_idx:03d}.jpg", format="JPEG")


async def run_direct_eval(
    *,
    session: DuplexSession,
    send_items: List[Dict[str, Any]],
    sessions_root: Path,
    ref_audio: Optional[str] = None,
    length_penalty: float = 1.0,
    send_interval_s: float = 1.0,
    speed: float = 1.0,
    pre_stop_wait_s: float = 2.0,
    post_stop_wait_s: float = 1.0,
    max_slice_nums: int = 1,
    quiet: bool = True,
    session_suffix: Optional[str] = None,
    on_chunk_done: Optional[Callable[[], None]] = None,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """跑一轮 duplex 评测，返回 session_dir。

    quiet=True（默认）：不向控制台刷进度 / 长报告；由调用方 print_final_summary。
    """
    def _vlog(msg: str) -> None:
        if not quiet:
            _log(msg)

    sessions_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{session_suffix}" if session_suffix else ""
    sid = f"{stamp}_omni_direct_{os.getpid()}{suffix}"
    session_dir = sessions_root / sid
    if session_dir.exists():
        n = 2
        while True:
            cand = sessions_root / f"{sid}_{n}"
            if not cand.exists():
                session_dir = cand
                sid = cand.name
                break
            n += 1
    session_dir.mkdir(parents=True, exist_ok=False)

    ref = ref_audio or session.ref_audio_path or _default_ref_audio()
    stage_timing = Path(session.output_dir) / "stage_timing.jsonl"
    if stage_timing.parent.is_dir():
        stage_timing.write_text("", encoding="utf-8")

    meta = {
        "session_id": sid,
        "app_type": "omni_duplex",
        "backend": "direct_omni",
        "cpp_port": session.cpp_port,
        "ref_audio": display_path(ref) if ref else None,
        "length_penalty": length_penalty,
        "n_send_items": len(send_items),
        **(meta_extra or {}),
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    session_start_perf = time.perf_counter()
    e2e = DuplexE2ETiming(
        session_dir=str(session_dir), session_start_perf=session_start_perf
    )

    await asyncio.to_thread(
        session.duplex_prepare,
        ref_audio_path=ref,
        length_penalty=float(length_penalty),
    )
    _vlog(f"[direct] prepared session_dir={display_path(session_dir)} ref={display_path(ref) if ref else None}")

    stop_poll = threading.Event()
    tts_dir = tts_wav_dir(session.output_dir)

    def _wav_poll_loop() -> None:
        while not stop_poll.is_set():
            try:
                poll_t0 = time.perf_counter()
                _audio_b64, _, new_wav_files, wav_info = session.collect_wav_nowait()
                if new_wav_files:
                    mtimes = {}
                    for wf in new_wav_files:
                        wp = os.path.join(tts_dir, wf)
                        rel = e2e.mtime_rel_ms(wp)
                        if rel is not None:
                            mtimes[wf] = rel
                    e2e.log_wav(
                        files=list(new_wav_files),
                        file_mtimes_rel_ms=mtimes,
                        t_poll_perf=poll_t0,
                        wav_info=wav_info,
                    )
            except Exception as e:
                logger.warning("[WAV poll] error: %s", e)
            stop_poll.wait(0.1)

    poll_thread = threading.Thread(target=_wav_poll_loop, daemon=True)
    poll_thread.start()

    speed = max(float(speed), 1e-6)
    start_send_wall = time.perf_counter()
    chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    dropped = {"n": 0}

    async def _process_one(chunk_idx: int, item: Dict[str, Any]) -> None:
        t_chunk_start = time.perf_counter()
        audio_waveform, frame_list, _msg = _decode_send_item(item)
        try:
            _save_user_media(session_dir, chunk_idx, audio_waveform, frame_list)
        except Exception as e:
            logger.warning("save user media failed: %s", e)

        def _duplex_step():
            t0 = time.perf_counter()
            prefill_result = session.duplex_prefill(
                audio_waveform=audio_waveform,
                frame_list=frame_list,
                max_slice_nums=max_slice_nums,
            )
            t_prefill = time.perf_counter()
            gen_result = session.duplex_generate(force_listen=False)
            t_gen = time.perf_counter()
            prefill_ms = (t_prefill - t0) * 1000
            return gen_result, prefill_ms, prefill_result, t_prefill, t_gen

        result, prefill_ms, prefill_cost, t_prefill_end, t_decode_end = (
            await asyncio.to_thread(_duplex_step)
        )
        wall_clock_ms = (time.perf_counter() - t_chunk_start) * 1000
        cpp_cnt = prefill_cost.cnt

        stage_extra = {}
        for k in (
            "vpm_ms",
            "apm_ms",
            "llm_prefill_ms",
            "cost_llm_ms",
            "cost_tts_ms",
            "cost_token2wav_ms",
            "stage_cnt",
        ):
            v = getattr(result, k, None)
            if v is not None:
                stage_extra[k] = v

        e2e.log_chunk(
            chunk_idx=chunk_idx,
            cnt=int(cpp_cnt),
            is_listen=bool(result.is_listen),
            end_of_turn=bool(result.end_of_turn),
            text=result.text or "",
            t_recv_perf=t_chunk_start,
            t_prefill_end_perf=t_prefill_end,
            t_decode_end_perf=t_decode_end,
            prefill_ms=prefill_ms,
            wall_clock_ms=wall_clock_ms,
            dropped_before=dropped["n"],
            turn_idle=bool(getattr(result, "turn_idle", False)),
            extra=stage_extra or None,
        )
        mode = (
            "LISTEN"
            if result.is_listen
            else ("IDLE" if getattr(result, "turn_idle", False) else "SPEAK")
        )
        _vlog(
            f"[direct] chunk={chunk_idx} cnt={cpp_cnt} {mode} "
            f"wall={wall_clock_ms:.0f}ms text={(result.text or '')[:40]!r}"
        )
        if on_chunk_done is not None:
            try:
                on_chunk_done()
            except Exception:
                pass

    async def _worker() -> None:
        while True:
            item = await chunk_queue.get()
            try:
                if item is None:
                    return
                chunk_idx, payload = item
                await _process_one(chunk_idx, payload)
            finally:
                chunk_queue.task_done()

    worker_task = asyncio.create_task(_worker())

    for i, item in enumerate(send_items):
        target = (i * send_interval_s) / speed
        sleep_s = target - (time.perf_counter() - start_send_wall)
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)
        if chunk_queue.full():
            try:
                chunk_queue.get_nowait()
                chunk_queue.task_done()
                dropped["n"] += 1
            except asyncio.QueueEmpty:
                pass
        await chunk_queue.put((i, item))

    await chunk_queue.put(None)
    await worker_task

    if pre_stop_wait_s > 0:
        await asyncio.sleep(pre_stop_wait_s)

    await asyncio.to_thread(session.duplex_stop)
    stop_poll.set()
    poll_thread.join(timeout=2.0)

    if post_stop_wait_s > 0:
        await asyncio.sleep(post_stop_wait_s)

    try:
        poll_t0 = time.perf_counter()
        _a, _, new_files, wav_info = session.collect_wav_nowait()
        if new_files:
            mtimes = {}
            for wf in new_files:
                wp = os.path.join(tts_dir, wf)
                rel = e2e.mtime_rel_ms(wp)
                if rel is not None:
                    mtimes[wf] = rel
            e2e.log_wav(
                files=list(new_files),
                file_mtimes_rel_ms=mtimes,
                t_poll_perf=poll_t0,
                wav_info=wav_info,
            )
    except Exception:
        pass

    summary = e2e.finalize()
    _vlog(f"[direct] e2e summary → {display_path(e2e.summary_path)}")

    try:
        if stage_timing.is_file():
            dest = session_dir / "stage_timing.jsonl"
            dest.write_bytes(stage_timing.read_bytes())
    except Exception as e:
        logger.warning("copy stage_timing failed: %s", e)

    os.environ["CPP_STAGE_TIMING"] = str(stage_timing)

    _analyze_e2e, _print_report, _stitch_turns = _import_analyze()
    from e2e_timing import stitch_speak_turn_wavs

    report = _analyze_e2e(session_dir)
    if quiet:
        stitch = stitch_speak_turn_wavs(str(session_dir), cpp_tts_dir=str(tts_dir))
    else:
        stitch = _stitch_turns(session_dir, cpp_tts_dir=str(tts_dir))
    report["speak_turn_wavs"] = stitch
    report["e2e_timing_summary"] = summary
    if not quiet:
        _print_report(report)
    out_path = session_dir / "eval_e2e_report.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _vlog(f"[direct] wrote {display_path(out_path)}")

    try:
        await asyncio.to_thread(session.full_reinit)
        _vlog("[direct] full_reinit done")
    except Exception as e:
        _vlog(f"[direct] full_reinit failed: {e}")
        logger.warning("full_reinit failed: %s", e)

    return session_dir
