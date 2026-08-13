#!/usr/bin/env python3
"""llama-omni-server 双工评测入口。

多视频：--video a.mp4 b.mp4 …；多条时对每次 mean 等权平均。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

JUDGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(JUDGE_ROOT))
sys.path.insert(0, str(JUDGE_ROOT / "scripts"))

from omni_client import DuplexSession  # noqa: E402
from console_ui import EvalProgress, print_judge_banner  # noqa: E402
from runner.duplex_eval_runner import run_direct_eval, _default_ref_audio  # noqa: E402

from judge_support import (  # noqa: E402
    _log as _log_always,
    _load_json,
    _port_free,
    average_latency_reports,
    display_path,
    find_free_gpu,
    print_latency_summary,
    run_plot,
    _pids_listening_on,
    _kill_pid_tree,
)

_VERBOSE = False
_JUDGE_LOG: Optional[Path] = None


def _log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    if _JUDGE_LOG is not None:
        try:
            with open(_JUDGE_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
    if _VERBOSE:
        print(line, flush=True)


def _resolve_python(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    venv = JUDGE_ROOT / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    return sys.executable


def _setup_file_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    root.addHandler(fh)
    logging.captureWarnings(True)


def build_send_items_from_video(
    video_path: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    from duplex_video_chunker import (  # type: ignore
        VideoChunkerConfig,
        build_send_items_from_video,
    )

    video_path = video_path.resolve()
    max_chunks = args.max_chunks
    max_duration = float(args.max_duration)
    if max_chunks is not None and max_chunks > 0:
        max_duration = min(max_duration, max_chunks * float(args.chunk_duration_s))
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
        f"[input] video={display_path(video_path)}  "
        f"audio_mode={cfg.audio_mode}  chunk={cfg.chunk_duration_s}s  "
        f"frame_offset={cfg.frame_offset_s}s  include_frames={cfg.include_frames}  "
        f"pad_before={cfg.pad_before_s}s pad_after={cfg.pad_after_s}s"
    )
    items = build_send_items_from_video(video_path, cfg)
    n_pad = sum(1 for it in items if it.get("is_pad"))
    _log(
        f"[input] extracted {len(items)} chunks "
        f"(main={len(items) - n_pad}, pad={n_pad})"
    )
    return items


def build_send_items_from_test_case(
    test_case_dir: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """直接读取已切分的 1 秒 WAV/JPG，不再从 MP4 重新取帧。"""
    import soundfile as sf

    test_case_dir = test_case_dir.resolve()
    wav_paths = sorted(test_case_dir.glob("*_test_case_*.wav"))
    if not wav_paths:
        raise ValueError(f"test case 缺少 WAV: {test_case_dir}")

    max_chunks = args.max_chunks
    if max_chunks is None or max_chunks <= 0:
        max_chunks = int(float(args.max_duration) / float(args.chunk_duration_s))
    if max_chunks > 0:
        wav_paths = wav_paths[:max_chunks]

    items: List[Dict[str, Any]] = []
    source_items: List[Dict[str, Any]] = []
    for index, wav_path in enumerate(wav_paths):
        jpg_path = wav_path.with_suffix(".jpg")
        if not jpg_path.is_file():
            raise ValueError(f"test case 缺少配对 JPG: {jpg_path}")
        audio, sample_rate = sf.read(
            str(wav_path), dtype="float32", always_2d=False
        )
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1, dtype=np.float32)
        if sample_rate != 16000:
            raise ValueError(
                f"test case WAV 采样率必须为 16000Hz: "
                f"{wav_path} ({sample_rate})"
            )
        msg: Dict[str, Any] = {
            "type": "audio_chunk",
            "audio_base64": base64.b64encode(
                np.ascontiguousarray(audio_np).tobytes()
            ).decode("ascii"),
        }
        if not args.no_video_frames:
            msg["frame_base64_list"] = [
                base64.b64encode(jpg_path.read_bytes()).decode("ascii")
            ]
        source_items.append({
            "ts_ms": float(index * float(args.chunk_duration_s) * 1000.0),
            "msg": msg,
            "samples": int(len(audio_np)),
            "is_pad": False,
        })

    if not source_items:
        raise ValueError(f"test case 没有可用 chunk: {test_case_dir}")
    first_msg = source_items[0]["msg"]
    last_msg = source_items[-1]["msg"]
    silence = np.zeros((16000,), dtype=np.float32)
    silence_b64 = base64.b64encode(silence.tobytes()).decode("ascii")
    pad_before = max(0, int(args.pad_before))
    pad_after = max(0, int(args.pad_after))
    for _ in range(pad_before):
        msg = {"type": "audio_chunk", "audio_base64": silence_b64}
        if first_msg.get("frame_base64_list"):
            msg["frame_base64_list"] = first_msg["frame_base64_list"]
        items.append({"ts_ms": 0.0, "msg": msg, "samples": 16000, "is_pad": True})
    items.extend(source_items)
    for _ in range(pad_after):
        msg = {"type": "audio_chunk", "audio_base64": silence_b64}
        if last_msg.get("frame_base64_list"):
            msg["frame_base64_list"] = last_msg["frame_base64_list"]
        items.append({
            "ts_ms": float(len(items) * float(args.chunk_duration_s) * 1000.0),
            "msg": msg,
            "samples": 16000,
            "is_pad": True,
        })
    _log(
        f"[input] test_case={display_path(test_case_dir)} "
        f"chunks={len(source_items)} pad_before={pad_before} "
        f"pad_after={pad_after}"
    )
    return items


def main() -> int:
    global _VERBOSE, _JUDGE_LOG

    p = argparse.ArgumentParser(description="Judge direct omni (no Demo)")
    p.add_argument("--model", type=Path, required=True, help="LLM GGUF 路径")
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--video",
        type=Path,
        nargs="+",
        help="多条目串行评测，最终输出跨次平均",
    )
    inputs.add_argument(
        "--test-case",
        type=Path,
        nargs="+",
        help="直接读取已切分的 *_test_case_NNNN.wav/.jpg 目录",
    )
    p.add_argument(
        "--llamacpp-root",
        type=Path,
        default=Path("../llama.cpp-omni"),
    )
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--cpp-port", type=int, default=19060)
    p.add_argument("--no-auto-port", action="store_true")
    p.add_argument("--batch-id", default="")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--rotation-round", type=int, default=1)
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=0)
    p.add_argument("--min-free-mib", type=int, default=12000)
    p.add_argument("--max-used-mib", type=int, default=2000)
    p.add_argument("--ready-timeout-s", type=float, default=300.0)
    p.add_argument("--ref-audio", type=Path, default=None)
    p.add_argument("--python", default=None)
    p.add_argument("--max-chunks", type=int, default=None)
    p.add_argument("--max-duration", type=float, default=120.0)
    p.add_argument("--chunk-duration-s", type=float, default=1.0)
    p.add_argument("--frame-offset", type=float, default=0.5)
    p.add_argument("--pad-before", type=int, default=0)
    p.add_argument("--pad-after", type=int, default=2)
    p.add_argument("--video-audio-mode", default="video")
    p.add_argument("--audio", default=None)
    p.add_argument("--no-video-frames", action="store_true")
    p.add_argument("--send-interval-s", type=float, default=1.0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--pre-stop-wait-s", type=float, default=2.0)
    p.add_argument("--post-stop-wait-s", type=float, default=1.0)
    p.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="length_penalty，默认 1.0",
    )
    p.add_argument("--ctx-size", type=int, default=4096)
    p.add_argument("--n-gpu-layers", type=int, default=99)
    p.add_argument("--keep-alive", action="store_true")
    p.add_argument(
        "--plot",
        action="store_true",
        help="额外画 turn-position 曲线（默认不做可视化；仅首个视频）",
    )
    p.add_argument(
        "--skip-plot",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="向控制台打印进度；默认只输出关键延迟摘要",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=JUDGE_ROOT / "tmp" / "runs",
    )
    p.add_argument(
        "--sessions-root",
        type=Path,
        default=JUDGE_ROOT / "sessions",
    )
    args = p.parse_args()
    _VERBOSE = bool(args.verbose)
    do_plot = bool(args.plot) and not bool(args.skip_plot)

    gguf = args.model.resolve()
    if not gguf.is_file():
        _log_always(f"ERROR: GGUF 不存在: {display_path(args.model)}")
        return 2

    videos: List[Path] = []
    video_labels: List[str] = []
    input_paths = args.test_case if args.test_case is not None else args.video
    assert input_paths is not None
    input_is_test_case = args.test_case is not None
    for v in input_paths:
        vp = v.resolve()
        if input_is_test_case and not vp.is_dir():
            _log_always(f"ERROR: test case 目录不存在: {display_path(v)}")
            return 2
        if not input_is_test_case and not vp.is_file():
            _log_always(f"ERROR: 视频不存在: {display_path(v)}")
            return 2
        videos.append(vp)
        video_labels.append(display_path(v))

    llamacpp = args.llamacpp_root.resolve()
    ref_audio = (
        str(args.ref_audio.resolve())
        if args.ref_audio is not None
        else _default_ref_audio()
    )

    cpp_port = args.cpp_port
    if not args.no_auto_port:
        c = cpp_port
        for _ in range(50):
            if _port_free(c):
                cpp_port = c
                break
            c += 1
        else:
            _log_always("ERROR: 找不到空闲 cpp 端口")
            return 2
        if cpp_port != args.cpp_port:
            _log(f"默认端口占用，改用 cpp={cpp_port}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.runs_dir.resolve() / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _JUDGE_LOG = log_dir / "judge.log"
    _setup_file_logging(log_dir / "python.log")
    _log(
        f"run_dir={display_path(run_dir)}  cpp.log={display_path(log_dir / 'cpp.log')}  "
        f"n_videos={len(videos)}"
    )

    import judge_support as _rj_mod

    _orig_rj_log = _rj_mod._log
    _rj_mod._log = _log

    try:
        try:
            _log("=== 扫描 GPU ===")
            gpu_id = find_free_gpu(
                max_used_mib=args.max_used_mib,
                min_free_mib=args.min_free_mib,
                prefer=args.gpu,
            )
        finally:
            _rj_mod._log = _orig_rj_log

        label = gguf.stem.replace("MiniCPM-o-4_5-", "")
        session: Optional[DuplexSession] = None
        owned = False

        def _cleanup() -> None:
            nonlocal session
            if session is not None and not args.keep_alive:
                try:
                    session.stop()
                except Exception as e:
                    _log(f"cleanup stop: {e}")
                for pid in _pids_listening_on(cpp_port):
                    _log(f"按端口清理 cpp :{cpp_port} pid={pid}")
                    _kill_pid_tree(pid)
                session = None

        signal.signal(signal.SIGINT, lambda *_: (_cleanup(), sys.exit(130)))
        signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), sys.exit(143)))

        try:
            _log(
                "=== 读取预切分 test case ==="
                if input_is_test_case
                else "=== 切分输入视频 ==="
            )
            all_send_items: List[List[Dict[str, Any]]] = []
            for video in videos:
                if input_is_test_case:
                    all_send_items.append(
                        build_send_items_from_test_case(video, args)
                    )
                else:
                    all_send_items.append(build_send_items_from_video(video, args))
            chunk_counts = [len(x) for x in all_send_items]
            n_chunks = sum(chunk_counts)

            print_judge_banner(
                model=display_path(args.model),
                n_videos=len(videos),
                n_chunks=n_chunks,
                gpu_id=gpu_id,
            )

            _log("=== 启动 llama-omni-server ===")
            session = DuplexSession(
                llamacpp_root=str(llamacpp),
                model_dir=str(gguf.parent),
                llm_model=gguf.name,
                gpu_id=gpu_id,
                cpp_server_port=cpp_port,
                ref_audio_path=ref_audio,
                ctx_size=args.ctx_size,
                n_gpu_layers=args.n_gpu_layers,
                log_path=str(log_dir / "cpp.log"),
                ready_timeout_s=args.ready_timeout_s,
            )
            owned = True
            session.start(duplex_mode=True)
            _log(f"服务就绪 cpp={session.base_url}  stdout→{display_path(log_dir / 'cpp.log')}")

            def _restart_session(reason: str) -> None:
                nonlocal session
                assert session is not None
                _log(f"[reset] {reason}: restart server and reload model")
                try:
                    session.full_reinit()
                    return
                except Exception as restart_exc:
                    _log(f"[reset] full_reinit failed, recreate server: {restart_exc}")
                session.stop()
                session = DuplexSession(
                    llamacpp_root=str(llamacpp),
                    model_dir=str(gguf.parent),
                    llm_model=gguf.name,
                    gpu_id=gpu_id,
                    cpp_server_port=cpp_port,
                    ref_audio_path=ref_audio,
                    ctx_size=args.ctx_size,
                    n_gpu_layers=args.n_gpu_layers,
                    log_path=str(log_dir / "cpp.log"),
                    ready_timeout_s=args.ready_timeout_s,
                )
                session.start(duplex_mode=True)

            progress = EvalProgress(chunk_counts)

            os.environ["CPP_STAGE_TIMING"] = str(
                Path(session.output_dir) / "stage_timing.jsonl"
            )

            per_video: List[Dict[str, Any]] = []
            reports: List[Dict[str, Any]] = []
            first_session: Optional[Path] = None

            for i, video in enumerate(videos):
                progress.start_video(i)
                send_items = all_send_items[i]
                report: Dict[str, Any] = {}
                session_dir = Path()
                used_attempt = int(args.attempt)
                for retry in range(max(0, int(args.max_retries)) + 1):
                    used_attempt = int(args.attempt) + retry
                    _log(
                        f"=== 评测（direct）[{i + 1}/{len(videos)}] "
                        f"run#{i + 1} attempt={used_attempt} {video_labels[i]} ==="
                    )
                    try:
                        session_dir = asyncio.run(
                            run_direct_eval(
                                session=session,
                                send_items=send_items,
                                sessions_root=args.sessions_root.resolve(),
                                ref_audio=ref_audio,
                                length_penalty=float(args.length_penalty),
                                send_interval_s=float(args.send_interval_s),
                                speed=float(args.speed),
                                pre_stop_wait_s=float(args.pre_stop_wait_s),
                                post_stop_wait_s=float(args.post_stop_wait_s),
                                quiet=not _VERBOSE,
                                session_suffix=f"w{args.worker_id}_r{i + 1}_a{used_attempt}",
                                on_chunk_done=progress.tick_chunk if retry == 0 else None,
                                meta_extra={
                                    "model": display_path(args.model),
                                    "video": video_labels[i],
                                    "input_kind": (
                                        "test_case"
                                        if input_is_test_case
                                        else "video"
                                    ),
                                    "gpu_id": gpu_id,
                                    "device_id": gpu_id,
                                    "worker_id": args.worker_id,
                                    "rotation_round": args.rotation_round,
                                    "batch_id": args.batch_id,
                                    "attempt": used_attempt,
                                    "server_port": cpp_port,
                                    "run_dir": display_path(run_dir),
                                    "video_index": i,
                                    "run_index": i + 1,
                                    "n_videos": len(videos),
                                },
                            )
                        )
                    except Exception as exc:
                        if retry >= int(args.max_retries):
                            raise
                        _log(
                            f"[retry] run#{i + 1} infrastructure failure on "
                            f"attempt={used_attempt}: {exc}"
                        )
                        _restart_session("infrastructure retry")
                        continue
                    report = _load_json(session_dir / "eval_e2e_report.json") or {}
                    report["batch_meta"] = {
                        "batch_id": args.batch_id,
                        "worker_id": args.worker_id,
                        "rotation_round": args.rotation_round,
                        "device_id": gpu_id,
                        "server_port": cpp_port,
                        "attempt": used_attempt,
                        "video": video_labels[i],
                        "input_kind": (
                            "test_case" if input_is_test_case else "video"
                        ),
                    }
                    (session_dir / "eval_e2e_report.json").write_text(
                        json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    validity = report.get("validity") or {}
                    if validity.get("data_valid", True) or retry >= int(args.max_retries):
                        break
                    _log(
                        f"[retry] run#{i + 1} data invalid; retry on gpu={gpu_id}: "
                        f"{validity.get('fatal_reasons') or []}"
                    )
                    _restart_session("data-validity retry")

                progress.finish_video()
                if first_session is None:
                    first_session = session_dir
                reports.append(report)
                e2e = (report.get("e2e_speak_recv_to_wav_poll_ms") or {}).get(
                    "mean_ms"
                )
                per_video.append(
                    {
                        "run_index": i + 1,
                        "video": video_labels[i],
                        "session_dir": display_path(session_dir),
                        "attempt": used_attempt,
                        "validity": report.get("validity") or {},
                        "e2e_mean_ms": e2e,
                    }
                )
                _log(
                    f"[batch] run#{i + 1} {Path(video_labels[i]).name} "
                    f"session={display_path(session_dir)} attempt={used_attempt} "
                    f"e2e_mean={e2e}"
                )
                if i + 1 < len(videos):
                    _restart_session("next video isolation")

            progress.finish_all()

            session_dir = first_session or Path()
            avg_report: Optional[Dict[str, Any]] = None
            meta: Dict[str, Any] = {
                "stamp": stamp,
                "mode": "direct_omni",
                "gpu_id": gpu_id,
                "device_id": gpu_id,
                "worker_id": args.worker_id,
                "rotation_round": args.rotation_round,
                "batch_id": args.batch_id,
                "model": display_path(args.model),
                "llamacpp_root": display_path(args.llamacpp_root),
                "cpp_port": cpp_port,
                "video": video_labels[0],
                "videos": video_labels,
                "input_kind": "test_case" if input_is_test_case else "video",
                "n_videos": len(videos),
                "label": label,
                "run_dir": display_path(run_dir),
                "session_dir": display_path(session_dir),
                "sessions": [x["session_dir"] for x in per_video],
                "ref_audio": display_path(ref_audio) if ref_audio else None,
                "per_video": per_video,
            }

            if len(videos) > 1:
                avg_report = average_latency_reports(reports)
                avg_path = run_dir / "batch_avg_report.json"
                avg_path.write_text(
                    json.dumps(
                        {**avg_report, "per_video": per_video},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                meta["batch_avg_report"] = display_path(avg_path)
                _log(f"batch avg → {display_path(avg_path)}")

            meta_path = run_dir / "run_meta.json"
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _log(f"run meta → {display_path(meta_path)}")

            if do_plot and first_session is not None:
                try:
                    _log("=== 可视化 ===")
                    py = _resolve_python(args.python)
                    _rj_mod._log = _log
                    try:
                        run_plot(
                            python=py,
                            session_dir=first_session,
                            out_dir=run_dir / "turn_position_curves",
                            label=label,
                        )
                    finally:
                        _rj_mod._log = _orig_rj_log
                except Exception as e:
                    _log(f"绘图跳过/失败: {e}")

            final_report = avg_report
            if final_report is None:
                final_report = reports[0] if reports else {}
            print_latency_summary(final_report)
            if not _VERBOSE:
                print(f"  log: {display_path(log_dir)}", flush=True)
            validities = [r.get("validity") or {} for r in reports]
            if any(v and not v.get("data_valid", False) for v in validities):
                return 3
            if any(v and not v.get("realtime_eligible", False) for v in validities):
                return 4
            return 0
        except Exception as e:
            _log_always(f"ERROR: {e}")
            raise
        finally:
            if owned and not args.keep_alive:
                _cleanup()
                _log("服务已停止")
            elif args.keep_alive:
                _log("keep-alive：llama-server 继续运行")
    finally:
        _rj_mod._log = _orig_rj_log


if __name__ == "__main__":
    raise SystemExit(main())
