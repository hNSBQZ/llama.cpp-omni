#!/usr/bin/env python3
"""将视频切成 1s 音画 chunk，供 duplex 回放。

输出格式与 replay_duplex_session._collect_send_items 一致。
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

INPUT_SR = 16000


@dataclass
class VideoChunkerConfig:
    chunk_duration_s: float = 1.0
    sample_rate: int = INPUT_SR
    video_fps: float = 1.0
    frame_offset_s: float = 0.5
    max_duration_s: float = 120.0
    max_chunks: Optional[int] = None
    audio_mode: str = "video"  # video | silent | external
    external_audio_path: Optional[str] = None
    jpeg_quality: int = 2
    include_frames: bool = True
    pad_before_s: int = 0
    pad_after_s: int = 2


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("未找到 ffmpeg，请先安装并加入 PATH")
    return exe


def _to_base64_f32(audio: np.ndarray) -> str:
    audio = np.asarray(audio, dtype=np.float32)
    return base64.b64encode(audio.tobytes()).decode("utf-8")


def _load_wav_mono(path: Path, sample_rate: int) -> np.ndarray:
    """优先 soundfile；失败则 ffmpeg+librosa。"""
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio[:, 0]
        audio = np.asarray(audio, dtype=np.float32)
        if sr != sample_rate and len(audio) > 0:
            old_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            new_len = max(1, int(round(len(audio) * sample_rate / sr)))
            new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
            audio = np.interp(new_x, old_x, audio).astype(np.float32)
        return audio
    except Exception:
        import librosa

        audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
        return np.asarray(audio, dtype=np.float32)


def _run_ffmpeg(cmd: List[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"ffmpeg 失败 (rc={r.returncode}): {err[-800:]}")


def extract_av_chunks(
    video_path: Path,
    cfg: VideoChunkerConfig,
) -> Tuple[List[np.ndarray], List[bytes]]:
    """返回长度一致的 (audio_chunks, jpeg_bytes_list)。

    - audio_mode=video: 从视频音轨抽 16k mono；无音轨则静音
    - audio_mode=silent: 全静音
    - audio_mode=external: 用 external_audio_path
    - include_frames=False: jpeg 列表为空字节占位，调用方可选择不塞进 msg
    """
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    ffmpeg = _require_ffmpeg()
    chunk_samples = int(round(cfg.chunk_duration_s * cfg.sample_rate))
    if chunk_samples <= 0:
        raise ValueError("chunk_duration_s 过小")

    max_s = float(cfg.max_duration_s)
    if cfg.max_chunks is not None and cfg.max_chunks > 0:
        max_s = min(max_s, cfg.max_chunks * cfg.chunk_duration_s)

    tmp_dir = tempfile.mkdtemp(prefix="duplex_video_chunk_")
    try:
        # 音频
        if cfg.audio_mode == "silent":
            # 先按视频时长估 chunk 数（用帧数）
            audio = None
        elif cfg.audio_mode == "external":
            if not cfg.external_audio_path:
                raise ValueError("audio_mode=external 需要 external_audio_path")
            audio = _load_wav_mono(Path(cfg.external_audio_path), cfg.sample_rate)
            max_samples = int(round(max_s * cfg.sample_rate))
            audio = audio[:max_samples]
        else:
            tmp_wav = os.path.join(tmp_dir, "audio.wav")
            try:
                _run_ffmpeg([
                    ffmpeg, "-hide_banner", "-loglevel", "error",
                    "-i", str(video_path),
                    "-ar", str(cfg.sample_rate), "-ac", "1",
                    "-t", str(max_s),
                    "-f", "wav", "-y", tmp_wav,
                ])
                audio = _load_wav_mono(Path(tmp_wav), cfg.sample_rate)
            except RuntimeError as e:
            # 无音轨时用静音
                if "does not contain any stream" in str(e) or "Output file does not contain" in str(e):
                    audio = None
                else:
                    audio = None

        # 视频帧
        jpeg_list: List[bytes] = []
        if cfg.include_frames:
            frames_dir = os.path.join(tmp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            if abs(cfg.video_fps - 1.0) < 1e-6 and abs(cfg.frame_offset_s) >= 1e-6:
                jpeg_list = _extract_frames_with_offset(
                    ffmpeg, video_path, frames_dir, max_s,
                    float(cfg.frame_offset_s), cfg.jpeg_quality,
                )
            else:
                _run_ffmpeg([
                    ffmpeg, "-hide_banner", "-loglevel", "error",
                    "-i", str(video_path),
                    "-t", str(max_s),
                    "-vf", f"fps={cfg.video_fps}",
                    "-q:v", str(cfg.jpeg_quality),
                    os.path.join(frames_dir, "frame_%04d.jpg"),
                ])
                frame_files = sorted(
                    f for f in os.listdir(frames_dir) if f.endswith(".jpg")
                )
                for fname in frame_files:
                    jpeg_list.append(Path(frames_dir, fname).read_bytes())

        if audio is not None and len(audio) > 0:
            n_from_audio = len(audio) // chunk_samples
        else:
            n_from_audio = 0

        n_from_frames = len(jpeg_list) if jpeg_list else 0
        if n_from_audio <= 0 and n_from_frames <= 0:
            # 纯估算
            n_chunks = max(1, int(max_s // cfg.chunk_duration_s))
        elif n_from_audio <= 0:
            n_chunks = n_from_frames
        elif n_from_frames <= 0 and cfg.include_frames:
            n_chunks = n_from_audio
        elif n_from_frames <= 0:
            n_chunks = n_from_audio
        else:
            n_chunks = min(n_from_audio, n_from_frames)

        if cfg.max_chunks is not None and cfg.max_chunks > 0:
            n_chunks = min(n_chunks, cfg.max_chunks)
        if n_chunks <= 0:
            raise RuntimeError(f"未能从视频切出任何 chunk: {video_path}")

        audio_chunks: List[np.ndarray] = []
        for i in range(n_chunks):
            if audio is not None and len(audio) >= (i + 1) * chunk_samples:
                seg = audio[i * chunk_samples : (i + 1) * chunk_samples].astype(np.float32)
            else:
                seg = np.zeros((chunk_samples,), dtype=np.float32)
            if len(seg) < chunk_samples:
                seg = np.pad(seg, (0, chunk_samples - len(seg)), mode="constant")
            audio_chunks.append(seg)

        if cfg.include_frames:
            # 帧不足时重复末帧或黑帧
            while len(jpeg_list) < n_chunks:
                if jpeg_list:
                    jpeg_list.append(jpeg_list[-1])
                else:
                    jpeg_list.append(_black_jpeg())
            jpeg_list = jpeg_list[:n_chunks]
        else:
            jpeg_list = [b""] * n_chunks

        return audio_chunks, jpeg_list
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _extract_frames_with_offset(
    ffmpeg: str,
    video_path: Path,
    frames_dir: str,
    max_s: float,
    offset_s: float,
    jpeg_quality: int,
) -> List[bytes]:
    """每秒在 t=i+offset 处抽一帧。"""
    out: List[bytes] = []
    n = int(max_s // 1.0)
    for i in range(n):
        t = i + offset_s
        if t >= max_s:
            break
        out_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
        try:
            _run_ffmpeg([
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-ss", f"{t:.3f}",
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", str(jpeg_quality),
                "-y", out_path,
            ])
        except RuntimeError:
            break
        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            break
        out.append(Path(out_path).read_bytes())
    return out


def _black_jpeg() -> bytes:
    """1x1 黑 JPEG 兜底，避免无帧时崩溃。"""
    try:
        from PIL import Image
        import io

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (0, 0, 0)).save(buf, format="JPEG", quality=70)
        return buf.getvalue()
    except Exception:
        # 最小合法 JPEG
        return base64.b64decode(
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
            "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
            "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAA"
            "AAAAAAAAAAj/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAA"
            "AAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAGfAP/EABQQ"
            "AQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAQUCf//EABQRAQAAAAAAAAAAAAAAAAAA"
            "AAD/2gAIAQMBAT8Bf//EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQIBAT8Bf//E"
            "ABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEABj8Cf//EABQQAQAAAAAAAAAAAAAAAA"
            "AAAAD/2gAIAQEAAT8hf//Z"
        )


def build_send_items_from_video(
    video_path: Path,
    cfg: Optional[VideoChunkerConfig] = None,
) -> List[dict]:
    """构造 replay 用的 send_items 列表。"""
    cfg = cfg or VideoChunkerConfig()
    audio_chunks, jpeg_list = extract_av_chunks(Path(video_path), cfg)
    if not audio_chunks:
        raise RuntimeError(f"视频切分结果为空: {video_path}")

    chunk_samples = int(round(cfg.chunk_duration_s * cfg.sample_rate))
    silence = np.zeros((chunk_samples,), dtype=np.float32)
    first_jpeg = jpeg_list[0] if jpeg_list else b""
    last_jpeg = jpeg_list[-1] if jpeg_list else b""

    pad_before = max(0, int(cfg.pad_before_s))
    pad_after = max(0, int(cfg.pad_after_s))

    all_audio: List[np.ndarray] = (
        [silence] * pad_before + list(audio_chunks) + [silence] * pad_after
    )
    if cfg.include_frames:
        all_jpeg: List[bytes] = (
            [first_jpeg] * pad_before + list(jpeg_list) + [last_jpeg] * pad_after
        )
    else:
        all_jpeg = [b""] * len(all_audio)

    items: List[dict] = []
    for i, audio_np in enumerate(all_audio):
        msg = {
            "type": "audio_chunk",
            "audio_base64": _to_base64_f32(audio_np),
        }
        if cfg.include_frames and all_jpeg[i]:
            msg["frame_base64_list"] = [
                base64.b64encode(all_jpeg[i]).decode("utf-8")
            ]
        items.append({
            "ts_ms": float(i * cfg.chunk_duration_s * 1000.0),
            "msg": msg,
            "samples": int(len(audio_np)),
            "is_pad": i < pad_before or i >= pad_before + len(audio_chunks),
        })
    return items
