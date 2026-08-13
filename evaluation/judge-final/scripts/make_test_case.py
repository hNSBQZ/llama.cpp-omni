#!/usr/bin/env python3
"""把视频切成预切分 test case 目录，供 RTS 自测使用。

正式评测的 RTS 输入是预切分好的 1 秒 WAV/JPG，不在评测时重新解码 MP4 —— 解码
耗时和帧提取抖动会混进速度成绩里。本脚本用与正式评测相同的切分参数产出同样布局
的目录，让自测走的是和线上完全一样的那条读取路径。

输出布局（RTS_TEST_CASE_DIR 指向 --out-dir 即可）：

    <out-dir>/<视频名>_test_case/<视频名>_test_case_0000.wav
                                <视频名>_test_case_0000.jpg
                                ...

WAV 为 16kHz 单声道 16-bit，每个 1 秒；JPG 为该秒中点抽的一帧。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from duplex_video_chunker import VideoChunkerConfig, extract_av_chunks  # noqa: E402

SAMPLE_RATE = 16000


def _to_pcm16(audio: np.ndarray) -> np.ndarray:
    """float32 → int16。

    显式截断而不是交给 libsndfile：两种取整方式会差 1 个 LSB，写死在这里才能保证
    任何机器上切出来的样本完全一致。
    """
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return np.trunc(clipped * 32767.0).astype(np.int16)


def make_one(video: Path, out_root: Path, max_duration_s: float) -> Path:
    cfg = VideoChunkerConfig(
        max_duration_s=max_duration_s,
        # test case 目录只存真实内容，前后静音 padding 由 runner 按
        # RTS_PAD_BEFORE / RTS_PAD_AFTER 在发送时补，不写进数据。
        pad_before_s=0,
        pad_after_s=0,
    )
    audio_chunks, jpegs = extract_av_chunks(video, cfg)
    if len(audio_chunks) != len(jpegs):
        raise RuntimeError(
            f"音画 chunk 数不一致: {len(audio_chunks)} vs {len(jpegs)} ({video})"
        )

    stem = video.stem
    out_dir = out_root / f"{stem}_test_case"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{stem}_test_case_*"):
        old.unlink()

    for i, (audio, jpeg) in enumerate(zip(audio_chunks, jpegs)):
        base = out_dir / f"{stem}_test_case_{i:04d}"
        sf.write(
            str(base.with_suffix(".wav")),
            _to_pcm16(audio),
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        base.with_suffix(".jpg").write_bytes(jpeg)

    print(f"[make_test_case] {video.name} -> {out_dir}  chunks={len(audio_chunks)}")
    return out_dir


def main(argv: list[str]) -> int:
    suite_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(
        description="从视频生成 RTS 自测用的预切分 test case 目录"
    )
    p.add_argument(
        "videos",
        nargs="*",
        type=Path,
        default=[suite_root / "judge-final/assets/video/omni_duplex1.mp4"],
        help="输入视频，默认用仓库自带的样例视频",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=suite_root / "judge-final/assets/test_case",
        help="输出根目录，对应 config.env 的 RTS_TEST_CASE_DIR",
    )
    p.add_argument(
        "--max-duration",
        type=float,
        default=120.0,
        help="单个视频最多切多少秒，需与 RTS_MAX_DURATION 一致",
    )
    args = p.parse_args(argv)

    for video in args.videos:
        video = video.resolve()
        if not video.is_file():
            print(f"ERROR: 视频不存在: {video}", file=sys.stderr)
            return 2
        make_one(video, args.out_dir.resolve(), args.max_duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
