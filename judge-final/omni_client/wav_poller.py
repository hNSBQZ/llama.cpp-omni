"""TTS 输出目录与 WAV 增量收集。"""

from __future__ import annotations

import base64
import logging
import os
import re
import shutil
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("omni_client.wav")


def reset_output_dir(output_dir: str) -> None:
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception:
                pass
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "round_000", "tts_wav"), exist_ok=True)


def find_latest_round_dir(output_dir: str) -> Optional[str]:
    if not os.path.exists(output_dir):
        return None
    rounds = sorted(
        [
            d
            for d in os.listdir(output_dir)
            if d.startswith("round_") and os.path.isdir(os.path.join(output_dir, d))
        ],
        reverse=True,
    )
    if rounds:
        return os.path.join(output_dir, rounds[0])
    return None


def collect_wav_output_nowait(
    output_dir: str,
    sent_wav_files: Set[str],
    sse_text: str = "",
) -> Tuple[Optional[str], str, List[str], Dict[str, Dict[str, float]]]:
    """增量收集新落盘的 wav。

    返回 (audio_b64, sse_text, accepted_files, wav_info)。

    wav_info[filename] = {"n_samples", "sample_rate", "duration_ms"} —— 反正这里
    已经把每个 wav 读进来了，顺手把时长带出去，下游算每包 RTF 就不必再读一遍文件。
    """
    import soundfile as sf

    direct_tts = os.path.join(output_dir, "tts_wav")
    if os.path.isdir(direct_tts) and any(
        f.startswith("wav_") and f.endswith(".wav") for f in os.listdir(direct_tts)
    ):
        tts_wav_dir = direct_tts
    else:
        round_dir = find_latest_round_dir(output_dir)
        if not round_dir:
            if os.path.isdir(direct_tts):
                tts_wav_dir = direct_tts
            else:
                return None, sse_text, []
        else:
            tts_wav_dir = os.path.join(round_dir, "tts_wav")
    if not os.path.exists(tts_wav_dir):
        return None, sse_text, [], {}

    wav_files = sorted(
        [
            f
            for f in os.listdir(tts_wav_dir)
            if f.startswith("wav_") and f.endswith(".wav")
        ],
        key=lambda f: int(re.search(r"wav_(\d+)", f).group(1))
        if re.search(r"wav_(\d+)", f)
        else 0,
    )
    new_files = [f for f in wav_files if f not in sent_wav_files]
    if not new_files:
        return None, sse_text, [], {}

    all_audio = []
    accepted_files: List[str] = []
    wav_info: Dict[str, Dict[str, float]] = {}
    for wf in new_files:
        wp = os.path.join(tts_wav_dir, wf)
        try:
            data, sr = sf.read(wp)
            if len(data) > 0:
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                all_audio.append(data)
                accepted_files.append(wf)
                sent_wav_files.add(wf)
                wav_info[wf] = {
                    "n_samples": int(len(data)),
                    "sample_rate": int(sr),
                    "duration_ms": round(len(data) / float(sr) * 1000.0, 3)
                    if sr
                    else 0.0,
                }
        except Exception:
            pass

    if not all_audio:
        return None, sse_text, [], {}

    combined = np.concatenate(all_audio)
    audio_b64 = base64.b64encode(combined.astype(np.float32).tobytes()).decode("utf-8")
    return audio_b64, sse_text, accepted_files, wav_info


def tts_wav_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "tts_wav")
