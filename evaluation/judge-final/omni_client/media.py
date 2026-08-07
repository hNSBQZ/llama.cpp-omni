"""评测用音视频临时文件读写。"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

_AUDIO_INPUT_SR = 16000
MIN_SAMPLES = 1600


def save_audio_to_temp(audio_np: np.ndarray, temp_dir: str, prefix: str) -> str:
    import soundfile as sf

    if len(audio_np) < MIN_SAMPLES:
        audio_np = np.pad(audio_np, (0, MIN_SAMPLES - len(audio_np)), mode="constant")

    path = os.path.join(temp_dir, f"{prefix}.wav")
    audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
    sf.write(path, audio_np, _AUDIO_INPUT_SR, format="WAV", subtype="PCM_16")
    return path


def save_pil_image_to_temp(pil_image, temp_dir: str, prefix: str) -> str:
    path = os.path.join(temp_dir, f"{prefix}.png")
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image.save(path, format="PNG")
    return path


def cleanup_temp_files(*paths: Optional[str]) -> None:
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
