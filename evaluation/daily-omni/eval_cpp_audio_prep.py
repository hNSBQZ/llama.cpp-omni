"""
音频加载、重采样、分段与临时 WAV 保存。

对齐 evalkit Daily-Omni 的音频处理逻辑：
  - 使用 librosa 加载并重采样到 16kHz mono float32
  - 按视频帧时间戳将完整音频切分为与帧一一对应的段
  - 保存为临时 WAV 文件供 C++ server prefill
"""
import os
import logging
import warnings
from typing import List, Tuple, Optional

import numpy as np

from eval_cpp_config import AUDIO_SR, AUDIO_TEMP_DIR

logger = logging.getLogger(__name__)

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    librosa = None
    _HAS_LIBROSA = False

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    sf = None
    _HAS_SOUNDFILE = False


def load_audio(
    audio_path: str,
    sr: int = AUDIO_SR,
    speed: float = 1.0,
    trim_end: float = 0.0,
) -> Optional[np.ndarray]:
    """
    加载音频并重采样。

    返回 float32 numpy 数组（16kHz mono），值域 [-1.0, 1.0]。
    加载失败返回 None。
    """
    if not _HAS_LIBROSA:
        raise ImportError("librosa is required for audio processing: pip install librosa")

    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None

    try:
        waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        logger.error(f"Failed to load audio {audio_path}: {e}")
        return None

    if speed > 1.0:
        waveform = _speedup_audio(waveform, sr, speed)

    if trim_end > 0:
        waveform = _trim_audio(waveform, sr, trim_end)

    return waveform


def _speedup_audio(waveform: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """音频变速（对齐 evalkit speedup_audio）。"""
    try:
        import pyrubberband as pyrb
        return pyrb.time_stretch(waveform, sr, speed)
    except ImportError:
        new_sr = int(sr * speed)
        return librosa.resample(waveform, orig_sr=new_sr, target_sr=sr)


def _trim_audio(waveform: np.ndarray, sr: int, trim_end: float) -> np.ndarray:
    """截断音频末尾（对齐 evalkit trim_audio_segment）。"""
    samples_to_trim = int(trim_end * sr)
    min_samples = int(0.1 * sr)
    if len(waveform) - samples_to_trim >= min_samples:
        return waveform[:-samples_to_trim]
    elif len(waveform) > min_samples:
        return waveform[:min_samples]
    return waveform


def segment_audio_by_timestamps(
    waveform: np.ndarray,
    timestamps: List[float],
    sr: int = AUDIO_SR,
) -> List[np.ndarray]:
    """
    按视频帧时间戳切分音频（对齐 evalkit get_video_frame_audio_segments）。

    每段音频对应一个帧，切分规则：
      segment[i] = waveform[timestamps[i]*sr : timestamps[i+1]*sr]
      segment[last] = waveform[timestamps[-1]*sr : end]
    """
    if waveform is None or len(waveform) == 0 or not timestamps:
        return []

    audio_duration = len(waveform) / sr
    segments = []

    for i, start_time in enumerate(timestamps):
        if i < len(timestamps) - 1:
            end_time = timestamps[i + 1]
        else:
            end_time = audio_duration

        start_sample = max(0, int(start_time * sr))
        end_sample = max(start_sample, int(end_time * sr))
        end_sample = min(end_sample, len(waveform))

        segment = waveform[start_sample:end_sample]
        segments.append(segment)

    num_pairs = min(len(timestamps), len(segments))
    segments = segments[:num_pairs]

    return segments


def save_audio_segments_as_wav(
    segments: List[np.ndarray],
    sample_id: str,
    sr: int = AUDIO_SR,
    tmp_dir: str = AUDIO_TEMP_DIR,
) -> List[str]:
    """
    将音频段列表保存为临时 WAV 文件。

    目录结构：{tmp_dir}/{sample_id}/audio_seg_000.wav, audio_seg_001.wav, ...
    返回文件绝对路径列表（有序）。
    """
    if not _HAS_SOUNDFILE:
        raise ImportError("soundfile is required for WAV writing: pip install soundfile")

    sample_dir = os.path.join(tmp_dir, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    paths: List[str] = []
    for idx, seg in enumerate(segments):
        path = os.path.join(sample_dir, f"audio_seg_{idx:03d}.wav")
        sf.write(path, seg, sr)
        paths.append(os.path.abspath(path))
    return paths


def prepare_audio_segments(
    audio_path: str,
    timestamps: List[float],
    sample_id: str,
    sr: int = AUDIO_SR,
    speed: float = 1.0,
    trim_end: float = 0.0,
) -> List[str]:
    """
    一步完成：加载音频 → 切分 → 保存为临时 WAV → 返回路径列表。

    如果音频不存在或加载失败，返回空列表。
    """
    waveform = load_audio(audio_path, sr=sr, speed=speed, trim_end=trim_end)
    if waveform is None:
        logger.warning(f"No audio loaded from: {audio_path}")
        return []

    segments = segment_audio_by_timestamps(waveform, timestamps, sr=sr)
    if not segments:
        logger.warning(f"No audio segments created for: {audio_path}")
        return []

    paths = save_audio_segments_as_wav(segments, sample_id, sr=sr)
    logger.info(f"Prepared {len(paths)} audio segments for sample {sample_id}")
    return paths
