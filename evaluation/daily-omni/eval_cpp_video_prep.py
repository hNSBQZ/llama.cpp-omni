"""
视频帧采样与临时 JPG 保存。

帧采样逻辑对齐 evalkit Daily-Omni 的 _sample_video_frame_indices：
  - 长视频 (duration > max_frames): 0.1s 粒度采样，超限则 uniform_sample
  - 短视频 (duration <= max_frames): 每秒取 1 帧
  - 解码器优先级：decord → ffmpeg → torchvision
  - 同时返回时间戳列表（供音频切分使用）
"""
import os
import shutil
import warnings
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image

from eval_cpp_config import MAX_NUM_FRAMES, MAX_FPS, FRAME_TEMP_DIR

logger = logging.getLogger(__name__)

try:
    from decord import VideoReader, cpu as decord_cpu
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False

try:
    import torch
    from torchvision import io as tv_io
    _HAS_TORCHVISION = True
except Exception:
    torch = None
    tv_io = None
    _HAS_TORCHVISION = False


# ==================== 帧采样核心逻辑 ====================

def _uniform_sample(seq, n):
    """均匀采样：从 seq 中等间隔取 n 个元素（取每段中点）。"""
    gap = len(seq) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [seq[i] for i in idxs]


def _sample_daily_omni_frame_indices(
    total_frames: int,
    avg_fps: float,
    max_frames: int = MAX_NUM_FRAMES,
) -> Tuple[List[int], List[float]]:
    """
    Daily-Omni 帧采样算法（对齐 evalkit _sample_video_frame_indices）。

    返回 (frame_indices, timestamps)，timestamps 用于后续音频切分。
    """
    if total_frames <= 0 or avg_fps <= 0:
        return [], []

    duration = total_frames / avg_fps

    if duration > max_frames:
        # 长视频：0.1s 粒度
        step = 0.1
        num_steps = int(duration / step)
        timestamps = [round(i * step, 1) for i in range(num_steps)]
        frame_idx = [min(int(ts * avg_fps), total_frames - 1) for ts in timestamps]

        if len(frame_idx) > max_frames:
            frame_idx = _uniform_sample(frame_idx, max_frames)
            timestamps = _uniform_sample(timestamps, max_frames)
    else:
        # 短视频：每秒取 1 帧
        int_duration = int(duration)
        if int_duration <= 0:
            int_duration = 1
        frame_idx = [int(i * avg_fps) for i in range(int_duration)]
        timestamps = [float(i) for i in range(int_duration)]

    frame_idx = [min(idx, total_frames - 1) for idx in frame_idx]

    return frame_idx, timestamps


# ==================== Backend: decord ====================

def _load_frames_decord(
    video_path: str, max_frames: int,
) -> Tuple[List[Image.Image], List[int], List[float], float, int]:
    """用 decord 加载帧。返回 (frames, frame_idx, timestamps, avg_fps, total_frames)。"""
    if not _HAS_DECORD:
        return [], [], [], 0.0, 0
    try:
        vr = VideoReader(video_path, ctx=decord_cpu(0))
    except Exception as e:
        warnings.warn(f"[decord] cannot open: {video_path}, {e}")
        return [], [], [], 0.0, 0

    try:
        avg_fps = float(vr.get_avg_fps())
        total_frames = len(vr)
        frame_idx, timestamps = _sample_daily_omni_frame_indices(total_frames, avg_fps, max_frames)
        if not frame_idx:
            return [], [], [], avg_fps, total_frames
        frames_np = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")).convert("RGB") for v in frames_np]
        return frames, frame_idx, timestamps, avg_fps, total_frames
    except Exception as e:
        warnings.warn(f"[decord] decode failed: {video_path}, {e}")
        return [], [], [], 0.0, 0


# ==================== Backend: ffmpeg CLI ====================

def _get_video_info_ffprobe(video_path: str) -> Tuple[float, int]:
    """用 ffprobe 获取 avg_fps 和 total_frames。"""
    try:
        cmd_fps = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,nb_frames",
            "-of", "csv=p=0",
            video_path,
        ]
        out = subprocess.check_output(cmd_fps, stderr=subprocess.DEVNULL).decode().strip()
        parts = out.split(",")
        fps_str = parts[0] if parts else "0/1"
        nb_str = parts[1] if len(parts) > 1 else "0"

        if "/" in fps_str:
            num, den = fps_str.split("/")
            avg_fps = float(num) / float(den) if float(den) > 0 else 25.0
        else:
            avg_fps = float(fps_str) if fps_str else 25.0

        total_frames = int(nb_str) if nb_str and nb_str != "N/A" else 0

        if total_frames <= 0:
            cmd_count = [
                "ffprobe", "-v", "error",
                "-count_frames",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0",
                video_path,
            ]
            out2 = subprocess.check_output(cmd_count, stderr=subprocess.DEVNULL).decode().strip()
            total_frames = int(out2) if out2 and out2 != "N/A" else 0

        return avg_fps, total_frames
    except Exception:
        return 25.0, 0


def _load_frames_ffmpeg(
    video_path: str, max_frames: int,
) -> Tuple[List[Image.Image], List[int], List[float], float, int]:
    """用 ffmpeg + ffprobe 抽帧（fallback #1）。"""
    if not os.path.exists(video_path):
        return [], [], [], 0.0, 0

    avg_fps, total_frames = _get_video_info_ffprobe(video_path)
    if total_frames <= 0:
        cmd_dur = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", video_path,
        ]
        try:
            dur_str = subprocess.check_output(cmd_dur, stderr=subprocess.DEVNULL).decode().strip()
            duration = float(dur_str)
            total_frames = int(duration * avg_fps)
        except Exception:
            total_frames = int(60 * avg_fps)

    frame_idx, timestamps = _sample_daily_omni_frame_indices(total_frames, avg_fps, max_frames)
    if not frame_idx:
        return [], [], [], avg_fps, total_frames

    tmp_dir = tempfile.mkdtemp(prefix="daily_omni_ffmpeg_")
    try:
        target_fps = min(MAX_FPS, avg_fps) if MAX_FPS > 0 else 1.0
        pattern = os.path.join(tmp_dir, "frame_%05d.jpg")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"fps={target_fps}",
            "-q:v", "2",
            "-vframes", str(max_frames),
            pattern,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            warnings.warn(f"[ffmpeg] failed: {video_path}, {e}")
            return [], [], [], avg_fps, total_frames

        frame_files = sorted(Path(tmp_dir).glob("frame_*.jpg"))
        if not frame_files:
            return [], [], [], avg_fps, total_frames

        frames = []
        actual_idx = []
        actual_ts = []
        for i, fp in enumerate(frame_files):
            try:
                with Image.open(fp) as img:
                    frames.append(img.convert("RGB").copy())
                if i < len(frame_idx):
                    actual_idx.append(frame_idx[i])
                    actual_ts.append(timestamps[i])
                else:
                    actual_idx.append(i)
                    actual_ts.append(float(i))
            except Exception as e:
                warnings.warn(f"[ffmpeg] read frame failed: {fp}, {e}")

        return frames, actual_idx, actual_ts, avg_fps, total_frames
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ==================== Backend: torchvision ====================

def _load_frames_torchvision(
    video_path: str, max_frames: int,
) -> Tuple[List[Image.Image], List[int], List[float], float, int]:
    """用 torchvision.io.read_video 加载帧（fallback #2）。"""
    if not _HAS_TORCHVISION:
        return [], [], [], 0.0, 0

    try:
        with torch.no_grad():
            video, _audio, info = tv_io.read_video(
                video_path, start_pts=0.0, end_pts=None,
                pts_unit="sec", output_format="TCHW",
            )
    except Exception as e:
        warnings.warn(f"[torchvision] read_video failed: {video_path}, {e}")
        return [], [], [], 0.0, 0

    total_frames = int(video.size(0))
    if total_frames == 0:
        return [], [], [], 0.0, 0

    video_fps = float(info.get("video_fps", 0.0) or 0.0)
    if video_fps <= 0:
        video_fps = 25.0

    frame_idx, timestamps = _sample_daily_omni_frame_indices(total_frames, video_fps, max_frames)
    if not frame_idx:
        return [], [], [], video_fps, total_frames

    with torch.no_grad():
        idx_tensor = torch.tensor(frame_idx, dtype=torch.long)
        sampled = video[idx_tensor]
        frames_np = sampled.permute(0, 2, 3, 1).contiguous().to(torch.uint8).cpu().numpy()

    frames = [Image.fromarray(v).convert("RGB") for v in frames_np]
    return frames, frame_idx, timestamps, video_fps, total_frames


# ==================== 统一入口 ====================

def encode_video(
    video_path: str,
    max_frames: int = MAX_NUM_FRAMES,
) -> Tuple[List[Image.Image], List[float]]:
    """
    从视频中采样帧，返回 (PIL.Image 列表, 时间戳列表)。

    解码器优先级：decord → ffmpeg → torchvision
    """
    frames, frame_idx, timestamps, avg_fps, total_frames = _load_frames_decord(video_path, max_frames)
    backend = "decord"

    if not frames:
        frames, frame_idx, timestamps, avg_fps, total_frames = _load_frames_ffmpeg(video_path, max_frames)
        backend = "ffmpeg" if frames else backend

    if not frames:
        frames, frame_idx, timestamps, avg_fps, total_frames = _load_frames_torchvision(video_path, max_frames)
        backend = "torchvision" if frames else "none"

    if not frames:
        warnings.warn(f"[encode_video] no frames decoded from: {video_path}")
        return [], []

    logger.info(
        f"Video: {video_path}, backend={backend}, "
        f"frames={len(frames)}, total={total_frames}, fps={avg_fps:.3f}, "
        f"timestamps=[{timestamps[0]:.1f}..{timestamps[-1]:.1f}]"
    )
    return frames, timestamps


# ==================== 帧文件管理 ====================

def save_frames_as_jpg(
    frames: List[Image.Image],
    sample_id: str,
    tmp_dir: str = FRAME_TEMP_DIR,
    quality: int = 95,
) -> List[str]:
    """
    将 PIL.Image 帧列表保存为临时 JPG 文件。

    目录结构：{tmp_dir}/{sample_id}/frame_000.jpg, frame_001.jpg, ...
    返回文件绝对路径列表（有序）。
    """
    sample_dir = os.path.join(tmp_dir, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    paths: List[str] = []
    for idx, frame in enumerate(frames):
        path = os.path.join(sample_dir, f"frame_{idx:03d}.jpg")
        frame.save(path, format="JPEG", quality=quality)
        paths.append(os.path.abspath(path))
    return paths


def cleanup_sample_media(sample_id: str, frame_dir: str = FRAME_TEMP_DIR, audio_dir: str = None):
    """清理单个样本的临时帧和音频文件。"""
    from eval_cpp_config import AUDIO_TEMP_DIR
    if audio_dir is None:
        audio_dir = AUDIO_TEMP_DIR

    for d in [frame_dir, audio_dir]:
        sample_dir = os.path.join(d, sample_id)
        if os.path.isdir(sample_dir):
            shutil.rmtree(sample_dir, ignore_errors=True)


def cleanup_all_media(temp_dir: str = None):
    """清理所有临时文件。"""
    from eval_cpp_config import TEMP_DIR
    if temp_dir is None:
        temp_dir = TEMP_DIR
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temp media dir: {temp_dir}")


def prepare_video_frames(
    video_path: str,
    sample_id: str,
    max_frames: int = MAX_NUM_FRAMES,
) -> Tuple[List[str], List[float]]:
    """
    一步完成：采样视频帧 → 保存为临时 JPG → 返回 (路径列表, 时间戳列表)。
    """
    frames, timestamps = encode_video(video_path, max_frames)
    if not frames:
        logger.warning(f"No frames extracted from video: {video_path}")
        return [], []
    paths = save_frames_as_jpg(frames, sample_id)
    logger.info(f"Prepared {len(paths)} frames for sample {sample_id}")
    return paths, timestamps
