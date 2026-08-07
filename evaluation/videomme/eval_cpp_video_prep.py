"""
视频帧采样与临时 JPG 保存。

帧采样逻辑对齐 Python videomme.py 的 default 策略：
  - 解码器优先级：decord → ffmpeg → torchvision
  - 每 round(avg_fps) 帧取 1 帧（≈1fps）
  - 超过 MAX_NUM_FRAMES 时 uniform_sample 到 MAX_NUM_FRAMES 帧
"""
import os
import shutil
import warnings
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple

from PIL import Image

from eval_cpp_config import MAX_NUM_FRAMES, MAX_FPS, FRAME_TEMP_DIR

logger = logging.getLogger(__name__)

# 尝试导入 decord
try:
    from decord import VideoReader, cpu as decord_cpu
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False

# 尝试导入 torchvision（可选 fallback）
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


def _sample_frame_indices(
    num_total_frames: int,
    avg_fps: float,
    max_num_frames: int,
) -> Tuple[List[int], List[float]]:
    """
    根据视频元信息计算采样帧下标。

    对齐 Python videomme.py 的 default 策略：
      sample_fps = round(avg_fps)
      frame_idx = range(0, total_frames, sample_fps)  # ≈1fps
      超过 max_num_frames 时 uniform_sample
    """
    sample_fps = round(avg_fps)
    if sample_fps < 1:
        sample_fps = 1
    frame_idx = list(range(0, num_total_frames, sample_fps))
    if len(frame_idx) > max_num_frames:
        frame_idx = _uniform_sample(frame_idx, max_num_frames)
    timestamps = []
    return frame_idx, timestamps


# ==================== Backend: decord ====================

def _load_frames_decord(
    video_path: str, max_num_frames: int, max_fps: float,
) -> Tuple[List[Image.Image], List[int], float, int]:
    """用 decord 加载帧。返回 (frames, frame_idx, avg_fps, total_frames)。"""
    if not _HAS_DECORD:
        return [], [], 0.0, 0
    try:
        vr = VideoReader(video_path, ctx=decord_cpu(0))
    except Exception as e:
        warnings.warn(f"[decord] cannot open: {video_path}, {e}")
        return [], [], 0.0, 0

    try:
        avg_fps = float(vr.get_avg_fps())
        total_frames = len(vr)
        frame_idx, _ = _sample_frame_indices(total_frames, avg_fps, max_num_frames)
        if not frame_idx:
            return [], [], avg_fps, total_frames
        frames_np = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")).convert("RGB") for v in frames_np]
        return frames, frame_idx, avg_fps, total_frames
    except Exception as e:
        warnings.warn(f"[decord] decode failed: {video_path}, {e}")
        return [], [], 0.0, 0


# ==================== Backend: ffmpeg CLI ====================

def _probe_video_ffprobe(video_path: str) -> Tuple[float, int]:
    """用 ffprobe 取 (avg_fps, total_frames)。nb_frames 缺失时用 duration×fps 估。"""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate,nb_frames",
             "-of", "csv=p=0", video_path],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 0.0, 0

    parts = out.split(",")
    fps_str = parts[0] if parts else ""
    nb_str = parts[1] if len(parts) > 1 else ""

    avg_fps = 0.0
    if "/" in fps_str:
        num, den = fps_str.split("/")
        try:
            avg_fps = float(num) / float(den) if float(den) else 0.0
        except ValueError:
            avg_fps = 0.0
    elif fps_str:
        try:
            avg_fps = float(fps_str)
        except ValueError:
            avg_fps = 0.0
    if avg_fps <= 0:
        avg_fps = 25.0

    total_frames = 0
    if nb_str and nb_str != "N/A":
        try:
            total_frames = int(nb_str)
        except ValueError:
            total_frames = 0
    if total_frames <= 0:
        try:
            dur = float(subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", video_path],
                stderr=subprocess.DEVNULL,
            ).decode().strip())
            total_frames = int(dur * avg_fps)
        except Exception:
            total_frames = 0

    return avg_fps, total_frames


def _load_frames_ffmpeg(
    video_path: str, max_num_frames: int, max_fps: float,
) -> Tuple[List[Image.Image], List[int], float, int]:
    """用 ffmpeg 命令行抽帧（fallback #1，装不上 decord 时走这条，例如 aarch64）。

    先算出 decord 路径会取的那批帧下标，再把下标换成时间戳，逐个用 input seek
    (`-ss` 在 `-i` 之前) 抓单帧。不能简单地 `-vf fps=1 -vframes 64`：那样只覆盖
    视频前 64 秒，而 Video-MME 的 medium/long 子集是十几分钟到一小时的长视频。
    """
    if max_num_frames <= 0:
        max_num_frames = 1
    if not os.path.exists(video_path):
        return [], [], 0.0, 0

    avg_fps, total_frames = _probe_video_ffprobe(video_path)
    if total_frames <= 0:
        return [], [], avg_fps, 0

    frame_idx, _ = _sample_frame_indices(total_frames, avg_fps, max_num_frames)
    if not frame_idx:
        return [], [], avg_fps, total_frames

    tmp_dir = tempfile.mkdtemp(prefix="cpp_eval_ffmpeg_")
    try:
        frames, kept_idx = [], []
        for i, fidx in enumerate(frame_idx):
            out_path = os.path.join(tmp_dir, f"frame_{i:05d}.jpg")
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", f"{fidx / avg_fps:.3f}",
                "-i", video_path,
                "-frames:v", "1", "-q:v", "2", "-y", out_path,
            ]
            try:
                subprocess.run(cmd, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                warnings.warn("[ffmpeg] ffmpeg not found on PATH")
                return [], [], avg_fps, total_frames
            except subprocess.CalledProcessError:
                continue  # 末尾几帧 seek 越界是常见情况，跳过即可
            if not os.path.exists(out_path):
                continue
            try:
                with Image.open(out_path) as img:
                    frames.append(img.convert("RGB").copy())
                kept_idx.append(fidx)
            except Exception as e:
                warnings.warn(f"[ffmpeg] read frame failed: {out_path}, {e}")

        return frames, kept_idx, avg_fps, total_frames
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ==================== Backend: torchvision ====================

def _load_frames_torchvision(
    video_path: str, max_num_frames: int, max_fps: float,
) -> Tuple[List[Image.Image], List[int], float, int]:
    """用 torchvision.io.read_video 加载帧（fallback #2）。"""
    if not _HAS_TORCHVISION:
        return [], [], 0.0, 0

    try:
        with torch.no_grad():
            video, _audio, info = tv_io.read_video(
                video_path, start_pts=0.0, end_pts=None,
                pts_unit="sec", output_format="TCHW",
            )
    except Exception as e:
        warnings.warn(f"[torchvision] read_video failed: {video_path}, {e}")
        return [], [], 0.0, 0

    total_frames = int(video.size(0))
    if total_frames == 0:
        return [], [], 0.0, 0

    video_fps = float(info.get("video_fps", 0.0) or 0.0)
    if video_fps <= 0:
        video_fps = 25.0

    duration = total_frames / video_fps
    if max_fps > 0:
        n_by_fps = int(duration * max_fps + 0.5)
        nframes = max(1, min(max_num_frames, total_frames, n_by_fps or max_num_frames))
    else:
        nframes = max(1, min(max_num_frames, total_frames))

    with torch.no_grad():
        idx = torch.linspace(0, total_frames - 1, nframes).round().long()
        sampled = video[idx]
        frames_np = sampled.permute(0, 2, 3, 1).contiguous().to(torch.uint8).cpu().numpy()

    frames = [Image.fromarray(v).convert("RGB") for v in frames_np]
    return frames, idx.tolist(), video_fps, total_frames


# ==================== 统一入口 ====================

def encode_video(
    video_path: str,
    max_num_frames: int = MAX_NUM_FRAMES,
    max_fps: float = MAX_FPS,
) -> List[Image.Image]:
    """
    从视频中采样帧，返回 PIL.Image 列表。

    解码器优先级：decord → ffmpeg → torchvision
    """
    # 1) decord
    frames, frame_idx, avg_fps, total_frames = _load_frames_decord(video_path, max_num_frames, max_fps)
    backend = "decord"

    # 2) ffmpeg fallback
    if not frames:
        frames, frame_idx, avg_fps, total_frames = _load_frames_ffmpeg(video_path, max_num_frames, max_fps)
        backend = "ffmpeg" if frames else backend

    # 3) torchvision fallback
    if not frames:
        frames, frame_idx, avg_fps, total_frames = _load_frames_torchvision(video_path, max_num_frames, max_fps)
        backend = "torchvision" if frames else "none"

    if not frames:
        warnings.warn(f"[encode_video] no frames decoded from: {video_path}")
        return []

    sample_fps = 0.0
    if total_frames > 0 and avg_fps > 0 and frame_idx:
        sample_fps = len(frame_idx) / max(total_frames, 1e-6) * avg_fps

    logger.info(
        f"Video: {video_path}, backend={backend}, sample_fps={sample_fps:.3f}, "
        f"frames={len(frames)}, total={total_frames}, raw_fps={avg_fps:.3f}"
    )
    return frames


# ==================== 帧文件管理 ====================

def save_frames_as_jpg(
    frames: List[Image.Image],
    video_id: str,
    tmp_dir: str = FRAME_TEMP_DIR,
    quality: int = 95,
) -> List[str]:
    """
    将 PIL.Image 帧列表保存为临时 JPG 文件。

    目录结构：{tmp_dir}/{video_id}/frame_000.jpg, frame_001.jpg, ...
    返回文件绝对路径列表（有序）。
    """
    video_dir = os.path.join(tmp_dir, video_id)
    os.makedirs(video_dir, exist_ok=True)

    paths: List[str] = []
    for idx, frame in enumerate(frames):
        path = os.path.join(video_dir, f"frame_{idx:03d}.jpg")
        frame.save(path, format="JPEG", quality=quality)
        paths.append(os.path.abspath(path))
    return paths


def cleanup_frames(video_id: str, tmp_dir: str = FRAME_TEMP_DIR):
    """清理单个视频的临时帧文件。"""
    video_dir = os.path.join(tmp_dir, video_id)
    if os.path.isdir(video_dir):
        shutil.rmtree(video_dir, ignore_errors=True)


def cleanup_all_frames(tmp_dir: str = FRAME_TEMP_DIR):
    """清理所有临时帧文件。"""
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temp frames dir: {tmp_dir}")


def prepare_video_frames(
    video_path: str,
    video_id: str,
    max_num_frames: int = MAX_NUM_FRAMES,
    max_fps: float = MAX_FPS,
) -> List[str]:
    """
    一步完成：采样视频帧 → 保存为临时 JPG → 返回路径列表。
    """
    frames = encode_video(video_path, max_num_frames, max_fps)
    if not frames:
        logger.warning(f"No frames extracted from video: {video_path}")
        return []
    paths = save_frames_as_jpg(frames, video_id)
    logger.info(f"Prepared {len(paths)} frames for video {video_id}")
    return paths
