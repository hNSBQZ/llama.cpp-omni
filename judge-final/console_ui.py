"""控制台横幅与评测进度条。"""

from __future__ import annotations

import sys
import time
from typing import List, Optional, TextIO


_LLAMA_CPP_LOGO = """\
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀"""

_JUDGE_ASCII_LOGO = """\
▄▄ ▄▄
██ ██                                                                      ▄▄
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄       ▄███▄ ███▄███▄ ███▄  ▄▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██  ███  ██ ██ ██ ██ ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀       ▀███▀ ██ ██ ██ ██ ██ ██
                                    ██    ██
                                    ▀▀    ▀▀
████▄              ▄▄▄
██ ██ ▄█▀▀▄ ███▄ ▄██▄ ▄███▄ ███▄ ███▄███▄  ▀▀█▄ ███▄  ▄████ ▄█▀▀▄    ▄█▀▀▄ ████▄ ▄█▀▀▄
████▀ ██▀▀▀ ██ ▀  ██  ██ ██ ██ ▀ ██ ██ ██ ▄█▀██ ██ ██ ██    ██▀▀▀    ██▀▀▀  ▄██▀ ██▀▀▀
██    ▀█▄▄▀ ██    ██  ▀███▀ ██   ██ ██ ██ ▀█▄██ ██ ██ ▀████ ▀█▄▄▀    ▀█▄▄▀ █████ ▀█▄▄▀


"""

_base_lines = _LLAMA_CPP_LOGO.splitlines()
_logo_lines = _JUDGE_ASCII_LOGO.splitlines()
for _i, _bl in enumerate(_base_lines):
    if not _logo_lines[_i].startswith(_bl):
        raise RuntimeError(
            f"judge logo corrupted llama.cpp row {_i}: "
            f"expected prefix {_bl!r}, got {_logo_lines[_i][: len(_bl) + 8]!r}"
        )

_non_empty_logo_lines = [line for line in _logo_lines if line.strip()]
JUDGE_BAR_WIDTH = len(_non_empty_logo_lines[-1]) if _non_empty_logo_lines else 0
JUDGE_BAR = "=" * JUDGE_BAR_WIDTH


def _logo_display_text() -> str:
    """去掉 logo 末尾一行空行，缩小与下方分隔线的间距。"""
    text = _JUDGE_ASCII_LOGO
    while text.endswith("\n\n\n"):
        text = text[:-1]
    if not text.endswith("\n"):
        text += "\n"
    return text


def print_judge_banner(
    *,
    model: str = "",
    n_videos: int = 1,
    n_chunks: int = 0,
    gpu_id: Optional[int] = None,
    file: TextIO = sys.stdout,
) -> None:
    """启动时打印正式横幅（始终到控制台）。"""
    print(JUDGE_BAR, file=file, flush=True)
    print(_logo_display_text(), file=file, end="", flush=True)
    print(JUDGE_BAR, file=file, flush=True)
    if model:
        print(f"  model : {model}", file=file, flush=True)
    if gpu_id is not None:
        print(f"  gpu   : {gpu_id}", file=file, flush=True)
    print(f"  videos: {n_videos}", file=file, flush=True)
    if n_chunks > 0:
        print(f"  chunks: {n_chunks}", file=file, flush=True)
    print(JUDGE_BAR, file=file, flush=True)
    print(file=file, flush=True)


def _fmt_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN
        return "--:--"
    s = int(round(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class EvalProgress:
    """按「已处理 chunk（帧）」+ 视频序号估计总进度。"""

    def __init__(
        self,
        chunk_counts: List[int],
        *,
        width: int = 28,
        file: TextIO = sys.stderr,
    ) -> None:
        self.chunk_counts = list(chunk_counts)
        self.total = max(sum(self.chunk_counts), 1)
        self.width = width
        self.file = file
        self.done = 0
        self.video_idx = 0  # 0-based current
        self.chunk_in_video = 0
        self._t0 = time.perf_counter()
        self._last_len = 0

    def start_video(self, video_idx: int) -> None:
        self.video_idx = video_idx
        self.chunk_in_video = 0
        expected = sum(self.chunk_counts[:video_idx])
        if self.done < expected:
            self.done = expected
        self.render()

    def tick_chunk(self) -> None:
        self.done = min(self.done + 1, self.total)
        self.chunk_in_video += 1
        self.render()

    def finish_video(self) -> None:
        """本视频结束：把进度补到该视频末尾（含被丢弃的 chunk）。"""
        expected = sum(self.chunk_counts[: self.video_idx + 1])
        if self.done < expected:
            self.done = expected
        self.chunk_in_video = self.chunk_counts[self.video_idx] if self.chunk_counts else 0
        self.render(newline=True)

    def finish_all(self) -> None:
        self.done = self.total
        self.render(newline=True)

    def render(self, *, newline: bool = False) -> None:
        pct = 100.0 * self.done / self.total
        filled = int(round(self.width * self.done / self.total))
        filled = min(max(filled, 0), self.width)
        bar = "#" * filled + "-" * (self.width - filled)

        n_vid = max(len(self.chunk_counts), 1)
        v_total = (
            self.chunk_counts[self.video_idx]
            if self.video_idx < len(self.chunk_counts)
            else 0
        )
        elapsed = time.perf_counter() - self._t0
        if self.done > 0:
            eta = elapsed * (self.total - self.done) / self.done
        else:
            eta = float("nan")

        line = (
            f"\r[{bar}] {pct:5.1f}%  "
            f"video {self.video_idx + 1}/{n_vid}  "
            f"chunk {self.chunk_in_video}/{v_total}  "
            f"ETA {_fmt_eta(eta)}"
        )
        pad = max(self._last_len - len(line) + 1, 0)  # +1 for \r
        out = line + (" " * pad)
        self._last_len = len(line)
        if newline:
            out += "\n"
            self._last_len = 0
        self.file.write(out)
        self.file.flush()
