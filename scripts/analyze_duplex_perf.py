#!/usr/bin/env python3
"""Parse duplex omni perf logs and generate SVG/CSV/Markdown assets.

This script uses only the Python standard library so it can run on the benchmark
host without creating a conda environment.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import html
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PERF_RE = re.compile(
    r"(?P<wall>\d{2}:\d{2}:\d{2}\.\d{3}) "
    r"\[DUPLEX_PERF\] (?:seq=(?P<seq>\d+) )?stage=(?P<stage>\S+) event=(?P<event>\S+) "
    r"chunk=(?P<chunk>-?\d+) t_ms=(?P<t_ms>[-\d.]+) dur_ms=(?P<dur_ms>[-\d.]+) "
    r"rss_mb=(?P<rss_mb>[-\d.]+|NA) gpu_used_mb=(?P<gpu_used_mb>[-\d.]+|NA) "
    r"gpu_total_mb=(?P<gpu_total_mb>[-\d.]+|NA) n_past=(?P<n_past>-?\d+) "
    r'detail="(?P<detail>[^"]*)"'
)
GPU_SAMPLE_RE = re.compile(
    r"(?P<wall>\d{2}:\d{2}:\d{2}\.\d{3}) "
    r"\[DUPLEX_GPU\] sample_id=(?P<sample_id>\d+) t_ms=(?P<t_ms>[-\d.]+) "
    r"device=(?P<device>\d+) sm_util_pct=(?P<sm_util_pct>[-\d.]+|NA) "
    r"mem_util_pct=(?P<mem_util_pct>[-\d.]+|NA) gpu_used_mb=(?P<gpu_used_mb>[-\d.]+|NA) "
    r"gpu_total_mb=(?P<gpu_total_mb>[-\d.]+|NA) power_w=(?P<power_w>[-\d.]+|NA) "
    r"temp_c=(?P<temp_c>[-\d.]+|NA) graphics_clock_mhz=(?P<graphics_clock_mhz>[-\d.]+|NA) "
    r"mem_clock_mhz=(?P<mem_clock_mhz>[-\d.]+|NA)"
)
GPU_STATUS_RE = re.compile(
    r"(?P<wall>\d{2}:\d{2}:\d{2}\.\d{3}) "
    r'\[DUPLEX_GPU\] event=(?P<event>\S+) reason="(?P<reason>[^"]*)"'
)
CHUNK_RE = re.compile(r"prefill:\s*([0-9.]+) s \| decode:\s*([0-9.]+) s \| total:\s*([0-9.]+) s \| n_past:\s*(\d+)")
OPT4_CHUNK_RE = re.compile(
    r"--- Chunk (?P<seq>\d+)/(?P<count>\d+) --- "
    r"prefill (?P<prefill_ms>[0-9.]+)ms \| decode (?P<decode_ms>[0-9.]+)ms \| "
    r"e2e (?P<e2e_ms>[0-9.]+)ms \| n_past (?P<n_past>\d+) \| "
    r"<\|(?P<decision>speak|listen)\|>(?: \"(?P<text>[^\"]*)\")?"
)
DECISION_RE = re.compile(r"决策:\s*<\|(speak|listen)\|>(?:\s*→\s*\"([^\"]*)\")?")

STAGE_ROWS = [
    ("vision.encode", "#4c78a8"),
    ("audio.encode", "#72b7b2"),
    ("queue.llm.wait_space", "#c7e9c0"),
    ("queue.llm.enqueue", "#74c476"),
    ("llm.prefill", "#b279a2"),
    ("wait.llm_prefill_done", "#f58518"),
    ("llm.decode", "#e45756"),
    ("queue.tts.wait_space", "#bae4b3"),
    ("queue.tts.enqueue", "#8cd17d"),
    ("tts.infer", "#54a24b"),
    ("queue.t2w.enqueue", "#9ecae9"),
    ("t2w.infer", "#3182bd"),
    ("t2w.write", "#9e9ac8"),
]

TOKEN_SPEED_STAGES = ["llm.prefill", "llm.decode", "tts.infer"]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def avg(values):
    values = list(values)
    return statistics.mean(values) if values else 0.0


def pctl(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def parse_metric(value: str) -> float:
    return -1.0 if value == "NA" else float(value)


def parse_detail(detail: str):
    result = {}
    for part in detail.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def detail_int(detail, key, default=0):
    try:
        return int(detail.get(key, default))
    except (TypeError, ValueError):
        return default


def token_count_for_event(event):
    detail = parse_detail(event["detail"])
    if "tokens" in detail:
        return detail_int(detail, "tokens", -1)
    if event["stage"] == "tts.infer":
        return detail_int(detail, "audio_tokens_generated", -1)
    if event["stage"] == "llm.prefill":
        # Old logs did not have a dedicated token field; n_past_delta is the
        # closest available KV-token proxy when no sliding happened.
        return detail_int(detail, "n_past_delta", -1)
    if event["stage"] == "llm.decode":
        return detail_int(detail, "tokens", -1)
    return -1


def parse_log(path: Path, gpu_log_path=None):
    text = path.read_text(errors="replace")
    gpu_text = ""
    if gpu_log_path and gpu_log_path != path:
        gpu_text = gpu_log_path.read_text(errors="replace")
    combined_gpu_text = text + "\n" + gpu_text

    events = []
    for match in PERF_RE.finditer(text):
        group = match.groupdict()
        events.append({
            "wall": group["wall"],
            "seq": int(group["seq"]) if group.get("seq") is not None else -1,
            "stage": group["stage"],
            "event": group["event"],
            "chunk": int(group["chunk"]),
            "t_ms": float(group["t_ms"]),
            "dur_ms": float(group["dur_ms"]),
            "rss_mb": parse_metric(group["rss_mb"]),
            "gpu_used_mb": parse_metric(group["gpu_used_mb"]),
            "gpu_total_mb": parse_metric(group["gpu_total_mb"]),
            "n_past": int(group["n_past"]),
            "detail": group["detail"],
        })

    gpu_samples = []
    for match in GPU_SAMPLE_RE.finditer(combined_gpu_text):
        group = match.groupdict()
        gpu_samples.append({
            "wall": group["wall"],
            "sample_id": int(group["sample_id"]),
            "t_ms": float(group["t_ms"]),
            "device": int(group["device"]),
            "sm_util_pct": parse_metric(group["sm_util_pct"]),
            "mem_util_pct": parse_metric(group["mem_util_pct"]),
            "gpu_used_mb": parse_metric(group["gpu_used_mb"]),
            "gpu_total_mb": parse_metric(group["gpu_total_mb"]),
            "power_w": parse_metric(group["power_w"]),
            "temp_c": parse_metric(group["temp_c"]),
            "graphics_clock_mhz": parse_metric(group["graphics_clock_mhz"]),
            "mem_clock_mhz": parse_metric(group["mem_clock_mhz"]),
        })
    gpu_samples.sort(key=lambda row: (row["device"], row["t_ms"], row["sample_id"]))

    gpu_statuses = []
    for match in GPU_STATUS_RE.finditer(combined_gpu_text):
        group = match.groupdict()
        gpu_statuses.append({
            "wall": group["wall"],
            "event": group["event"],
            "reason": group["reason"],
        })

    chunks = []
    for match in OPT4_CHUNK_RE.finditer(text):
        group = match.groupdict()
        chunks.append({
            "chunk": int(group["seq"]),
            "prefill_ms": float(group["prefill_ms"]),
            "decode_ms": float(group["decode_ms"]),
            "total_ms": float(group["e2e_ms"]),
            "n_past": int(group["n_past"]),
            "decision": group["decision"],
            "text": group["text"] or "",
        })
    if not chunks:
        decisions = list(DECISION_RE.finditer(text))
        for idx, match in enumerate(CHUNK_RE.finditer(text)):
            decision = decisions[idx].group(1) if idx < len(decisions) else ""
            decision_text = decisions[idx].group(2) if idx < len(decisions) and decisions[idx].group(2) else ""
            chunks.append({
                "chunk": idx,
                "prefill_ms": float(match.group(1)) * 1000.0,
                "decode_ms": float(match.group(2)) * 1000.0,
                "total_ms": float(match.group(3)) * 1000.0,
                "n_past": int(match.group(4)),
                "decision": decision,
                "text": decision_text,
            })
    return text, events, chunks, gpu_samples, gpu_statuses


def stage_stats(events):
    grouped = defaultdict(list)
    for event in events:
        if event["event"] == "end" and event["dur_ms"] >= 0:
            grouped[event["stage"]].append(event["dur_ms"])
    stats = {}
    for stage, values in grouped.items():
        stats[stage] = {
            "n": len(values),
            "total": sum(values),
            "avg": avg(values),
            "p50": statistics.median(values),
            "p90": pctl(values, 0.90),
            "min": min(values),
            "max": max(values),
        }
    return stats


def stage_token_stats(events):
    grouped = defaultdict(lambda: {
        "n": 0,
        "tokens": 0,
        "total_ms": 0.0,
        "event_tokens_per_s": [],
        "event_ms_per_token": [],
    })
    for event in events:
        if event["stage"] not in TOKEN_SPEED_STAGES or event["event"] != "end" or event["dur_ms"] < 0:
            continue
        tokens = token_count_for_event(event)
        if tokens < 0:
            continue
        row = grouped[event["stage"]]
        row["n"] += 1
        row["tokens"] += tokens
        row["total_ms"] += event["dur_ms"]
        if tokens > 0 and event["dur_ms"] > 0:
            row["event_tokens_per_s"].append(tokens * 1000.0 / event["dur_ms"])
            row["event_ms_per_token"].append(event["dur_ms"] / tokens)

    stats = {}
    for stage, row in grouped.items():
        tokens = row["tokens"]
        total_ms = row["total_ms"]
        stats[stage] = {
            "n": row["n"],
            "tokens": tokens,
            "total_ms": total_ms,
            "tokens_per_s": tokens * 1000.0 / total_ms if tokens > 0 and total_ms > 0 else 0.0,
            "ms_per_token": total_ms / tokens if tokens > 0 else 0.0,
            "avg_event_tokens_per_s": avg(row["event_tokens_per_s"]),
            "p50_event_tokens_per_s": statistics.median(row["event_tokens_per_s"]) if row["event_tokens_per_s"] else 0.0,
            "avg_event_ms_per_token": avg(row["event_ms_per_token"]),
        }
    return stats


def tts_token_rows(events):
    rows = []
    for event in sorted(events, key=lambda item: (item["t_ms"], item.get("seq", -1))):
        if event["stage"] != "tts.infer" or event["event"] != "end":
            continue
        detail = parse_detail(event["detail"])
        def to_int(key, default=0):
            try:
                return int(detail.get(key, default))
            except ValueError:
                return default
        llm_tokens = to_int("llm_tokens")
        filtered_llm_tokens = to_int("filtered_llm_tokens", llm_tokens)
        condition_tokens = to_int("condition_tokens")
        compute_tokens = to_int("compute_tokens", to_int("tokens", -1))
        audio_tokens = to_int("audio_tokens_generated", -1)
        if compute_tokens < 0:
            compute_tokens = audio_tokens
        rows.append({
            "chunk": event["chunk"],
            "tts_chunk": to_int("tts_chunk", event["chunk"]),
            "dur_ms": event["dur_ms"],
            "llm_tokens": llm_tokens,
            "filtered_llm_tokens": filtered_llm_tokens,
            "condition_tokens": condition_tokens,
            "compute_tokens": compute_tokens,
            "audio_tokens_generated": audio_tokens,
            "is_end_of_turn": to_int("is_end_of_turn"),
            "flush_only": to_int("flush_only"),
            "compute_tokens_per_s": compute_tokens * 1000.0 / event["dur_ms"] if compute_tokens >= 0 and event["dur_ms"] > 0 else 0.0,
            "audio_tokens_per_ms": audio_tokens / event["dur_ms"] if audio_tokens >= 0 and event["dur_ms"] > 0 else 0.0,
            "audio_tokens_per_s": audio_tokens * 1000.0 / event["dur_ms"] if audio_tokens >= 0 and event["dur_ms"] > 0 else 0.0,
            "ms_per_audio_token": event["dur_ms"] / audio_tokens if audio_tokens > 0 else 0.0,
            "has_audio_token_detail": int(audio_tokens >= 0),
        })
    return rows


def build_stage_intervals(events):
    starts = defaultdict(list)
    intervals = []
    for event in sorted(events, key=lambda item: (item["t_ms"], item.get("seq", -1))):
        key = (event["stage"], event["chunk"])
        if event["event"] == "start":
            starts[key].append(event)
        elif event["event"] == "end" and event["dur_ms"] >= 0:
            start = starts[key].pop(0) if starts[key] else None
            begin = start["t_ms"] if start else event["t_ms"] - event["dur_ms"]
            finish = event["t_ms"]
            if finish >= begin:
                intervals.append({
                    "stage": event["stage"],
                    "chunk": event["chunk"],
                    "begin_ms": begin,
                    "end_ms": finish,
                    "duration_ms": finish - begin,
                })
    return intervals


def nearest_sample(samples, times, target):
    if not samples:
        return None
    idx = bisect.bisect_left(times, target)
    candidates = []
    if idx < len(samples):
        candidates.append(samples[idx])
    if idx > 0:
        candidates.append(samples[idx - 1])
    return min(candidates, key=lambda sample: abs(sample["t_ms"] - target)) if candidates else None


def stage_gpu_stats(intervals, gpu_samples):
    by_device = defaultdict(list)
    for sample in gpu_samples:
        by_device[sample["device"]].append(sample)
    for samples in by_device.values():
        samples.sort(key=lambda sample: sample["t_ms"])

    grouped = defaultdict(lambda: {
        "n_intervals": 0,
        "n_samples": 0,
        "total_ms": 0.0,
        "estimated_samples": 0,
        "sm": [],
        "mem": [],
        "power": [],
    })

    for interval in intervals:
        for device, samples in by_device.items():
            times = [sample["t_ms"] for sample in samples]
            lo = bisect.bisect_left(times, interval["begin_ms"])
            hi = bisect.bisect_right(times, interval["end_ms"])
            selected = samples[lo:hi]
            estimated = False
            if not selected:
                sample = nearest_sample(samples, times, (interval["begin_ms"] + interval["end_ms"]) / 2.0)
                selected = [sample] if sample else []
                estimated = bool(sample)

            row = grouped[(interval["stage"], device)]
            row["n_intervals"] += 1
            row["total_ms"] += interval["duration_ms"]
            if estimated:
                row["estimated_samples"] += 1
            row["n_samples"] += len(selected)
            row["sm"].extend(sample["sm_util_pct"] for sample in selected if sample["sm_util_pct"] >= 0)
            row["mem"].extend(sample["mem_util_pct"] for sample in selected if sample["mem_util_pct"] >= 0)
            row["power"].extend(sample["power_w"] for sample in selected if sample["power_w"] >= 0)

    stats = {}
    for (stage, device), row in grouped.items():
        avg_sm = avg(row["sm"])
        stats[(stage, device)] = {
            "stage": stage,
            "device": device,
            "n_intervals": row["n_intervals"],
            "n_samples": row["n_samples"],
            "total_ms": row["total_ms"],
            "avg_sm_util_pct": avg_sm,
            "p50_sm_util_pct": pctl(row["sm"], 0.50),
            "p90_sm_util_pct": pctl(row["sm"], 0.90),
            "max_sm_util_pct": max(row["sm"]) if row["sm"] else 0.0,
            "avg_mem_util_pct": avg(row["mem"]),
            "max_mem_util_pct": max(row["mem"]) if row["mem"] else 0.0,
            "avg_power_w": avg(row["power"]),
            "max_power_w": max(row["power"]) if row["power"] else 0.0,
            "estimated_samples": row["estimated_samples"],
            "stage_gpu_busy_ms": row["total_ms"] * avg_sm / 100.0,
        }
    return stats


def svg_base(width, height, body):
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,'Noto Sans CJK SC','Microsoft YaHei',sans-serif;fill:#172033}",
        ".title{font-size:22px;font-weight:700}.subtitle{font-size:13px;fill:#5b6475}",
        ".label{font-size:13px}.small{font-size:11px;fill:#5b6475}.tiny{font-size:10px;fill:#5b6475}",
        ".lane{fill:#f6f8fb;stroke:#d6dde8}.box{rx:12;ry:12;stroke:#29415f;stroke-width:1.2}",
        "</style>",
        *body,
        "</svg>",
        "",
    ])


def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def pipeline_svg(path: Path):
    body = [
        '<rect width="1180" height="640" fill="#ffffff"/>',
        '<text x="40" y="42" class="title">Omni Duplex 实测流水线</text>',
        '<text x="40" y="66" class="subtitle">主线程按 chunk 串行推进；LLM KV prefill、TTS 和 T2W 在后台线程中异步执行</text>',
    ]
    lanes = [("主线程 / API", 110), ("LLM 线程", 235), ("TTS 线程", 360), ("T2W 线程", 485)]
    for lane, y in lanes:
        body += [f'<rect x="30" y="{y - 42}" width="1120" height="92" class="lane" rx="14"/>', f'<text x="48" y="{y - 14}" class="label" font-weight="700">{esc(lane)}</text>']
    boxes = [
        (170, 96, 160, 48, "api.stream_prefill", "vision + audio encode", "#dcecff"),
        (365, 96, 150, 48, "queue.llm", "enqueue embedding", "#d8f3dc"),
        (560, 96, 150, 48, "api.stream_decode", "wait + sample", "#fff2cc"),
        (760, 96, 150, 48, "decision", "LISTEN / SPEAK", "#ffe4e6"),
        (365, 222, 165, 48, "llm.prefill", "write KV cache", "#e6dcff"),
        (560, 222, 165, 48, "llm.decode", "token loop", "#ffd6a5"),
        (760, 347, 180, 48, "tts.infer", "token/hidden -> audio token", "#d0f4de"),
        (760, 472, 180, 48, "t2w.infer", "audio tokens -> wav", "#cde7ff"),
    ]
    for x, y, w, h, title, sub, color in boxes:
        body += [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" class="box"/>', f'<text x="{x + 12}" y="{y + 20}" class="label" font-weight="700">{esc(title)}</text>', f'<text x="{x + 12}" y="{y + 38}" class="small">{esc(sub)}</text>']
    body.append('<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#29415f"/></marker></defs>')
    for x1, y1, x2, y2 in [(330,120,365,120),(515,120,560,120),(710,120,760,120),(440,144,440,222),(530,246,560,246),(635,222,635,144),(835,144,835,347),(850,395,850,472),(940,496,1040,496)]:
        body.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#29415f" stroke-width="1.8" marker-end="url(#arrow)"/>')
    for x, text in [(60, "index=0 是 session/ref audio 初始化"), (420, "decode 包含等待异步 KV prefill"), (795, "TTS/T2W 与下一轮 chunk 重叠")]:
        body.append(f'<text x="{x}" y="585" class="small">• {esc(text)}</text>')
    write(path, svg_base(1180, 640, body))


def stage_latency_svg(path: Path, stats):
    width = 1080
    left, top, bar_w, row_h = 270, 88, 650, 42
    chart_bottom = top + len(STAGE_ROWS) * row_h + 10
    height = max(470, chart_bottom + 50)
    max_value = max([stats.get(stage, {}).get("avg", 0) for stage, _ in STAGE_ROWS] + [260])
    body = [f'<rect width="1080" height="{height}" fill="#ffffff"/>', '<text x="40" y="42" class="title">阶段耗时均值</text>', '<text x="40" y="66" class="subtitle">单位 ms；横条为平均值，灰线为 p50 到 max</text>']
    for tick in range(0, 301, 50):
        x = left + tick / max_value * bar_w
        body += [f'<line x1="{x:.1f}" y1="78" x2="{x:.1f}" y2="{chart_bottom}" stroke="#edf1f7"/>', f'<text x="{x - 8:.1f}" y="{chart_bottom + 19}" class="tiny">{tick}</text>']
    for idx, (stage, color) in enumerate(STAGE_ROWS):
        y = top + idx * row_h
        row = stats.get(stage, {})
        avgv, p50, maxv, n = row.get("avg", 0), row.get("p50", 0), row.get("max", 0), int(row.get("n", 0))
        w = avgv / max_value * bar_w
        p50x, maxx = left + p50 / max_value * bar_w, left + maxv / max_value * bar_w
        body += [f'<text x="40" y="{y + 18}" class="label">{esc(stage)}</text>', f'<rect x="{left}" y="{y}" width="{w:.1f}" height="22" fill="{color}" rx="5"/>', f'<line x1="{p50x:.1f}" y1="{y + 31}" x2="{maxx:.1f}" y2="{y + 31}" stroke="#7f8897" stroke-width="2"/>', f'<circle cx="{p50x:.1f}" cy="{y + 31}" r="3" fill="#7f8897"/>', f'<circle cx="{maxx:.1f}" cy="{y + 31}" r="3" fill="#7f8897"/>', f'<text x="{left + w + 8:.1f}" y="{y + 16}" class="small">{avgv:.1f} ms</text>', f'<text x="940" y="{y + 18}" class="tiny">n={n} p50={p50:.1f} max={maxv:.1f}</text>']
    write(path, svg_base(width, height, body))


def chunk_latency_svg(path: Path, chunks):
    width, height = 1180, 560
    left, top, chart_w, chart_h = 70, 90, 1030, 350
    max_total = max([c["total_ms"] for c in chunks] + [1])
    max_n_past = max([c["n_past"] for c in chunks] + [1])
    bar_gap = 3
    bar_w = max(6, (chart_w - bar_gap * max(0, len(chunks) - 1)) / max(1, len(chunks)))
    body = ['<rect width="1180" height="560" fill="#ffffff"/>', '<text x="40" y="42" class="title">逐 chunk API 延迟</text>', '<text x="40" y="66" class="subtitle">蓝色=prefill，橙色=decode；圆点表示决策，绿=SPEAK，灰=LISTEN；紫线=n_past</text>']
    for tick in range(0, int(max_total) + 51, 50):
        y = top + chart_h - tick / max_total * chart_h
        body += [f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#edf1f7"/>', f'<text x="30" y="{y + 4:.1f}" class="tiny">{tick}</text>']
    prev = None
    for idx, chunk in enumerate(chunks):
        x = left + idx * (bar_w + bar_gap)
        pre_h, dec_h = chunk["prefill_ms"] / max_total * chart_h, chunk["decode_ms"] / max_total * chart_h
        y_pre, y_dec = top + chart_h - pre_h, top + chart_h - pre_h - dec_h
        color = "#2ca25f" if chunk["decision"] == "speak" else "#8b95a5"
        body += [f'<rect x="{x:.1f}" y="{y_pre:.1f}" width="{bar_w:.1f}" height="{pre_h:.1f}" fill="#4c78a8"/>', f'<rect x="{x:.1f}" y="{y_dec:.1f}" width="{bar_w:.1f}" height="{dec_h:.1f}" fill="#f58518"/>', f'<circle cx="{x + bar_w / 2:.1f}" cy="{y_dec - 7:.1f}" r="3.2" fill="{color}"/>']
        if idx % 5 == 0:
            body.append(f'<text x="{x - 2:.1f}" y="{top + chart_h + 18}" class="tiny">{idx}</text>')
        point = (x + bar_w / 2, top + chart_h - chunk["n_past"] / max_n_past * chart_h)
        if prev:
            body.append(f'<line x1="{prev[0]:.1f}" y1="{prev[1]:.1f}" x2="{point[0]:.1f}" y2="{point[1]:.1f}" stroke="#6f4e7c" stroke-width="1.5" opacity="0.65"/>')
        prev = point
    body += ['<rect x="845" y="92" width="250" height="88" fill="#ffffff" stroke="#d6dde8" rx="10"/>', '<rect x="865" y="112" width="22" height="12" fill="#4c78a8"/><text x="895" y="122" class="small">prefill</text>', '<rect x="865" y="136" width="22" height="12" fill="#f58518"/><text x="895" y="146" class="small">decode</text>', '<circle cx="876" cy="164" r="4" fill="#2ca25f"/><text x="895" y="168" class="small">SPEAK；紫线=n_past</text>']
    write(path, svg_base(width, height, body))


def overlap_svg(path: Path, events, start_ms=850.0, end_ms=1400.0):
    width, height = 1180, 530
    left, top, chart_w = 210, 86, 900
    lanes = [
        ("main.prefill", ["api.duplex.frame_total", "api.stream_prefill", "vision.encode", "audio.encode", "queue.llm.wait_space", "queue.llm.enqueue"]),
        ("llm", ["llm.prefill", "wait.llm_prefill_done", "llm.decode", "api.llm_decode_loop"]),
        ("tts", ["queue.tts.wait_space", "queue.tts.enqueue", "queue.tts.wait_data", "tts.infer"]),
        ("t2w", ["queue.t2w.enqueue", "queue.t2w.wait_data", "t2w.infer", "t2w.write"]),
    ]
    colors = {
        "api.duplex.frame_total":"#8da0cb", "api.stream_prefill":"#4c78a8", "vision.encode":"#77aadd", "audio.encode":"#72b7b2",
        "queue.llm.wait_space":"#c7e9c0", "queue.llm.enqueue":"#74c476",
        "llm.prefill":"#b279a2", "wait.llm_prefill_done":"#f58518", "llm.decode":"#e45756", "api.llm_decode_loop":"#ff9da6",
        "queue.tts.wait_space":"#bae4b3", "queue.tts.enqueue":"#8cd17d", "queue.tts.wait_data":"#a1d99b", "tts.infer":"#54a24b",
        "queue.t2w.enqueue":"#9ecae9", "queue.t2w.wait_data":"#c6dbef", "t2w.infer":"#3182bd", "t2w.write":"#9e9ac8",
    }
    stage_lane = {stage: lane for lane, stages in lanes for stage in stages}
    lane_y = {}
    body = ['<rect width="1180" height="530" fill="#ffffff"/>', '<text x="40" y="42" class="title">异步重叠时间线：约 0.85s 到 1.40s</text>', '<text x="40" y="66" class="subtitle">TTS/T2W 与下一轮 prefill/decode 同时运行；chunk 标签按日志原值显示</text>']
    for tick in range(int(start_ms), int(end_ms) + 1, 50):
        x = left + (tick - start_ms) / (end_ms - start_ms) * chart_w
        body += [f'<line x1="{x:.1f}" y1="80" x2="{x:.1f}" y2="455" stroke="#edf1f7"/>', f'<text x="{x - 14:.1f}" y="478" class="tiny">{tick}</text>']
    for idx, (lane, _) in enumerate(lanes):
        y = top + idx * 92
        lane_y[lane] = y
        body += [f'<rect x="40" y="{y - 20}" width="1070" height="62" fill="#f6f8fb" stroke="#d6dde8" rx="10"/>', f'<text x="58" y="{y + 5}" class="label" font-weight="700">{esc(lane)}</text>']
    starts = defaultdict(list)
    bars = []
    for event in sorted(events, key=lambda item: item["t_ms"]):
        key = (event["stage"], event["chunk"])
        if event["event"] == "start":
            starts[key].append(event)
        elif event["event"] == "end" and event["stage"] in stage_lane:
            start = starts[key].pop(0) if starts[key] else None
            begin = start["t_ms"] if start else event["t_ms"] - max(event["dur_ms"], 0)
            finish = event["t_ms"]
            if finish >= start_ms and begin <= end_ms:
                bars.append((begin, finish, event["stage"], event["chunk"]))
    offsets = {stage: (i % 3) * 16 for i, stage in enumerate(colors)}
    for begin, finish, stage, chunk in bars:
        lane = stage_lane[stage]
        x = left + (max(begin, start_ms) - start_ms) / (end_ms - start_ms) * chart_w
        w = max(2, (min(finish, end_ms) - max(begin, start_ms)) / (end_ms - start_ms) * chart_w)
        y = lane_y[lane] - 4 + offsets.get(stage, 0)
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="13" fill="{colors[stage]}" rx="3"/>')
        if w > 62:
            body.append(f'<text x="{x + 4:.1f}" y="{y + 10:.1f}" class="tiny">c{chunk} {esc(stage.split(".")[-1])}</text>')
    write(path, svg_base(width, height, body))


def gpu_utilization_svg(path: Path, gpu_samples, intervals):
    width = 1180
    left, chart_w = 80, 1010
    device_ids = sorted({sample["device"] for sample in gpu_samples})
    if not gpu_samples or not device_ids:
        body = ['<rect width="1180" height="180" fill="#ffffff"/>', '<text x="40" y="42" class="title">GPU 利用率时间线</text>', '<text x="40" y="72" class="subtitle">未解析到 [DUPLEX_GPU] sample</text>']
        write(path, svg_base(width, 180, body))
        return

    sample_min = min(sample["t_ms"] for sample in gpu_samples)
    sample_max = max(sample["t_ms"] for sample in gpu_samples)
    interval_min = min([interval["begin_ms"] for interval in intervals] + [sample_min])
    interval_max = max([interval["end_ms"] for interval in intervals] + [sample_max])
    start_ms = min(sample_min, interval_min)
    end_ms = max(sample_max, interval_max)
    span = max(1.0, end_ms - start_ms)
    device_h = 115
    stage_top = 96 + len(device_ids) * device_h
    height = stage_top + 220
    body = ['<rect width="1180" height="%d" fill="#ffffff"/>' % height, '<text x="40" y="42" class="title">GPU 利用率时间线</text>', '<text x="40" y="66" class="subtitle">蓝线=SM util，橙线=memory controller util；下方为主要阶段 interval</text>']

    def x_of(t_ms):
        return left + (t_ms - start_ms) / span * chart_w

    for tick in range(int(start_ms // 100 * 100), int(end_ms) + 101, 100):
        x = x_of(tick)
        body += [f'<line x1="{x:.1f}" y1="84" x2="{x:.1f}" y2="{height - 40}" stroke="#edf1f7"/>', f'<text x="{x - 18:.1f}" y="{height - 18}" class="tiny">{tick}</text>']

    for idx, device in enumerate(device_ids):
        y0 = 90 + idx * device_h
        body += [f'<rect x="40" y="{y0 - 10}" width="1070" height="92" fill="#f6f8fb" stroke="#d6dde8" rx="10"/>', f'<text x="50" y="{y0 + 10}" class="label" font-weight="700">device {device}</text>']
        for pct in (0, 50, 100):
            y = y0 + 70 - pct / 100.0 * 62
            body += [f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e8edf5"/>', f'<text x="46" y="{y + 4:.1f}" class="tiny">{pct}%</text>']
        samples = [sample for sample in gpu_samples if sample["device"] == device]
        for key, color in (("sm_util_pct", "#2f6fed"), ("mem_util_pct", "#f58518")):
            points = []
            for sample in samples:
                value = sample[key]
                if value < 0:
                    continue
                points.append(f'{x_of(sample["t_ms"]):.1f},{y0 + 70 - min(100.0, value) / 100.0 * 62:.1f}')
            if len(points) >= 2:
                body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.8"/>')

    stage_colors = {stage: color for stage, color in STAGE_ROWS}
    focus_stages = ["llm.prefill", "llm.decode", "tts.infer", "t2w.infer", "vision.encode", "audio.encode"]
    lane_y = {stage: stage_top + 22 + idx * 26 for idx, stage in enumerate(focus_stages)}
    body += [f'<text x="40" y="{stage_top}" class="label" font-weight="700">stage intervals</text>']
    for stage in focus_stages:
        y = lane_y[stage]
        body += [f'<text x="40" y="{y + 10}" class="tiny">{esc(stage)}</text>', f'<line x1="{left}" y1="{y + 5}" x2="{left + chart_w}" y2="{y + 5}" stroke="#edf1f7"/>']
    for interval in intervals:
        stage = interval["stage"]
        if stage not in lane_y:
            continue
        x = x_of(interval["begin_ms"])
        w = max(1.5, (interval["end_ms"] - interval["begin_ms"]) / span * chart_w)
        y = lane_y[stage]
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="11" fill="{stage_colors.get(stage, "#8b95a5")}" opacity="0.58" rx="3"/>')

    body += ['<rect x="875" y="90" width="220" height="52" fill="#ffffff" stroke="#d6dde8" rx="10"/>', '<line x1="895" y1="110" x2="930" y2="110" stroke="#2f6fed" stroke-width="2"/><text x="940" y="114" class="small">SM util</text>', '<line x1="895" y1="132" x2="930" y2="132" stroke="#f58518" stroke-width="2"/><text x="940" y="136" class="small">mem util</text>']
    write(path, svg_base(width, height, body))


def write_csvs(out_dir: Path, stats, chunks, gpu_samples, gpu_stats, tts_tokens, token_stats):
    with (out_dir / "omni-duplex-stage-stats.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "n", "total_ms", "avg_ms", "p50_ms", "p90_ms", "min_ms", "max_ms"])
        for stage in sorted(stats):
            row = stats[stage]
            writer.writerow([stage, row["n"], f'{row["total"]:.3f}', f'{row["avg"]:.3f}', f'{row["p50"]:.3f}', f'{row["p90"]:.3f}', f'{row["min"]:.3f}', f'{row["max"]:.3f}'])
    with (out_dir / "omni-duplex-chunk-summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk", "prefill_ms", "decode_ms", "total_ms", "n_past", "decision", "text"])
        for chunk in chunks:
            writer.writerow([chunk["chunk"], f'{chunk["prefill_ms"]:.3f}', f'{chunk["decode_ms"]:.3f}', f'{chunk["total_ms"]:.3f}', chunk["n_past"], chunk["decision"], chunk["text"]])
    with (out_dir / "omni-duplex-gpu-samples.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "t_ms", "device", "sm_util_pct", "mem_util_pct", "gpu_used_mb", "gpu_total_mb", "power_w", "temp_c", "graphics_clock_mhz", "mem_clock_mhz"])
        for sample in gpu_samples:
            writer.writerow([sample["sample_id"], f'{sample["t_ms"]:.3f}', sample["device"], sample["sm_util_pct"], sample["mem_util_pct"], sample["gpu_used_mb"], sample["gpu_total_mb"], sample["power_w"], sample["temp_c"], sample["graphics_clock_mhz"], sample["mem_clock_mhz"]])
    with (out_dir / "omni-duplex-stage-gpu-stats.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "device", "n_intervals", "n_samples", "total_ms", "avg_sm_util_pct", "p50_sm_util_pct", "p90_sm_util_pct", "max_sm_util_pct", "avg_mem_util_pct", "max_mem_util_pct", "avg_power_w", "max_power_w", "estimated_samples", "stage_gpu_busy_ms"])
        for key in sorted(gpu_stats):
            row = gpu_stats[key]
            writer.writerow([row["stage"], row["device"], row["n_intervals"], row["n_samples"], f'{row["total_ms"]:.3f}', f'{row["avg_sm_util_pct"]:.3f}', f'{row["p50_sm_util_pct"]:.3f}', f'{row["p90_sm_util_pct"]:.3f}', f'{row["max_sm_util_pct"]:.3f}', f'{row["avg_mem_util_pct"]:.3f}', f'{row["max_mem_util_pct"]:.3f}', f'{row["avg_power_w"]:.3f}', f'{row["max_power_w"]:.3f}', row["estimated_samples"], f'{row["stage_gpu_busy_ms"]:.3f}'])
    with (out_dir / "omni-duplex-stage-token-speed.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "n", "tokens", "total_ms", "tokens_per_s", "ms_per_token", "avg_event_tokens_per_s", "p50_event_tokens_per_s", "avg_event_ms_per_token"])
        for stage in TOKEN_SPEED_STAGES:
            row = token_stats.get(stage, {})
            writer.writerow([
                stage,
                int(row.get("n", 0)),
                int(row.get("tokens", 0)),
                f'{row.get("total_ms", 0.0):.3f}',
                f'{row.get("tokens_per_s", 0.0):.3f}',
                f'{row.get("ms_per_token", 0.0):.6f}',
                f'{row.get("avg_event_tokens_per_s", 0.0):.3f}',
                f'{row.get("p50_event_tokens_per_s", 0.0):.3f}',
                f'{row.get("avg_event_ms_per_token", 0.0):.6f}',
            ])
    with (out_dir / "omni-duplex-tts-token-stats.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk", "tts_chunk", "dur_ms", "llm_tokens", "filtered_llm_tokens", "condition_tokens", "compute_tokens", "audio_tokens_generated", "is_end_of_turn", "flush_only", "compute_tokens_per_s", "audio_tokens_per_ms", "audio_tokens_per_s", "ms_per_audio_token", "has_audio_token_detail"])
        for row in tts_tokens:
            writer.writerow([row["chunk"], row["tts_chunk"], f'{row["dur_ms"]:.3f}', row["llm_tokens"], row["filtered_llm_tokens"], row["condition_tokens"], row["compute_tokens"], row["audio_tokens_generated"], row["is_end_of_turn"], row["flush_only"], f'{row["compute_tokens_per_s"]:.3f}', f'{row["audio_tokens_per_ms"]:.6f}', f'{row["audio_tokens_per_s"]:.3f}', f'{row["ms_per_audio_token"]:.3f}', row["has_audio_token_detail"]])


def stage_kind(stage: str) -> str:
    if stage.startswith("api."):
        return "api"
    if stage.startswith("queue."):
        return "queue"
    if stage.startswith("wait."):
        return "wait"
    if stage.startswith("control."):
        return "control"
    if stage.startswith("session."):
        return "session"
    if stage.endswith(".write"):
        return "io"
    return "compute"


def ordered_stage_names(stats):
    preferred = [stage for stage, _ in STAGE_ROWS]
    return preferred + [stage for stage in sorted(stats) if stage not in preferred]


def write_stage_timing_table(path: Path, stats):
    lines = [
        "# Duplex 阶段用时统计表",
        "",
        "单位均为 ms。`total` 是该 stage 所有 end 事件的耗时总和；异步/父子阶段不要直接相加。",
        "",
        "| type | stage | n | total | avg | p50 | p90 | min | max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in ordered_stage_names(stats):
        row = stats.get(stage, {})
        if not row:
            continue
        lines.append(
            f"| `{stage_kind(stage)}` | `{stage}` | {int(row['n'])} | "
            f"{row['total']:.1f} | {row['avg']:.1f} | {row['p50']:.1f} | "
            f"{row['p90']:.1f} | {row['min']:.1f} | {row['max']:.1f} |"
        )
    lines.append("")
    write(path, "\n".join(lines))


def gpu_stage_summary(gpu_stats, stage):
    rows = [row for (row_stage, _), row in gpu_stats.items() if row_stage == stage]
    if not rows:
        return None
    total_samples = sum(row["n_samples"] for row in rows)
    avg_sm = avg(row["avg_sm_util_pct"] for row in rows)
    max_sm = max(row["max_sm_util_pct"] for row in rows)
    avg_power = avg(row["avg_power_w"] for row in rows)
    estimated = sum(row["estimated_samples"] for row in rows)
    devices = ",".join(str(row["device"]) for row in sorted(rows, key=lambda item: item["device"]))
    return {
        "devices": devices,
        "total_samples": total_samples,
        "avg_sm": avg_sm,
        "max_sm": max_sm,
        "avg_power": avg_power,
        "estimated": estimated,
    }


def write_report(path: Path, log_path: Path, text: str, events, chunks, stats, gpu_samples, gpu_stats, gpu_statuses, tts_tokens, token_stats):
    decisions = Counter(c["decision"] for c in chunks)
    event_counts = Counter((e["stage"], e["event"]) for e in events)
    user_chunks = chunks[1:] if len(chunks) > 1 else chunks
    speak = [c for c in chunks if c["decision"] == "speak"]
    listen = [c for c in chunks if c["decision"] == "listen"]
    gpu = [e["gpu_used_mb"] for e in events if e["gpu_used_mb"] >= 0]
    rss = [e["rss_mb"] for e in events if e["rss_mb"] >= 0]
    def st(stage, key="avg"):
        return stats.get(stage, {}).get(key, 0.0)
    tts_tokens_with_detail = [row for row in tts_tokens if row["has_audio_token_detail"]]
    tts_audio_values = [row["audio_tokens_generated"] for row in tts_tokens_with_detail]
    rss_range = f"{min(rss):.0f}-{max(rss):.0f} MB" if rss else "NA"
    gpu_range = f"{min(gpu):.0f}-{max(gpu):.0f} MB" if gpu else "NA"
    def speed(stage):
        return token_stats.get(stage, {}).get("tokens_per_s", 0.0)
    lines = [
        "# Omni Duplex TTS GPU 实测性能报告",
        "",
        f"- 日志：`{log_path.name}`",
        f"- 解析到 `{len(events)}` 条结构化 `[DUPLEX_PERF]` 事件；原始 marker 为 `{text.count('[DUPLEX_PERF]')}` 条。",
        f"- Chunk：`{len(chunks)}`；SPEAK/LISTEN：`{decisions.get('speak', 0)}` / `{decisions.get('listen', 0)}`。",
        f"- API 口径：平均 prefill `{avg(c['prefill_ms'] for c in chunks):.1f} ms`，平均 decode `{avg(c['decode_ms'] for c in chunks):.1f} ms`，平均每 chunk `{avg(c['total_ms'] for c in chunks):.1f} ms`。",
        f"- 用户 chunk 口径（排除 `index=0`）：平均 prefill `{avg(c['prefill_ms'] for c in user_chunks):.1f} ms`，平均 decode `{avg(c['decode_ms'] for c in user_chunks):.1f} ms`，平均每 chunk `{avg(c['total_ms'] for c in user_chunks):.1f} ms`。",
        f"- 阶段平均速度：`llm.prefill` `{speed('llm.prefill'):.2f} tok/s`，`llm.decode` `{speed('llm.decode'):.2f} tok/s`，`TTS.infer` `{speed('tts.infer'):.2f} compute tok/s`。",
        f"- n_past：峰值 `{max(c['n_past'] for c in chunks)}`，最终 `{chunks[-1]['n_past']}`；RSS 约 `{rss_range}`，GPU used 约 `{gpu_range}`。",
        "",
        "## 图表",
        "",
        "![Omni Duplex 实测流水线](figures/omni-duplex-pipeline.svg)",
        "",
        "![阶段耗时均值](figures/omni-duplex-stage-latency.svg)",
        "",
        "![逐 chunk API 延迟](figures/omni-duplex-chunk-latency.svg)",
        "",
        "![异步重叠时间线](figures/omni-duplex-overlap-timeline.svg)",
        "",
        "![GPU 利用率时间线](figures/omni-duplex-gpu-utilization.svg)",
        "",
        "完整阶段用时统计表见 [`omni-duplex-stage-timing-table.md`](omni-duplex-stage-timing-table.md)，CSV 见 `omni-duplex-stage-stats.csv`、`omni-duplex-stage-token-speed.csv`、`omni-duplex-gpu-samples.csv`、`omni-duplex-stage-gpu-stats.csv` 和 `omni-duplex-tts-token-stats.csv`。",
        "",
        "## 主要结论",
        "",
        f"1. `vision.encode` / `audio.encode` 是输入侧 encoder 口径，平均约 `{st('vision.encode'):.1f}` / `{st('audio.encode'):.1f} ms`。",
        f"2. LLM KV prefill 在后台线程执行，平均约 `{st('llm.prefill'):.1f} ms`，平均速度 `{speed('llm.prefill'):.2f} tok/s`；API `decode` 通过 `wait.llm_prefill_done` 等待它，因此当前 decode API 不是纯采样耗时。",
        f"3. LLM decode 平均速度 `{speed('llm.decode'):.2f} tok/s`；SPEAK chunk 的 decode 均值约 `{avg(c['decode_ms'] for c in speak):.1f} ms`，LISTEN chunk 约 `{avg(c['decode_ms'] for c in listen):.1f} ms`。",
        f"4. TTS/T2W 是后台链路。`tts.infer` 平均约 `{st('tts.infer'):.1f} ms`、`{speed('tts.infer'):.2f} compute tok/s`，`t2w.infer` 平均约 `{st('t2w.infer'):.1f} ms`，但它们可能与后续 chunk 重叠。",
        "5. `[DUPLEX_PERF]` 带 `seq` 后可以按 `(t_ms, seq)` 稳定排序；TTS/T2W 仍建议继续增加稳定的 `utterance_id` / `audio_chunk_id` 来做跨线程归因。",
        "",
        "## GPU 利用率",
        "",
    ]
    if gpu_samples:
        lines.append(f"- 解析到 `{len(gpu_samples)}` 条 `[DUPLEX_GPU]` sample，覆盖 device：`{', '.join(str(d) for d in sorted({s['device'] for s in gpu_samples}))}`。")
        for stage in ["tts.infer", "llm.decode", "llm.prefill", "t2w.infer"]:
            row = gpu_stage_summary(gpu_stats, stage)
            if row:
                lines.append(f"- `{stage}`: avg SM `{row['avg_sm']:.1f}%`, max `{row['max_sm']:.1f}%`, avg power `{row['avg_power']:.1f} W`, samples `{row['total_samples']}`, estimated `{row['estimated']}`。")
        lines.append("- 低 SM util + 高耗时的阶段优先检查 CPU、IO、队列和同步点；`stage_gpu_busy_ms` 仅用于单阶段观察，不应跨重叠阶段相加。")
    elif gpu_statuses:
        reasons = ", ".join(f"{status['event']}:{status['reason']}" for status in gpu_statuses)
        lines.append(f"- 未解析到 GPU sample；状态：`{reasons}`。")
    else:
        lines.append("- 未解析到 `[DUPLEX_GPU]` sample。运行时设置 `OMNI_GPU_PROF=1` 可启用 NVML 采样。")
    lines += [
        "",
        "## 阶段平均速度",
        "",
        "速度口径为该阶段所有 end 事件的 `compute tokens / total duration`；`llm.prefill` 的 token 包含文本控制标记 token 和音频/视觉 embedding token，`llm.decode` 包含模型采样 token 和手动 feed 的控制 token，`tts.infer` 包含 TTS 采样步数（含 EOS 等未输出 token）。",
        "",
        "| stage | n | tokens | total ms | tokens/s | ms/token |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in TOKEN_SPEED_STAGES:
        row = token_stats.get(stage, {})
        display_stage = "TTS.infer" if stage == "tts.infer" else stage
        lines.append(
            f"| `{display_stage}` | {int(row.get('n', 0))} | {int(row.get('tokens', 0))} | "
            f"{row.get('total_ms', 0.0):.1f} | {row.get('tokens_per_s', 0.0):.2f} | "
            f"{row.get('ms_per_token', 0.0):.4f} |"
        )
    lines += [
        "",
        "## TTS Token 规模",
        "",
    ]
    if tts_tokens_with_detail:
        lines.append(f"- 解析到 `{len(tts_tokens_with_detail)}` 次 `tts.infer` audio token 输出；总计 `{sum(tts_audio_values)}` 个 audio token。")
        lines.append(f"- 每次 `tts.infer` 输出 token：平均 `{avg(tts_audio_values):.1f}`，p50 `{statistics.median(tts_audio_values):.1f}`，p90 `{pctl(tts_audio_values, 0.90):.1f}`，范围 `{min(tts_audio_values)}-{max(tts_audio_values)}`。")
        per_token = [row["ms_per_audio_token"] for row in tts_tokens_with_detail if row["ms_per_audio_token"] > 0]
        if per_token:
            lines.append(f"- TTS 每 audio token 成本：平均 `{avg(per_token):.2f} ms/token`，p50 `{statistics.median(per_token):.2f} ms/token`。")
            output_total_ms = sum(row["dur_ms"] for row in tts_tokens_with_detail)
            output_tok_s = sum(tts_audio_values) * 1000.0 / output_total_ms if output_total_ms > 0 else 0.0
            lines.append(f"- TTS compute 速度：`{speed('tts.infer'):.2f} tok/s`；有效 audio 输出速度：`{output_tok_s:.2f} audio tok/s`（总体口径）。")
        lines += [
            "",
            "| chunk | tts_chunk | dur ms | llm tokens | filtered | condition | compute tokens | audio tokens | ms/audio token | end_of_turn | flush_only |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in tts_tokens_with_detail:
            lines.append(
                f"| {row['chunk']} | {row['tts_chunk']} | {row['dur_ms']:.1f} | "
                f"{row['llm_tokens']} | {row['filtered_llm_tokens']} | {row['condition_tokens']} | "
                f"{row['compute_tokens']} | {row['audio_tokens_generated']} | {row['ms_per_audio_token']:.2f} | "
                f"{row['is_end_of_turn']} | {row['flush_only']} |"
            )
    elif tts_tokens:
        lines.append("- 当前日志里的 `tts.infer` detail 还没有 `audio_tokens_generated` 字段；请用更新后的二进制重新跑一次 benchmark 后再分析。")
    else:
        lines.append("- 未解析到 `tts.infer` end 事件。")
    lines += [
        "",
        "## 阶段耗时表",
        "",
        "| type | stage | n | total ms | avg ms | p50 ms | p90 ms | max ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage, _ in STAGE_ROWS:
        row = stats.get(stage, {})
        lines.append(f"| `{stage_kind(stage)}` | `{stage}` | {int(row.get('n', 0))} | {row.get('total', 0):.1f} | {row.get('avg', 0):.1f} | {row.get('p50', 0):.1f} | {row.get('p90', 0):.1f} | {row.get('max', 0):.1f} |")
    lines += [
        "",
        "## 计时口径说明",
        "",
        "- `index=0` 是 session/ref audio 初始化，不能和普通用户音频 chunk 混在一起解释。",
        "- `api.duplex.frame_total` 是 push frame 到 result 出队的端到端 API 包络；`api.stream_prefill` 主要覆盖 encoder submit 和入队。",
        "- 异步 LLM KV 写入体现在 `llm.prefill` 和 `wait.llm_prefill_done`；这些内部阶段不要和 API 包络相加。",
        "- TTS/T2W 的 `chunk` 来自随队列传递的 `perf_chunk_index`；若一次 drain 多个队列项，T2W window 仍按第一个有效 chunk 标注。",
        "- 对端到端首响、RTF 和尾包 flush 的精确测量，仍建议在 LLM enqueue、TTS audio token、T2W wav 输出之间增加同一个稳定请求 ID。",
        "",
        "## 结构化事件完整性",
        "",
        "| stage/event | count |",
        "| --- | ---: |",
    ]
    for (stage, event), count in sorted(event_counts.items()):
        lines.append(f"| `{stage}` / `{event}` | {count} |")
    lines.append("")
    write(path, "\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=Path("/cache/hanqingzhe/llama.cpp-omni/duplex_omni_tts_gpu_4_7.log"))
    parser.add_argument("--gpu-log", type=Path, default=None, help="Optional separate OMNI_GPU_PROF_FILE log")
    parser.add_argument("--out-dir", type=Path, default=Path("/cache/hanqingzhe/llama.cpp-omni/docs/development"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    text, events, chunks, gpu_samples, gpu_statuses = parse_log(args.log, args.gpu_log)
    stats = stage_stats(events)
    tts_tokens = tts_token_rows(events)
    token_stats = stage_token_stats(events)
    intervals = build_stage_intervals(events)
    gpu_stats = stage_gpu_stats(intervals, gpu_samples)
    figures = args.out_dir / "figures"
    pipeline_svg(figures / "omni-duplex-pipeline.svg")
    stage_latency_svg(figures / "omni-duplex-stage-latency.svg", stats)
    chunk_latency_svg(figures / "omni-duplex-chunk-latency.svg", chunks)
    overlap_svg(figures / "omni-duplex-overlap-timeline.svg", events)
    gpu_utilization_svg(figures / "omni-duplex-gpu-utilization.svg", gpu_samples, intervals)
    write_csvs(args.out_dir, stats, chunks, gpu_samples, gpu_stats, tts_tokens, token_stats)
    write_stage_timing_table(args.out_dir / "omni-duplex-stage-timing-table.md", stats)
    write_report(args.out_dir / "omni-duplex-tts-gpu-4-7-report.md", args.log, text, events, chunks, stats, gpu_samples, gpu_stats, gpu_statuses, tts_tokens, token_stats)
    print(f"parsed_events={len(events)} raw_markers={text.count('[DUPLEX_PERF]')} chunks={len(chunks)} gpu_samples={len(gpu_samples)} tts_token_rows={len(tts_tokens)}")
    print(f"wrote={args.out_dir / 'omni-duplex-tts-gpu-4-7-report.md'}")
    print(f"stage_table={args.out_dir / 'omni-duplex-stage-timing-table.md'}")
    print(f"figures={figures}")


if __name__ == "__main__":
    main()
