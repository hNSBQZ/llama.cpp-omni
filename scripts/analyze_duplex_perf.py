#!/usr/bin/env python3
"""Parse duplex omni perf logs and generate SVG/CSV/Markdown assets.

This script uses only the Python standard library so it can run on the benchmark
host without creating a conda environment.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PERF_RE = re.compile(
    r"(?P<wall>\d{2}:\d{2}:\d{2}\.\d{3}) "
    r"\[DUPLEX_PERF\] stage=(?P<stage>\S+) event=(?P<event>\S+) "
    r"chunk=(?P<chunk>-?\d+) t_ms=(?P<t_ms>[-\d.]+) dur_ms=(?P<dur_ms>[-\d.]+) "
    r"rss_mb=(?P<rss_mb>[-\d.]+) gpu_used_mb=(?P<gpu_used_mb>[-\d.]+) "
    r"gpu_total_mb=(?P<gpu_total_mb>[-\d.]+) n_past=(?P<n_past>-?\d+) "
    r'detail="(?P<detail>[^"]*)"'
)
CHUNK_RE = re.compile(r"prefill:\s*([0-9.]+) s \| decode:\s*([0-9.]+) s \| total:\s*([0-9.]+) s \| n_past:\s*(\d+)")
DECISION_RE = re.compile(r"决策:\s*<\|(speak|listen)\|>(?:\s*→\s*\"([^\"]*)\")?")

STAGE_ROWS = [
    ("prefill.vision_encode", "#4c78a8"),
    ("prefill.audio_encode", "#72b7b2"),
    ("llm.kv_prefill", "#b279a2"),
    ("decode.wait_llm_prefill", "#f58518"),
    ("decode.llm_sample", "#e45756"),
    ("stream_decode", "#ff9da6"),
    ("tts.llm_to_audio_tokens", "#54a24b"),
    ("t2w.tokens_to_wav", "#9ecae9"),
]


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


def parse_log(path: Path):
    text = path.read_text(errors="replace")
    events = []
    for match in PERF_RE.finditer(text):
        group = match.groupdict()
        events.append({
            "wall": group["wall"],
            "stage": group["stage"],
            "event": group["event"],
            "chunk": int(group["chunk"]),
            "t_ms": float(group["t_ms"]),
            "dur_ms": float(group["dur_ms"]),
            "rss_mb": float(group["rss_mb"]),
            "gpu_used_mb": float(group["gpu_used_mb"]),
            "gpu_total_mb": float(group["gpu_total_mb"]),
            "n_past": int(group["n_past"]),
            "detail": group["detail"],
        })

    decisions = list(DECISION_RE.finditer(text))
    chunks = []
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
    return text, events, chunks


def stage_stats(events):
    grouped = defaultdict(list)
    for event in events:
        if event["event"] == "end" and event["dur_ms"] >= 0:
            grouped[event["stage"]].append(event["dur_ms"])
    stats = {}
    for stage, values in grouped.items():
        stats[stage] = {
            "n": len(values),
            "avg": avg(values),
            "p50": statistics.median(values),
            "p90": pctl(values, 0.90),
            "min": min(values),
            "max": max(values),
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
        (170, 96, 160, 48, "stream_prefill", "vision + audio encode", "#dcecff"),
        (365, 96, 150, 48, "enqueue LLM", "queue item", "#d8f3dc"),
        (560, 96, 150, 48, "stream_decode", "wait + sample", "#fff2cc"),
        (760, 96, 150, 48, "decision", "LISTEN / SPEAK", "#ffe4e6"),
        (365, 222, 165, 48, "llm.kv_prefill", "write KV cache", "#e6dcff"),
        (560, 222, 165, 48, "decode sample", "first token / eos", "#ffd6a5"),
        (760, 347, 180, 48, "tts.llm_to_audio", "text tokens -> audio tokens", "#d0f4de"),
        (760, 472, 180, 48, "t2w.tokens_to_wav", "audio tokens -> wav", "#cde7ff"),
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
    width, height = 1080, 470
    left, top, bar_w, row_h = 270, 88, 650, 42
    max_value = max([stats.get(stage, {}).get("avg", 0) for stage, _ in STAGE_ROWS] + [260])
    body = ['<rect width="1080" height="470" fill="#ffffff"/>', '<text x="40" y="42" class="title">阶段耗时均值</text>', '<text x="40" y="66" class="subtitle">单位 ms；横条为平均值，灰线为 p50 到 max</text>']
    for tick in range(0, 301, 50):
        x = left + tick / max_value * bar_w
        body += [f'<line x1="{x:.1f}" y1="78" x2="{x:.1f}" y2="425" stroke="#edf1f7"/>', f'<text x="{x - 8:.1f}" y="444" class="tiny">{tick}</text>']
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
    lanes = [("main.prefill", ["stream_prefill", "prefill.vision_encode", "prefill.audio_encode"]), ("llm", ["llm.kv_prefill", "decode.wait_llm_prefill", "decode.llm_sample"]), ("tts", ["tts.llm_to_audio_tokens", "tts.enqueue_from_llm"]), ("t2w", ["t2w.tokens_to_wav"])]
    colors = {"stream_prefill":"#4c78a8","prefill.vision_encode":"#77aadd","prefill.audio_encode":"#72b7b2","llm.kv_prefill":"#b279a2","decode.wait_llm_prefill":"#f58518","decode.llm_sample":"#e45756","tts.enqueue_from_llm":"#8cd17d","tts.llm_to_audio_tokens":"#54a24b","t2w.tokens_to_wav":"#9ecae9"}
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


def write_csvs(out_dir: Path, stats, chunks):
    with (out_dir / "omni-duplex-stage-stats.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "n", "avg_ms", "p50_ms", "p90_ms", "min_ms", "max_ms"])
        for stage in sorted(stats):
            row = stats[stage]
            writer.writerow([stage, row["n"], f'{row["avg"]:.3f}', f'{row["p50"]:.3f}', f'{row["p90"]:.3f}', f'{row["min"]:.3f}', f'{row["max"]:.3f}'])
    with (out_dir / "omni-duplex-chunk-summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk", "prefill_ms", "decode_ms", "total_ms", "n_past", "decision", "text"])
        for chunk in chunks:
            writer.writerow([chunk["chunk"], f'{chunk["prefill_ms"]:.3f}', f'{chunk["decode_ms"]:.3f}', f'{chunk["total_ms"]:.3f}', chunk["n_past"], chunk["decision"], chunk["text"]])


def write_report(path: Path, log_path: Path, text: str, events, chunks, stats):
    decisions = Counter(c["decision"] for c in chunks)
    event_counts = Counter((e["stage"], e["event"]) for e in events)
    user_chunks = chunks[1:] if len(chunks) > 1 else chunks
    speak = [c for c in chunks if c["decision"] == "speak"]
    listen = [c for c in chunks if c["decision"] == "listen"]
    gpu = [e["gpu_used_mb"] for e in events if e["gpu_used_mb"] >= 0]
    rss = [e["rss_mb"] for e in events if e["rss_mb"] >= 0]
    def st(stage, key="avg"):
        return stats.get(stage, {}).get(key, 0.0)
    lines = [
        "# Omni Duplex TTS GPU 实测性能报告",
        "",
        f"- 日志：`{log_path.name}`",
        f"- 解析到 `{len(events)}` 条结构化 `[DUPLEX_PERF]` 事件；原始 marker 为 `{text.count('[DUPLEX_PERF]')}` 条。",
        f"- Chunk：`{len(chunks)}`；SPEAK/LISTEN：`{decisions.get('speak', 0)}` / `{decisions.get('listen', 0)}`。",
        f"- API 口径：平均 prefill `{avg(c['prefill_ms'] for c in chunks):.1f} ms`，平均 decode `{avg(c['decode_ms'] for c in chunks):.1f} ms`，平均每 chunk `{avg(c['total_ms'] for c in chunks):.1f} ms`。",
        f"- 用户 chunk 口径（排除 `index=0`）：平均 prefill `{avg(c['prefill_ms'] for c in user_chunks):.1f} ms`，平均 decode `{avg(c['decode_ms'] for c in user_chunks):.1f} ms`，平均每 chunk `{avg(c['total_ms'] for c in user_chunks):.1f} ms`。",
        f"- n_past：峰值 `{max(c['n_past'] for c in chunks)}`，最终 `{chunks[-1]['n_past']}`；RSS 约 `{min(rss):.0f}-{max(rss):.0f} MB`，GPU used 约 `{min(gpu):.0f}-{max(gpu):.0f} MB`。",
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
        "## 主要结论",
        "",
        f"1. `prefill.vision_encode` 是每轮 prefill 的主耗时，平均约 `{st('prefill.vision_encode'):.1f} ms`；`prefill.audio_encode` 平均约 `{st('prefill.audio_encode'):.1f} ms`。",
        f"2. LLM KV prefill 在后台线程执行，平均约 `{st('llm.kv_prefill'):.1f} ms`；API `decode` 通过 `decode.wait_llm_prefill` 等待它，因此当前 decode 不是纯采样耗时。",
        f"3. SPEAK 比 LISTEN 更慢：SPEAK chunk 的 decode 均值约 `{avg(c['decode_ms'] for c in speak):.1f} ms`，LISTEN chunk 约 `{avg(c['decode_ms'] for c in listen):.1f} ms`。",
        f"4. TTS/T2W 没有计入 `stream_decode` 完成时间。`tts.llm_to_audio_tokens` 平均约 `{st('tts.llm_to_audio_tokens'):.1f} ms`，`t2w.tokens_to_wav` 平均约 `{st('t2w.tokens_to_wav'):.1f} ms`，但它们与后续 chunk 重叠。",
        "5. 当前日志存在多线程输出交织，部分 `[DUPLEX_PERF]` 行被普通日志插入，导致 start/end 数量不完全一致。建议给性能日志加互斥或写入独立文件，并为 TTS/T2W 增加稳定的 `utterance_id` / `audio_chunk_id`。",
        "",
        "## 阶段耗时表",
        "",
        "| stage | n | avg ms | p50 ms | p90 ms | max ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage, _ in STAGE_ROWS:
        row = stats.get(stage, {})
        lines.append(f"| `{stage}` | {int(row.get('n', 0))} | {row.get('avg', 0):.1f} | {row.get('p50', 0):.1f} | {row.get('p90', 0):.1f} | {row.get('max', 0):.1f} |")
    lines += [
        "",
        "## 计时口径说明",
        "",
        "- `index=0` 是 session/ref audio 初始化，不能和普通用户音频 chunk 混在一起解释。",
        "- `test.chunk_prefill_api` / `stream_prefill` 主要覆盖 encoder 和入队；异步 LLM KV 写入大多体现在 `llm.kv_prefill` 和 `decode.wait_llm_prefill`。",
        "- TTS/T2W 是后台链路，图中的 TTS/T2W `chunk` 标签按日志原值展示，但从时间线看它可能对应上一轮已经入队的文本，不能作为严格因果 ID。",
        "- 对端到端首响、RTF 和尾包 flush 的精确测量，需要在 LLM enqueue、TTS audio token、T2W wav 输出之间增加同一个稳定请求 ID。",
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
    parser.add_argument("--out-dir", type=Path, default=Path("/cache/hanqingzhe/llama.cpp-omni/docs/development"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    text, events, chunks = parse_log(args.log)
    stats = stage_stats(events)
    figures = args.out_dir / "figures"
    pipeline_svg(figures / "omni-duplex-pipeline.svg")
    stage_latency_svg(figures / "omni-duplex-stage-latency.svg", stats)
    chunk_latency_svg(figures / "omni-duplex-chunk-latency.svg", chunks)
    overlap_svg(figures / "omni-duplex-overlap-timeline.svg", events)
    write_csvs(args.out_dir, stats, chunks)
    write_report(args.out_dir / "omni-duplex-tts-gpu-4-7-report.md", args.log, text, events, chunks, stats)
    print(f"parsed_events={len(events)} raw_markers={text.count('[DUPLEX_PERF]')} chunks={len(chunks)}")
    print(f"wrote={args.out_dir / 'omni-duplex-tts-gpu-4-7-report.md'}")
    print(f"figures={figures}")


if __name__ == "__main__":
    main()
