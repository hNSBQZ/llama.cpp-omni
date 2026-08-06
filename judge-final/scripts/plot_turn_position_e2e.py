#!/usr/bin/env python3
"""Aggregate within-turn SPEAK chunk e2e by position and plot curves.

For each session:
  - Align SPEAK chunk <-> wav via last_speak_chunk_idx (first wav per chunk)
  - Drop only trailing empty-text SPEAK chunks (turn-end flush)
  - Mid-turn empty text should not happen: warn if seen, but keep in order
  - Position 1 = first kept chunk in a speak turn (after trailing-empty strip)
  - Mean e2e across turns at each position

Writes JSON + SVG (stdlib only). Optional PNG if matplotlib is installed.

Example:
  .venv/bin/python scripts/plot_turn_position_e2e.py \\
    --case A=sessions/<session_id> \\
    --out-dir tmp/turn_position_curves
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from judge_support import display_path  # noqa: E402


def _is_empty_text(text: Any) -> bool:
    return not str(text or "").strip()


def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0}
    out: Dict[str, float] = {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 1),
        "median": round(statistics.median(vals), 1),
        "min": round(min(vals), 1),
        "max": round(max(vals), 1),
    }
    if len(vals) > 1:
        out["stdev"] = round(statistics.pstdev(vals), 1)
    else:
        out["stdev"] = 0.0
    return out


def load_speak_and_pairs(
    session_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (all SPEAK chunk events, wav-aligned SPEAK pairs)."""
    jsonl = session_dir / "e2e_timing.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"missing e2e_timing.jsonl: {jsonl}")

    chunks: List[Dict[str, Any]] = []
    wavs: List[Dict[str, Any]] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event") == "chunk":
            chunks.append(e)
        elif e.get("event") == "wav":
            wavs.append(e)

    speak_chunks: List[Dict[str, Any]] = []
    speak_by_idx: Dict[Any, Dict[str, Any]] = {}
    for c in chunks:
        if c.get("mode") != "SPEAK":
            continue
        text = c.get("text") or ""
        row = {
            "chunk_idx": c["chunk_idx"],
            "speak_turn_id": c.get("speak_turn_id"),
            "text": text,
            "empty": _is_empty_text(text),
        }
        speak_chunks.append(row)
        speak_by_idx[c["chunk_idx"]] = c

    pairs: List[Dict[str, Any]] = []
    seen = set()
    for w in wavs:
        idx = w.get("last_speak_chunk_idx")
        if idx is None or idx in seen:
            continue
        c = speak_by_idx.get(idx)
        if not c:
            continue
        seen.add(idx)
        text = c.get("text") or ""
        t_recv = float(c["t_recv_ms"])
        t_poll = float(w["t_poll_ms"])
        pairs.append(
            {
                "chunk_idx": idx,
                "speak_turn_id": c.get("speak_turn_id"),
                "text": text,
                "empty": _is_empty_text(text),
                "e2e": round(t_poll - t_recv, 1),
                "wall": c.get("wall_clock_ms"),
            }
        )
    return speak_chunks, pairs


def _strip_trailing_empty(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    end = len(rows)
    while end > 0 and rows[end - 1]["empty"]:
        end -= 1
    return rows[:end]


def _mid_turn_empty_warnings(
    speak_chunks: List[Dict[str, Any]], case_id: str
) -> List[str]:
    """Warn if an empty SPEAK chunk is followed by a nonempty one in the same turn."""
    by_turn: Dict[Any, List[Dict[str, Any]]] = {}
    for r in speak_chunks:
        by_turn.setdefault(r["speak_turn_id"], []).append(r)

    warnings: List[str] = []
    for tid in sorted(by_turn.keys(), key=lambda x: (x is None, x)):
        rows = sorted(by_turn[tid], key=lambda r: r["chunk_idx"])
        for i, r in enumerate(rows):
            if not r["empty"]:
                continue
            later_nonempty = [x for x in rows[i + 1 :] if not x["empty"]]
            if not later_nonempty:
                continue  # trailing empty (flush) — expected
            nxt = later_nonempty[0]
            warnings.append(
                f"[WARN] {case_id}: mid-turn empty text "
                f"turn={tid} chunk_idx={r['chunk_idx']} "
                f"before nonempty chunk_idx={nxt['chunk_idx']} "
                f"text={nxt['text'][:40]!r} — kept in position order"
            )
    return warnings


def case_curves(
    session_dir: Path, label: str, case_id: str = ""
) -> Tuple[Dict[str, Any], List[str]]:
    speak_chunks, pairs = load_speak_and_pairs(session_dir)
    warnings = _mid_turn_empty_warnings(speak_chunks, case_id or label)

    by_turn: Dict[Any, List[Dict[str, Any]]] = {}
    for p in pairs:
        by_turn.setdefault(p["speak_turn_id"], []).append(p)

    per_turn: List[Dict[str, Any]] = []
    pos_e2e: Dict[int, List[float]] = {}
    pos_wall: Dict[int, List[float]] = {}

    for tid in sorted(by_turn.keys(), key=lambda x: (x is None, x)):
        rows = sorted(by_turn[tid], key=lambda r: r["chunk_idx"])
        # Only drop turn-end flush empties; keep any mid-turn empty in order.
        kept = _strip_trailing_empty(rows)
        if not kept:
            continue
        e2e_by_pos = [r["e2e"] for r in kept]
        texts = [("" if r["empty"] else r["text"][:40]) for r in kept]
        per_turn.append(
            {
                "turn": tid,
                "e2e_by_pos": e2e_by_pos,
                "texts": texts,
                "empty_flags": [bool(r["empty"]) for r in kept],
            }
        )
        for i, r in enumerate(kept, start=1):
            pos_e2e.setdefault(i, []).append(r["e2e"])
            if r.get("wall") is not None:
                pos_wall.setdefault(i, []).append(float(r["wall"]))

    positions = []
    for pos in sorted(pos_e2e.keys()):
        e_stats = _stats(pos_e2e[pos])
        w_stats = _stats(pos_wall.get(pos, []))
        positions.append(
            {
                "pos": pos,
                "n": e_stats["n"],
                "e2e_mean": e_stats.get("mean"),
                "e2e_stdev": e_stats.get("stdev"),
                "e2e_median": e_stats.get("median"),
                "wall_mean": w_stats.get("mean"),
                "values": pos_e2e[pos],
            }
        )

    return (
        {
            "label": label,
            "session": display_path(session_dir),
            "n_turns": len(per_turn),
            "positions": positions,
            "per_turn": per_turn,
            "mid_turn_empty_warnings": warnings,
        },
        warnings,
    )


def write_svg(
    cases: Dict[str, Dict[str, Any]],
    out_path: Path,
    title: str,
    y_min: float = 550.0,
    y_max: float = 900.0,
) -> None:
    # Distinct colors (colorblind-friendlier-ish)
    palette = ["#CF2D56", "#D75C4E", "#3685BF", "#1F8A65", "#7754D9", "#C08532"]
    max_pos = max(
        (p["pos"] for c in cases.values() for p in c["positions"]),
        default=1,
    )
    W, H = 900, 480
    pad = {"l": 64, "r": 24, "t": 56, "b": 64}
    iw = W - pad["l"] - pad["r"]
    ih = H - pad["t"] - pad["b"]

    def x_of(pos: int) -> float:
        if max_pos <= 1:
            return pad["l"] + iw / 2
        return pad["l"] + (pos - 1) / (max_pos - 1) * iw

    def y_of(v: float) -> float:
        return pad["t"] + (y_max - v) / (y_max - y_min) * ih

    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        width=str(W),
        height=str(H),
        viewBox=f"0 0 {W} {H}",
    )
    ET.SubElement(svg, "rect", x="0", y="0", width=str(W), height=str(H), fill="#FFFFFF")

    title_el = ET.SubElement(
        svg,
        "text",
        x=str(W / 2),
        y="28",
        fill="#141414",
        style="font-family:sans-serif;font-size:16px;font-weight:600",
        **{"text-anchor": "middle"},
    )
    title_el.text = title

    # grid + axes
    for v in range(int(y_min), int(y_max) + 1, 50):
        y = y_of(float(v))
        ET.SubElement(
            svg,
            "line",
            x1=str(pad["l"]),
            x2=str(W - pad["r"]),
            y1=str(y),
            y2=str(y),
            stroke="#E5E5E5",
            **{"stroke-width": "1"},
        )
        t = ET.SubElement(
            svg,
            "text",
            x=str(pad["l"] - 10),
            y=str(y + 4),
            fill="#666666",
            style="font-family:sans-serif;font-size:11px",
            **{"text-anchor": "end"},
        )
        t.text = str(v)

    for pos in range(1, max_pos + 1):
        t = ET.SubElement(
            svg,
            "text",
            x=str(x_of(pos)),
            y=str(H - 28),
            fill="#666666",
            style="font-family:sans-serif;font-size:12px",
            **{"text-anchor": "middle"},
        )
        t.text = str(pos)

    xlab = ET.SubElement(
        svg,
        "text",
        x=str(W / 2),
        y=str(H - 10),
        fill="#888888",
        style="font-family:sans-serif;font-size:12px",
        **{"text-anchor": "middle"},
    )
    xlab.text = "within-turn chunk position (1=first; trailing empty stripped)"

    ylab = ET.SubElement(
        svg,
        "text",
        x="18",
        y=str(H / 2),
        fill="#888888",
        style="font-family:sans-serif;font-size:12px",
        transform=f"rotate(-90 18 {H / 2})",
        **{"text-anchor": "middle"},
    )
    ylab.text = "e2e mean (ms)  recv->wav poll"

    # series
    legend_x = pad["l"]
    legend_y = 44
    for i, (cid, case) in enumerate(cases.items()):
        color = palette[i % len(palette)]
        pts = case["positions"]
        if not pts:
            continue
        d = []
        for j, p in enumerate(pts):
            x, y = x_of(int(p["pos"])), y_of(float(p["e2e_mean"]))
            d.append(f"{'M' if j == 0 else 'L'} {x:.1f} {y:.1f}")
            ET.SubElement(
                svg,
                "circle",
                cx=f"{x:.1f}",
                cy=f"{y:.1f}",
                r="4",
                fill="#FFFFFF",
                stroke=color,
                **{"stroke-width": "2"},
            )
        ET.SubElement(
            svg,
            "path",
            d=" ".join(d),
            fill="none",
            stroke=color,
            **{"stroke-width": "2.5"},
        )
        lx = legend_x + i * 170
        ET.SubElement(
            svg,
            "line",
            x1=str(lx),
            x2=str(lx + 18),
            y1=str(legend_y),
            y2=str(legend_y),
            stroke=color,
            **{"stroke-width": "2.5"},
        )
        lt = ET.SubElement(
            svg,
            "text",
            x=str(lx + 24),
            y=str(legend_y + 4),
            fill="#141414",
            style="font-family:sans-serif;font-size:12px",
        )
        lt.text = f"{cid}: {case['label']}"

    # caption
    cap = ET.SubElement(
        svg,
        "text",
        x=str(pad["l"]),
        y=str(H - 48),
        fill="#999999",
        style="font-family:sans-serif;font-size:10px",
    )
    cap.text = (
        "Trailing empty-text SPEAK chunks stripped. "
        "Mid-turn empty (if any) is kept and warned."
    )

    tree = ET.ElementTree(svg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def try_write_png(
    cases: Dict[str, Dict[str, Any]],
    out_path: Path,
    title: str,
    y_min: float,
    y_max: float,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=140)
    colors = ["#CF2D56", "#D75C4E", "#3685BF", "#1F8A65", "#7754D9", "#C08532"]
    all_xs: List[int] = []
    for i, (cid, case) in enumerate(cases.items()):
        xs = [int(p["pos"]) for p in case["positions"]]
        ys = [p["e2e_mean"] for p in case["positions"]]
        if not xs:
            continue
        all_xs.extend(xs)
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            color=colors[i % len(colors)],
            label=f"{cid}: {case['label']}",
        )
    ax.set_xlabel("within-turn chunk position (1=first; trailing empty stripped)")
    ax.set_ylabel("e2e mean (ms)  recv→wav poll")
    ax.set_title(title)
    ax.set_ylim(y_min, y_max)
    if all_xs:
        ticks = list(range(min(all_xs), max(all_xs) + 1))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def parse_case(arg: str) -> Tuple[str, Path]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"--case expects ID=path, got: {arg!r}"
        )
    cid, path = arg.split("=", 1)
    cid = cid.strip()
    p = Path(path.strip())
    if not cid:
        raise argparse.ArgumentTypeError("empty case id")
    if not p.exists():
        raise argparse.ArgumentTypeError(f"session not found: {p}")
    return cid, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--case",
        action="append",
        required=True,
        type=parse_case,
        help="Case as ID=session_dir (repeatable)",
    )
    ap.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional ID=label override (repeatable)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp/bench_quant/turn_position_curves"),
    )
    ap.add_argument("--title", default="Speak turn position e2e curves")
    ap.add_argument("--y-min", type=float, default=550.0)
    ap.add_argument("--y-max", type=float, default=900.0)
    args = ap.parse_args()

    label_map: Dict[str, str] = {}
    for item in args.label:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        label_map[k.strip()] = v.strip()

    cases: Dict[str, Dict[str, Any]] = {}
    all_warnings: List[str] = []
    for cid, session in args.case:
        label = label_map.get(cid, session.name)
        case, warns = case_curves(session, label, case_id=cid)
        cases[cid] = case
        all_warnings.extend(warns)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "turn_position_curves.json"
    svg_path = out_dir / "turn_position_e2e.svg"
    png_path = out_dir / "turn_position_e2e.png"

    payload = {
        "metric": "e2e_recv_to_wav_poll_ms",
        "rule": (
            "strip trailing empty-text SPEAK chunks only; "
            "mid-turn empty text is unexpected: warn and keep in position order"
        ),
        "title": args.title,
        "cases": cases,
        "mid_turn_empty_warnings": all_warnings,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_svg(cases, svg_path, args.title, args.y_min, args.y_max)
    png_ok = try_write_png(cases, png_path, args.title, args.y_min, args.y_max)

    for w in all_warnings:
        print(w)
    if not all_warnings:
        print("[ok] no mid-turn empty text in any case")

    print(f"[ok] json -> {json_path}")
    print(f"[ok] svg  -> {svg_path}")
    if png_ok:
        print(f"[ok] png  -> {png_path}")
    else:
        print("[note] matplotlib not installed; skipped PNG (SVG written)")

    # compact table
    print("\npositions (pos: n, e2e_mean):")
    for cid, case in cases.items():
        row = ", ".join(
            f"{p['pos']}:{p['n']}@{p['e2e_mean']}" for p in case["positions"]
        )
        print(f"  {cid} ({case['label']}): {row}")


if __name__ == "__main__":
    main()
