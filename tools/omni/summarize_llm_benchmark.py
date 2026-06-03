#!/usr/bin/env python3
"""Summarize llama-bench JSONL output for the omni LLM benchmark wrapper."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TEST_MEANINGS = {
    "pp128": "prefill 128 tokens",
    "pp512": "prefill 512 tokens",
    "pp2048": "prefill 2048 tokens",
    "tg32": "decode 32 tokens",
    "tg128": "decode 128 tokens",
    "tg256": "decode 256 tokens",
}

TEST_ORDER = {
    "pp128": 0,
    "pp512": 1,
    "pp2048": 2,
    "tg32": 3,
    "tg128": 4,
    "tg256": 5,
}


@dataclass(frozen=True)
class ModelInfo:
    name: str
    gpu: str
    path: str
    safe_name: str


@dataclass(frozen=True)
class BenchRow:
    model: str
    gpu: str
    test: str
    avg_ts: float
    stddev_ts: float
    meaning: str
    raw_file: str


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_base = script_dir / "output" / "llm_bench"

    parser = argparse.ArgumentParser(
        description="Create a Markdown summary from run_llm_benchmark.sh JSONL outputs.",
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Benchmark run directory. Defaults to the latest directory under tools/omni/output/llm_bench.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base,
        help=f"Base directory used when run_dir is omitted. Default: {default_base}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output Markdown path. Default: <run_dir>/summary.md",
    )
    return parser.parse_args()


def latest_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise SystemExit(f"benchmark base dir not found: {base_dir}")

    candidates = [path for path in base_dir.iterdir() if path.is_dir() and (path / "raw").is_dir()]
    if not candidates:
        raise SystemExit(f"no benchmark run directories found under: {base_dir}")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_models(run_dir: Path) -> dict[str, ModelInfo]:
    models_path = run_dir / "models.txt"
    if not models_path.exists():
        return {}

    models: dict[str, ModelInfo] = {}
    with models_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line or line_no == 1:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            name = parts[0]
            gpu = parts[1]
            path = parts[2]
            safe_name = parts[3] if len(parts) >= 4 else Path(path).stem
            models[safe_name] = ModelInfo(name=name, gpu=gpu, path=path, safe_name=safe_name)

    return models


def iter_jsonl_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def test_name(record: dict) -> str:
    n_prompt = int(record.get("n_prompt") or 0)
    n_gen = int(record.get("n_gen") or 0)
    n_depth = int(record.get("n_depth") or 0)

    if n_prompt > 0 and n_gen == 0:
        name = f"pp{n_prompt}"
    elif n_gen > 0 and n_prompt == 0:
        name = f"tg{n_gen}"
    else:
        name = f"pp{n_prompt}+tg{n_gen}"

    if n_depth > 0:
        name = f"{name}@d{n_depth}"
    return name


def format_rate(value: float) -> str:
    return f"{value:.2f}"


def collect_rows(run_dir: Path, models: dict[str, ModelInfo]) -> list[BenchRow]:
    raw_dir = run_dir / "raw"
    if not raw_dir.is_dir():
        raise SystemExit(f"raw directory not found: {raw_dir}")

    rows: list[BenchRow] = []
    for jsonl_path in sorted(raw_dir.glob("*.jsonl")):
        safe_name = jsonl_path.stem
        model_info = models.get(
            safe_name,
            ModelInfo(name=safe_name, gpu="", path="", safe_name=safe_name),
        )

        for record in iter_jsonl_records(jsonl_path):
            test = test_name(record)
            avg_ts = float(record.get("avg_ts") or 0.0)
            stddev_ts = float(record.get("stddev_ts") or 0.0)
            rows.append(
                BenchRow(
                    model=model_info.name,
                    gpu=model_info.gpu,
                    test=test,
                    avg_ts=avg_ts,
                    stddev_ts=stddev_ts,
                    meaning=TEST_MEANINGS.get(test, ""),
                    raw_file=jsonl_path.name,
                )
            )

    if not rows:
        raise SystemExit(f"no JSONL benchmark records found under: {raw_dir}")

    rows.sort(key=lambda row: (TEST_ORDER.get(row.test, 10_000), row.test, -row.avg_ts, row.model))
    return rows


def render_markdown(run_dir: Path, rows: list[BenchRow]) -> str:
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# LLM Benchmark Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: `{generated_at}`",
        "- sort: each test group is ordered by `tokens/s` from fast to slow",
        "",
    ]

    for test in sorted({row.test for row in rows}, key=lambda item: (TEST_ORDER.get(item, 10_000), item)):
        test_rows = sorted(
            (row for row in rows if row.test == test),
            key=lambda row: (-row.avg_ts, row.model),
        )
        meaning = TEST_MEANINGS.get(test, "-")
        lines.extend(
            [
                f"## {test}",
                "",
                f"{meaning}.",
                "",
                "| rank | model | gpu | tokens/s |",
                "| ---: | --- | ---: | ---: |",
            ]
        )
        for rank, row in enumerate(test_rows, start=1):
            gpu = row.gpu if row.gpu else "-"
            tokens_per_second = f"{format_rate(row.avg_ts)} ± {format_rate(row.stddev_ts)}"
            lines.append(f"| {rank} | `{row.model}` | `{gpu}` | `{tokens_per_second}` |")
        lines.append("")

    lines.extend(
        [
            "## Raw Files",
            "",
        ]
    )
    for raw_file in sorted({row.raw_file for row in rows}):
        lines.append(f"- `raw/{raw_file}`")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else latest_run_dir(args.base_dir).resolve()
    output_path = args.output.resolve() if args.output else run_dir / "summary.md"

    models = read_models(run_dir)
    rows = collect_rows(run_dir, models)
    output_path.write_text(render_markdown(run_dir, rows), encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
