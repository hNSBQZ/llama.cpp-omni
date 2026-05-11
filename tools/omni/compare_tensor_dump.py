#!/usr/bin/env python3
"""Compare two plain-text tensor dumps produced by dumptensor()."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def load_tensor(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    expected_dim: int | None = None

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            try:
                row = [float(part) for part in parts]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: non-numeric value in tensor dump") from exc

            if expected_dim is None:
                expected_dim = len(row)
                if expected_dim == 0:
                    raise ValueError(f"{path}:{line_no}: empty row")
            elif len(row) != expected_dim:
                raise ValueError(
                    f"{path}:{line_no}: row dim {len(row)} != expected dim {expected_dim}"
                )

            rows.append(row)

    if not rows:
        raise ValueError(f"{path}: empty tensor dump")
    return rows


def shape(rows: list[list[float]]) -> tuple[int, int]:
    return len(rows), len(rows[0])


def compare(lhs: list[list[float]], rhs: list[list[float]]) -> tuple[float, float, float, float, tuple[int, int]]:
    diffs: list[float] = []
    max_pos = (0, 0)
    max_diff = -1.0

    for i, (lhs_row, rhs_row) in enumerate(zip(lhs, rhs)):
        for j, (lhs_val, rhs_val) in enumerate(zip(lhs_row, rhs_row)):
            diff = abs(lhs_val - rhs_val)
            diffs.append(diff)
            if diff > max_diff:
                max_diff = diff
                max_pos = (i, j)

    mean = sum(diffs) / len(diffs)
    min_diff = min(diffs)
    variance = sum((diff - mean) ** 2 for diff in diffs) / len(diffs)
    std = math.sqrt(variance)
    return mean, max_diff, min_diff, std, max_pos


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two dumptensor text files and print absolute-difference stats."
    )
    parser.add_argument("lhs", type=Path, help="first tensor dump")
    parser.add_argument("rhs", type=Path, help="second tensor dump")
    args = parser.parse_args()

    try:
        lhs = load_tensor(args.lhs)
        rhs = load_tensor(args.rhs)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    lhs_shape = shape(lhs)
    rhs_shape = shape(rhs)
    print(f"lhs shape: rows={lhs_shape[0]} dim={lhs_shape[1]}")
    print(f"rhs shape: rows={rhs_shape[0]} dim={rhs_shape[1]}")

    if lhs_shape != rhs_shape:
        print(f"error: shape mismatch: {lhs_shape} != {rhs_shape}", file=sys.stderr)
        return 2

    mean, max_diff, min_diff, std, max_pos = compare(lhs, rhs)
    print("absolute diff:")
    print(f"  mean: {mean:.10g}")
    print(f"  max:  {max_diff:.10g} at row={max_pos[0]} col={max_pos[1]}")
    print(f"  min:  {min_diff:.10g}")
    print(f"  std:  {std:.10g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
