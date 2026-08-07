import re
from pathlib import Path


# 在这里硬编码结果文件路径，按需增删
RESULT_FILES = [
    "tts_seed/results/q8_1.txt",
    "tts_seed/results/q8_2.txt",
#    "tts_seed/results/f16_3.txt",
]


PATTERNS = {
    "WER": re.compile(r"^WER:\s*([0-9]+(?:\.[0-9]+)?)%\s*$", re.MULTILINE),
    "WER_BELOW50": re.compile(
        r"^WER_BELOW50:\s*([0-9]+(?:\.[0-9]+)?)%\s*(?:,|$)", re.MULTILINE
    ),
    "WER_NORMALIZED": re.compile(
        r"^WER_NORMALIZED:\s*([0-9]+(?:\.[0-9]+)?)%\s*$", re.MULTILINE
    ),
    "ASV": re.compile(r"^ASV:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE),
    "ASV-var": re.compile(r"^ASV-var:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE),
}


def extract_metric(text: str, metric_name: str) -> float:
    match = PATTERNS[metric_name].search(text)
    if not match:
        raise ValueError(f"无法在文件中找到字段: {metric_name}")
    return float(match.group(1))


def main() -> None:
    sums = {name: 0.0 for name in PATTERNS}
    count = 0

    for file_path in RESULT_FILES:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        text = path.read_text(encoding="utf-8")
        for metric_name in PATTERNS:
            sums[metric_name] += extract_metric(text, metric_name)
        count += 1

    print(f"已统计文件数: {count}\n")
    print("字段汇总（sum）和平均（avg）:")
    for metric_name, total in sums.items():
        avg = total / count if count else 0.0
        if metric_name.startswith("WER"):
            print(f"- {metric_name}: sum={total:.3f}% , avg={avg:.3f}%")
        else:
            print(f"- {metric_name}: sum={total:.6f} , avg={avg:.6f}")


if __name__ == "__main__":
    main()
