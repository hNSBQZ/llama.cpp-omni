"""
Daily-Omni 评测评分脚本。

按 qa_type、content_parent_category、video_category、video_duration 分组统计准确率。
"""
import json
import re
import argparse
from typing import Dict, List, Optional


def extract_characters_regex(s: str) -> str:
    """从模型输出中提取 A/B/C/D 答案（对齐 evalkit MQAEvaluator 逻辑）。"""
    s = s.strip()
    if not s:
        return ""

    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for prefix in answer_prefixes:
        s = s.replace(prefix, "")

    s = s.replace("<|tts_eos|>", "").strip()

    cleaned = s.rstrip(".").strip()
    if len(cleaned) == 1 and cleaned.upper() in "ABCD":
        return cleaned.upper()

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ""
    return matches[0]


def eval_daily_omni_results(
    results_path: str,
    group_by: Optional[List[str]] = None,
):
    """
    评测 Daily-Omni 结果。

    Args:
        results_path: pipeline 输出的 JSON 文件路径
        group_by: 分组统计的字段列表，默认全部
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    if not predictions:
        print("No predictions found.")
        return

    if group_by is None:
        group_by = ["qa_type", "content_parent_category", "video_category", "video_duration"]

    total_correct = 0
    total_answered = 0
    group_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

    for field in group_by:
        group_stats[field] = {}

    for pred in predictions:
        prediction = pred.get("prediction", "")
        annotation = pred.get("annotation", {})
        gt_answer = annotation.get("gt_answer", "")

        if not prediction:
            prediction = extract_characters_regex(pred.get("raw_response", ""))

        if prediction:
            total_answered += 1
            is_correct = prediction == gt_answer
            if is_correct:
                total_correct += 1

            for field in group_by:
                group_val = annotation.get(field, "unknown")
                if group_val not in group_stats[field]:
                    group_stats[field][group_val] = {"correct": 0, "answered": 0}
                group_stats[field][group_val]["answered"] += 1
                if is_correct:
                    group_stats[field][group_val]["correct"] += 1

    # 打印结果
    print("=" * 60)
    print("Daily-Omni Evaluation Results")
    print("=" * 60)

    for field in group_by:
        print(f"\n--- By {field} ---")
        stats = group_stats[field]
        for key in sorted(stats.keys()):
            s = stats[key]
            acc = 100.0 * s["correct"] / s["answered"] if s["answered"] > 0 else 0.0
            print(f"  {key}: {acc:.1f}% ({s['correct']}/{s['answered']})")

    print("\n--- Overall ---")
    acc = 100.0 * total_correct / total_answered if total_answered > 0 else 0.0
    print(f"  Total: {acc:.1f}% ({total_correct}/{total_answered})")
    print(f"  Total samples: {len(predictions)}")
    if total_answered < len(predictions):
        print(f"  Unanswered: {len(predictions) - total_answered}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily-Omni Evaluation Scoring")
    parser.add_argument("--results-file", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--group-by", type=str, nargs="*", default=None,
                        help="Fields to group by (default: all)")
    args = parser.parse_args()
    eval_daily_omni_results(args.results_file, group_by=args.group_by)
