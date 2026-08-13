"""RTS 数据完整性与实时资格判定。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ValidityResult:
    fatal_reasons: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    realtime_reasons: List[Dict[str, Any]] = field(default_factory=list)
    n_core_candidates: int = 0
    n_core_valid: int = 0

    def fatal(self, code: str, *, cnt: Optional[int] = None, detail: str = "") -> None:
        item: Dict[str, Any] = {"code": code}
        if cnt is not None:
            item["cnt"] = int(cnt)
        if detail:
            item["detail"] = detail
        if item not in self.fatal_reasons:
            self.fatal_reasons.append(item)

    def warn(self, code: str, *, cnt: Optional[int] = None, detail: str = "") -> None:
        item: Dict[str, Any] = {"code": code}
        if cnt is not None:
            item["cnt"] = int(cnt)
        if detail:
            item["detail"] = detail
        if item not in self.warnings:
            self.warnings.append(item)

    def realtime(self, code: str, *, cnt: Optional[int] = None, detail: str = "") -> None:
        item: Dict[str, Any] = {"code": code}
        if cnt is not None:
            item["cnt"] = int(cnt)
        if detail:
            item["detail"] = detail
        if item not in self.realtime_reasons:
            self.realtime_reasons.append(item)

    def as_dict(self) -> Dict[str, Any]:
        data_valid = not self.fatal_reasons
        realtime_eligible = not self.realtime_reasons
        return {
            "data_valid": data_valid,
            "realtime_eligible": realtime_eligible,
            "eligible_for_pool": data_valid and realtime_eligible,
            "fatal_reasons": self.fatal_reasons,
            "realtime_reasons": self.realtime_reasons,
            "warnings": self.warnings,
            "n_core_candidates": self.n_core_candidates,
            "n_core_valid": self.n_core_valid,
        }


def _cnt(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _events_by_cnt(events: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for event in events:
        cnt = _cnt(event.get("src_cnt"))
        if cnt is not None:
            out.setdefault(cnt, []).append(event)
    return out


def evaluate_run_validity(
    *,
    chunks: List[Dict[str, Any]],
    stage: Dict[str, List[Dict[str, Any]]],
    rtf: Dict[str, Any],
    wav_integrity: Dict[str, Any],
    send_interval_ms: float = 1000.0,
    input_dropped_count: int = 0,
    causal_latencies_ms: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """根据完整输入时间线和 C++ 事件判定单视频执行是否可计分。"""
    result = ValidityResult()
    tts_events = stage.get("tts") or []
    t2w_events = stage.get("t2w") or []
    dequeue_events = stage.get("t2w_dequeue") or []
    tts_by_cnt = _events_by_cnt(tts_events)
    t2w_by_cnt = _events_by_cnt(t2w_events)

    observed_drops = max(
        [int(c.get("dropped_before") or 0) for c in chunks] + [input_dropped_count]
    )
    if observed_drops > 0:
        result.fatal("INPUT_CHUNK_DROPPED", detail=f"count={observed_drops}")

    for event in [*tts_events, *t2w_events]:
        if _cnt(event.get("src_cnt")) is None:
            result.fatal("SRC_CNT_MISSING", detail=f"event={event.get('event')}")

    if wav_integrity and not wav_integrity.get("ok", True):
        result.fatal(
            "WAV_INTEGRITY_MISMATCH",
            detail=(
                f"missing_in_poll={wav_integrity.get('missing_in_poll') or []} "
                f"missing_in_cpp={wav_integrity.get('missing_in_cpp') or []}"
            ),
        )

    chunk_cnts = {_cnt(c.get("cnt")) for c in chunks}
    for cnt in sorted((set(t2w_by_cnt) | set(tts_by_cnt)) - chunk_cnts):
        result.fatal("SRC_CNT_NOT_FOUND", cnt=cnt)

    wav_names = [event.get("wav") for event in t2w_events if event.get("wav")]
    if len(wav_names) != len(set(wav_names)):
        result.fatal("DUPLICATE_WAV_ID")

    for chunk in chunks:
        cnt = _cnt(chunk.get("cnt"))
        if cnt is None:
            result.fatal("SRC_CNT_MISSING", detail="chunk")
            continue
        stage_cnt = _cnt(chunk.get("stage_cnt"))
        if stage_cnt is None:
            result.fatal("SRC_CNT_MISSING", cnt=cnt, detail="decode metrics")
        elif stage_cnt != cnt:
            result.fatal(
                "SRC_CNT_MISMATCH",
                cnt=cnt,
                detail=f"stage_cnt={stage_cnt}",
            )
        if chunk.get("audio_expected") and not chunk.get("audio_ok", False):
            result.fatal("MEDIA_AUDIO_MISSING", cnt=cnt)
        if chunk.get("vision_expected") and not chunk.get("vision_ok", False):
            result.fatal("MEDIA_VISION_MISSING", cnt=cnt)
        if chunk.get("mode") == "SPEAK" and not t2w_by_cnt.get(cnt):
            result.fatal("SPEAK_NO_WAV", cnt=cnt)

    frames = rtf.get("frames") or []
    core_frames = [frame for frame in frames if frame.get("role") == "core"]
    result.n_core_candidates = len(core_frames)
    for frame in core_frames:
        cnt = _cnt(frame.get("cnt"))
        ok = True
        checks = (
            ("CORE_MODE_NOT_SPEAK", frame.get("mode") == "SPEAK"),
            ("CORE_CHUNK_COUNT_NOT_ONE", int(frame.get("n_chunk") or 0) == 1),
            ("CORE_TTS_COUNT_NOT_ONE", int(frame.get("n_tts") or 0) == 1),
            ("CORE_WAV_COUNT_NOT_ONE", int(frame.get("n_wav") or 0) == 1),
            ("CORE_WAV_NOT_24000_SAMPLES", int(frame.get("n_samples") or 0) == 24000),
            ("CORE_WAV_NOT_24000_HZ", int(frame.get("sample_rate") or 0) == 24000),
        )
        for code, passed in checks:
            if not passed:
                result.fatal(code, cnt=cnt)
                ok = False

        matching_tts = tts_by_cnt.get(cnt or -1, [])
        if matching_tts:
            event = matching_tts[0]
            generated = event.get("generated_audio_tokens")
            if generated is not None and int(generated) != 26:
                result.fatal(
                    "TTS_NONFINAL_TOKEN_COUNT_NOT_26",
                    cnt=cnt,
                    detail=f"actual={generated}",
                )
                ok = False
            if event.get("generation_ok") is False:
                result.fatal("TTS_GENERATION_FAILED", cnt=cnt)
                ok = False
        if ok:
            result.n_core_valid += 1

    for event in dequeue_events:
        cnt = _cnt(event.get("src_cnt"))
        if int(event.get("distinct_src_cnt") or 0) > 1:
            result.fatal("T2W_MIXED_SRC_BATCH", cnt=cnt)
        if event.get("cross_src_backlog"):
            result.realtime("T2W_CROSS_SRC_QUEUE_BACKLOG", cnt=cnt)
        wait_ms = float(event.get("oldest_wait_ms") or 0.0)
        if wait_ms >= send_interval_ms:
            result.realtime(
                "T2W_QUEUE_DEADLINE_MISS",
                cnt=cnt,
                detail=f"wait_ms={wait_ms:.3f} deadline_ms={send_interval_ms:.3f}",
            )

    for latency in causal_latencies_ms or []:
        if float(latency) < -5.0:
            result.fatal(
                "NEGATIVE_CAUSAL_LATENCY",
                detail=f"latency_ms={float(latency):.3f}",
            )

    return result.as_dict()
