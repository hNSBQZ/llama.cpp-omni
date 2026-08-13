from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


JUDGE_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = JUDGE_ROOT.parent
sys.path.insert(0, str(JUDGE_ROOT))
sys.path.insert(0, str(EVALUATION_ROOT))

from judge_support import build_batch_pooled_report  # noqa: E402
from run_validity import evaluate_run_validity  # noqa: E402
import run_eval  # noqa: E402


def _valid_inputs():
    chunks = [{"cnt": 1, "stage_cnt": 1, "mode": "SPEAK", "dropped_before": 0}]
    stage = {
        "tts": [{
            "event": "tts",
            "src_cnt": 1,
            "generated_audio_tokens": 26,
            "generation_ok": True,
        }],
        "t2w": [{
            "event": "t2w",
            "src_cnt": 1,
            "n_samples": 24000,
            "sample_rate": 24000,
        }],
        "t2w_dequeue": [{
            "event": "t2w_dequeue",
            "src_cnt": 1,
            "distinct_src_cnt": 1,
            "oldest_wait_ms": 10.0,
        }],
    }
    rtf = {
        "frames": [{
            "cnt": 1,
            "role": "core",
            "mode": "SPEAK",
            "n_chunk": 1,
            "n_tts": 1,
            "n_wav": 1,
            "n_samples": 24000,
            "sample_rate": 24000,
        }]
    }
    return chunks, stage, rtf


class ValidityTest(unittest.TestCase):
    def test_valid_core_is_eligible(self):
        chunks, stage, rtf = _valid_inputs()
        result = evaluate_run_validity(
            chunks=chunks,
            stage=stage,
            rtf=rtf,
            wav_integrity={"ok": True},
        )
        self.assertTrue(result["eligible_for_pool"])
        self.assertEqual(result["n_core_valid"], 1)

    def test_deadline_only_removes_realtime_eligibility(self):
        chunks, stage, rtf = _valid_inputs()
        stage["t2w_dequeue"][0]["oldest_wait_ms"] = 1000.0
        result = evaluate_run_validity(
            chunks=chunks,
            stage=stage,
            rtf=rtf,
            wav_integrity={"ok": True},
        )
        self.assertTrue(result["data_valid"])
        self.assertFalse(result["realtime_eligible"])

    def test_core_shape_failure_is_fatal(self):
        chunks, stage, rtf = _valid_inputs()
        rtf["frames"][0]["n_wav"] = 2
        result = evaluate_run_validity(
            chunks=chunks,
            stage=stage,
            rtf=rtf,
            wav_integrity={"ok": True},
        )
        self.assertFalse(result["data_valid"])
        self.assertIn(
            "CORE_WAV_COUNT_NOT_ONE",
            {reason["code"] for reason in result["fatal_reasons"]},
        )


class BatchPoolingTest(unittest.TestCase):
    @staticmethod
    def _report(compute_ms: float, *, eligible: bool = True):
        return {
            "session_dir": "session",
            "validity": {
                "data_valid": eligible,
                "realtime_eligible": eligible,
                "eligible_for_pool": eligible,
            },
            "rtf": {
                "available": True,
                "n_wav": 1,
                "frames": [{
                    "role": "core",
                    "audio_ms": 1000.0,
                    "compute_ms": compute_ms,
                    "encode_ms": 100.0,
                    "llm_prefill_ms": 10.0,
                    "llm_decode_ms": 200.0,
                    "tts_ms": 300.0,
                    "token2wav_ms": compute_ms - 610.0,
                }],
                "core": {"available": True, "n_turns": 1},
            },
        }

    def test_batch_rtf_is_pooled_by_frame(self):
        report = build_batch_pooled_report(
            [self._report(1000.0), self._report(2000.0)],
            min_core_frames=2,
            batch_id="batch",
        )
        self.assertEqual(report["batch_core_rtf"], 1.5)
        self.assertTrue(report["batch_validity"]["score_eligible"])

    def test_invalid_video_invalidates_batch(self):
        report = build_batch_pooled_report(
            [self._report(1000.0), self._report(2000.0, eligible=False)],
            min_core_frames=1,
            batch_id="batch",
        )
        self.assertIsNone(report["batch_core_rtf"])
        self.assertFalse(report["batch_validity"]["data_valid"])


class ConfigTest(unittest.TestCase):
    def test_rts_counts_limit_devices_and_videos(self):
        env = {
            "RTS_DEVICE_IDS": "0,1,2,3",
            "RTS_DEVICE_COUNT": "2",
            "RTS_VIDEO": "a.mp4 b.mp4 c.mp4",
            "RTS_VIDEO_COUNT": "2",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(run_eval.rts_device_ids(), ["0", "1"])
            self.assertEqual(run_eval.rts_videos(), ["a.mp4", "b.mp4"])

    def test_round_robin_assignment_is_deterministic(self):
        self.assertEqual(
            run_eval.split_round_robin(["a", "b", "c", "d", "e"], 2),
            [["a", "c", "e"], ["b", "d"]],
        )

    def test_video_directory_is_sorted_and_filterable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.mp4").write_bytes(b"")
            (root / "a.mov").write_bytes(b"")
            (root / "ignore.txt").write_text("x")
            nested = root / "nested"
            nested.mkdir()
            (nested / "c.mkv").write_bytes(b"")
            env = {
                "RTS_VIDEO_DIR": str(root),
                "RTS_VIDEO_COUNT": "2",
            }
            with patch.dict(os.environ, env, clear=False):
                videos = run_eval.rts_videos()
            self.assertEqual(
                [Path(video).name for video in videos],
                ["a.mov", "b.mp4"],
            )

    def test_prechunked_test_case_directory_has_highest_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_b = root / "b_test_case"
            case_a = root / "a_test_case"
            for case, prefix in ((case_b, "b"), (case_a, "a")):
                case.mkdir()
                (case / f"{prefix}_test_case_0000.wav").write_bytes(b"wav")
                (case / f"{prefix}_test_case_0000.jpg").write_bytes(b"jpg")
            env = {
                "RTS_TEST_CASE_DIR": str(root),
                "RTS_VIDEO_DIR": "",
                "RTS_VIDEO_COUNT": "0",
            }
            with patch.dict(os.environ, env, clear=False):
                cases = run_eval.rts_videos()
                kind = run_eval.rts_input_kind()
            self.assertEqual(
                [Path(case).name for case in cases],
                ["a_test_case", "b_test_case"],
            )
            self.assertEqual(kind, "test_case")

    def test_four_round_rotation_covers_every_video_on_every_worker(self):
        videos = [str(i) for i in range(1, 11)]
        groups = run_eval.split_balanced_contiguous(videos, 4)
        self.assertEqual(
            groups,
            [["1", "2", "3"], ["4", "5", "6"], ["7", "8"], ["9", "10"]],
        )
        seen = {worker_id: [] for worker_id in range(4)}
        for round_index in range(4):
            assignments = run_eval.rotated_group_assignments(
                groups, round_index
            )
            for worker_id, assigned in enumerate(assignments):
                seen[worker_id].extend(assigned)
        for assigned in seen.values():
            self.assertEqual(sorted(assigned, key=int), videos)


if __name__ == "__main__":
    unittest.main()
