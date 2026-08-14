import unittest

from stratified_sampling import select_video_ids


class StratifiedSamplingTest(unittest.TestCase):
    def test_half_preserves_every_videomme_stratum(self):
        records = []
        for duration in ("short", "medium", "long"):
            for sub_category in range(30):
                domain = sub_category // 5
                for index in range(10):
                    video_id = f"{duration}-{sub_category}-{index}"
                    records.append((video_id, duration, domain, sub_category))

        selected = set(select_video_ids(records, 0.5))

        self.assertEqual(len(selected), 450)
        for duration in ("short", "medium", "long"):
            for sub_category in range(30):
                prefix = f"{duration}-{sub_category}-"
                self.assertEqual(sum(video_id.startswith(prefix)
                                     for video_id in selected), 5)

    def test_uses_evenly_spaced_midpoints(self):
        records = [(str(index), "short", "domain", "category")
                   for index in range(10)]
        self.assertEqual(select_video_ids(records, 0.5),
                         ["1", "3", "5", "7", "9"])

    def test_largest_remainder_hits_global_target(self):
        records = [(f"a-{i}", "short", "a", "a") for i in range(3)]
        records += [(f"b-{i}", "short", "b", "b") for i in range(3)]
        self.assertEqual(len(select_video_ids(records, 0.5)), 3)

    def test_rejects_invalid_ratio(self):
        for ratio in (0, -0.1, 1.1):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                select_video_ids([], ratio)


if __name__ == "__main__":
    unittest.main()
