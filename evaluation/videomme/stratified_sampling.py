"""Deterministic video-level sampling for Video-MME."""

from collections import defaultdict
from math import floor
from typing import Any, Dict, Iterable, List, Tuple


VideoRecord = Tuple[Any, Any, Any, Any]


def select_video_ids(records: Iterable[VideoRecord], ratio: float) -> List[Any]:
    """Select a proportional, evenly spaced sample from each dataset stratum.

    Each record is ``(video_id, duration, domain, sub_category)``. Quotas use
    largest-remainder allocation so the total is exactly ``round(N * ratio)``.
    Within each stratum, samples are taken from the midpoints of equal-width
    intervals in the original dataset order.
    """
    if not 0 < ratio <= 1:
        raise ValueError(f"sample ratio must be in (0, 1], got {ratio}")

    strata: Dict[Tuple[Any, Any, Any], List[Any]] = defaultdict(list)
    seen = set()
    total = 0
    for video_id, duration, domain, sub_category in records:
        if video_id in seen:
            raise ValueError(f"duplicate video_id: {video_id}")
        seen.add(video_id)
        strata[(duration, domain, sub_category)].append(video_id)
        total += 1

    if ratio == 1 or total == 0:
        return [video_id for ids in strata.values() for video_id in ids]

    target = max(1, floor(total * ratio + 0.5))
    quotas = {}
    ranked_remainders = []
    assigned = 0
    for order, (key, ids) in enumerate(strata.items()):
        exact = len(ids) * ratio
        quota = floor(exact)
        quotas[key] = quota
        assigned += quota
        ranked_remainders.append((exact - quota, order, key))

    ranked_remainders.sort(key=lambda item: (-item[0], item[1]))
    for _, _, key in ranked_remainders[:target - assigned]:
        quotas[key] += 1

    selected = []
    for key, ids in strata.items():
        quota = quotas[key]
        if quota == len(ids):
            selected.extend(ids)
            continue
        # Midpoints of equal-width intervals cover the full stratum without
        # depending on random state.
        selected.extend(ids[((2 * i + 1) * len(ids)) // (2 * quota)]
                        for i in range(quota))
    return selected
