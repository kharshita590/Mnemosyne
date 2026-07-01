from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memory.decay import compute_decay_weight


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TestComputeDecayWeight:
    def test_accessed_today_is_near_one(self):
        now = utc_now()
        weight = compute_decay_weight(
            created_at=now - timedelta(days=1),
            last_accessed_at=now,
            access_count=0,
        )
        assert weight > 0.9

    def test_not_accessed_in_30_days_is_near_half(self):
        now = utc_now()
        weight = compute_decay_weight(
            created_at=now - timedelta(days=60),
            last_accessed_at=now - timedelta(days=30),
            access_count=0,
        )
        # Half-life is 30 days: recency = 0.5, frequency_boost = 1, raw = 0.5
        assert 0.4 <= weight <= 0.6

    def test_high_access_count_boosts_weight(self):
        now = utc_now()
        stale = now - timedelta(days=15)

        low_access = compute_decay_weight(
            created_at=stale,
            last_accessed_at=stale,
            access_count=0,
        )
        high_access = compute_decay_weight(
            created_at=stale,
            last_accessed_at=stale,
            access_count=10,
        )
        assert high_access > low_access

    def test_weight_is_clamped_to_min_0_01(self):
        ancient = utc_now() - timedelta(days=3650)
        weight = compute_decay_weight(
            created_at=ancient,
            last_accessed_at=ancient,
            access_count=0,
        )
        assert weight >= 0.01

    def test_weight_never_exceeds_1(self):
        now = utc_now()
        weight = compute_decay_weight(
            created_at=now,
            last_accessed_at=now,
            access_count=9999,
        )
        assert weight <= 1.0

    def test_naive_datetime_handled(self):
        naive_now = datetime.utcnow()
        weight = compute_decay_weight(
            created_at=naive_now,
            last_accessed_at=None,
            access_count=0,
        )
        assert 0.01 <= weight <= 1.0
