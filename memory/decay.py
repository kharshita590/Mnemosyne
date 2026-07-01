from __future__ import annotations

import math
from datetime import datetime, timezone


def compute_decay_weight(
    created_at: datetime,
    last_accessed_at: datetime | None,
    access_count: int,
    half_life_days: float = 30.0,
) -> float:
    """
    Composite decay score combining:
    - Recency: exponential decay from last access (half-life = half_life_days)
    - Frequency: logarithmic boost from access count

    Formula:
        recency = exp(-lambda * days_since_access)   where lambda = ln(2) / half_life_days
        frequency = log(1 + access_count)
        final = clip(recency * frequency_boost, 0.01, 1.0)
    """
    now = datetime.now(timezone.utc)
    reference = last_accessed_at or created_at
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)

    days_since_access = max(0.0, (now - reference).total_seconds() / 86400)
    lam = math.log(2) / half_life_days
    recency = math.exp(-lam * days_since_access)

    frequency_boost = math.log1p(access_count)  # log(1 + n), = 0 when n=0

    raw = recency * (1.0 + frequency_boost * 0.1)
    return max(0.01, min(1.0, raw))
