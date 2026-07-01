from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from retrieval.rrf import reciprocal_rank_fusion


def make_memory(content: str = "test") -> MagicMock:
    mem = MagicMock()
    mem.id = uuid.uuid4()
    mem.content = content
    return mem


class TestReciprocalRankFusion:
    def test_empty_input_returns_empty(self):
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_rank_1_in_both_lists_scores_highest(self):
        top = make_memory("top")
        middle = make_memory("middle")
        bottom = make_memory("bottom")

        list1 = [(top, 0.9), (middle, 0.5), (bottom, 0.1)]
        list2 = [(top, 0.8), (bottom, 0.4), (middle, 0.2)]

        result = reciprocal_rank_fusion(list1, list2)
        assert result[0][0].id == top.id

    def test_rank1_single_beats_rank10_both(self):
        rank1_single = make_memory("rank1_single")
        rank10_both = make_memory("rank10_both")
        other = [make_memory(f"other_{i}") for i in range(9)]

        list1 = [(rank1_single, 0.99)] + [(m, 0.1) for m in other] + [(rank10_both, 0.05)]
        list2 = [make_memory(f"x_{i}") for i in range(9)]
        list2_scored = [(m, 0.5) for m in list2] + [(rank10_both, 0.4)]

        result = reciprocal_rank_fusion(list1, list2_scored)
        ids = [r[0].id for r in result]
        assert ids.index(rank1_single.id) < ids.index(rank10_both.id)

    def test_single_list_preserves_order(self):
        mems = [make_memory(f"m{i}") for i in range(5)]
        ranked = [(m, 1.0 / (i + 1)) for i, m in enumerate(mems)]

        result = reciprocal_rank_fusion(ranked)
        result_ids = [r[0].id for r in result]
        expected_ids = [m.id for m in mems]
        assert result_ids == expected_ids

    def test_scores_are_positive(self):
        mems = [make_memory(f"m{i}") for i in range(3)]
        ranked = [(m, float(i)) for i, m in enumerate(mems)]
        result = reciprocal_rank_fusion(ranked, ranked)
        for _, score in result:
            assert score > 0
