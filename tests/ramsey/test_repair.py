from __future__ import annotations

import random

import pytest

from ramsey.cayley import circulant
from ramsey.graph import check_ramsey_graph, cycle_graph, find_triangle, graph_from_edges
from ramsey.repair import joinable_pairs, repair_search, with_edge, without_edge


def test_edges_can_be_added_and_removed() -> None:
    graph = cycle_graph(5)
    added = with_edge(graph, 0, 2)
    assert added.has_edge(0, 2)
    assert without_edge(added, 0, 2) == graph


def test_joinable_pairs_excludes_triangle_closing_and_saturated_pairs() -> None:
    path = graph_from_edges(4, ((0, 1), (1, 2)))
    # 0 and 2 share neighbour 1, so joining them closes a triangle.
    assert (0, 2) not in joinable_pairs(path, (0, 2), 3)
    # 0 and 3 share nothing and have room.
    assert (0, 3) in joinable_pairs(path, (0, 3), 3)
    # the degree cap blocks everything.
    assert joinable_pairs(path, (0, 3), 0) == ()


def test_repair_rejects_a_seed_with_a_triangle() -> None:
    with pytest.raises(ValueError, match="must be triangle-free"):
        repair_search(cycle_graph(3), 2, 3, steps=1, rng=random.Random(0), budget=10)


def test_repair_never_returns_something_worse_than_its_seed() -> None:
    seed = circulant(17, (6, 7))
    outcome = repair_search(seed, 5, 5, steps=30, rng=random.Random(1), budget=100)
    assert outcome.best_independence <= outcome.start_independence
    assert find_triangle(outcome.best_graph) is None


def test_repair_reaches_a_witness_no_cayley_graph_can() -> None:
    """R(3,6) > 17 holds, but no circulant on 17 vertices witnesses it."""
    seed = circulant(17, (6, 7))
    rng = random.Random(11)
    for _ in range(400):
        outcome = repair_search(seed, 5, 5, steps=60, rng=rng, budget=200)
        if outcome.accepted:
            assert check_ramsey_graph(outcome.best_graph, 6).accepted
            assert outcome.improved
            assert "ACCEPTED" in outcome.render()
            return
    pytest.fail("repair did not reach a (3,6,17)-graph")


def test_repair_respects_its_evaluation_budget() -> None:
    outcome = repair_search(circulant(17, (6, 7)), 5, 5, steps=500, rng=random.Random(2), budget=9)
    assert outcome.evaluations <= 9


def test_a_failed_repair_reports_its_gap() -> None:
    outcome = repair_search(cycle_graph(9), 2, 2, steps=5, rng=random.Random(3), budget=20)
    assert not outcome.accepted
    assert outcome.gap > 0
    assert "above the cap" in outcome.render()
