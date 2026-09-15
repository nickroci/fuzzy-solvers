from __future__ import annotations

import pytest

from ramsey.cayley import AbelianGroup, cayley_graph, circulant, symmetric_closure
from ramsey.graph import (
    RamseyGraph,
    capped_independence_number,
    cycle_graph,
    independence_number,
)
from ramsey.kernel import accelerated, capped_independence, kernel_max_order

CASES = ((13, (1, 5)), (16, (3, 7, 8)), (17, (6, 7)), (39, (1, 3, 9, 14)))


def test_the_kernel_reports_whether_it_is_compiled() -> None:
    assert isinstance(accelerated(), bool)
    assert kernel_max_order() >= 0


@pytest.mark.parametrize(("order", "steps"), CASES)
@pytest.mark.parametrize("ceiling", [1, 3, 6, 10, 99])
def test_the_kernel_agrees_with_the_referee(
    order: int, steps: tuple[int, ...], ceiling: int
) -> None:
    """The compiled kernel must agree with the reference on every input."""
    group = AbelianGroup(moduli=(order,))
    graph = cayley_graph(group, symmetric_closure(group, steps))
    assert capped_independence(graph, ceiling) == capped_independence_number(graph, ceiling)


def test_the_kernel_saturates_at_its_ceiling() -> None:
    graph = cycle_graph(12)
    assert capped_independence(graph, 2) == 2
    assert capped_independence(graph, 99) == independence_number(graph)


def test_a_non_positive_ceiling_is_zero() -> None:
    assert capped_independence(cycle_graph(5), 0) == 0
    assert capped_independence(cycle_graph(5), -3) == 0


def test_an_empty_graph_falls_back_cleanly() -> None:
    assert capped_independence(RamseyGraph(order=0, adjacency=()), 5) == 0


def test_a_graph_beyond_the_kernel_capacity_falls_back() -> None:
    """Above the compiled capacity the reference answers, so nothing breaks."""
    graph = circulant(300, (1, 7))
    assert capped_independence(graph, 8) == capped_independence_number(graph, 8)


def test_an_edgeless_graph_is_wholly_independent() -> None:
    graph = RamseyGraph(order=9, adjacency=(0,) * 9)
    assert capped_independence(graph, 99) == 9
