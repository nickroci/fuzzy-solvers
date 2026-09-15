from __future__ import annotations

import pytest

from ramsey.extension import (
    extend_graph,
    extension_constraints,
    find_extension,
)
from ramsey.graph import (
    RamseyGraph,
    check_ramsey_graph,
    cycle_graph,
    graph_from_edges,
)

RAMSEY_3_T = {3: 6, 4: 9, 5: 14, 6: 18}


def cyclic(order: int, connections: tuple[int, ...]) -> RamseyGraph:
    edges = tuple(
        (vertex, (vertex + step) % order) for vertex in range(order) for step in connections
    )
    return graph_from_edges(order, edges)


def wagner() -> RamseyGraph:
    cycle = tuple((v, (v + 1) % 8) for v in range(8))
    chords = tuple((v, v + 4) for v in range(4))
    return graph_from_edges(8, cycle + chords)


def petersen() -> RamseyGraph:
    return graph_from_edges(
        10,
        (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),
            (0, 5),
            (1, 6),
            (2, 7),
            (3, 8),
            (4, 9),
            (5, 7),
            (7, 9),
            (9, 6),
            (6, 8),
            (8, 5),
        ),
    )


def without_vertex(graph: RamseyGraph, vertex: int) -> RamseyGraph:
    return graph.induced(graph.full_mask & ~(1 << vertex))


def greedy_chain(start: RamseyGraph, second_colour: int) -> RamseyGraph:
    """Extend one vertex at a time until the search reports no extension."""
    current = start
    while True:
        report = find_extension(current, second_colour)
        if not report.extends:
            return current
        assert report.witness is not None
        assert report.witness.check.accepted
        current = report.witness.extended


def test_extend_graph_adds_one_vertex_with_the_given_neighbourhood() -> None:
    extended = extend_graph(cycle_graph(5), 0b00101)
    assert extended.order == 6
    assert extended.degree(5) == 2
    assert extended.has_edge(0, 5)
    assert extended.has_edge(2, 5)
    assert not extended.has_edge(1, 5)


def test_extend_graph_rejects_a_neighbourhood_outside_the_graph() -> None:
    with pytest.raises(ValueError, match="outside the graph"):
        extend_graph(cycle_graph(5), 0b100000)


# --- the extension test must agree with established Ramsey numbers ---


def test_c5_does_not_extend_because_r_3_3_is_6() -> None:
    """C5 is a (3,3,5)-graph and R(3,3) = 6, so it must be maximal."""
    report = find_extension(cycle_graph(5), 3)
    assert not report.extends
    assert report.exhaustive


def test_wagner_does_not_extend_because_r_3_4_is_9() -> None:
    """V8 is a (3,4,8)-graph and R(3,4) = 9, so it must be maximal."""
    assert check_ramsey_graph(wagner(), 4).accepted
    report = find_extension(wagner(), 4)
    assert not report.extends
    assert report.exhaustive


def test_the_maximal_3_5_graph_is_refuted_by_the_degree_range_alone() -> None:
    """No (3,5,14)-graph exists, and the constraints alone prove it.

    A new vertex would need degree at least ``13 - R(3,4) + 1 = 5`` so that the
    residual is a (3,4)-graph, but at most ``4`` because every neighbourhood of
    a triangle-free graph is independent.  That contradiction is the classical
    ``R(3,t) <= R(3,t-1) + t`` bound, and it costs no search.
    """
    graph = cyclic(13, (1, 5))
    assert check_ramsey_graph(graph, 5).accepted
    constraints = extension_constraints(graph, 5)
    assert constraints.minimum_degree == 5
    assert constraints.maximum_degree == 4
    assert not constraints.feasible
    report = find_extension(graph, 5)
    assert not report.extends
    assert report.nodes == 0


@pytest.mark.parametrize(
    ("graph", "second_colour"),
    [(wagner(), 4), (petersen(), 5), (cyclic(13, (1, 5)), 5)],
)
def test_deleting_a_vertex_yields_a_graph_that_extends_back(
    graph: RamseyGraph, second_colour: int
) -> None:
    """A vertex-deleted maximal graph must admit at least the vertex it lost."""
    reduced = without_vertex(graph, 0)
    assert check_ramsey_graph(reduced, second_colour).accepted
    report = find_extension(reduced, second_colour)
    assert report.extends
    witness = report.witness
    assert witness is not None
    assert witness.extended.order == graph.order
    assert witness.check.accepted
    assert all(
        not reduced.has_edge(first, second)
        for first in witness.neighbourhood
        for second in witness.neighbourhood
        if first != second
    )


@pytest.mark.parametrize("second_colour", [3, 4, 5, 6])
def test_no_extension_chain_ever_reaches_the_ramsey_number(second_colour: int) -> None:
    """Repeated extension must stop below R(3,t) vertices, by definition.

    This is the end-to-end check on both halves of the module: every accepted
    step is referee-verified, and the terminal order is a bound the literature
    already fixes.
    """
    start = cycle_graph(4) if second_colour == 3 else cycle_graph(5)
    assert check_ramsey_graph(start, second_colour).accepted
    terminal = greedy_chain(start, second_colour)
    assert check_ramsey_graph(terminal, second_colour).accepted
    assert terminal.order <= RAMSEY_3_T[second_colour] - 1


def test_the_chain_for_r_3_4_reaches_the_extremal_order() -> None:
    """Greedy extension attains the maximum 8 vertices for R(3,4) = 9."""
    assert greedy_chain(cycle_graph(5), 4).order == 8


# --- constraints ---


def test_constraints_for_the_campaign_case() -> None:
    """A (3,10,39)-graph can only gain a vertex of degree 4 to 9."""
    graph = cyclic(39, (1, 7))
    constraints = extension_constraints(graph, 10)
    assert constraints.minimum_degree == 4
    assert constraints.maximum_degree == 9
    assert constraints.residual_independence == 8
    assert constraints.eligible_count == 39
    assert constraints.feasible


def test_high_degree_vertices_are_ineligible() -> None:
    """A vertex already at the degree cap cannot gain another edge."""
    star = graph_from_edges(5, ((0, 1), (0, 2), (0, 3)))
    constraints = extension_constraints(star, 4)
    assert constraints.maximum_degree == 3
    assert not constraints.eligible >> 0 & 1
    assert constraints.eligible_count == 4


def test_infeasible_constraints_short_circuit_the_search() -> None:
    graph = cyclic(39, (1, 2, 3, 4, 5))
    constraints = extension_constraints(graph, 10)
    assert constraints.eligible_count == 0
    assert not constraints.feasible
    assert constraints.diagnostics
    report = find_extension(graph, 10)
    assert not report.extends
    assert report.nodes == 0


def test_unknown_smaller_ramsey_number_gives_a_trivial_lower_bound() -> None:
    constraints = extension_constraints(cyclic(20, (1, 5)), 11)
    assert constraints.minimum_degree == 0


# --- coverage accounting ---


def test_a_budget_exhausted_search_is_not_reported_as_exhaustive() -> None:
    report = find_extension(petersen(), 5, node_limit=2)
    assert not report.extends
    assert not report.exhaustive
    assert report.nodes == 2


def test_an_exhaustive_negative_reports_its_search_volume() -> None:
    terminal = greedy_chain(cycle_graph(5), 5)
    report = find_extension(terminal, 5)
    assert not report.extends
    assert report.exhaustive
    assert report.constraints.feasible
    assert report.nodes > 0
    assert report.residual_checks > 0


def test_a_found_witness_counts_as_a_closed_search() -> None:
    report = find_extension(petersen(), 5, node_limit=1_000_000)
    assert report.extends
    assert report.exhaustive
