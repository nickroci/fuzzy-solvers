from __future__ import annotations

import pytest

from ramsey.graph import (
    RamseyGraph,
    check_ramsey_graph,
    cycle_graph,
    find_triangle,
    graph_from_edges,
    has_independence_at_most,
    independence_number,
    independence_number_within,
    independent_set_of_size,
    independent_set_within,
    is_triangle_free,
    maximum_independent_set,
    parse_graph6,
    to_graph6,
)

PETERSEN_EDGES = (
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
)


def petersen() -> RamseyGraph:
    return graph_from_edges(10, PETERSEN_EDGES)


def wagner() -> RamseyGraph:
    """The Möbius-Kantor graph V8: an 8-cycle with all four long chords."""
    cycle = tuple((v, (v + 1) % 8) for v in range(8))
    chords = tuple((v, v + 4) for v in range(4))
    return graph_from_edges(8, cycle + chords)


def cyclic(order: int, connections: tuple[int, ...]) -> RamseyGraph:
    edges = tuple(
        (vertex, (vertex + step) % order) for vertex in range(order) for step in connections
    )
    return graph_from_edges(order, edges)


def test_rejects_asymmetric_adjacency() -> None:
    with pytest.raises(ValueError, match="not symmetric"):
        RamseyGraph(order=2, adjacency=(0b10, 0b00))


def test_rejects_self_loop() -> None:
    with pytest.raises(ValueError, match="adjacent to itself"):
        RamseyGraph(order=2, adjacency=(0b01, 0b00))


def test_rejects_vertex_outside_graph() -> None:
    with pytest.raises(ValueError, match="outside the graph"):
        RamseyGraph(order=2, adjacency=(0b100, 0b000))


def test_rejects_wrong_mask_count() -> None:
    with pytest.raises(ValueError, match="expected 3 adjacency masks"):
        RamseyGraph(order=3, adjacency=(0, 0))


def test_rejects_negative_order() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RamseyGraph(order=-1, adjacency=())


def test_edge_endpoint_outside_graph_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside a graph of order"):
        graph_from_edges(3, ((0, 3),))


def test_cycle_needs_three_vertices() -> None:
    with pytest.raises(ValueError, match="at least 3 vertices"):
        cycle_graph(2)


def test_cycle_basics() -> None:
    graph = cycle_graph(5)
    assert graph.order == 5
    assert graph.size == 5
    assert graph.degrees() == (2, 2, 2, 2, 2)
    assert graph.has_edge(0, 1)
    assert not graph.has_edge(0, 2)
    assert graph.edges() == ((0, 1), (0, 4), (1, 2), (2, 3), (3, 4))


def test_triangle_is_found_and_reported() -> None:
    graph = cycle_graph(3)
    triangle = find_triangle(graph)
    assert triangle is not None
    assert sorted(triangle.vertices) == [0, 1, 2]
    assert not is_triangle_free(graph)


def test_petersen_is_triangle_free_with_independence_four() -> None:
    graph = petersen()
    assert is_triangle_free(graph)
    assert independence_number(graph) == 4
    assert has_independence_at_most(graph, 4)
    assert not has_independence_at_most(graph, 3)


def test_maximum_independent_set_is_independent() -> None:
    graph = petersen()
    witness = maximum_independent_set(graph)
    assert witness.size == 4
    assert all(
        not graph.has_edge(first, second)
        for first in witness.vertices
        for second in witness.vertices
        if first != second
    )


def test_independent_set_of_size_is_a_decision_procedure() -> None:
    graph = petersen()
    assert independent_set_of_size(graph, 4) is not None
    assert independent_set_of_size(graph, 5) is None
    empty = independent_set_of_size(graph, 0)
    assert empty is not None
    assert empty.size == 0


def test_independence_restricted_to_a_mask() -> None:
    graph = cycle_graph(6)
    assert independence_number(graph) == 3
    first_four = 0b001111
    assert independence_number_within(graph, first_four) == 2
    assert independent_set_within(graph, first_four, 3) is None
    assert independent_set_within(graph, first_four, 2) is not None


def test_induced_subgraph_renumbers_from_zero() -> None:
    graph = cycle_graph(5)
    induced = graph.induced(0b00111)
    assert induced.order == 3
    assert induced.edges() == ((0, 1), (1, 2))


def test_empty_graph_is_degenerate_but_legal() -> None:
    graph = RamseyGraph(order=0, adjacency=())
    assert graph.size == 0
    assert is_triangle_free(graph)
    assert independence_number(graph) == 0


# --- established Ramsey facts: the referee must agree with the literature ---


def test_c5_is_a_3_3_5_graph() -> None:
    """C5 witnesses R(3,3) > 5, and R(3,3) = 6."""
    check = check_ramsey_graph(cycle_graph(5), 3)
    assert check.accepted
    assert check.diagnostics == ()
    assert check.independence_target == 2


def test_wagner_is_a_3_4_8_graph() -> None:
    """V8 witnesses R(3,4) > 8, and R(3,4) = 9."""
    check = check_ramsey_graph(wagner(), 4)
    assert check.accepted
    assert independence_number(wagner()) == 3


def test_petersen_is_a_3_5_10_graph() -> None:
    check = check_ramsey_graph(petersen(), 5)
    assert check.accepted


def test_cyclic_13_is_a_3_5_13_graph() -> None:
    """The circulant C13(1,5) witnesses R(3,5) > 13, and R(3,5) = 14."""
    graph = cyclic(13, (1, 5))
    check = check_ramsey_graph(graph, 5)
    assert check.accepted
    assert independence_number(graph) == 4


def test_rejection_carries_both_witnesses() -> None:
    check = check_ramsey_graph(cycle_graph(3), 3)
    assert not check.accepted
    assert check.triangle is not None
    assert check.independent_set is None
    assert "contains a triangle" in check.diagnostics[0]


def test_rejection_reports_an_oversized_independent_set() -> None:
    check = check_ramsey_graph(cycle_graph(6), 3)
    assert not check.accepted
    assert check.triangle is None
    assert check.independent_set is not None
    assert check.independent_set.size == 3
    assert "independent set of size 3" in check.diagnostics[0]


def test_second_colour_must_be_at_least_two() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        check_ramsey_graph(cycle_graph(5), 1)


# --- graph6 ---


@pytest.mark.parametrize(
    "graph",
    [
        cycle_graph(5),
        petersen(),
        wagner(),
        cyclic(13, (1, 5)),
        RamseyGraph(order=1, adjacency=(0,)),
    ],
)
def test_graph6_round_trip(graph: RamseyGraph) -> None:
    assert parse_graph6(to_graph6(graph)) == graph


def test_graph6_accepts_the_optional_header() -> None:
    encoded = to_graph6(cycle_graph(5))
    assert parse_graph6(f">>graph6<<{encoded}") == cycle_graph(5)


def test_graph6_round_trip_for_a_large_order() -> None:
    graph = cyclic(70, (1, 7))
    encoded = to_graph6(graph)
    assert ord(encoded[0]) == 126
    assert parse_graph6(encoded) == graph


def test_graph6_rejects_empty_record() -> None:
    with pytest.raises(ValueError, match="empty graph6 record"):
        parse_graph6("   ")


def test_graph6_rejects_out_of_range_characters() -> None:
    with pytest.raises(ValueError, match="outside the printable range"):
        parse_graph6("\x01\x02")


def test_graph6_rejects_wrong_payload_length() -> None:
    with pytest.raises(ValueError, match="payload bytes"):
        parse_graph6(to_graph6(cycle_graph(5)) + "A")


def test_graph6_rejects_truncated_long_prefix() -> None:
    with pytest.raises(ValueError, match="truncated inside its order prefix"):
        parse_graph6("~A")
