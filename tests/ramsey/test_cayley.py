from __future__ import annotations

import pytest

from ramsey.cayley import (
    AbelianGroup,
    cayley_graph,
    circulant,
    is_sum_free,
    sum_violation,
    symmetric_closure,
)
from ramsey.graph import check_ramsey_graph, cycle_graph, independence_number

CLEBSCH_GROUP = AbelianGroup(moduli=(2, 2, 2, 2))
CLEBSCH_CONNECTION = (1, 2, 4, 8, 15)


def test_group_rejects_an_empty_factor_list() -> None:
    with pytest.raises(ValueError, match="at least one factor"):
        AbelianGroup(moduli=())


def test_group_rejects_a_non_positive_modulus() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        AbelianGroup(moduli=(4, 0))


def test_group_order_and_name() -> None:
    group = AbelianGroup(moduli=(3, 5))
    assert group.order == 15
    assert group.name == "Z3 x Z5"


def test_coordinates_round_trip() -> None:
    group = AbelianGroup(moduli=(2, 3, 4))
    for index in range(group.order):
        assert group.index(group.coordinates(index)) == index


def test_index_rejects_wrong_arity() -> None:
    with pytest.raises(ValueError, match="expected 2 coordinates"):
        AbelianGroup(moduli=(3, 5)).index((1,))


def test_addition_and_negation_are_group_operations() -> None:
    group = AbelianGroup(moduli=(6, 4))
    for index in range(group.order):
        assert group.add(index, group.negate(index)) == 0
        assert group.add(index, 0) == index


def test_symmetric_closure_drops_identity_and_adds_inverses() -> None:
    group = AbelianGroup(moduli=(8,))
    assert symmetric_closure(group, (0, 1, 4)) == (1, 4, 7)


def test_symmetric_closure_wraps_out_of_range_generators() -> None:
    group = AbelianGroup(moduli=(8,))
    assert symmetric_closure(group, (9,)) == (1, 7)


def test_sum_violation_reports_a_triangle_through_the_identity() -> None:
    group = AbelianGroup(moduli=(9,))
    connection = symmetric_closure(group, (1, 2))
    violation = sum_violation(group, connection)
    assert violation is not None
    assert group.add(*violation) in set(connection)
    assert not is_sum_free(group, connection)


def test_a_sum_free_set_gives_a_triangle_free_graph() -> None:
    group = AbelianGroup(moduli=(13,))
    connection = symmetric_closure(group, (1, 5))
    assert is_sum_free(group, connection)
    assert check_ramsey_graph(cayley_graph(group, connection), 5).triangle is None


def test_cayley_graph_rejects_the_identity_in_the_connection_set() -> None:
    with pytest.raises(ValueError, match="must not contain the identity"):
        cayley_graph(AbelianGroup(moduli=(7,)), (0, 1, 6))


def test_cayley_graph_rejects_an_asymmetric_connection_set() -> None:
    with pytest.raises(ValueError, match="closed under negation"):
        cayley_graph(AbelianGroup(moduli=(7,)), (1,))


def test_cayley_graph_is_regular_of_the_connection_set_size() -> None:
    group = AbelianGroup(moduli=(11,))
    connection = symmetric_closure(group, (1, 3))
    graph = cayley_graph(group, connection)
    assert set(graph.degrees()) == {len(connection)}


# --- the constructions must reproduce known Ramsey witnesses ---


def test_circulant_reproduces_the_five_cycle() -> None:
    assert circulant(5, (1,)) == cycle_graph(5)


def test_circulant_reproduces_the_wagner_3_4_8_graph() -> None:
    """C8(1,4) is the Wagner graph, witnessing R(3,4) > 8."""
    graph = circulant(8, (1, 4))
    assert check_ramsey_graph(graph, 4).accepted
    assert independence_number(graph) == 3


def test_circulant_reproduces_the_3_5_13_graph() -> None:
    """C13(1,5) witnesses R(3,5) > 13, and R(3,5) = 14."""
    graph = circulant(13, (1, 5))
    assert check_ramsey_graph(graph, 5).accepted
    assert independence_number(graph) == 4


def test_clebsch_graph_is_a_non_cyclic_3_6_16_witness() -> None:
    """The Clebsch graph over Z2^4 witnesses R(3,6) > 16 and is not a circulant."""
    graph = cayley_graph(CLEBSCH_GROUP, CLEBSCH_CONNECTION)
    assert graph.order == 16
    assert set(graph.degrees()) == {5}
    assert check_ramsey_graph(graph, 6).accepted
    assert independence_number(graph) == 5


def test_maximum_degree_bounds_the_independence_number() -> None:
    """A triangle-free graph has alpha >= max degree, which caps usable degree."""
    graph = cayley_graph(CLEBSCH_GROUP, CLEBSCH_CONNECTION)
    assert independence_number(graph) >= max(graph.degrees())


def test_coprime_factors_make_a_cyclic_group() -> None:
    """Z2 x Z37 is Z74 by the Chinese remainder theorem."""
    assert AbelianGroup(moduli=(2, 37)).is_cyclic
    assert AbelianGroup(moduli=(9, 11)).is_cyclic
    assert AbelianGroup(moduli=(131,)).is_cyclic


def test_repeated_prime_factors_make_a_non_cyclic_group() -> None:
    assert not AbelianGroup(moduli=(3, 3, 11)).is_cyclic
    assert not AbelianGroup(moduli=(2, 2, 23)).is_cyclic
    assert not AbelianGroup(moduli=(2, 2, 2, 2)).is_cyclic
