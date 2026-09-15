from __future__ import annotations

import pytest

from ramsey.cayley import AbelianGroup, cayley_graph, symmetric_closure
from ramsey.graph import check_ramsey_graph, find_triangle, independence_number
from ramsey.lift import (
    VoltageGraph,
    cayley_as_lift,
    is_triangle_free_lift,
    lift_graph,
    triangle_voltage,
)

# The Petersen graph is the Z5-lift of a two-vertex base, and is not a Cayley graph.
PETERSEN = VoltageGraph(
    base_size=2, moduli=(5,), voltages={(0, 0): (1, 4), (1, 1): (2, 3), (0, 1): (0,)}
)


def test_lift_rejects_a_non_positive_base() -> None:
    with pytest.raises(ValueError, match="base_size must be positive"):
        VoltageGraph(base_size=0, moduli=(5,), voltages={})


def test_lift_rejects_a_key_outside_the_base() -> None:
    with pytest.raises(ValueError, match="outside the base"):
        VoltageGraph(base_size=2, moduli=(5,), voltages={(0, 2): (1,)})


def test_lift_rejects_an_unordered_key() -> None:
    with pytest.raises(ValueError, match="outside the base"):
        VoltageGraph(base_size=2, moduli=(5,), voltages={(1, 0): (1,)})


def test_lift_rejects_a_repeated_voltage() -> None:
    with pytest.raises(ValueError, match="repeats an element"):
        VoltageGraph(base_size=2, moduli=(5,), voltages={(0, 1): (1, 1)})


def test_lift_rejects_a_voltage_outside_the_group() -> None:
    with pytest.raises(ValueError, match="outside the group"):
        VoltageGraph(base_size=2, moduli=(5,), voltages={(0, 1): (7,)})


def test_a_fibre_voltage_set_must_exclude_the_identity() -> None:
    with pytest.raises(ValueError, match="must not contain the identity"):
        VoltageGraph(base_size=1, moduli=(5,), voltages={(0, 0): (0, 1, 4)})


def test_a_fibre_voltage_set_must_be_symmetric() -> None:
    with pytest.raises(ValueError, match="closed under negation"):
        VoltageGraph(base_size=1, moduli=(5,), voltages={(0, 0): (1,)})


def test_reading_a_base_edge_backwards_negates_its_voltages() -> None:
    assert PETERSEN.directed(0, 1) == (0,)
    assert PETERSEN.directed(1, 0) == (0,)
    other = VoltageGraph(base_size=2, moduli=(7,), voltages={(0, 1): (2,)})
    assert other.directed(1, 0) == (5,)


def test_lift_order_and_degrees() -> None:
    assert PETERSEN.order == 10
    assert PETERSEN.degrees() == (3, 3)
    assert "Z5" in PETERSEN.name


def test_the_petersen_lift_is_a_non_cayley_ramsey_witness() -> None:
    """Petersen witnesses R(3,5) > 10 and is not a Cayley graph."""
    graph = lift_graph(PETERSEN)
    assert graph.order == 10
    assert find_triangle(graph) is None
    assert independence_number(graph) == 4
    assert check_ramsey_graph(graph, 5).accepted


def test_triangle_freeness_is_decided_on_the_base() -> None:
    assert is_triangle_free_lift(PETERSEN)
    assert triangle_voltage(PETERSEN) is None


def test_a_triangle_is_reported_with_its_voltages() -> None:
    bad = VoltageGraph(base_size=1, moduli=(9,), voltages={(0, 0): (1, 2, 7, 8)})
    violation = triangle_voltage(bad)
    assert violation is not None
    assert not is_triangle_free_lift(bad)
    assert find_triangle(lift_graph(bad)) is not None


def test_the_language_strictly_contains_cayley_graphs() -> None:
    group = AbelianGroup(moduli=(13,))
    connection = symmetric_closure(group, (1, 5))
    assert lift_graph(cayley_as_lift((13,), connection)) == cayley_graph(group, connection)


def test_a_trivial_group_makes_the_lift_the_base_graph() -> None:
    """base_size = order with the trivial group is an unconstrained graph."""
    construction = VoltageGraph(base_size=4, moduli=(1,), voltages={(0, 1): (0,), (2, 3): (0,)})
    graph = lift_graph(construction)
    assert graph.order == 4
    assert graph.edges() == ((0, 1), (2, 3))
