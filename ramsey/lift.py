"""Voltage-graph lifts: a construction language wider than Cayley graphs.

A Cayley graph is the most symmetric thing you can build from a group, and
that symmetry is what made the search tractable — but it is also a ceiling.
``R(3, 6) > 17`` is true and has no Cayley witness at all, because 17 is prime
and no circulant works.  A search that can only speak Cayley cannot reach
graphs like that however many evaluations it spends.

A lift is the same idea with room to breathe.  Take a small **base** graph on
``b`` vertices, a group ``G``, and assign each base edge a set of **voltages**
in ``G``.  The lift has vertex set ``V(base) x G``, with ``(i, g)`` joined to
``(j, g + s)`` for every voltage ``s`` on the base edge ``{i, j}``.  Taking
``b = 1`` recovers exactly a Cayley graph, so nothing is lost; taking ``b > 1``
reaches graphs no Cayley construction can — the Petersen graph, for instance,
is the ``Z5``-lift of a two-vertex base and is famously not a Cayley graph.

The triangle condition generalises sum-freeness cleanly.  Write ``V(i, j)``
for the voltages carried from base vertex ``i`` to ``j`` (negated when read
backwards).  Two lift vertices are adjacent exactly when their group
difference lies in ``V(i, j)``, so a triangle is a pair of voltages ``a`` on
``i -> j`` and ``b`` on ``j -> l`` whose sum lies on ``i -> l``.  The lift is
triangle-free precisely when no such pair exists, which is decidable on the
base alone and never needs the graph.

The model authors the base size, the group, and the voltages — a few dozen
numbers it can reason about structurally, rather than an adjacency matrix it
could only guess at.
"""

from __future__ import annotations

from dataclasses import dataclass

from ramsey.cayley import AbelianGroup
from ramsey.graph import RamseyGraph

VoltageMap = dict[tuple[int, int], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class VoltageGraph:
    """A base graph with group voltages on its edges.

    ``voltages`` is keyed by ordered pairs ``(i, j)`` with ``i <= j``.  A
    diagonal entry describes edges within one base vertex's fibre and must be
    symmetric under negation and identity-free, exactly as a Cayley connection
    set is; an off-diagonal entry is unconstrained, because reading it
    backwards negates it.
    """

    base_size: int
    moduli: tuple[int, ...]
    voltages: VoltageMap

    def __post_init__(self) -> None:
        """Reject a specification that does not describe an undirected lift."""
        if self.base_size < 1:
            message = f"base_size must be positive, got {self.base_size}"
            raise ValueError(message)
        group = self.group
        for (first, second), assignment in self.voltages.items():
            if not (0 <= first <= second < self.base_size):
                message = f"voltage key {(first, second)} is outside the base"
                raise ValueError(message)
            if len(set(assignment)) != len(assignment):
                message = f"voltage set for {(first, second)} repeats an element"
                raise ValueError(message)
            if any(not 0 <= value < group.order for value in assignment):
                message = f"voltage set for {(first, second)} names an element outside the group"
                raise ValueError(message)
            if first == second:
                if 0 in assignment:
                    message = f"the fibre voltage set at {first} must not contain the identity"
                    raise ValueError(message)
                if any(group.negate(value) not in set(assignment) for value in assignment):
                    message = f"the fibre voltage set at {first} must be closed under negation"
                    raise ValueError(message)

    @property
    def group(self) -> AbelianGroup:
        """Return the voltage group."""
        return AbelianGroup(moduli=self.moduli)

    @property
    def order(self) -> int:
        """Return the number of vertices in the lift."""
        return self.base_size * self.group.order

    @property
    def name(self) -> str:
        """Render the construction in base-and-group form."""
        return f"lift(base {self.base_size} x {self.group.name})"

    def directed(self, first: int, second: int) -> tuple[int, ...]:
        """Return the voltages read from ``first`` to ``second``.

        Reading a base edge backwards negates its voltages, which is what makes
        the lift undirected without storing both directions.
        """
        if first <= second:
            return self.voltages.get((first, second), ())
        group = self.group
        return tuple(group.negate(value) for value in self.voltages.get((second, first), ()))

    def degree(self, base_vertex: int) -> int:
        """Return the degree of every lift vertex in one base vertex's fibre."""
        return sum(len(self.directed(base_vertex, other)) for other in range(self.base_size))

    def degrees(self) -> tuple[int, ...]:
        """Return the degree of each base vertex's fibre."""
        return tuple(self.degree(vertex) for vertex in range(self.base_size))


@dataclass(frozen=True, slots=True)
class TriangleVoltage:
    """A base triple and voltages that close a triangle in the lift."""

    base: tuple[int, int, int]
    first: int
    second: int


def triangle_voltage(construction: VoltageGraph) -> TriangleVoltage | None:
    """Return a voltage pair closing a triangle, or ``None`` if there is none.

    This is the lift's analogue of a sum-freeness violation: it is decided on
    the base and the group, never by looking at the lifted graph, so a
    construction can be rejected before anything is built.
    """
    group = construction.group
    size = construction.base_size
    for first in range(size):
        for middle in range(size):
            outgoing = construction.directed(first, middle)
            if not outgoing:
                continue
            for last in range(size):
                closing = set(construction.directed(first, last))
                if not closing:
                    continue
                onward = construction.directed(middle, last)
                for left in outgoing:
                    for right in onward:
                        if group.add(left, right) in closing:
                            return TriangleVoltage(
                                base=(first, middle, last), first=left, second=right
                            )
    return None


def is_triangle_free_lift(construction: VoltageGraph) -> bool:
    """Report whether the lift is triangle-free, decided on the base."""
    return triangle_voltage(construction) is None


def lift_graph(construction: VoltageGraph) -> RamseyGraph:
    """Build the lifted graph.

    Vertex ``(i, g)`` is indexed ``i * |G| + g``, so a fibre is a contiguous
    block and the base structure stays legible in the vertex numbering.
    """
    group = construction.group
    size = group.order
    masks = [0] * construction.order
    for first in range(construction.base_size):
        for second in range(construction.base_size):
            for value in construction.directed(first, second):
                for element in range(size):
                    source = first * size + element
                    destination = second * size + group.add(element, value)
                    if source != destination:
                        masks[source] |= 1 << destination
                        masks[destination] |= 1 << source
    return RamseyGraph(order=construction.order, adjacency=tuple(masks))


def cayley_as_lift(moduli: tuple[int, ...], connection: tuple[int, ...]) -> VoltageGraph:
    """Return a Cayley graph expressed as a one-vertex lift.

    This is the statement that the language strictly generalises what came
    before: every Cayley construction is a lift, so widening the search space
    gives up nothing.
    """
    return VoltageGraph(base_size=1, moduli=moduli, voltages={(0, 0): tuple(sorted(connection))})
