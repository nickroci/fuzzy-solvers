"""The one-vertex extension test for triangle-free Ramsey graphs.

Deciding ``R(3, 10)`` reduces to a single existence question: is there a
triangle-free graph on 40 vertices with independence number at most 9?  Every
such graph loses a vertex to give a ``(3, 10, 39)``-graph — triangle-freeness
and a bound on the independence number are both hereditary — so every
``(3, 10, 40)``-graph is a one-vertex extension of some ``(3, 10, 39)``-graph.
This module is the exact test for that step.

Adding a vertex ``w`` with neighbourhood ``S`` to a ``(3, t, n)``-graph ``G``
yields a ``(3, t, n + 1)``-graph if and only if:

1. ``S`` is independent in ``G`` — otherwise ``w`` closes a triangle; and
2. ``alpha(G - S) <= t - 2`` — an independent set containing ``w`` is ``w``
   together with an independent set avoiding ``N[w]``.

Three further conditions are implied rather than assumed, and are used only to
prune.  In a triangle-free graph every neighbourhood is independent, so no
degree may exceed ``t - 1``; that caps ``|S|`` and restricts ``S`` to vertices
whose degree in ``G`` is at most ``t - 2``.  And because ``G - S`` must itself
be a ``(3, t - 1)``-graph it has fewer than ``R(3, t - 1)`` vertices, which
forces ``|S| >= n - R(3, t - 1) + 1``.  For the campaign case ``t = 10``,
``n = 39`` and ``R(3, 9) = 36`` this pins the new vertex's degree to the range
``4 <= |S| <= 9``.

The search is exhaustive over that space and reports its own coverage, so a
negative result is a measured fact rather than an absence of evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ramsey.graph import (
    RamseyCheck,
    RamseyGraph,
    check_ramsey_graph,
    independent_set_within,
)

SMALL_RAMSEY_NUMBERS: Final[dict[int, int]] = {
    3: 6,
    4: 9,
    5: 14,
    6: 18,
    7: 23,
    8: 28,
    9: 36,
}
"""Established values of ``R(3, t)``; ``R(3, 10)`` is open and absent."""


@dataclass(frozen=True, slots=True)
class ExtensionConstraints:
    """The necessary conditions on the neighbourhood of an added vertex."""

    minimum_degree: int
    maximum_degree: int
    eligible: int
    eligible_count: int
    residual_independence: int
    feasible: bool

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Return the reasons the constraint set admits no extension at all."""
        lines = []
        if self.minimum_degree > self.maximum_degree:
            lines.append(
                f"required degree range is empty: {self.minimum_degree} > {self.maximum_degree}"
            )
        if self.eligible_count < self.minimum_degree:
            lines.append(
                f"only {self.eligible_count} vertices may gain an edge, "
                f"but the new vertex needs at least {self.minimum_degree}"
            )
        return tuple(lines)


def extension_constraints(graph: RamseyGraph, second_colour: int) -> ExtensionConstraints:
    """Derive the necessary conditions on a one-vertex extension of ``graph``.

    ``second_colour`` is the ``t`` of the target ``(3, t)``-graph.  When
    ``R(3, t - 1)`` is known the vertex-count argument gives a lower bound on
    the new degree; otherwise the lower bound is the trivial zero.
    """
    maximum_degree = second_colour - 1
    eligible_degree = second_colour - 2
    eligible = 0
    for vertex in range(graph.order):
        if graph.degree(vertex) <= eligible_degree:
            eligible |= 1 << vertex
    smaller = SMALL_RAMSEY_NUMBERS.get(second_colour - 1)
    minimum_degree = 0 if smaller is None else max(0, graph.order - smaller + 1)
    eligible_count = eligible.bit_count()
    return ExtensionConstraints(
        minimum_degree=minimum_degree,
        maximum_degree=maximum_degree,
        eligible=eligible,
        eligible_count=eligible_count,
        residual_independence=second_colour - 2,
        feasible=minimum_degree <= maximum_degree and eligible_count >= minimum_degree,
    )


@dataclass(frozen=True, slots=True)
class ExtensionWitness:
    """A neighbourhood that extends a graph by one vertex, with the result.

    ``check`` is the referee's verdict on ``extended``, run at construction so
    that a witness is never reported on the strength of the search alone.
    """

    neighbourhood: tuple[int, ...]
    extended: RamseyGraph
    check: RamseyCheck


@dataclass(frozen=True, slots=True)
class ExtensionReport:
    """The verdict for one source graph, with the coverage that produced it.

    ``exhaustive`` is the load-bearing field.  It is true only when the search
    closed the entire constrained space; a report that ran out of budget says
    nothing about the graphs it did not reach, and must never be counted as a
    negative.
    """

    witness: ExtensionWitness | None
    exhaustive: bool
    nodes: int
    residual_checks: int
    pruned_branches: int
    constraints: ExtensionConstraints

    @property
    def extends(self) -> bool:
        """Report whether an extension was found."""
        return self.witness is not None


def extend_graph(graph: RamseyGraph, neighbourhood: int) -> RamseyGraph:
    """Return ``graph`` with one vertex added, adjacent to ``neighbourhood``."""
    if neighbourhood & ~graph.full_mask:
        message = "neighbourhood names a vertex outside the graph"
        raise ValueError(message)
    added = graph.order
    masks = [
        mask | (1 << added) if neighbourhood >> vertex & 1 else mask
        for vertex, mask in enumerate(graph.adjacency)
    ]
    masks.append(neighbourhood)
    return RamseyGraph(order=added + 1, adjacency=tuple(masks))


def find_extension(
    graph: RamseyGraph,
    second_colour: int,
    *,
    node_limit: int | None = None,
) -> ExtensionReport:
    """Search exhaustively for a one-vertex extension of a ``(3, t, n)``-graph.

    Returns the first witness found, or an exhaustive negative.  ``node_limit``
    caps the number of search nodes; when the cap is reached the report is
    returned with ``exhaustive`` false, which callers must propagate rather
    than treat as a proof of non-extendability.
    """
    constraints = extension_constraints(graph, second_colour)
    state = _SearchState(node_limit=node_limit)
    if not constraints.feasible:
        return _report(None, state, constraints)
    witness = _search(graph, second_colour, constraints, state, constraints.eligible, 0, 0)
    return _report(witness, state, constraints)


@dataclass(slots=True)
class _SearchState:
    """Mutable counters and the budget for one extension search."""

    node_limit: int | None
    nodes: int = 0
    residual_checks: int = 0
    pruned: int = 0
    exhausted: bool = True

    def visit(self) -> bool:
        """Count one search node, reporting whether budget remains."""
        if self.node_limit is not None and self.nodes >= self.node_limit:
            self.exhausted = False
            return False
        self.nodes += 1
        return True


def _report(
    witness: ExtensionWitness | None,
    state: _SearchState,
    constraints: ExtensionConstraints,
) -> ExtensionReport:
    """Assemble the report, treating a found witness as a closed search."""
    return ExtensionReport(
        witness=witness,
        exhaustive=state.exhausted or witness is not None,
        nodes=state.nodes,
        residual_checks=state.residual_checks,
        pruned_branches=state.pruned,
        constraints=constraints,
    )


def _search(
    graph: RamseyGraph,
    second_colour: int,
    constraints: ExtensionConstraints,
    state: _SearchState,
    candidates: int,
    chosen: int,
    size: int,
) -> ExtensionWitness | None:
    """Enumerate independent neighbourhoods, deepest-constraint-first.

    Each node either accepts the current set as a neighbourhood, prunes the
    whole branch on the residual-independence lower bound, or splits on the
    lowest remaining candidate.  Splitting on a specific vertex — included in
    one branch, excluded in the other — enumerates every independent subset
    exactly once.
    """
    if not state.visit():
        return None
    if size >= constraints.minimum_degree and _residual_is_clear(graph, constraints, state, chosen):
        return _accept(graph, second_colour, chosen)
    if _branch_is_dead(graph, constraints, state, chosen, candidates, size):
        return None
    lowest = candidates & -candidates
    pivot = lowest.bit_length() - 1
    witness = _search(
        graph,
        second_colour,
        constraints,
        state,
        candidates & ~lowest & ~graph.adjacency[pivot],
        chosen | lowest,
        size + 1,
    )
    if witness is not None:
        return witness
    return _search(graph, second_colour, constraints, state, candidates & ~lowest, chosen, size)


def _residual_is_clear(
    graph: RamseyGraph,
    constraints: ExtensionConstraints,
    state: _SearchState,
    chosen: int,
) -> bool:
    """Report whether deleting ``chosen`` leaves the required independence bound."""
    state.residual_checks += 1
    residual = graph.full_mask & ~chosen
    return independent_set_within(graph, residual, constraints.residual_independence + 1) is None


def _branch_is_dead(
    graph: RamseyGraph,
    constraints: ExtensionConstraints,
    state: _SearchState,
    chosen: int,
    candidates: int,
    size: int,
) -> bool:
    """Report whether no descendant of this node can be a valid neighbourhood.

    Three reasons close a branch.  The first two are structural: the degree cap
    is reached, or nothing is left to add.  The third is the residual bound —
    deleting more vertices can only shrink the independence number, so the
    smallest residual any descendant can reach is the one deleting every
    remaining candidate.  If even that still holds a large independent set, no
    superset of ``chosen`` can work.
    """
    if size == constraints.maximum_degree or candidates == 0:
        return True
    if size + candidates.bit_count() < constraints.minimum_degree:
        state.pruned += 1
        return True
    state.residual_checks += 1
    floor = graph.full_mask & ~chosen & ~candidates
    if independent_set_within(graph, floor, constraints.residual_independence + 1) is not None:
        state.pruned += 1
        return True
    return False


def _accept(graph: RamseyGraph, second_colour: int, chosen: int) -> ExtensionWitness:
    """Build the extended graph and put it through the referee before returning."""
    extended = extend_graph(graph, chosen)
    check = check_ramsey_graph(extended, second_colour)
    if not check.accepted:
        message = (
            "internal error: the extension search accepted a neighbourhood the referee "
            f"rejects: {check.diagnostics}"
        )
        raise AssertionError(message)
    neighbourhood = tuple(vertex for vertex in range(graph.order) if chosen >> vertex & 1)
    return ExtensionWitness(neighbourhood=neighbourhood, extended=extended, check=check)
