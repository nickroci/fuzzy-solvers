"""Edge-level repair: leaving the symmetric space from a good seed.

Every construction language has a ceiling.  A Cayley search cannot reach a
witness that is not a Cayley graph, and a lift search cannot reach one that is
not a lift — but *any* graph is reachable by editing edges, and the only
question is where the editing starts.  Starting from nothing is hopeless; the
space of graphs on a hundred vertices is astronomically larger than any search
budget.  Starting from a construction that a structured search already brought
within a few of the cap is not.

So this module takes a seed and repairs it.  The obstruction is an independent
set that is too large, and the only way to destroy it is to add an edge inside
it — every other edge leaves it independent.  Adding that edge is legal when it
closes no triangle and breaks no degree cap, and when no legal repair exists
the search frees capacity by deleting an edge and tries again.

The result is a graph with no symmetry left to speak of, which is exactly the
point: the seed supplies the structure, and the repair supplies what the
structure could not.  Nothing here is clever — it is the standard local search
— so any advantage still comes from the seed the model chose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ramsey.graph import (
    RamseyGraph,
    find_triangle,
    maximum_independent_set,
)
from ramsey.kernel import capped_independence as capped_independence_number

if TYPE_CHECKING:
    import random


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    """What an edge-level repair reached, and what it cost."""

    evaluations: int
    start_independence: int
    best_independence: int
    best_graph: RamseyGraph
    independence_cap: int
    edges_added: int
    edges_removed: int

    @property
    def accepted(self) -> bool:
        """Report whether the repair reached a witness."""
        return self.best_independence <= self.independence_cap

    @property
    def gap(self) -> int:
        """Return how far the best graph sits above the cap."""
        return self.best_independence - self.independence_cap

    @property
    def improved(self) -> bool:
        """Report whether the repair beat the seed it started from."""
        return self.best_independence < self.start_independence

    def render(self) -> str:
        """Render the outcome, stating only measured facts."""
        head = (
            f"repair: {self.evaluations} evaluations, {self.edges_added} edges added, "
            f"{self.edges_removed} removed, independence "
            f"{self.start_independence} -> {self.best_independence}"
        )
        if self.accepted:
            return f"{head}. ACCEPTED (cap {self.independence_cap})."
        return f"{head}, {self.gap} above the cap of {self.independence_cap}."


def joinable_pairs(
    graph: RamseyGraph, vertices: tuple[int, ...], degree_cap: int
) -> tuple[tuple[int, int], ...]:
    """Return the pairs in ``vertices`` that may legally be joined.

    A pair is legal when the two are not already adjacent, share no common
    neighbour — joining them would close a triangle — and both have room under
    the degree cap.
    """
    legal = []
    for index, first in enumerate(vertices):
        for second in vertices[index + 1 :]:
            if graph.has_edge(first, second):
                continue
            if graph.adjacency[first] & graph.adjacency[second]:
                continue
            if graph.degree(first) >= degree_cap or graph.degree(second) >= degree_cap:
                continue
            legal.append((first, second))
    return tuple(legal)


def with_edge(graph: RamseyGraph, first: int, second: int) -> RamseyGraph:
    """Return the graph with one edge added."""
    masks = list(graph.adjacency)
    masks[first] |= 1 << second
    masks[second] |= 1 << first
    return RamseyGraph(order=graph.order, adjacency=tuple(masks))


def without_edge(graph: RamseyGraph, first: int, second: int) -> RamseyGraph:
    """Return the graph with one edge removed."""
    masks = list(graph.adjacency)
    masks[first] &= ~(1 << second)
    masks[second] &= ~(1 << first)
    return RamseyGraph(order=graph.order, adjacency=tuple(masks))


def repair_search(
    seed: RamseyGraph,
    independence_cap: int,
    degree_cap: int,
    *,
    steps: int,
    rng: random.Random,
    budget: int,
) -> RepairOutcome:
    """Repair a seed by adding edges inside its oversized independent sets.

    Each step destroys one obstruction if it legally can, and otherwise frees
    degree capacity by deleting a random edge.  The best graph seen is kept,
    so a run that wanders never returns something worse than its seed.
    """
    if find_triangle(seed) is not None:
        message = "the seed must be triangle-free"
        raise ValueError(message)
    ceiling = independence_cap + 1 + _GRADIENT_HEADROOM
    current = seed
    start = capped_independence_number(seed, ceiling)
    best_score, best_graph = start, seed
    evaluations = 1
    added = 0
    removed = 0
    for _ in range(steps):
        if evaluations >= budget or best_score <= independence_cap:
            break
        obstruction = maximum_independent_set(current).vertices
        legal = joinable_pairs(current, obstruction, degree_cap)
        if legal:
            first, second = legal[rng.randrange(len(legal))]
            current = with_edge(current, first, second)
            added += 1
        else:
            edges = current.edges()
            if not edges:
                break
            first, second = edges[rng.randrange(len(edges))]
            current = without_edge(current, first, second)
            removed += 1
        score = capped_independence_number(current, ceiling)
        evaluations += 1
        if score < best_score:
            best_score, best_graph = score, current
    return RepairOutcome(
        evaluations=evaluations,
        start_independence=start,
        best_independence=best_score,
        best_graph=best_graph,
        independence_cap=independence_cap,
        edges_added=added,
        edges_removed=removed,
    )


_GRADIENT_HEADROOM = 6
