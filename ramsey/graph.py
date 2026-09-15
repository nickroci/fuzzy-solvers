"""Exact referee for triangle-free Ramsey graphs.

A ``(3, t, n)``-graph is a triangle-free graph on ``n`` vertices whose
independence number is at most ``t - 1``.  Such a graph witnesses
``R(3, t) > n``, so the classical Ramsey number ``R(3, t)`` is the least ``n``
for which no ``(3, t, n)``-graph exists.

This module owns validation and exact measurement only.  It contains no
search heuristic and no construction: everything here is the referee that
later search code must satisfy, and every acceptance carries a witness that
can be rechecked independently.

Graphs are stored as one integer bitmask per vertex, which makes the
independence search a sequence of machine-word operations rather than set
allocations.  ``order`` is small by construction (the campaign target is 40),
so the exact maximum-independent-set search below is affordable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

_GRAPH6_OFFSET: Final = 63
_GRAPH6_BITS: Final = 6
_GRAPH6_SMALL_ORDER: Final = 62
_GRAPH6_LONG_MARKER: Final = 126
_GRAPH6_LONG_BYTES: Final = 3
_GRAPH6_MAX_ORDER: Final = 258047


@dataclass(frozen=True, slots=True)
class IndependentSet:
    """An independent set of a graph, returned as a rechecked witness."""

    vertices: tuple[int, ...]

    @property
    def size(self) -> int:
        """Return the number of vertices in the set."""
        return len(self.vertices)


@dataclass(frozen=True, slots=True)
class Triangle:
    """Three mutually adjacent vertices, returned as a refutation witness."""

    vertices: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class RamseyGraph:
    """An undirected simple graph held as one adjacency bitmask per vertex.

    The constructor enforces the representation invariants that every other
    function in this package relies on and never re-checks: the mask tuple has
    exactly ``order`` entries, no vertex is adjacent to itself, no mask names a
    vertex outside the graph, and adjacency is symmetric.
    """

    order: int
    adjacency: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject any mask tuple that is not a simple undirected graph."""
        if self.order < 0:
            message = f"order must be non-negative, got {self.order}"
            raise ValueError(message)
        if len(self.adjacency) != self.order:
            message = f"expected {self.order} adjacency masks, got {len(self.adjacency)}"
            raise ValueError(message)
        full = (1 << self.order) - 1
        for vertex, mask in enumerate(self.adjacency):
            if mask & ~full:
                message = f"vertex {vertex} is adjacent to a vertex outside the graph"
                raise ValueError(message)
            if mask >> vertex & 1:
                message = f"vertex {vertex} is adjacent to itself"
                raise ValueError(message)
        for vertex, mask in enumerate(self.adjacency):
            for other in _iter_bits(mask):
                if not self.adjacency[other] >> vertex & 1:
                    message = f"adjacency is not symmetric for the pair ({vertex}, {other})"
                    raise ValueError(message)

    @property
    def full_mask(self) -> int:
        """Return the bitmask naming every vertex of the graph."""
        return (1 << self.order) - 1

    @property
    def size(self) -> int:
        """Return the number of edges."""
        return sum(mask.bit_count() for mask in self.adjacency) // 2

    def degree(self, vertex: int) -> int:
        """Return the degree of one vertex."""
        return self.adjacency[vertex].bit_count()

    def degrees(self) -> tuple[int, ...]:
        """Return every vertex degree, indexed by vertex."""
        return tuple(mask.bit_count() for mask in self.adjacency)

    def has_edge(self, first: int, second: int) -> bool:
        """Report whether two vertices are adjacent."""
        return bool(self.adjacency[first] >> second & 1)

    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return every edge once, as sorted vertex pairs in ascending order."""
        return tuple(
            (vertex, other)
            for vertex, mask in enumerate(self.adjacency)
            for other in _iter_bits(mask)
            if other > vertex
        )

    def closed_neighbourhood(self, vertex: int) -> int:
        """Return the mask of one vertex together with its neighbours."""
        return self.adjacency[vertex] | 1 << vertex

    def induced(self, vertices: int) -> RamseyGraph:
        """Return the subgraph induced on a vertex mask, renumbered from zero.

        The returned graph's vertices follow the ascending order of the set
        bits in ``vertices``; callers that need the original labels keep the
        mask alongside the result.
        """
        kept = tuple(_iter_bits(vertices))
        index = {vertex: position for position, vertex in enumerate(kept)}
        masks = []
        for vertex in kept:
            mask = 0
            for neighbour in _iter_bits(self.adjacency[vertex] & vertices):
                mask |= 1 << index[neighbour]
            masks.append(mask)
        return RamseyGraph(order=len(kept), adjacency=tuple(masks))


def graph_from_edges(order: int, edges: tuple[tuple[int, int], ...]) -> RamseyGraph:
    """Build a graph from an explicit edge list.

    Repeated edges are accepted and collapse to one; a self-loop or an
    out-of-range endpoint is rejected by the graph constructor.
    """
    masks = [0] * order
    for first, second in edges:
        if not (0 <= first < order and 0 <= second < order):
            message = f"edge ({first}, {second}) names a vertex outside a graph of order {order}"
            raise ValueError(message)
        masks[first] |= 1 << second
        masks[second] |= 1 << first
    return RamseyGraph(order=order, adjacency=tuple(masks))


def cycle_graph(order: int) -> RamseyGraph:
    """Build the cycle on ``order`` vertices, the smallest useful test case."""
    if order < _MIN_CYCLE_ORDER:
        message = f"a cycle needs at least {_MIN_CYCLE_ORDER} vertices, got {order}"
        raise ValueError(message)
    return graph_from_edges(order, tuple((v, (v + 1) % order) for v in range(order)))


def find_triangle(graph: RamseyGraph) -> Triangle | None:
    """Return one triangle of the graph, or ``None`` when it is triangle-free.

    The scan tests each edge's endpoints for a common neighbour, so it costs
    ``O(|E|)`` word operations rather than enumerating vertex triples.
    """
    for first, second in graph.edges():
        shared = graph.adjacency[first] & graph.adjacency[second]
        if shared:
            third = (shared & -shared).bit_length() - 1
            return Triangle(vertices=(first, second, third))
    return None


def is_triangle_free(graph: RamseyGraph) -> bool:
    """Report whether the graph contains no triangle."""
    return find_triangle(graph) is None


def maximum_independent_set(graph: RamseyGraph) -> IndependentSet:
    """Return a maximum independent set, computed exactly."""
    _, mask = _search_independent(graph, graph.full_mask, 0, 0, -1, 0, None)
    return IndependentSet(vertices=tuple(_iter_bits(mask)))


def independence_number(graph: RamseyGraph) -> int:
    """Return the exact independence number."""
    return maximum_independent_set(graph).size


def independent_set_of_size(graph: RamseyGraph, size: int) -> IndependentSet | None:
    """Return an independent set of at least ``size`` vertices, or ``None``.

    This is the decision form of :func:`maximum_independent_set` and stops at
    the first witness, so it is the cheap way to refute a bound.  A
    non-positive ``size`` is satisfied by the empty set.
    """
    return independent_set_within(graph, graph.full_mask, size)


def independent_set_within(graph: RamseyGraph, vertices: int, size: int) -> IndependentSet | None:
    """Return an independent set of ``size`` vertices drawn only from ``vertices``.

    Restricting by mask rather than by building an induced subgraph is what
    makes the extension search affordable: the residual independence of a
    vertex-deleted subgraph is queried once per search node, and each query
    must avoid re-allocating a graph.
    """
    if size <= 0:
        return IndependentSet(vertices=())
    best_size, mask = _search_independent(graph, vertices, 0, 0, -1, 0, size)
    if best_size < size:
        return None
    return IndependentSet(vertices=tuple(_iter_bits(mask)))


def capped_independence_number(graph: RamseyGraph, ceiling: int) -> int:
    """Return ``min(alpha, ceiling)``: exact below the ceiling, saturated at it.

    Hill-climbing needs a gradient, not an exact value: two hopeless
    constructions differ only in how hopeless they are, and paying for that
    difference exactly is waste.  Saturating the search at a ceiling keeps a
    refinement step affordable while still separating candidates near the
    target.
    """
    if ceiling <= 0:
        return 0
    best, _ = _search_independent(graph, graph.full_mask, 0, 0, -1, 0, ceiling)
    return best


def independence_number_within(graph: RamseyGraph, vertices: int) -> int:
    """Return the exact independence number of the subgraph induced on a mask."""
    best_size, _ = _search_independent(graph, vertices, 0, 0, -1, 0, None)
    return best_size


def has_independence_at_most(graph: RamseyGraph, bound: int) -> bool:
    """Report whether the independence number is at most ``bound``."""
    return independent_set_of_size(graph, bound + 1) is None


@dataclass(frozen=True, slots=True)
class RamseyCheck:
    """The referee's verdict on a candidate ``(3, t, n)``-graph.

    ``accepted`` is true only when both structural conditions hold.  Each
    failure carries the witness that refutes it, so a rejection can be
    rechecked without re-running the search that found it.
    """

    accepted: bool
    order: int
    clique_target: int
    independence_target: int
    triangle: Triangle | None
    independent_set: IndependentSet | None

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Return one line per violated condition, empty when accepted."""
        lines = []
        if self.triangle is not None:
            lines.append(f"contains a triangle on vertices {self.triangle.vertices}")
        if self.independent_set is not None:
            lines.append(
                f"contains an independent set of size {self.independent_set.size} "
                f"(at most {self.independence_target} allowed): {self.independent_set.vertices}"
            )
        return tuple(lines)


def check_ramsey_graph(graph: RamseyGraph, second_colour: int) -> RamseyCheck:
    """Check that ``graph`` is a ``(3, second_colour, order)``-graph.

    Accepts exactly when the graph is triangle-free and has independence
    number at most ``second_colour - 1``; such a graph proves
    ``R(3, second_colour) > graph.order``.
    """
    if second_colour < _MIN_SECOND_COLOUR:
        message = f"second_colour must be at least {_MIN_SECOND_COLOUR}, got {second_colour}"
        raise ValueError(message)
    triangle = find_triangle(graph)
    witness = independent_set_of_size(graph, second_colour)
    return RamseyCheck(
        accepted=triangle is None and witness is None,
        order=graph.order,
        clique_target=_TRIANGLE_ORDER,
        independence_target=second_colour - 1,
        triangle=triangle,
        independent_set=witness,
    )


def parse_graph6(line: str) -> RamseyGraph:
    """Parse one graph6 record, the format the published Ramsey censuses use.

    Handles the one-byte and the ``126``-prefixed three-byte order encodings;
    the four-byte encoding is beyond any order this campaign uses and is
    rejected rather than silently mis-parsed.
    """
    text = line.strip().removeprefix(">>graph6<<")
    if not text:
        message = "empty graph6 record"
        raise ValueError(message)
    data = [ord(character) - _GRAPH6_OFFSET for character in text]
    if any(value < 0 or value >= 1 << _GRAPH6_BITS for value in data):
        message = "graph6 record contains a character outside the printable range"
        raise ValueError(message)
    order, cursor = _parse_graph6_order(data)
    expected_bits = order * (order - 1) // 2
    expected_bytes = (expected_bits + _GRAPH6_BITS - 1) // _GRAPH6_BITS
    payload = data[cursor:]
    if len(payload) != expected_bytes:
        message = (
            f"graph6 record for order {order} needs {expected_bytes} payload bytes, "
            f"got {len(payload)}"
        )
        raise ValueError(message)
    masks = [0] * order
    bit = 0
    for column in range(1, order):
        for row in range(column):
            byte, offset = divmod(bit, _GRAPH6_BITS)
            if payload[byte] >> (_GRAPH6_BITS - 1 - offset) & 1:
                masks[row] |= 1 << column
                masks[column] |= 1 << row
            bit += 1
    return RamseyGraph(order=order, adjacency=tuple(masks))


def to_graph6(graph: RamseyGraph) -> str:
    """Render a graph as a graph6 record, inverting :func:`parse_graph6`."""
    if graph.order > _GRAPH6_MAX_ORDER:
        message = f"order {graph.order} exceeds the supported graph6 range"
        raise ValueError(message)
    if graph.order <= _GRAPH6_SMALL_ORDER:
        prefix = [graph.order]
    else:
        # Every byte is written with the +63 offset, so the literal 126 marker
        # is carried here as the value that offset restores.
        prefix = [
            _GRAPH6_LONG_MARKER - _GRAPH6_OFFSET,
            *_split_base64(graph.order, _GRAPH6_LONG_BYTES),
        ]
    bits = [
        graph.adjacency[row] >> column & 1
        for column in range(1, graph.order)
        for row in range(column)
    ]
    padding = (-len(bits)) % _GRAPH6_BITS
    bits.extend([0] * padding)
    payload = [
        int("".join(str(bit) for bit in bits[start : start + _GRAPH6_BITS]), 2)
        for start in range(0, len(bits), _GRAPH6_BITS)
    ]
    return "".join(chr(value + _GRAPH6_OFFSET) for value in (*prefix, *payload))


_MIN_CYCLE_ORDER: Final = 3
_MIN_SECOND_COLOUR: Final = 2
_TRIANGLE_ORDER: Final = 3


def _parse_graph6_order(data: list[int]) -> tuple[int, int]:
    """Return the graph order and the index at which the payload starts."""
    if data[0] != _GRAPH6_LONG_MARKER - _GRAPH6_OFFSET:
        return data[0], 1
    if len(data) < 1 + _GRAPH6_LONG_BYTES:
        message = "graph6 record is truncated inside its order prefix"
        raise ValueError(message)
    order = 0
    for value in data[1 : 1 + _GRAPH6_LONG_BYTES]:
        order = order << _GRAPH6_BITS | value
    return order, 1 + _GRAPH6_LONG_BYTES


def _split_base64(value: int, count: int) -> tuple[int, ...]:
    """Split an integer into ``count`` six-bit groups, most significant first."""
    return tuple(value >> (_GRAPH6_BITS * (count - 1 - index)) & 0x3F for index in range(count))


def _iter_bits(mask: int) -> tuple[int, ...]:
    """Return the ascending indices of the set bits of ``mask``."""
    indices = []
    remaining = mask
    while remaining:
        lowest = remaining & -remaining
        indices.append(lowest.bit_length() - 1)
        remaining ^= lowest
    return tuple(indices)


def _search_independent(
    graph: RamseyGraph,
    candidates: int,
    chosen: int,
    size: int,
    best_size: int,
    best_mask: int,
    target: int | None,
) -> tuple[int, int]:
    """Branch and bound for the maximum independent set over a candidate mask.

    ``target`` turns the search into a decision procedure: when set, the
    recursion returns as soon as a set of that size exists.  The bound prunes
    a branch whose remaining candidates cannot beat the incumbent, which is
    what keeps the search affordable on the sparse, small-independence graphs
    this campaign works with.
    """
    if size > best_size:
        best_size, best_mask = size, chosen
    if target is not None and best_size >= target:
        return best_size, best_mask
    if candidates == 0 or size + _clique_cover_bound(graph, candidates) <= best_size:
        return best_size, best_mask
    pivot = _highest_degree(graph, candidates)
    best_size, best_mask = _search_independent(
        graph,
        candidates & ~graph.closed_neighbourhood(pivot),
        chosen | 1 << pivot,
        size + 1,
        best_size,
        best_mask,
        target,
    )
    if target is not None and best_size >= target:
        return best_size, best_mask
    return _search_independent(
        graph,
        candidates & ~(1 << pivot),
        chosen,
        size,
        best_size,
        best_mask,
        target,
    )


def _clique_cover_bound(graph: RamseyGraph, candidates: int) -> int:
    """Bound the independence number of an induced subgraph by a clique cover.

    An independent set takes at most one vertex from each clique of a cover, so
    ``alpha <= |S| - nu`` for any matching of size ``nu``: each matched edge is
    a two-vertex clique that absorbs two vertices while contributing one.  In a
    triangle-free graph every clique is a vertex or an edge, so a matching *is*
    the best cover of this shape, and a greedy maximal one is enough to make
    the bound sound.  Equivalently this is Gallai's identity relaxed through
    ``tau >= nu``.

    This replaces the trivial ``|S|`` bound and is what makes the search
    tractable past a few dozen vertices.
    """
    remaining = candidates
    matched = 0
    while remaining:
        lowest = remaining & -remaining
        remaining ^= lowest
        vertex = lowest.bit_length() - 1
        partners = graph.adjacency[vertex] & remaining
        if partners:
            remaining ^= partners & -partners
            matched += 1
    return candidates.bit_count() - matched


def _highest_degree(graph: RamseyGraph, candidates: int) -> int:
    """Return the candidate vertex with the most neighbours still in play.

    Branching on the busiest vertex shrinks the candidate mask fastest in the
    inclusion branch, which is the cheaper of the two to exhaust.
    """
    best_vertex = -1
    best_degree = -1
    for vertex in _iter_bits(candidates):
        degree = (graph.adjacency[vertex] & candidates).bit_count()
        if degree > best_degree:
            best_vertex, best_degree = vertex, degree
    return best_vertex
