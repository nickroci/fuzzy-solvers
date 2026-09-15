"""Cayley-graph construction primitives: the instruction set for the solver.

Every published lower bound on ``R(3, k)`` in the higher range is an explicit
graph, and almost all of them are Cayley graphs — circulants when the group is
cyclic, and non-cyclic Cayley colorings otherwise.  This module owns the
construction and nothing else: it turns a group and a connection set into a
graph, and it decides triangle-freeness structurally rather than by search.

The model chooses the group and the connection set.  This module builds what
it asks for; :mod:`ramsey.graph` measures the result exactly.  Nothing
here scores, ranks, or searches, so a construction can never flatter itself.

For an abelian group ``G`` and a connection set ``S`` closed under negation
and excluding the identity, the Cayley graph has an edge ``{x, x + s}`` for
every ``s in S``.  A triangle is ``x, x + a, x + a + b`` with ``a``, ``b`` and
``a + b`` all in ``S``, so the graph is triangle-free exactly when ``S`` is
**sum-free**.  That reduces the expensive structural condition to arithmetic
on the connection set, which is what makes wide sweeps affordable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, prod

from ramsey.graph import RamseyGraph


@dataclass(frozen=True, slots=True)
class AbelianGroup:
    """A finite abelian group ``Z_m1 x ... x Z_mr`` with elements indexed.

    Elements are addressed by a single integer in ``[0, order)`` under
    mixed-radix positional encoding, so a connection set is a set of ints and
    group arithmetic stays cheap.
    """

    moduli: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject a factor list that does not describe a finite group."""
        if not self.moduli:
            message = "a group needs at least one factor"
            raise ValueError(message)
        if any(modulus < 1 for modulus in self.moduli):
            message = f"every modulus must be positive, got {self.moduli}"
            raise ValueError(message)

    @property
    def order(self) -> int:
        """Return the number of group elements."""
        return prod(self.moduli)

    @property
    def is_cyclic(self) -> bool:
        """Report whether the group is cyclic.

        By the Chinese remainder theorem a product of cyclic factors is cyclic
        exactly when the factor orders are pairwise coprime, so ``Z2 x Z37`` is
        ``Z74`` and every Cayley graph over it is a circulant.  Presentation is
        not structure, and only structure decides whether a published cyclic
        exhaustion already covers an order.
        """
        factors = [modulus for modulus in self.moduli if modulus > 1]
        return all(
            gcd(first, second) == 1
            for index, first in enumerate(factors)
            for second in factors[index + 1 :]
        )

    @property
    def name(self) -> str:
        """Render the group in the usual product notation."""
        return " x ".join(f"Z{modulus}" for modulus in self.moduli)

    def coordinates(self, index: int) -> tuple[int, ...]:
        """Return the mixed-radix coordinates of an element index."""
        digits = []
        remaining = index
        for modulus in reversed(self.moduli):
            digits.append(remaining % modulus)
            remaining //= modulus
        return tuple(reversed(digits))

    def index(self, coordinates: tuple[int, ...]) -> int:
        """Return the element index of a coordinate tuple."""
        if len(coordinates) != len(self.moduli):
            message = f"expected {len(self.moduli)} coordinates, got {len(coordinates)}"
            raise ValueError(message)
        value = 0
        for digit, modulus in zip(coordinates, self.moduli, strict=True):
            value = value * modulus + digit % modulus
        return value

    def add(self, first: int, second: int) -> int:
        """Return the index of the sum of two elements."""
        return self.index(
            tuple(
                (left + right) % modulus
                for left, right, modulus in zip(
                    self.coordinates(first), self.coordinates(second), self.moduli, strict=True
                )
            )
        )

    def negate(self, index: int) -> int:
        """Return the index of the additive inverse of an element."""
        return self.index(
            tuple(
                (-digit) % modulus
                for digit, modulus in zip(self.coordinates(index), self.moduli, strict=True)
            )
        )


def symmetric_closure(group: AbelianGroup, generators: tuple[int, ...]) -> tuple[int, ...]:
    """Close a generator list under negation, dropping the identity.

    A Cayley graph is undirected only when its connection set is symmetric, so
    this is the canonical way to turn a model's proposal into a legal one.
    """
    closed: set[int] = set()
    for generator in generators:
        element = generator % group.order
        if element == 0:
            continue
        closed.add(element)
        closed.add(group.negate(element))
    return tuple(sorted(closed))


def sum_violation(group: AbelianGroup, connection: tuple[int, ...]) -> tuple[int, int] | None:
    """Return a pair of connection elements whose sum is also in the set.

    Such a pair is exactly a triangle through the identity, so a ``None``
    result certifies triangle-freeness without touching the graph.
    """
    members = set(connection)
    for first in connection:
        for second in connection:
            if group.add(first, second) in members:
                return (first, second)
    return None


def is_sum_free(group: AbelianGroup, connection: tuple[int, ...]) -> bool:
    """Report whether a connection set induces a triangle-free Cayley graph."""
    return sum_violation(group, connection) is None


def cayley_graph(group: AbelianGroup, connection: tuple[int, ...]) -> RamseyGraph:
    """Build the Cayley graph of ``group`` with the given connection set.

    The connection set must already be symmetric and identity-free; use
    :func:`symmetric_closure` first.  Degree is ``len(connection)`` at every
    vertex, which is the constraint that matters: a triangle-free graph has
    independence number at least its maximum degree, so a connection set
    larger than ``k - 1`` cannot witness ``R(3, k)``.
    """
    order = group.order
    members = set(connection)
    if 0 in members:
        message = "the connection set must not contain the identity"
        raise ValueError(message)
    if any(group.negate(element) not in members for element in connection):
        message = "the connection set must be closed under negation"
        raise ValueError(message)
    masks = [0] * order
    for vertex in range(order):
        mask = 0
        for element in connection:
            mask |= 1 << group.add(vertex, element)
        masks[vertex] = mask
    return RamseyGraph(order=order, adjacency=tuple(masks))


def circulant(order: int, steps: tuple[int, ...]) -> RamseyGraph:
    """Build a circulant graph, the cyclic special case of a Cayley graph."""
    group = AbelianGroup(moduli=(order,))
    return cayley_graph(group, symmetric_closure(group, steps))


def integer_partitions(total: int) -> tuple[tuple[int, ...], ...]:
    """Return every partition of ``total`` in non-increasing order."""
    if total == 0:
        return ((),)
    found: list[tuple[int, ...]] = []
    for first in range(total, 0, -1):
        found.extend(
            (first, *rest)
            for rest in integer_partitions(total - first)
            if not rest or rest[0] <= first
        )
    return tuple(found)


def prime_factorisation(order: int) -> tuple[tuple[int, int], ...]:
    """Return the prime factorisation of ``order`` as (prime, exponent) pairs."""
    if order < 1:
        message = f"order must be positive, got {order}"
        raise ValueError(message)
    factors: list[tuple[int, int]] = []
    remaining = order
    divisor = 2
    while divisor * divisor <= remaining:
        exponent = 0
        while remaining % divisor == 0:
            remaining //= divisor
            exponent += 1
        if exponent:
            factors.append((divisor, exponent))
        divisor += 1
    if remaining > 1:
        factors.append((remaining, 1))
    return tuple(factors)


def abelian_groups(order: int) -> tuple[AbelianGroup, ...]:
    """Return every abelian group of the given order, up to isomorphism.

    By the structure theorem an abelian group is a product of prime-power
    cyclic factors, one choice of partition per prime exponent.  Enumerating
    them is what tells the search whether an order admits anything beyond the
    circulants — and for a squarefree order it does not, which is exactly why
    the published cyclic exhaustions settle those cases.
    """
    options: list[tuple[tuple[int, ...], ...]] = []
    for prime, exponent in prime_factorisation(order):
        options.append(
            tuple(
                tuple(prime**part for part in partition)
                for partition in integer_partitions(exponent)
            )
        )
    groups: list[tuple[int, ...]] = [()]
    for choices in options:
        groups = [(*existing, *choice) for existing in groups for choice in choices]
    return tuple(AbelianGroup(moduli=tuple(sorted(moduli))) for moduli in groups)
