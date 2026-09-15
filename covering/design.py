"""Exact referee for covering designs.

A ``(v, k, t)``-covering is a family of ``k``-subsets of a ``v``-set such that
every ``t``-subset lies inside at least one of them.  ``C(v, k, t)`` is the
least number of blocks that suffices, and the published repositories record the
best construction anyone has found.  Beating a listed block count by one is a
new entry.

Two properties make this a better-shaped target than the two that failed
before.  Verification is a scan, not a search: mark every ``t``-subset of every
block and check nothing is left, which is ``B * C(k, t)`` set operations with
no branching anywhere.  And the published counts sit well above the lower
bounds - 53 against 39 for ``C(28, 13, 4)`` - so an improvement is very likely
to *exist*, unlike a code table cell whose gap of one may simply mean the
published value is already optimal.

This module owns validation and exact measurement only.  Blocks are bitmasks
over the point set and subsets are addressed by combinatorial rank, so coverage
is an array of counts rather than a set of tuples: a block edit then touches a
computable handful of entries instead of forcing a rescan.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb


@dataclass(frozen=True, slots=True)
class Parameters:
    """The shape of a covering problem."""

    points: int
    block_size: int
    strength: int

    def __post_init__(self) -> None:
        """Reject a shape that does not describe a covering."""
        if not 1 <= self.strength <= self.block_size <= self.points:
            message = (
                f"need 1 <= t <= k <= v, got t={self.strength}, "
                f"k={self.block_size}, v={self.points}"
            )
            raise ValueError(message)

    @property
    def requirements(self) -> int:
        """Return how many ``t``-subsets must be covered."""
        return comb(self.points, self.strength)

    @property
    def per_block(self) -> int:
        """Return how many requirements a single block covers."""
        return comb(self.block_size, self.strength)

    @property
    def schonheim(self) -> int:
        """Return the Schonheim lower bound on the number of blocks.

        The recursion is ``L(v,k,t) = ceil(v/k * L(v-1,k-1,t-1))`` anchored at
        ``L(v,k,1) = ceil(v/k)``, so the ceilings nest from the innermost level
        outwards.  Evaluating them in the other order silently gives a weaker
        number: for ``C(6,3,2)`` it returns 5 where the true bound is 6.

        This is what says how much room a published count still leaves, so it
        is worth getting right: for ``C(28,13,4)`` it gives 39, which is the
        lower bound the repository publishes against an incumbent of 53.
        """
        bound = 1
        for level in range(self.strength - 1, -1, -1):
            bound = -(-(self.points - level) * bound // (self.block_size - level))
        return bound

    def describe(self) -> str:
        """Render the problem in repository notation."""
        return (
            f"C({self.points},{self.block_size},{self.strength}): "
            f"{self.requirements:,} subsets to cover, "
            f"{self.per_block:,} per block, Schonheim bound {self.schonheim}"
        )


def rank_table(parameters: Parameters) -> dict[tuple[int, ...], int]:
    """Return a map from each ``t``-subset to its index.

    Indexing subsets by position rather than by tuple is what lets coverage be
    a flat array of counts, so a block edit updates a known set of entries.
    """
    return {
        subset: index
        for index, subset in enumerate(combinations(range(parameters.points), parameters.strength))
    }


def block_requirements(
    parameters: Parameters, block: int, ranks: dict[tuple[int, ...], int]
) -> tuple[int, ...]:
    """Return the indices of every requirement one block covers."""
    members = tuple(point for point in range(parameters.points) if block >> point & 1)
    return tuple(ranks[subset] for subset in combinations(members, parameters.strength))


@dataclass(frozen=True, slots=True)
class CoveringCheck:
    """The referee's verdict on a claimed covering."""

    accepted: bool
    parameters: Parameters
    blocks: int
    uncovered: int
    malformed: tuple[str, ...]

    def render(self) -> str:
        """Render the verdict as a repository entry or the reason it is not one."""
        if self.accepted:
            return (
                f"verified ({self.parameters.points},{self.parameters.block_size},"
                f"{self.parameters.strength}) covering with {self.blocks} blocks"
            )
        if self.malformed:
            return "; ".join(self.malformed)
        return f"{self.uncovered:,} of {self.parameters.requirements:,} subsets uncovered"


def check_covering(parameters: Parameters, blocks: tuple[int, ...]) -> CoveringCheck:
    """Check that every ``t``-subset lies in some block.

    Deliberately independent of any search state: it rebuilds coverage from the
    blocks alone, so a bug in an incremental counter cannot make a bad design
    look valid.
    """
    malformed = []
    limit = 1 << parameters.points
    for index, block in enumerate(blocks):
        if block < 0 or block >= limit:
            malformed.append(f"block {index} names a point outside the {parameters.points}-set")
        elif block.bit_count() != parameters.block_size:
            malformed.append(
                f"block {index} has {block.bit_count()} points, not {parameters.block_size}"
            )
    if malformed:
        return CoveringCheck(
            accepted=False,
            parameters=parameters,
            blocks=len(blocks),
            uncovered=parameters.requirements,
            malformed=tuple(malformed),
        )
    ranks = rank_table(parameters)
    covered = bytearray(parameters.requirements)
    for block in blocks:
        for index in block_requirements(parameters, block, ranks):
            covered[index] = 1
    uncovered = parameters.requirements - sum(covered)
    return CoveringCheck(
        accepted=uncovered == 0,
        parameters=parameters,
        blocks=len(blocks),
        uncovered=uncovered,
        malformed=(),
    )


def uncovered_subsets(
    parameters: Parameters, blocks: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """Return the ``t``-subsets no block contains, in rank order.

    A search that reports only *how many* requirements are missing throws away
    what a constructor needs most.  Fourteen missing subsets that form a single
    orbit under the group are one hole in the algebra; fourteen scattered ones
    are a design that is simply short of blocks, and the two call for opposite
    moves.
    """
    ranks = rank_table(parameters)
    covered = bytearray(parameters.requirements)
    for block in blocks:
        for index in block_requirements(parameters, block, ranks):
            covered[index] = 1
    return tuple(
        subset
        for index, subset in enumerate(combinations(range(parameters.points), parameters.strength))
        if not covered[index]
    )


def point_deficits(parameters: Parameters, missing: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Return, per point, how many missing requirements contain it."""
    counts = [0] * parameters.points
    for subset in missing:
        for point in subset:
            counts[point] += 1
    return tuple(counts)
