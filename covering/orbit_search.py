"""Searching over base blocks rather than over designs.

This is the change that matters. A design of 53 blocks on 28 points is a point
in a space of about ``C(28,13)`` choose 53, and local search from a random
design does not find one - measured, it stalls around 500 uncovered subsets
where the published design has none.

Developing base blocks under a group collapses that space. Four base blocks
under the cyclic group already give 53 blocks, so the search is over four
13-subsets rather than 53, and a single point moved in one base block moves its
entire orbit coherently. The move is structural, not a perturbation of one
block among dozens.

The block count is also constrained arithmetically rather than chosen: an orbit
has size ``|G|`` over the size of its stabiliser, so every design in this space
has a block count that is a sum of divisors of the group order. A search here
cannot even express most of the designs that local search wastes its time on.

The model supplies the group and the number of base blocks. This module moves
the base blocks and measures; it never chooses the structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

from covering.design import Parameters, check_covering, rank_table
from covering.orbit import PointGroup, develop, orbit_sizes

if TYPE_CHECKING:
    import random


def random_block(parameters: Parameters, rng: random.Random) -> int:
    """Draw one uniformly random block of the right size."""
    block = 0
    for point in rng.sample(range(parameters.points), parameters.block_size):
        block |= 1 << point
    return block


def uncovered_count(
    parameters: Parameters, blocks: tuple[int, ...], ranks: dict[tuple[int, ...], int]
) -> int:
    """Return how many requirements the design leaves uncovered."""
    covered = bytearray(parameters.requirements)
    for block in blocks:
        members = [point for point in range(parameters.points) if block >> point & 1]
        for subset in combinations(members, parameters.strength):
            covered[ranks[subset]] = 1
    return parameters.requirements - sum(covered)


@dataclass(slots=True)
class OrbitSearch:
    """Hill-climb over base blocks developed under a fixed group."""

    parameters: Parameters
    group: PointGroup
    base_blocks: list[int]
    rng: random.Random
    ranks: dict[tuple[int, ...], int] = field(init=False)
    current: int = field(init=False)
    best: int = field(init=False)
    best_blocks: tuple[int, ...] = field(init=False)
    evaluations: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Develop the starting base blocks and adopt them."""
        self.ranks = rank_table(self.parameters)
        self.current = self.score(tuple(self.base_blocks))
        self.best = self.current
        self.best_blocks = tuple(self.base_blocks)

    def score(self, base: tuple[int, ...]) -> int:
        """Return the uncovered count of the design these base blocks develop to."""
        self.evaluations += 1
        return uncovered_count(self.parameters, develop(self.group, base), self.ranks)

    @property
    def blocks(self) -> int:
        """Return how many blocks the current base blocks develop to."""
        return len(develop(self.group, tuple(self.base_blocks)))

    def propose(self) -> tuple[int, ...]:
        """Move one point in one base block, which moves its whole orbit."""
        index = self.rng.randrange(len(self.base_blocks))
        block = self.base_blocks[index]
        inside = [p for p in range(self.parameters.points) if block >> p & 1]
        outside = [p for p in range(self.parameters.points) if not block >> p & 1]
        updated = (block | (1 << self.rng.choice(outside))) & ~(1 << self.rng.choice(inside))
        return tuple(
            updated if position == index else value
            for position, value in enumerate(self.base_blocks)
        )

    def step(self) -> bool:
        """Take one first-improvement step, reporting full coverage."""
        proposal = self.propose()
        score = self.score(proposal)
        if score <= self.current:
            self.base_blocks = list(proposal)
            self.current = score
        if score < self.best:
            self.best, self.best_blocks = score, proposal
        return self.best == 0

    def certify(self) -> tuple[int, bool]:
        """Return the block count of the best design and whether it is valid."""
        blocks = develop(self.group, self.best_blocks)
        return len(blocks), check_covering(self.parameters, blocks).accepted


def start(parameters: Parameters, group: PointGroup, count: int, rng: random.Random) -> OrbitSearch:
    """Begin a search from random base blocks."""
    return OrbitSearch(
        parameters=parameters,
        group=group,
        base_blocks=[random_block(parameters, rng) for _ in range(count)],
        rng=rng,
    )


def describe(group: PointGroup, base_blocks: tuple[int, ...]) -> str:
    """Render what a set of base blocks develops to."""
    sizes = orbit_sizes(group, base_blocks)
    return (
        f"{group.name} (order {group.order}) developing {len(base_blocks)} base blocks "
        f"-> orbit sizes {sizes}, {sum(sizes)} blocks before dedup"
    )
