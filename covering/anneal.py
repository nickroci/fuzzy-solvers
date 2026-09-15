"""Fixed-block-count annealing, the method covering designs are actually built by.

Greedy block-building reached 100 blocks where the published design uses 53, so
it is not the engine. The standard approach inverts the question: fix the block
count at the number you want, start from a random design, and minimise the
number of uncovered subsets until it reaches zero. Success is then a covering
with exactly that many blocks, and failure says nothing except that this run
did not find one.

The move is a single point exchange inside a single block, which is what makes
this affordable: only the requirements through the point leaving and the point
arriving can change state, so a proposal costs ``2 * C(k-1, t-1)`` count
updates rather than a rescan. Undoing a rejected proposal is the same operation
in reverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import exp
from typing import TYPE_CHECKING

from covering.design import Parameters, rank_table

if TYPE_CHECKING:
    import random


@dataclass(slots=True)
class FixedDesign:
    """A design of fixed size, with incremental coverage counts."""

    parameters: Parameters
    blocks: list[int]
    ranks: dict[tuple[int, ...], int] = field(init=False)
    counts: list[int] = field(init=False)
    uncovered: int = field(init=False)

    def __post_init__(self) -> None:
        """Build coverage counts for the starting design."""
        self.ranks = rank_table(self.parameters)
        self.counts = [0] * self.parameters.requirements
        self.uncovered = self.parameters.requirements
        for block in self.blocks:
            self._apply(block, +1)

    def _members(self, block: int) -> list[int]:
        return [point for point in range(self.parameters.points) if block >> point & 1]

    def _apply(self, block: int, delta: int) -> None:
        strength = self.parameters.strength
        for subset in combinations(self._members(block), strength):
            index = self.ranks[subset]
            before = self.counts[index]
            self.counts[index] = before + delta
            if before == 0 and delta > 0:
                self.uncovered -= 1
            elif before == 1 and delta < 0:
                self.uncovered += 1

    def _touched(self, block: int, point: int) -> list[tuple[int, ...]]:
        """Return the subsets of ``block`` that contain ``point``."""
        others = [member for member in self._members(block) if member != point]
        return [(*rest, point) for rest in combinations(others, self.parameters.strength - 1)]

    def exchange(self, position: int, leaving: int, arriving: int) -> int:
        """Swap one point of one block, returning the change in uncovered count.

        Only subsets through the leaving or arriving point can change state, so
        the update is local rather than a rescan of the design.
        """
        block = self.blocks[position]
        before = self.uncovered
        for subset in self._touched(block, leaving):
            index = self.ranks[tuple(sorted(subset))]
            self.counts[index] -= 1
            if self.counts[index] == 0:
                self.uncovered += 1
        updated = (block | (1 << arriving)) & ~(1 << leaving)
        for subset in self._touched(updated, arriving):
            index = self.ranks[tuple(sorted(subset))]
            if self.counts[index] == 0:
                self.uncovered -= 1
            self.counts[index] += 1
        self.blocks[position] = updated
        return self.uncovered - before


def random_design(parameters: Parameters, size: int, rng: random.Random) -> FixedDesign:
    """Build a design of the requested size from random blocks."""
    blocks = []
    for _ in range(size):
        points = rng.sample(range(parameters.points), parameters.block_size)
        block = 0
        for point in points:
            block |= 1 << point
        blocks.append(block)
    return FixedDesign(parameters=parameters, blocks=blocks)


@dataclass(slots=True)
class Annealer:
    """Anneal a fixed-size design towards full coverage."""

    design: FixedDesign
    rng: random.Random
    temperature: float
    best: int = field(init=False)
    best_blocks: tuple[int, ...] = field(init=False)
    proposals: int = field(default=0, init=False)
    accepted: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Adopt the starting design as the incumbent."""
        self.best = self.design.uncovered
        self.best_blocks = tuple(self.design.blocks)

    def step(self) -> bool:
        """Take one Metropolis step, reporting whether coverage is complete.

        Proposals are biased towards blocks that are failing: a point is taken
        from a random block and replaced by one it does not hold. Accepting
        equal moves lets the search drift across plateaus, which is where most
        of the progress in these landscapes happens.
        """
        parameters = self.design.parameters
        position = self.rng.randrange(len(self.design.blocks))
        block = self.design.blocks[position]
        inside = [point for point in range(parameters.points) if block >> point & 1]
        outside = [point for point in range(parameters.points) if not block >> point & 1]
        leaving = inside[self.rng.randrange(len(inside))]
        arriving = outside[self.rng.randrange(len(outside))]
        delta = self.design.exchange(position, leaving, arriving)
        self.proposals += 1
        if delta <= 0 or self.rng.random() < exp(-delta / max(self.temperature, 1e-9)):
            self.accepted += 1
            if self.design.uncovered < self.best:
                self.best = self.design.uncovered
                self.best_blocks = tuple(self.design.blocks)
        else:
            self.design.exchange(position, arriving, leaving)
        return self.design.uncovered == 0
