"""Construction and repair for covering designs.

Two stages, deliberately ordinary so that any success is attributable to where
the search was aimed rather than to cleverness hidden here.

A greedy pass builds a covering from nothing: grow each block one point at a
time, always taking the point that covers the most still-uncovered subsets.
That reaches a valid design quickly but not a small one.

Local search then tries to shrink it.  Delete a block and the design becomes
invalid in a known, small way - only the subsets that block alone covered are
now uncovered - and repair moves exchange points between blocks to cover them
again.  Coverage is kept as a count per subset rather than a flag, so deleting
a block is exact rather than a rescan: a subset becomes uncovered precisely
when its count reaches zero.

The gate this exists to pass comes first: the engine must independently reach
the *published* block count before any attempt on one fewer is worth running.
A search that cannot find what is known to be findable has not earned a
frontier campaign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

from covering.design import Parameters, block_requirements, rank_table

if TYPE_CHECKING:
    import random


@dataclass(slots=True)
class Coverage:
    """Coverage counts over every requirement, maintained incrementally."""

    parameters: Parameters
    ranks: dict[tuple[int, ...], int] = field(init=False)
    counts: list[int] = field(init=False)
    uncovered: int = field(init=False)
    blocks: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Start from an empty design, with everything uncovered."""
        self.ranks = rank_table(self.parameters)
        self.counts = [0] * self.parameters.requirements
        self.uncovered = self.parameters.requirements

    def requirements_of(self, block: int) -> tuple[int, ...]:
        """Return the requirement indices a block covers."""
        return block_requirements(self.parameters, block, self.ranks)

    def add(self, block: int) -> None:
        """Add a block, updating only the counts it touches."""
        for index in self.requirements_of(block):
            if self.counts[index] == 0:
                self.uncovered -= 1
            self.counts[index] += 1
        self.blocks.append(block)

    def remove(self, position: int) -> int:
        """Remove the block at a position and return it."""
        block = self.blocks.pop(position)
        for index in self.requirements_of(block):
            self.counts[index] -= 1
            if self.counts[index] == 0:
                self.uncovered += 1
        return block

    def gain(self, block: int) -> int:
        """Return how many currently uncovered requirements a block would cover."""
        return sum(1 for index in self.requirements_of(block) if self.counts[index] == 0)

    def uncovered_subsets(self, limit: int = 0) -> list[int]:
        """Return indices of uncovered requirements, optionally truncated."""
        found = [index for index, count in enumerate(self.counts) if count == 0]
        return found[:limit] if limit else found


def _members(block: int, points: int) -> list[int]:
    return [point for point in range(points) if block >> point & 1]


def greedy_block(coverage: Coverage, rng: random.Random, seed_points: tuple[int, ...] = ()) -> int:
    """Grow one block greedily, taking the most productive point at each step.

    Ties are broken at random so that repeated calls explore different designs
    rather than returning the same block forever.
    """
    parameters = coverage.parameters
    chosen = list(seed_points)
    block = 0
    for point in chosen:
        block |= 1 << point
    while len(chosen) < parameters.block_size:
        best_gain = -1
        best: list[int] = []
        for point in range(parameters.points):
            if block >> point & 1:
                continue
            candidate = block | (1 << point)
            if candidate.bit_count() < parameters.strength:
                score = 0
            else:
                score = sum(
                    1
                    for subset in combinations(
                        _members(candidate, parameters.points), parameters.strength
                    )
                    if point in subset and coverage.counts[coverage.ranks[subset]] == 0
                )
            if score > best_gain:
                best_gain, best = score, [point]
            elif score == best_gain:
                best.append(point)
        pick = best[rng.randrange(len(best))]
        block |= 1 << pick
        chosen.append(pick)
    return block


def greedy_cover(parameters: Parameters, rng: random.Random) -> Coverage:
    """Build a valid covering from nothing, greedily."""
    coverage = Coverage(parameters=parameters)
    while coverage.uncovered:
        coverage.add(greedy_block(coverage, rng))
    return coverage


def repair(coverage: Coverage, rng: random.Random, steps: int) -> int:
    """Try to recover full coverage by exchanging points between blocks.

    Each step takes an uncovered subset and looks for a block already holding
    most of it, swapping one point in for one point out. The move is accepted
    when it does not increase the number of uncovered subsets, which lets the
    search drift sideways rather than only downhill.
    """
    parameters = coverage.parameters
    inverse = {index: subset for subset, index in coverage.ranks.items()}
    for _ in range(steps):
        if coverage.uncovered == 0:
            return 0
        target = inverse[rng.choice(coverage.uncovered_subsets())]
        needed = 0
        for point in target:
            needed |= 1 << point
        order = list(range(len(coverage.blocks)))
        rng.shuffle(order)
        for position in order:
            block = coverage.blocks[position]
            missing = [point for point in target if not block >> point & 1]
            if len(missing) != 1:
                continue
            present = [
                point for point in _members(block, parameters.points) if not needed >> point & 1
            ]
            if not present:
                continue
            before = coverage.uncovered
            removed = coverage.remove(position)
            outgoing = present[rng.randrange(len(present))]
            proposal = (removed | (1 << missing[0])) & ~(1 << outgoing)
            coverage.add(proposal)
            if coverage.uncovered > before:
                coverage.remove(len(coverage.blocks) - 1)
                coverage.add(removed)
            break
    return coverage.uncovered
