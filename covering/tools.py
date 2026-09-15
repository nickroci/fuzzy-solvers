"""The model-facing surface: the agent authors the algebra, the kernel searches.

A covering of this size is not found by perturbing a design, it is
*constructed*: a few base blocks developed under a group.  So the model's job
here is the algebra.  It names a group -- by a known family, or by writing its
own permutation in cycle notation -- and says how many base blocks to develop.

Before any search runs, the surface answers an arithmetic question the model
should be reasoning about and a random search cannot even ask: a block is a
union of its stabiliser's orbits on points, so the achievable orbit sizes are
fixed by the group, and a target block count is either a sum of them or it is
unreachable in that group however long anyone searches.  That check is not
advice; it is a fact about the group the model just named, reported before it
spends anything.

When a search falls short the surface reports *what* is missing rather than
only how much.  Fourteen uncovered subsets forming one orbit under the group is
a hole in the algebra and calls for a different stabiliser; fourteen scattered
ones are a design short of blocks and call for another base block.  A bare
count cannot tell those apart, and it is the distinction the whole method turns
on.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from covering.design import (
    Parameters,
    check_covering,
    point_deficits,
    uncovered_subsets,
)
from covering.kernel import Schedule, anneal
from covering.orbit import (
    PointGroup,
    admissible_orbit_sizes,
    cyclic,
    cyclic_with_fixed,
    develop,
    expressible,
    frobenius,
    from_cycles,
    grid_translations,
    multiplier,
    orbit_sizes,
    transitive_block_count_constraint,
)
from covering.orbit_search import random_block

_MAX_SEARCH_SHARE = 0.35
_MOVES_PER_SECOND = 600_000
_MAX_MOVES_PER_RESTART = 20_000_000
_REPORTED_ORBITS = 6
_REPORTED_POINTS = 8


class BudgetExhaustedError(RuntimeError):
    """Raised when a search is requested past the session's move budget."""

    def __init__(self, budget: int) -> None:
        super().__init__(f"move budget of {budget:,} is exhausted")
        self.budget = budget


@dataclass(frozen=True, slots=True)
class Feasibility:
    """Whether a group can express a block count at all."""

    group: str
    order: int
    sizes: tuple[int, ...]
    target: int
    reachable: bool
    divisor: int

    def render(self) -> str:
        """State the arithmetic, and what it rules in or out."""
        if not self.reachable:
            verdict = (
                f"{self.target} blocks is NOT a sum of these orbit sizes, so no base "
                f"blocks in this group can produce that count. A block is a union of "
                f"its stabiliser's orbits on points, which fixes the achievable orbit "
                f"sizes. Choose a group whose order shares a factor with the block size."
            )
        elif self.target % self.divisor:
            verdict = (
                f"{self.target} blocks is a sum of these orbit sizes, but under a "
                f"point-transitive group v*r = B*k forces {self.divisor} to divide the "
                f"block count, and it does not divide {self.target}. Unreachable."
            )
        else:
            verdict = f"{self.target} blocks is reachable in this group."
        return f"{self.group}: order {self.order}, achievable orbit sizes {self.sizes}. {verdict}"


@dataclass(frozen=True, slots=True)
class Banked:
    """A verified covering that beats the published block count."""

    parameters: Parameters
    blocks: tuple[int, ...]
    published: int
    group: str

    def render(self) -> str:
        """Render the result as a repository entry."""
        return (
            f"({self.parameters.points},{self.parameters.block_size},"
            f"{self.parameters.strength}) covering with {len(self.blocks)} blocks, "
            f"beating the published {self.published}. Built on {self.group}."
        )


@dataclass(frozen=True, slots=True)
class Incumbent:
    """The best design a session has reached, kept so it can be resumed."""

    group: PointGroup
    base_blocks: tuple[int, ...]
    uncovered: int
    block_count: int

    @property
    def blocks(self) -> tuple[int, ...]:
        """Return the developed design."""
        return develop(self.group, self.base_blocks)


def _subset_orbits(group: PointGroup, missing: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Return the sizes of the group orbits the missing requirements fall into."""
    remaining = {tuple(subset) for subset in missing}
    generators = group.generators
    sizes: list[int] = []
    while remaining:
        start = next(iter(remaining))
        seen = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for generator in generators:
                image = tuple(sorted(generator[point] for point in current))
                if image not in seen:
                    seen.add(image)
                    frontier.append(image)
        sizes.append(len(seen & remaining))
        remaining -= seen
    return tuple(sorted(sizes, reverse=True))


@dataclass(slots=True)
class CoveringTools:
    """Tool surface over one covering target and one move budget."""

    parameters: Parameters
    published: int
    lower_bound: int
    budget: int
    provenance: str = ""
    _used: int = field(default=0, init=False)
    _incumbent: Incumbent | None = field(default=None, init=False)
    _last: Incumbent | None = field(default=None, init=False)
    _banked: Banked | None = field(default=None, init=False)

    @property
    def used(self) -> int:
        """Return how many moves have been spent."""
        return self._used

    @property
    def remaining(self) -> int:
        """Return the moves still available."""
        return self.budget - self._used

    @property
    def banked(self) -> Banked | None:
        """Return the secured result, if one was found."""
        return self._banked

    @property
    def last_outcome(self) -> Incumbent | None:
        """Return the design just evaluated, which is not always the best one.

        The board records what an arm tried. Reporting the incumbent instead
        attributes the best design's block count and shortfall to whatever
        group was named last, which is how the shared history came to claim
        that 24 base blocks under the trivial group develop to 312 blocks.
        """
        return self._last

    @property
    def incumbent_uncovered(self) -> int:
        """Return the shortfall of the best design in contention, or -1 if there is none.

        A design with too many blocks reports the full requirement count,
        because reporting zero for something that cannot be an entry reads as
        success in every log and summary downstream - and one live session
        finished with "best_uncovered=0" having never built a design of the
        right size.
        """
        best = self._incumbent
        if best is None:
            return -1
        if best.block_count > self.target_blocks:
            return self.parameters.requirements
        return best.uncovered

    @property
    def incumbent_blocks(self) -> int:
        """Return how many blocks the best design so far develops to."""
        return 0 if self._incumbent is None else len(self._incumbent.blocks)

    @property
    def target_blocks(self) -> int:
        """Return the block count a new repository entry needs."""
        return self.published - 1

    def describe_target(self) -> str:
        """State the problem, the incumbent, and what a new entry needs."""
        entry = (
            f"Published best: {self.published} blocks. A valid covering with "
            f"{self.target_blocks} blocks is a NEW REPOSITORY ENTRY."
        )
        room = (
            f"Best known lower bound: {self.lower_bound}, so {self.target_blocks} "
            f"blocks is not ruled out by anything published."
        )
        lines = [self.parameters.describe(), entry, room]
        if self.provenance:
            lines.append(f"The published value was set by: {self.provenance}.")
        lines.append(
            "Designs of this size are built by developing a few base blocks under a "
            "group, not by local search over arbitrary designs. Name a group, check "
            "it can express the block count, then search inside it."
        )
        incidences = self.target_blocks * self.parameters.block_size
        degree = incidences / self.parameters.points
        if incidences % self.parameters.points:
            lines.append(
                f"Counting: {self.target_blocks} blocks of {self.parameters.block_size} "
                f"points is {incidences} incidences over {self.parameters.points} points, "
                f"or {degree:.2f} each. That is not a whole number, so the points cannot "
                f"all lie in equally many blocks, so NO POINT-TRANSITIVE GROUP can act on "
                f"a design of this size. Groups that move every point to every other are "
                f"ruled out before you try them."
            )
        else:
            lines.append(
                f"Counting: {self.target_blocks} blocks of {self.parameters.block_size} "
                f"points is {incidences} incidences over {self.parameters.points} points, "
                f"exactly {int(degree)} each, so a point-transitive group is not excluded "
                f"by counting alone."
            )
        lines.append(
            f"The kernel runs roughly {_MOVES_PER_SECOND:,} moves a second, so ten "
            f"million moves is about fifteen seconds and a hundred million is about "
            f"three minutes. You have {self.budget:,} moves. Spend them in many "
            f"moderate searches rather than a few enormous ones: you learn nothing "
            f"while one is running, and inspect() between them is free."
        )
        return "\n".join(lines)

    def named_group(self, family: str, argument: int = 0) -> PointGroup:
        """Return a group from a named family."""
        points = self.parameters.points
        builders = {
            "cyclic": lambda: cyclic(points),
            "cyclic_fixed": lambda: cyclic_with_fixed(points, argument or 1),
            "multiplier": lambda: multiplier(points, argument),
            "frobenius": lambda: frobenius(points, argument),
        }
        builder = builders.get(family)
        if builder is None:
            message = f"unknown family {family!r}; try {sorted(builders)}"
            raise ValueError(message)
        return builder()

    def authored_group(self, cycles: tuple[tuple[int, ...], ...], name: str) -> PointGroup:
        """Return a group the model wrote itself, in cycle notation."""
        return from_cycles(self.parameters.points, cycles, name)

    def grid_group(self, rows: int, columns: int) -> PointGroup:
        """Return the translation group of a rows-by-columns grid on the points.

        A single permutation only ever generates a cyclic group, so a product
        group has to be named by its generators; this is the common case.
        """
        return grid_translations(rows, columns)

    def trivial_group(self) -> PointGroup:
        """Return the one-element group, for searching designs with no symmetry."""
        points = self.parameters.points
        return PointGroup(points=points, generators=(tuple(range(points)),), name="1")

    def check_feasible(self, group: PointGroup, target: int) -> Feasibility:
        """Report whether a group can express a block count. Costs nothing."""
        sizes = admissible_orbit_sizes(group, self.parameters.block_size)
        divisor = transitive_block_count_constraint(
            group, self.parameters.points, self.parameters.block_size
        )
        return Feasibility(
            group=group.name,
            order=group.order,
            sizes=sizes,
            target=target,
            reachable=expressible(sizes, target),
            divisor=divisor,
        )

    def _spend(self, moves: int) -> int:
        """Return the moves actually affordable, refusing an exhausted budget.

        A single restart is capped as well as a single call. Nothing is learned
        while one is running, and a session that asks for its whole budget in
        one anneal spends the wall clock it needed for a dozen restarts and
        comes back with one sample.
        """
        if self.remaining <= 0:
            raise BudgetExhaustedError(self.budget)
        ceiling = min(int(self.budget * _MAX_SEARCH_SHARE), _MAX_MOVES_PER_RESTART, self.remaining)
        return max(min(moves, ceiling), 1)

    def _rank(self, block_count: int, missing: int) -> tuple[int, int, int]:
        """Order designs as candidates, lowest first.

        A design with more blocks than the target cannot be a new entry however
        well it covers, so it never displaces one that can. Without this, a
        session that stumbles on a perfect oversized covering adopts it as its
        incumbent and then resumes from it, throwing away the design that was
        actually in contention - which is how one live session spent its last
        ten tool calls.
        """
        return (0 if block_count <= self.target_blocks else 1, missing, block_count)

    def _record(self, group: PointGroup, base_blocks: tuple[int, ...], missing: int) -> str:
        """Adopt a result as the incumbent, banking it when it is a new entry."""
        blocks = develop(group, base_blocks)
        self._last = Incumbent(
            group=group,
            base_blocks=base_blocks,
            uncovered=missing,
            block_count=len(blocks),
        )
        best = self._incumbent
        if best is None or self._rank(len(blocks), missing) < self._rank(
            best.block_count, best.uncovered
        ):
            self._incumbent = Incumbent(
                group=group,
                base_blocks=base_blocks,
                uncovered=missing,
                block_count=len(blocks),
            )
        note = ""
        if missing == 0 and len(blocks) < self.published:
            check = check_covering(self.parameters, blocks)
            if check.accepted:
                self._banked = Banked(
                    parameters=self.parameters,
                    blocks=blocks,
                    published=self.published,
                    group=group.name,
                )
                note = f" NEW ENTRY: verified covering with {len(blocks)} blocks."
        if len(blocks) > self.target_blocks:
            note += (
                f" WARNING: this develops to {len(blocks)} blocks and a new entry "
                f"needs at most {self.target_blocks}, so this design cannot be one "
                f"however well it covers. Change the base block count or the group "
                f"order so the orbit sizes sum to {self.target_blocks} or fewer."
            )
        return (
            f"{group.name}, {len(base_blocks)} base blocks -> orbit sizes "
            f"{orbit_sizes(group, base_blocks)}, {len(blocks)} blocks. "
            f"Uncovered: {missing} of {self.parameters.requirements:,}.{note}"
        )

    def search(
        self,
        group: PointGroup,
        base_count: int,
        moves: int,
        *,
        seed: int = 0,
        restarts: int = 1,
        start_temp: float = 1.2,
    ) -> str:
        """Develop random base blocks under a group and anneal them, `restarts` times.

        Restarts are the currency. Annealing this cost function finds a design
        on a minority of attempts however long each one runs, so a single long
        search is a worse buy than several independent ones -- and with a cap
        on tool calls, one restart per call is what stops a session ever
        getting enough of them. The spread of outcomes is reported because it
        is informative in itself: restarts landing on wildly different counts
        mean the budget is the limit, and restarts all landing on the same
        small number mean the target probably cannot be reached at all.
        """
        if base_count < 1:
            message = f"base_count must be positive, got {base_count}"
            raise ValueError(message)
        if restarts < 1:
            message = f"restarts must be positive, got {restarts}"
            raise ValueError(message)
        reached: list[int] = []
        best: int | None = None
        best_blocks: tuple[int, ...] = ()
        for index in range(restarts):
            if index and self.remaining <= 0:
                break
            allowed = self._spend(moves)
            rng = random.Random(seed + index * 7919)
            base = tuple(random_block(self.parameters, rng) for _ in range(base_count))
            result = anneal(
                self.parameters,
                group,
                base,
                Schedule(steps=allowed, seed=seed + index, start_temp=start_temp),
            )
            self._used += result.moves or allowed
            reached.append(result.uncovered)
            if best is None or result.uncovered < best:
                best, best_blocks = result.uncovered, result.base_blocks
            if result.uncovered == 0:
                break
        summary = self._record(group, best_blocks, best or 0)
        return f"{summary} Restarts reached: {reached}."

    def resume(self, moves: int, *, seed: int = 0, start_temp: float = 0.6) -> str:
        """Anneal further from the best design reached so far."""
        best = self._incumbent
        if best is None:
            message = "no incumbent to resume; run search first"
            raise ValueError(message)
        allowed = self._spend(moves)
        result = anneal(
            self.parameters,
            best.group,
            best.base_blocks,
            Schedule(steps=allowed, seed=seed, start_temp=start_temp),
        )
        self._used += result.moves or allowed
        return self._record(best.group, result.base_blocks, result.uncovered)

    def adopt(self, group: PointGroup, base_blocks: tuple[int, ...]) -> str:
        """Take base blocks the model wrote itself as the incumbent. Costs nothing."""
        for block in base_blocks:
            if block.bit_count() != self.parameters.block_size:
                message = (
                    f"base block has {block.bit_count()} points, not {self.parameters.block_size}"
                )
                raise ValueError(message)
        blocks = develop(group, base_blocks)
        missing = check_covering(self.parameters, blocks).uncovered
        return self._record(group, base_blocks, missing)

    def inspect(self) -> str:
        """Report what the incumbent is missing, not merely how much."""
        best = self._incumbent
        if best is None:
            return "No design yet; run search first."
        blocks = best.blocks
        missing = uncovered_subsets(self.parameters, blocks)
        if not missing:
            return f"{len(blocks)} blocks, nothing uncovered."
        orbits = _subset_orbits(best.group, missing)
        deficits = point_deficits(self.parameters, missing)
        worst = Counter(dict(enumerate(deficits))).most_common(_REPORTED_POINTS)
        shown = orbits[:_REPORTED_ORBITS]
        ellipsis = " ..." if len(orbits) > _REPORTED_ORBITS else ""
        busiest = ", ".join(f"{point}({count})" for point, count in worst if count)
        headline = (
            f"{len(blocks)} blocks under {best.group.name}, {len(missing)} of "
            f"{self.parameters.requirements:,} requirements uncovered."
        )
        shape = f"They fall into {len(orbits)} orbit(s) under the group, sizes {shown}{ellipsis}."
        lines = [
            headline,
            shape,
            f"Representative uncovered requirement: {missing[0]}.",
            f"Points appearing most often in uncovered requirements: {busiest}",
        ]
        if len(orbits) == 1 and orbits[0] == len(missing):
            lines.append(
                "The whole deficit is a single group orbit, so this is a hole in the "
                "algebra rather than a shortage of blocks: a different stabiliser, "
                "not another random restart."
            )
        return "\n".join(lines)

    def status(self) -> str:
        """Render budget use and the best coverage reached."""
        secured = "none" if self._banked is None else self._banked.render()
        best = self._incumbent
        reached = "n/a" if best is None else str(best.uncovered)
        return (
            f"{self._used:,}/{self.budget:,} moves spent, best uncovered {reached}, "
            f"banked: {secured}"
        )
