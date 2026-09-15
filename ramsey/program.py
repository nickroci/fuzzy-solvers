"""Mechanical search programs the model commissions and the kernel runs.

One model call evaluating one graph is the wrong ratio.  A session gets a few
hundred model turns, so a search priced at one graph per turn samples a space
that a plain loop covers tens of thousands of times over in the same wall
clock — and against records set by dedicated multi-year searches, a few
hundred samples is noise.

So the model does not propose graphs here.  It proposes a *program*: which
group to work in, what degree, which generators to force in or keep out, how
many restarts and how many hill-climbing steps.  The kernel then runs that
program at full mechanical speed and reports what it found.  The model's
judgement is spent on **where to point a large budget**, which is the one
thing it might do better than a loop, and the loop does the part loops are
good at.

The search itself is deliberately ordinary: random restarts from greedy
sum-free connection sets, first-improvement hill-climbing on the independence
number under a swap neighbourhood.  Nothing here is clever, and that is the
point — any advantage the campaign shows has to come from where the model
aimed it, not from a heuristic hidden in the kernel.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ramsey.cayley import AbelianGroup, cayley_graph, sum_violation, symmetric_closure
from ramsey.kernel import capped_independence as capped_independence_number
from ramsey.lift import (
    VoltageGraph,
    VoltageMap,
    lift_graph,
    triangle_voltage,
)


@dataclass(frozen=True, slots=True)
class SearchProgram:
    """A commissioned mechanical search over one group and degree.

    ``required`` and ``forbidden`` are the model's structural hypothesis: the
    generators it believes a good construction must contain or must avoid.
    Everything else the kernel decides by search.
    """

    moduli: tuple[int, ...]
    degree: int
    restarts: int
    steps: int
    seed: int = 0
    required: tuple[int, ...] = ()
    forbidden: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Reject a program that could never run."""
        if self.degree < 1:
            message = f"degree must be positive, got {self.degree}"
            raise ValueError(message)
        if self.restarts < 1:
            message = f"restarts must be positive, got {self.restarts}"
            raise ValueError(message)
        if self.steps < 0:
            message = f"steps must be non-negative, got {self.steps}"
            raise ValueError(message)

    def describe(self) -> str:
        """Render the program as the model specified it."""
        group = " x ".join(f"Z{m}" for m in self.moduli)
        parts = [f"{group}, degree {self.degree}, {self.restarts} restarts x {self.steps} steps"]
        if self.required:
            parts.append(f"requiring {self.required}")
        if self.forbidden:
            parts.append(f"forbidding {self.forbidden}")
        return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """What a commissioned search found, and what it cost."""

    program: SearchProgram
    evaluations: int
    restarts_done: int
    best_independence: int
    best_generators: tuple[int, ...]
    independence_cap: int
    ceiling: int

    @property
    def accepted(self) -> bool:
        """Report whether the search reached a witness."""
        return self.best_independence <= self.independence_cap

    @property
    def gap(self) -> int:
        """Return how far the best construction sits above the cap."""
        return self.best_independence - self.independence_cap

    def render(self) -> str:
        """Render the outcome, stating only measured facts."""
        head = (
            f"search[{self.program.describe()}]: {self.evaluations} evaluations over "
            f"{self.restarts_done} restarts."
        )
        if self.accepted:
            return (
                f"{head} ACCEPTED: independence {self.best_independence} "
                f"(cap {self.independence_cap}). Generators {self.best_generators}."
            )
        saturated = (
            " (saturated at the search ceiling)" if self.best_independence >= self.ceiling else ""
        )
        tail = ""
        if 0 < self.gap <= _REPAIR_SIGNAL_GAP:
            tail = (
                f" This is within {self.gap} of the cap. Structured search has taken this "
                f"family as far as it goes; commission_repair edits edges directly and is "
                f"what reaches a witness that is not algebraic at all."
            )
        return (
            f"{head} Best independence {self.best_independence}{saturated}, "
            f"{self.gap} above the cap of {self.independence_cap}. "
            f"Best generators {self.best_generators}.{tail}"
        )


def representative_pool(group: AbelianGroup, forbidden: frozenset[int]) -> tuple[int, ...]:
    """Return one representative per inverse pair, minus the forbidden ones."""
    pool = {min(element, group.negate(element)) for element in range(1, group.order)}
    return tuple(sorted(pool - {min(f, group.negate(f)) for f in forbidden}))


def legal_connection(
    group: AbelianGroup, generators: tuple[int, ...], degree: int
) -> tuple[int, ...] | None:
    """Return the symmetric connection set if it is legal, else ``None``."""
    connection = symmetric_closure(group, generators)
    if not connection or len(connection) > degree:
        return None
    if sum_violation(group, connection) is not None:
        return None
    return connection


def greedy_sum_free(
    group: AbelianGroup,
    pool: tuple[int, ...],
    degree: int,
    required: tuple[int, ...],
    rng: random.Random,
) -> tuple[int, ...]:
    """Grow a maximal sum-free generator set from a shuffled pool.

    Starting from the model's required generators keeps its hypothesis intact;
    everything after that is the kernel filling the set as far as the degree
    allows.
    """
    chosen: list[int] = []
    for generator in required:
        if legal_connection(group, (*chosen, generator), degree) is not None:
            chosen.append(generator)
    order = list(pool)
    rng.shuffle(order)
    for candidate in order:
        if candidate in chosen:
            continue
        if legal_connection(group, (*chosen, candidate), degree) is not None:
            chosen.append(candidate)
    return tuple(sorted(chosen))


def run_program(program: SearchProgram, independence_cap: int, *, budget: int) -> SearchOutcome:
    """Run a commissioned search, stopping at a witness or at ``budget``.

    ``budget`` is the number of graph evaluations the caller is willing to pay
    for; the walk stops early the moment a construction meets the cap, because
    nothing better is needed.
    """
    group = AbelianGroup(moduli=program.moduli)
    ceiling = independence_cap + 1 + _GRADIENT_HEADROOM
    forbidden = frozenset(program.forbidden)
    pool = representative_pool(group, forbidden)
    rng = random.Random(program.seed)
    best_score = ceiling
    best_generators: tuple[int, ...] = ()
    evaluations = 0
    restarts_done = 0

    for _ in range(program.restarts):
        if evaluations >= budget:
            break
        restarts_done += 1
        current = greedy_sum_free(group, pool, program.degree, program.required, rng)
        connection = legal_connection(group, current, program.degree)
        if connection is None:
            continue
        score = capped_independence_number(cayley_graph(group, connection), ceiling)
        evaluations += 1
        if score < best_score:
            best_score, best_generators = score, current
        if best_score <= independence_cap:
            break
        current, score, spent = _climb(
            _Context(group=group, program=program, pool=pool, ceiling=ceiling, rng=rng),
            current,
            score,
            budget - evaluations,
        )
        evaluations += spent
        if score < best_score:
            best_score, best_generators = score, current
        if best_score <= independence_cap:
            break

    return SearchOutcome(
        program=program,
        evaluations=evaluations,
        restarts_done=restarts_done,
        best_independence=best_score,
        best_generators=best_generators,
        independence_cap=independence_cap,
        ceiling=ceiling,
    )


_GRADIENT_HEADROOM = 6
_REPAIR_SIGNAL_GAP = 2


@dataclass(frozen=True, slots=True)
class _Context:
    """The fixed part of a walk: group, candidate pool, ceiling and generator."""

    group: AbelianGroup
    program: SearchProgram
    pool: tuple[int, ...]
    ceiling: int
    rng: random.Random


def _climb(
    context: _Context,
    current: tuple[int, ...],
    score: int,
    budget: int,
) -> tuple[tuple[int, ...], int, int]:
    """First-improvement hill-climb under a swap neighbourhood."""
    group, program = context.group, context.program
    pool, ceiling, rng = context.pool, context.ceiling, context.rng
    spent = 0
    required = set(program.required)
    for _ in range(program.steps):
        if spent >= budget:
            break
        swappable = [g for g in current if g not in required]
        outside = [g for g in pool if g not in set(current)]
        if not swappable or not outside:
            break
        leaving = swappable[rng.randrange(len(swappable))]
        entering = outside[rng.randrange(len(outside))]
        proposal = tuple(sorted((*(g for g in current if g != leaving), entering)))
        connection = legal_connection(group, proposal, program.degree)
        if connection is None:
            continue
        trial = capped_independence_number(cayley_graph(group, connection), ceiling)
        spent += 1
        if trial < score:
            current, score = proposal, trial
        if score <= 0:
            break
    return current, score, spent


@dataclass(frozen=True, slots=True)
class LiftProgram:
    """A commissioned search over voltage-graph lifts.

    ``base_size`` times the group order must equal the target order.  A base
    size of one is exactly the Cayley case, so this program strictly contains
    the Cayley search; larger bases reach graphs no Cayley construction can.
    """

    base_size: int
    moduli: tuple[int, ...]
    degree: int
    restarts: int
    steps: int
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject a program that could never run."""
        if self.base_size < 1:
            message = f"base_size must be positive, got {self.base_size}"
            raise ValueError(message)
        if self.degree < 1:
            message = f"degree must be positive, got {self.degree}"
            raise ValueError(message)
        if self.restarts < 1:
            message = f"restarts must be positive, got {self.restarts}"
            raise ValueError(message)
        if self.steps < 0:
            message = f"steps must be non-negative, got {self.steps}"
            raise ValueError(message)

    @property
    def degenerate(self) -> bool:
        """Report whether this lift is just a Cayley graph in disguise.

        A base of one vertex has no base structure to lift, so the result is
        exactly the Cayley graph on the group.  Commissioning that through the
        lift tool buys nothing the Cayley tool did not already offer.
        """
        return self.base_size == 1

    def describe(self) -> str:
        """Render the program as the model specified it."""
        group = " x ".join(f"Z{m}" for m in self.moduli)
        return (
            f"lift base {self.base_size} x {group}, degree {self.degree}, "
            f"{self.restarts} restarts x {self.steps} steps"
        )


@dataclass(frozen=True, slots=True)
class LiftOutcome:
    """What a commissioned lift search found, and what it cost."""

    program: LiftProgram
    evaluations: int
    restarts_done: int
    best_independence: int
    best_voltages: VoltageMap
    independence_cap: int
    ceiling: int

    @property
    def accepted(self) -> bool:
        """Report whether the search reached a witness."""
        return self.best_independence <= self.independence_cap

    @property
    def gap(self) -> int:
        """Return how far the best construction sits above the cap."""
        return self.best_independence - self.independence_cap

    def render(self) -> str:
        """Render the outcome, stating only measured facts."""
        voltages = {f"{a}-{b}": list(v) for (a, b), v in sorted(self.best_voltages.items())}
        head = (
            f"search[{self.program.describe()}]: {self.evaluations} evaluations over "
            f"{self.restarts_done} restarts."
        )
        if self.program.degenerate:
            head += (
                " NOTE: base_size=1 makes this exactly a Cayley graph, so this search "
                "covered no ground the Cayley tool does not already cover. A base above "
                "one is what reaches graphs Cayley constructions cannot."
            )
        if self.accepted:
            return f"{head} ACCEPTED: independence {self.best_independence}. Voltages {voltages}."
        return (
            f"{head} Best independence {self.best_independence}, {self.gap} above the cap "
            f"of {self.independence_cap}. Best voltages {voltages}."
        )


def _base_pairs(base_size: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (first, second) for first in range(base_size) for second in range(first, base_size)
    )


def greedy_lift(program: LiftProgram, rng: random.Random) -> VoltageMap:
    """Grow a triangle-free voltage assignment greedily under the degree cap.

    Diagonal entries are added in inverse pairs, since a fibre's own voltages
    must be symmetric; off-diagonal entries are free, because reading a base
    edge backwards already negates it.
    """
    group = AbelianGroup(moduli=program.moduli)
    voltages: VoltageMap = {}
    candidates = [
        (pair, value)
        for pair in _base_pairs(program.base_size)
        for value in range(group.order)
        if not (pair[0] == pair[1] and value == 0)
    ]
    rng.shuffle(candidates)
    for (first, second), value in candidates:
        existing = voltages.get((first, second), ())
        if value in existing:
            continue
        addition = (
            (value, group.negate(value))
            if first == second and group.negate(value) != value
            else (value,)
        )
        proposal = dict(voltages)
        proposal[(first, second)] = tuple(sorted({*existing, *addition}))
        trial = VoltageGraph(base_size=program.base_size, moduli=program.moduli, voltages=proposal)
        if max(trial.degrees(), default=0) > program.degree:
            continue
        if triangle_voltage(trial) is not None:
            continue
        voltages = proposal
    return voltages


def run_lift_program(program: LiftProgram, independence_cap: int, *, budget: int) -> LiftOutcome:
    """Run a commissioned lift search, stopping at a witness or at ``budget``."""
    ceiling = independence_cap + 1 + _GRADIENT_HEADROOM
    rng = random.Random(program.seed)
    best_score = ceiling
    best_voltages: VoltageMap = {}
    evaluations = 0
    restarts_done = 0
    for _ in range(program.restarts):
        if evaluations >= budget:
            break
        restarts_done += 1
        voltages = greedy_lift(program, rng)
        if not voltages:
            continue
        construction = VoltageGraph(
            base_size=program.base_size, moduli=program.moduli, voltages=voltages
        )
        score = capped_independence_number(lift_graph(construction), ceiling)
        evaluations += 1
        if score < best_score:
            best_score, best_voltages = score, voltages
        if best_score <= independence_cap:
            break
    return LiftOutcome(
        program=program,
        evaluations=evaluations,
        restarts_done=restarts_done,
        best_independence=best_score,
        best_voltages=best_voltages,
        independence_cap=independence_cap,
        ceiling=ceiling,
    )
