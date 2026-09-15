"""Commissioned quasi-cyclic searches: the model says where, the kernel runs it.

One model call evaluating one code is the wrong ratio, for exactly the reason
it was wrong in the Ramsey campaign: a session gets a few hundred turns, and a
search priced at one candidate per turn samples a space a plain loop covers
tens of thousands of times over.

So the model proposes a *program* - which block length and index, which
polynomial degrees to force in or keep out, how many restarts and hill-climbing
steps - and the kernel runs it at full speed.  The search is deliberately
ordinary: random restarts over generator polynomials, first-improvement
hill-climbing on a single coefficient flip.  Any advantage has to come from
where the model aimed it, not from a heuristic hidden in here.

Rejection is nearly free.  A candidate only needs to beat the published bound,
so the referee stops the moment a lighter codeword appears and only a genuinely
promising code pays for a full enumeration.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from codes.kernel import minimum_distance
from codes.quasicyclic import QuasiCyclic


@dataclass(frozen=True, slots=True)
class QcProgram:
    """A commissioned search over quasi-cyclic codes of one shape."""

    block: int
    index: int
    target_distance: int
    restarts: int
    steps: int
    seed: int = 0
    weight: int = 0

    def __post_init__(self) -> None:
        """Reject a program that could never run."""
        for name, value in (("block", self.block), ("index", self.index)):
            if value < 1:
                message = f"{name} must be positive, got {value}"
                raise ValueError(message)
        if self.restarts < 1:
            message = f"restarts must be positive, got {self.restarts}"
            raise ValueError(message)
        if self.steps < 0:
            message = f"steps must be non-negative, got {self.steps}"
            raise ValueError(message)

    @property
    def length(self) -> int:
        """Return the code length this program searches."""
        return self.block * self.index

    def describe(self) -> str:
        """Render the program as the model specified it."""
        weight = f", weight {self.weight}" if self.weight else ""
        return (
            f"[{self.length},{self.block}] QC index {self.index}, block {self.block}{weight}, "
            f"{self.restarts} restarts x {self.steps} steps, need d >= {self.target_distance}"
        )


@dataclass(frozen=True, slots=True)
class QcOutcome:
    """What a commissioned search found, and what it cost."""

    program: QcProgram
    evaluations: int
    best_distance: int
    best_polynomials: tuple[int, ...]
    best_dimension: int

    @property
    def accepted(self) -> bool:
        """Report whether the search reached a new table entry."""
        return (
            self.best_distance >= self.program.target_distance
            and self.best_dimension == self.program.block
        )

    @property
    def shortfall(self) -> int:
        """Return how far the best code falls short of the target distance."""
        return self.program.target_distance - self.best_distance

    def render(self) -> str:
        """Render the outcome, stating only measured facts."""
        head = f"search[{self.program.describe()}]: {self.evaluations:,} codes evaluated."
        if self.accepted:
            return (
                f"{head} ACCEPTED: [{self.program.length},{self.best_dimension},"
                f"{self.best_distance}] beats the published bound. "
                f"Polynomials {self.best_polynomials}."
            )
        return (
            f"{head} Best [{self.program.length},{self.best_dimension},{self.best_distance}], "
            f"{self.shortfall} short of d >= {self.program.target_distance}. "
            f"Polynomials {self.best_polynomials}."
        )


def _random_polynomials(program: QcProgram, rng: random.Random) -> tuple[int, ...]:
    """Draw one random generator polynomial per block."""
    if not program.weight:
        return tuple(rng.getrandbits(program.block) for _ in range(program.index))
    drawn = []
    for _ in range(program.index):
        value = 0
        for position in rng.sample(range(program.block), min(program.weight, program.block)):
            value |= 1 << position
        drawn.append(value)
    return tuple(drawn)


def _measure(program: QcProgram, polynomials: tuple[int, ...]) -> tuple[int, int]:
    """Return the dimension and minimum distance of one candidate."""
    code = QuasiCyclic(block=program.block, polynomials=polynomials).code()
    dimension = code.dimension
    if dimension != program.block:
        return dimension, 0
    return dimension, minimum_distance(code, floor=program.target_distance)


def _climb(
    program: QcProgram,
    current: tuple[int, ...],
    distance: int,
    rng: random.Random,
    budget: int,
) -> tuple[tuple[int, ...], int, int, int]:
    """Hill-climb one restart by flipping single polynomial coefficients."""
    spent = 0
    dimension = program.block
    best = (current, distance, dimension)
    for _ in range(program.steps):
        if spent >= budget or distance >= program.target_distance:
            break
        position = rng.randrange(program.index)
        bit = rng.randrange(program.block)
        proposal = tuple(
            value ^ (1 << bit) if slot == position else value for slot, value in enumerate(current)
        )
        trial_dimension, trial_distance = _measure(program, proposal)
        spent += 1
        if trial_distance >= distance:
            current, distance, dimension = proposal, trial_distance, trial_dimension
        if trial_distance > best[1]:
            best = (proposal, trial_distance, trial_dimension)
    return best[0], best[1], best[2], spent


def run_qc_program(program: QcProgram, *, budget: int) -> QcOutcome:
    """Run a commissioned search, stopping at a new entry or at ``budget``."""
    rng = random.Random(program.seed)
    best_distance = 0
    best_polynomials: tuple[int, ...] = ()
    best_dimension = 0
    evaluations = 0
    for _ in range(program.restarts):
        if evaluations >= budget or best_distance >= program.target_distance:
            break
        current = _random_polynomials(program, rng)
        dimension, distance = _measure(program, current)
        evaluations += 1
        if distance > best_distance:
            best_distance, best_polynomials, best_dimension = distance, current, dimension
        climbed, reached, reached_dimension, spent = _climb(
            program, current, distance, rng, budget - evaluations
        )
        evaluations += spent
        if reached > best_distance:
            best_distance, best_polynomials, best_dimension = (
                reached,
                climbed,
                reached_dimension,
            )
    return QcOutcome(
        program=program,
        evaluations=evaluations,
        best_distance=best_distance,
        best_polynomials=best_polynomials,
        best_dimension=best_dimension,
    )
