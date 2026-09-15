"""The model-facing tool surface for the code search.

This is the solver boundary. The model chooses *what to search* - which
quasi-cyclic shape, which structural constraints, how large a budget - and
this module does exactly that, measures the result with the referee, and
reports the measurement. It contains no construction heuristic and no ranking:
anything the model gets credit for, the model authored.

A result is one code. If a search reaches a distance above the published lower
bound, the output is a handful of generator polynomials anyone can recheck in
milliseconds, and it is a new entry in a maintained table.

Every evaluated code costs one call against one global ledger, so a
commissioned search and the same number of hand-proposed codes cost the same
and the model arm stays comparable to a matched mechanical arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codes.board import Attempt, Board
from codes.kernel import minimum_distance
from codes.linear import check_code
from codes.quasicyclic import QuasiCyclic
from codes.search import QcOutcome, QcProgram, run_qc_program

if TYPE_CHECKING:
    from codes.tables import Cell

_MAX_SEARCH_SHARE = 0.35


class BudgetExhaustedError(RuntimeError):
    """Raised when an evaluation is requested past the global budget."""

    def __init__(self, budget: int) -> None:
        super().__init__(f"evaluation budget of {budget:,} codes is exhausted")
        self.budget = budget


@dataclass(frozen=True, slots=True)
class Banked:
    """A verified code that beats the published bound."""

    length: int
    dimension: int
    distance: int
    block: int
    polynomials: tuple[int, ...]
    published: int

    def render(self) -> str:
        """Render the result as a table entry."""
        return (
            f"[{self.length},{self.dimension},{self.distance}] beats the published "
            f"lower bound of {self.published}. QC block {self.block}, "
            f"polynomials {self.polynomials}."
        )


@dataclass(slots=True)
class CodeTools:
    """Tool surface over one table cell and one evaluation budget."""

    cell: Cell
    budget: int
    board: Board | None = None
    _used: int = field(default=0, init=False)
    _best: int = field(default=0, init=False)
    _best_detail: str = field(default="", init=False)
    _banked: Banked | None = field(default=None, init=False)

    @property
    def used(self) -> int:
        """Return the number of codes evaluated."""
        return self._used

    @property
    def remaining(self) -> int:
        """Return the evaluations still available."""
        return self.budget - self._used

    @property
    def banked(self) -> Banked | None:
        """Return the secured result, if the search found one."""
        return self._banked

    @property
    def best_distance(self) -> int:
        """Return the best distance reached anywhere in this session."""
        return self._best

    @property
    def cell_key(self) -> str:
        """Return the identifier this cell is recorded under."""
        return f"[{self.cell.length},{self.cell.dimension}]"

    def describe_target(self) -> str:
        """State the cell, the published bounds, and what a new entry needs."""
        return (
            f"Target: improve the published lower bound for a binary "
            f"[{self.cell.length},{self.cell.dimension}] linear code.\n"
            f"Published bounds: d is between {self.cell.lower} and {self.cell.upper}. "
            f"A code with d >= {self.cell.target} and dimension exactly "
            f"{self.cell.dimension} is a NEW TABLE ENTRY.\n"
            f"A quasi-cyclic code of index {self.cell.quasi_cyclic_index} and block "
            f"{self.cell.dimension} has exactly this length and can reach this "
            f"dimension, so it takes {self.cell.quasi_cyclic_index} generator "
            f"polynomials of degree below {self.cell.dimension}."
        )

    def read_board(self) -> str:
        """Report what every other arm has already tried on this cell."""
        if self.board is None:
            return "No shared history is available for this run."
        return self.board.render(self.cell_key)

    def status(self) -> str:
        """Render budget use and the best distance reached."""
        secured = "none" if self._banked is None else self._banked.render()
        return (
            f"{self._used:,}/{self.budget:,} codes evaluated, "
            f"best d={self._best} (need {self.cell.target}), banked: {secured}"
        )

    def commission_search(
        self,
        index: int,
        restarts: int,
        steps: int,
        *,
        seed: int = 0,
        weight: int = 0,
    ) -> QcOutcome:
        """Run a mechanical search over quasi-cyclic codes of the cell's shape.

        This is where the budget should go: one call buys thousands of codes.
        A single call is capped at a share of what remains, so one over-large
        request cannot consume the session and several structural hypotheses
        still fit.
        """
        if index * self.cell.dimension != self.cell.length:
            message = (
                f"index {index} times block {self.cell.dimension} must equal "
                f"the length {self.cell.length}"
            )
            raise ValueError(message)
        program = QcProgram(
            block=self.cell.dimension,
            index=index,
            target_distance=self.cell.target,
            restarts=restarts,
            steps=steps,
            seed=seed,
            weight=weight,
        )
        share = max(int(self.remaining * _MAX_SEARCH_SHARE), 1)
        outcome = run_qc_program(program, budget=share)
        self._used += outcome.evaluations
        self._publish(program.describe(), outcome.best_distance, outcome.evaluations)
        if outcome.best_distance > self._best:
            self._best = outcome.best_distance
            self._best_detail = program.describe()
        if outcome.accepted:
            self._secure(outcome.best_polynomials)
        return outcome

    def evaluate(self, polynomials: tuple[int, ...]) -> str:
        """Measure one specific quasi-cyclic construction exactly."""
        if self._used >= self.budget:
            raise BudgetExhaustedError(self.budget)
        self._used += 1
        block = self.cell.dimension
        if any(p < 0 or p >= (1 << block) for p in polynomials):
            return f"rejected: every polynomial must have degree below {block}"
        if len(polynomials) * block != self.cell.length:
            return (
                f"rejected: {len(polynomials)} polynomials of block {block} give length "
                f"{len(polynomials) * block}, not {self.cell.length}"
            )
        code = QuasiCyclic(block=block, polynomials=tuple(polynomials)).code()
        dimension = code.dimension
        if dimension != self.cell.dimension:
            return (
                f"dimension is {dimension}, not {self.cell.dimension}: the generator "
                f"polynomials share a non-trivial factor with x^{block} - 1."
            )
        distance = minimum_distance(code)
        if distance > self._best:
            self._best = distance
            self._best_detail = f"explicit polynomials {tuple(polynomials)}"
        if distance >= self.cell.target:
            self._secure(tuple(polynomials))
            return f"ACCEPTED: [{self.cell.length},{dimension},{distance}] is a new entry."
        return (
            f"[{self.cell.length},{dimension},{distance}], "
            f"{self.cell.target - distance} short of d >= {self.cell.target}."
        )

    def _secure(self, polynomials: tuple[int, ...]) -> None:
        """Verify a candidate with the referee and bank it if it stands."""
        code = QuasiCyclic(block=self.cell.dimension, polynomials=polynomials).code()
        check = check_code(code, self.cell.dimension, self.cell.target)
        if not check.accepted:
            return
        self._banked = Banked(
            length=self.cell.length,
            dimension=check.dimension,
            distance=check.distance,
            block=self.cell.dimension,
            polynomials=polynomials,
            published=self.cell.lower,
        )

    def _publish(self, detail: str, best: int, evaluations: int) -> None:
        """Record an attempt so later arms inherit it."""
        if self.board is None or not evaluations:
            return
        self.board.record(
            Attempt(
                cell=self.cell_key,
                detail=detail,
                best_distance=best,
                target_distance=self.cell.target,
                evaluations=evaluations,
            )
        )
