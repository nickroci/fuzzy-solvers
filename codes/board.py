"""Shared history of what every arm has tried on a code cell.

The lesson from the Ramsey campaign arrived too late to help it: arms that
cannot see each other spend most of a fan-out rediscovering the same dead
ends. Here it is present from the start. Every commissioned search appends the
shape it tried and the distance it reached, and any arm can read the history
for its cell before spending a budget.

Append-only JSONL with ``O_APPEND``, because real parallelism means separate
processes and those cannot share memory; short lines are written atomically,
which is all the coordination this needs. A torn line is skipped rather than
raised on, since one bad append must not cost every reader the history.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class Attempt:
    """One recorded search: the shape tried, and the distance it reached."""

    cell: str
    detail: str
    best_distance: int
    target_distance: int
    evaluations: int

    @property
    def shortfall(self) -> int:
        """Return how far the attempt fell short of a new table entry."""
        return self.target_distance - self.best_distance

    def render(self) -> str:
        """Render one line of history."""
        return (
            f"{self.detail:<58} reached d={self.best_distance} "
            f"({self.shortfall} short) in {self.evaluations:,} codes"
        )


@dataclass(frozen=True, slots=True)
class Board:
    """An append-only shared history, safe across processes."""

    path: Path

    def record(self, attempt: Attempt) -> None:
        """Append one attempt atomically."""
        line = json.dumps(asdict(attempt), sort_keys=True, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(handle, line.encode())
        finally:
            os.close(handle)

    def attempts(self, cell: str = "") -> tuple[Attempt, ...]:
        """Return recorded attempts, optionally filtered to one cell."""
        if not self.path.exists():
            return ()
        found = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                attempt = Attempt(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
            if not cell or attempt.cell == cell:
                found.append(attempt)
        return tuple(found)

    def render(self, cell: str, *, limit: int = 30) -> str:
        """Render the history for one cell, closest first."""
        attempts = sorted(self.attempts(cell), key=lambda a: (a.shortfall, a.evaluations))
        if not attempts:
            return "No other arm has reported an attempt on this cell yet."
        lines = [f"{len(attempts)} attempt(s) on this cell by all arms, closest first:"]
        lines.extend(f"  {a.render()}" for a in attempts[:limit])
        if len(attempts) > limit:
            lines.append(f"  ... and {len(attempts) - limit} more")
        best = attempts[0]
        lines.append(f"Best reached anywhere: d={best.best_distance} via {best.detail}.")
        return "\n".join(lines)


BRIEFING = """\
Established facts, so you do not rediscover them:

* A binary [n,k,d] code is a k-dimensional subspace of GF(2)^n; d is the least
  weight of a non-zero codeword. The published tables give a lower and an upper
  bound on the best possible d. Raising the LOWER bound by one is a new table
  entry - you do not need to close the gap.
* A quasi-cyclic code of index m and block p has length n = m*p and dimension
  at most p, and is given by m polynomials of degree below p. It is spanned by
  the p cyclic shifts of the concatenated generator. Most best-known binary
  entries in this range are quasi-cyclic, which is why this is the language.
* For the code to have full dimension p, the generator polynomials must not
  share a non-trivial factor with x^p - 1. Choosing polynomials whose supports
  respect the cyclotomic cosets modulo p is the standard way to control this.
* A plain random-restart search reaches the PUBLISHED distance on these cells
  within seconds. It does not reach one more. Repeating an unstructured search
  at a larger budget is therefore unlikely to help; what has not been tried is
  structure - particular weights, supports built from cyclotomic cosets,
  polynomials with prescribed factors, or relations between the blocks.
* Rejection is nearly free and a promising code is cheap too, so commission
  large searches rather than evaluating candidates one at a time.
"""
