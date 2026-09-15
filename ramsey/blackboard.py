"""A shared record of what every arm has already tried.

Each search session starts cold.  It does not know that another arm spent
thirty thousand evaluations on the same group at the same degree and reached
two above the cap, so it spends them again.  Across a fan-out of thirty-five
arms that is most of the budget, burnt re-learning the same dead ends.

The blackboard is the fix: every commissioned search appends what it tried and
what it reached, and any arm can read the whole history for its target before
choosing.  It turns a fan-out from thirty-five independent guesses into one
search with thirty-five workers, which is the only version of parallelism that
compounds.

It is append-only JSONL so that arms in separate processes can share it — the
kernel is CPU-bound and Python threads share a GIL, so a real fan-out means
separate processes, and those cannot share memory.  Short lines opened with
``O_APPEND`` are written atomically by the operating system, which is all the
coordination this needs.

Nothing here ranks or advises.  It reports what was tried and what it reached,
and leaves the inference to the reader.
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
    """One recorded search: what was tried, and how far it got."""

    target: str
    family: str
    detail: str
    best_independence: int
    independence_cap: int
    evaluations: int

    @property
    def gap(self) -> int:
        """Return how far the attempt landed above the cap."""
        return self.best_independence - self.independence_cap

    def render(self) -> str:
        """Render one line of history."""
        return (
            f"{self.family:<8} {self.detail:<52} reached alpha={self.best_independence} "
            f"(gap +{self.gap}) in {self.evaluations:,} evals"
        )


@dataclass(frozen=True, slots=True)
class Blackboard:
    """An append-only shared history, safe across processes."""

    path: Path

    def record(self, attempt: Attempt) -> None:
        """Append one attempt.

        Opened with ``O_APPEND`` so concurrent writers interleave whole lines
        rather than corrupting each other; a partial line would poison the
        history for every reader.
        """
        line = json.dumps(asdict(attempt), sort_keys=True, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(descriptor, line.encode())
        finally:
            os.close(descriptor)

    def attempts(self, target: str = "") -> tuple[Attempt, ...]:
        """Return recorded attempts, optionally filtered to one target.

        A malformed line is skipped rather than raising: a reader that dies on
        a torn write would lose the whole history to one bad append.
        """
        if not self.path.exists():
            return ()
        found = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                attempt = Attempt(**payload)
            except (json.JSONDecodeError, TypeError):
                continue
            if not target or attempt.target == target:
                found.append(attempt)
        return tuple(found)

    def render(self, target: str, *, limit: int = 40) -> str:
        """Render the history for one target, best first."""
        attempts = sorted(self.attempts(target), key=lambda a: (a.gap, a.evaluations))
        if not attempts:
            return "No other arm has reported an attempt on this target yet."
        lines = [f"{len(attempts)} attempt(s) recorded on this target by all arms, best first:"]
        lines.extend(f"  {attempt.render()}" for attempt in attempts[:limit])
        if len(attempts) > limit:
            lines.append(f"  ... and {len(attempts) - limit} more")
        best = attempts[0]
        lines.append(
            f"Best reached anywhere: alpha={best.best_independence}, "
            f"gap +{best.gap}, by {best.family} {best.detail}."
        )
        return "\n".join(lines)


DOMAIN_BRIEFING = """\
Established facts about this problem, so you do not rediscover them:

* A witness for R(3,k) > n is a graph on n vertices that is triangle-free with
  independence number at most k-1. Every neighbourhood of a triangle-free graph
  is independent, so no vertex may exceed degree k-1 either.
* Published lower bounds in this range were mostly obtained from circulant
  (cyclic Cayley) constructions. Harborth and Krause exhaustively searched
  cyclic graphs below 102 vertices, so a circulant below that order is very
  unlikely to be new. Orders 106, 111, 122, 131 and 139 are above that limit.
* A product of cyclic groups is cyclic exactly when the factor orders are
  pairwise coprime, so Z2 x Z37 is Z74 and every Cayley graph over it is a
  circulant. Of the open orders only 92 (Z2 x Z2 x Z23) and 99 (Z3 x Z3 x Z11)
  admit a genuinely non-cyclic abelian group. All groups of order 99 are
  abelian, so those two exhaust it.
* Not every witness is a Cayley graph. R(3,6) > 17 is true, yet no Cayley graph
  on 17 vertices witnesses it: the best circulant reaches independence 6 against
  a cap of 5, and an exhaustive check over every degree-4 circulant confirms it.
  Repairing that circulant edge by edge reached a witness in under a second.
  A plateau one or two above the cap is the signal to repair, not to keep
  searching the same family.
* A lift with base size 1 is exactly a Cayley graph; a larger base reaches
  graphs no Cayley construction can; a base equal to the order over the trivial
  group is an unconstrained graph. base_size times group order must equal the
  target order.
"""
