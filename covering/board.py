"""A shared record of what every arm has already tried on a covering cell.

Each session starts cold.  It does not know that another arm already developed
four base blocks under the cyclic group and stalled two requirements short, so
it spends its budget discovering that again.  Across a fan-out that is most of
the budget, burnt re-learning the same dead ends.

The board is the fix: every search appends the structure it tried and how close
it got, and any arm can read the whole history for its cell before choosing.
It turns a fan-out from many independent guesses into one search with many
workers, which is the only version of parallelism that compounds.

It is append-only JSONL so arms in separate processes can share it.  Short
lines opened with ``O_APPEND`` are written atomically by the operating system,
which is all the coordination this needs.

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
    """One recorded search: the structure tried, and how close it came."""

    cell: str
    group: str
    base_blocks: int
    blocks: int
    uncovered: int
    moves: int

    def render(self) -> str:
        """Render one line of history."""
        return (
            f"{self.group:<14} {self.base_blocks} base -> {self.blocks} blocks, "
            f"left {self.uncovered} uncovered in {self.moves:,} moves"
        )


@dataclass(frozen=True, slots=True)
class Board:
    """An append-only shared history, safe across processes."""

    path: Path

    def record(self, attempt: Attempt) -> None:
        """Append one attempt."""
        line = json.dumps(asdict(attempt), sort_keys=True, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(descriptor, line.encode())
        finally:
            os.close(descriptor)

    def attempts(self, cell: str = "") -> tuple[Attempt, ...]:
        """Return recorded attempts, optionally filtered to one cell.

        A malformed line is skipped rather than raising: a reader that died on
        a torn write would lose the whole history to one bad append.
        """
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

    def render(self, cell: str, *, target: int = 0, limit: int = 30) -> str:
        """Render the history for one cell, closest first.

        ``target`` is the block count a new entry needs. Attempts that develop
        to more blocks than that are listed last however little they leave
        uncovered, because a perfect covering of the wrong size is not close to
        anything -- and sorting on shortfall alone put exactly those at the top
        of the history every arm reads first.
        """

        def rank(attempt: Attempt) -> tuple[int, int, int]:
            too_big = 1 if target and attempt.blocks > target else 0
            return (too_big, attempt.uncovered, attempt.moves)

        attempts = sorted(self.attempts(cell), key=rank)
        if not attempts:
            return "No other arm has reported an attempt on this cell yet."
        lines = [f"{len(attempts)} attempt(s) on this cell by all arms, closest first:"]
        lines.extend(f"  {attempt.render()}" for attempt in attempts[:limit])
        if len(attempts) > limit:
            lines.append(f"  ... and {len(attempts) - limit} more")
        best = attempts[0]
        note = (
            ""
            if not target or best.blocks <= target
            else f" (but {best.blocks} blocks, and a new entry needs {target})"
        )
        lines.append(
            f"Closest anywhere: {best.uncovered} uncovered, by {best.group} "
            f"with {best.base_blocks} base block(s){note}."
        )
        return "\n".join(lines)


BRIEFING = """\
Established facts about this problem, so you do not rediscover them:

* A (v,k,t)-covering is a family of k-subsets of a v-set such that every
  t-subset lies inside at least one of them. Fewer blocks is better. The
  referee checks every t-subset, so you never assert that a design works.
* Designs of this size are built by developing a few BASE BLOCKS under a group,
  not by perturbing a design block by block. A base block's orbit under a group
  G has size |G| divided by the size of its stabiliser, so the block count of
  any G-invariant design is a sum of those orbit sizes. If your target count is
  not such a sum, no amount of searching in that group will reach it - so call
  check_feasible before you spend anything. It costs nothing.
* A block is a union of its stabiliser's orbits ON POINTS, and those orbits need
  not be free. A stabiliser of order 2 can fix points, so its order need not
  divide the block size. Do not assume otherwise.
* Under a point-transitive group, counting incidences gives v*r = B*k, so
  v/gcd(v,k) divides the block count B. This rules out counts the orbit sizes
  alone would allow.
* The trivial group is a legitimate choice: it develops a design into itself, so
  searching under it is an ordinary free search over independent blocks. That is
  the mechanical baseline. Structure earns its place by beating it, and on many
  cells it does not - large symmetry groups force block counts that are simply
  too coarse near the target.
* When a search falls short, call inspect(). If the whole deficit is a single
  orbit under your group, the algebra has a hole and you need a different
  stabiliser, not another restart. If the deficit is scattered, you are short of
  blocks or short of search.
* check_feasible, inspect, read_board, status and adopt spend no moves AND no
  search allowance. Use them freely. Only search and resume are rationed.
* RESTARTS ARE THE CURRENCY. Annealing finds a design on a minority of attempts
  however long each one runs; on a cell that was solved this way the successful
  restart was one in twelve, and the failures scored 7, 8, 10 and 12 uncovered.
  So a restart coming back 10 short is NOT evidence the target is unreachable,
  and a single long search is a worse buy than several independent ones. Pass a
  high `restarts` to search rather than spending a whole tool call per attempt.
* Count your blocks. A design that develops to more blocks than the target
  cannot be a new entry however well it covers, so base_count times the orbit
  size must land on the target or below. Getting zero uncovered with too many
  blocks is not progress.
* A cell whose published value came from a generic recursive construction
  ("Simple construction from ...") has probably never had a dedicated search run
  on it. One whose value came from simulated annealing has. describe_target
  tells you which.
"""
