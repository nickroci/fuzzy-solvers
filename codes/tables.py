"""The published bounds, and which cells are worth attacking.

``codetables.de`` publishes, for each length and dimension, a lower and an
upper bound on the minimum distance of the best binary linear code.  Where the
two differ the cell is open, and *raising the lower bound by one* is a new
entry — which is why this frontier is thousands of cells wide rather than a
handful of exact values.

Choosing which cell to attack is the step the Ramsey campaign skipped, and it
cost that campaign everything: targets were picked from a survey's passing
remark rather than from measured weakness.  Here the choice is structural.

Most best-known binary entries are quasi-cyclic, and a quasi-cyclic code of
index ``m`` and block ``p`` has length ``m * p`` and dimension at most ``p``.
So the cells a quasi-cyclic search can even *reach* at full dimension are
exactly those where ``k`` divides ``n``, and the search is only affordable
where ``2^k`` enumeration is.  Ranking by gap within that set aims the budget
at cells that are both reachable and loose.

Nothing here fetches: the table is parsed from a saved page, so a run is
reproducible and the site is asked for a page once rather than per cell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"<[^>]*>")
_RANGE = re.compile(r"(\d+)\s*-\s*(\d+)")


@dataclass(frozen=True, slots=True)
class Cell:
    """One published table entry: the bounds on ``d`` for a length and dimension."""

    length: int
    dimension: int
    lower: int
    upper: int

    @property
    def gap(self) -> int:
        """Return how much room the published bounds leave."""
        return self.upper - self.lower

    @property
    def is_open(self) -> bool:
        """Report whether the bounds differ, so the cell can still be improved."""
        return self.upper > self.lower

    @property
    def target(self) -> int:
        """Return the distance a new code must reach to be a new entry."""
        return self.lower + 1

    @property
    def quasi_cyclic_index(self) -> int:
        """Return ``n / k`` when a full-dimension quasi-cyclic code could fit, else 0.

        A quasi-cyclic code of index ``m`` and block ``p`` has length ``m * p``
        and dimension at most ``p``, so reaching dimension ``k`` at length ``n``
        needs ``k`` to divide ``n``.
        """
        if self.dimension < 1 or self.length % self.dimension:
            return 0
        return self.length // self.dimension

    def render(self) -> str:
        """Render the cell as the tables present it."""
        return (
            f"[{self.length},{self.dimension}] d in [{self.lower},{self.upper}] "
            f"gap {self.gap}, need d >= {self.target}"
        )


def parse_table(html: str) -> tuple[Cell, ...]:
    """Parse a saved bounds page into cells.

    A cell reading ``lo-hi`` is open; a bare number is settled and carries no
    room, so it is recorded with equal bounds rather than dropped — a caller
    filtering on :attr:`Cell.is_open` should not have to know the difference.
    """
    cells: list[Cell] = []
    for row in _ROW.findall(html):
        entries = _CELL.findall(row)
        if not entries:
            continue
        head = _TAG.sub("", entries[0]).replace("&nbsp;", " ").strip()
        if not head.isdigit():
            continue
        length = int(head)
        for dimension, entry in enumerate(entries[1:], start=1):
            text = _TAG.sub("", entry).replace("&nbsp;", " ").strip()
            ranged = _RANGE.fullmatch(text)
            if ranged:
                cells.append(
                    Cell(
                        length=length,
                        dimension=dimension,
                        lower=int(ranged.group(1)),
                        upper=int(ranged.group(2)),
                    )
                )
            elif text.isdigit():
                value = int(text)
                cells.append(Cell(length=length, dimension=dimension, lower=value, upper=value))
    return tuple(cells)


def load_table(path: Path) -> tuple[Cell, ...]:
    """Parse a bounds page saved to disk."""
    return parse_table(path.read_text(errors="ignore"))


def quasi_cyclic_targets(
    cells: tuple[Cell, ...],
    *,
    max_dimension: int = 22,
    min_index: int = 2,
    max_index: int = 6,
) -> tuple[Cell, ...]:
    """Return open cells a quasi-cyclic search can reach, loosest first.

    ``max_dimension`` is an affordability bound, not a mathematical one: the
    referee enumerates ``2^k`` codewords, so a cell beyond it cannot be
    searched at the rate a campaign needs however promising it looks.
    """
    reachable = [
        cell
        for cell in cells
        if cell.is_open
        and 1 <= cell.dimension <= max_dimension
        and min_index <= cell.quasi_cyclic_index <= max_index
    ]
    return tuple(sorted(reachable, key=lambda cell: (-cell.gap, cell.length)))
