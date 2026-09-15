"""Exact referee for binary linear codes.

A binary linear ``[n, k, d]`` code is a ``k``-dimensional subspace of
``GF(2)^n``; ``d`` is the least Hamming weight of a non-zero codeword, and it
is what the published tables bound.  Improving the best known ``d`` for any
``(n, k)`` by even one is a new table entry, which is why the target here is a
frontier of thousands of cells rather than a handful of exact values.

This module owns validation and exact measurement only, exactly as the Ramsey
referee did.  It contains no construction and no search: everything here is
the authority that later search code must satisfy.

Codewords are bitmasks and a generator is a tuple of them, so the whole
representation is machine words.  Minimum distance is computed by enumerating
the ``2^k`` codewords in Gray-code order: successive codewords differ by one
generator row, so each costs a single XOR and a popcount rather than a fresh
linear combination.  That makes the cost ``2^k`` bounded operations — large,
but *bounded*, which is the decisive difference from the independent-set
search that made the Ramsey campaign unaffordable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

_MIN_REDUNDANCY: Final = 2


@dataclass(frozen=True, slots=True)
class LinearCode:
    """A binary linear code, held as generator rows over ``GF(2)``.

    The rows need not be independent; :attr:`dimension` reports the rank, which
    is the ``k`` a table entry refers to.
    """

    length: int
    rows: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject rows that do not fit the declared length."""
        if self.length < 1:
            message = f"length must be positive, got {self.length}"
            raise ValueError(message)
        limit = 1 << self.length
        for index, row in enumerate(self.rows):
            if row < 0 or row >= limit:
                message = f"row {index} has a bit outside a code of length {self.length}"
                raise ValueError(message)

    @property
    def dimension(self) -> int:
        """Return the rank of the generator rows, which is ``k``."""
        return len(reduced_basis(self.rows))

    def basis(self) -> tuple[int, ...]:
        """Return an independent basis in row-echelon form."""
        return reduced_basis(self.rows)


def reduced_basis(rows: tuple[int, ...]) -> tuple[int, ...]:
    """Return a reduced row-echelon basis of the span, dropping dependent rows.

    Elimination is by leading bit, so each surviving row owns a pivot no other
    row has and the rank is the basis length.  The second pass clears those
    pivot columns from every *other* row, which is what makes the basis
    *reduced* — and the dual construction below relies on that, since it reads
    a null-space row straight off the pivot columns.
    """
    basis: list[int] = []
    for row in rows:
        residue = row
        for member in basis:
            residue = min(residue, residue ^ member)
        if residue:
            basis.append(residue)
            basis.sort(reverse=True)
    for index, member in enumerate(basis):
        pivot = member.bit_length() - 1
        for other in range(len(basis)):
            if other != index and basis[other] >> pivot & 1:
                basis[other] ^= member
    return tuple(basis)


def minimum_distance(code: LinearCode, *, floor: int = 0) -> int:
    """Return the minimum non-zero codeword weight, or ``0`` for a trivial code.

    ``floor`` short-circuits the enumeration: the search stops as soon as a
    codeword lighter than ``floor`` appears, because a candidate that already
    fails the target needs no exact value.  That is what makes rejecting a bad
    construction nearly free while a promising one pays in full.
    """
    basis = reduced_basis(code.rows)
    if not basis:
        return 0
    best = code.length
    current = 0
    for step in range(1, 1 << len(basis)):
        current ^= basis[(step & -step).bit_length() - 1]
        weight = current.bit_count()
        if weight and weight < best:
            best = weight
            if floor and best < floor:
                return best
    return best


@dataclass(frozen=True, slots=True)
class CodeCheck:
    """The referee's verdict on a claimed ``[n, k, d]`` code."""

    accepted: bool
    length: int
    dimension: int
    distance: int
    claimed_dimension: int
    claimed_distance: int

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Return one line per violated condition, empty when accepted."""
        lines = []
        if self.dimension != self.claimed_dimension:
            lines.append(f"dimension is {self.dimension}, not the claimed {self.claimed_dimension}")
        if self.distance < self.claimed_distance:
            lines.append(
                f"minimum distance is {self.distance}, below the claimed {self.claimed_distance}"
            )
        return tuple(lines)

    def render(self) -> str:
        """Render the verdict as a table entry or the reason it is not one."""
        if self.accepted:
            return f"verified [{self.length},{self.dimension},{self.distance}] binary code"
        return "; ".join(self.diagnostics)


def check_code(code: LinearCode, dimension: int, distance: int) -> CodeCheck:
    """Check that ``code`` really is an ``[n, k, d]`` code with the claimed parameters.

    Acceptance requires the rank to be exactly ``dimension`` and the minimum
    distance to be at least ``distance``; a code that beats its claim is
    accepted, since the claim is a lower bound on what was found.
    """
    measured_dimension = code.dimension
    measured_distance = minimum_distance(code)
    return CodeCheck(
        accepted=measured_dimension == dimension and measured_distance >= distance,
        length=code.length,
        dimension=measured_dimension,
        distance=measured_distance,
        claimed_dimension=dimension,
        claimed_distance=distance,
    )


def hamming_code(parity: int) -> LinearCode:
    """Build the binary Hamming code of redundancy ``parity``.

    A known family with known parameters — ``[2^r - 1, 2^r - 1 - r, 3]`` — so
    the referee can be checked against something the literature already fixes.
    """
    if parity < _MIN_REDUNDANCY:
        message = f"redundancy must be at least {_MIN_REDUNDANCY}, got {parity}"
        raise ValueError(message)
    length = (1 << parity) - 1
    columns = list(range(1, length + 1))
    rows = []
    for bit in range(parity):
        row = 0
        for position, column in enumerate(columns):
            if column >> bit & 1:
                row |= 1 << position
        rows.append(row)
    parity_check = LinearCode(length=length, rows=tuple(rows))
    return dual_code(parity_check)


def dual_code(code: LinearCode) -> LinearCode:
    """Return the dual code, the null space of the generator rows."""
    basis = reduced_basis(code.rows)
    pivots = [member.bit_length() - 1 for member in basis]
    free = [bit for bit in range(code.length) if bit not in set(pivots)]
    rows = []
    for position in free:
        row = 1 << position
        for member, pivot in zip(basis, pivots, strict=True):
            if member >> position & 1:
                row |= 1 << pivot
        rows.append(row)
    return LinearCode(length=code.length, rows=tuple(rows))
