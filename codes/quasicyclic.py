"""Quasi-cyclic codes: the construction language for the code search.

Most best-known entries in the binary tables are quasi-cyclic, and the reason
matters here: a quasi-cyclic code of index ``m`` and block length ``p`` is
determined by ``m`` polynomials of degree below ``p``.  That is a handful of
integers, not a generator matrix — an object the model can reason about
algebraically, in terms of factorisations of ``x^p - 1``, cyclotomic cosets and
idempotents, rather than guess at bit by bit.

The code of length ``n = m * p`` is spanned by the ``p`` cyclic shifts of the
concatenated generator ``(g_1 | ... | g_m)``: row ``i`` places ``x^i g_j mod
(x^p - 1)`` in block ``j``.  Multiplying by ``x`` modulo ``x^p - 1`` is a
rotation of a ``p``-bit mask, so the whole construction is bit rotations.

This module builds only.  :mod:`codes.linear` measures.
"""

from __future__ import annotations

from dataclasses import dataclass

from codes.linear import LinearCode


def rotate(polynomial: int, shift: int, block: int) -> int:
    """Return ``x^shift * polynomial`` modulo ``x^block - 1``."""
    shift %= block
    mask = (1 << block) - 1
    value = polynomial & mask
    return ((value << shift) | (value >> (block - shift))) & mask if shift else value


@dataclass(frozen=True, slots=True)
class QuasiCyclic:
    """A quasi-cyclic code given by ``height`` generators across ``index`` blocks.

    ``polynomials`` is row-major: ``height * index`` entries, one row of
    ``index`` block polynomials per generator.  The code is spanned by the
    cyclic shifts of every row, so the dimension is generically ``height *
    block`` and the length is ``index * block``.

    Height one is the familiar single-generator case.  Height above one is what
    the published tables call *stacked*, and it is not a curiosity: the
    ``[50,20,13]`` entry of May 2025 is quasi-cyclic of degree 5 stacked to
    height 2, and ``[56,24,13]`` is stacked to height 3.  Searching only height
    one covers a slice of the class and misses where recent records came from.
    """

    block: int
    polynomials: tuple[int, ...]
    height: int = 1

    def __post_init__(self) -> None:
        """Reject a specification that does not describe a code."""
        if self.block < 1:
            message = f"block length must be positive, got {self.block}"
            raise ValueError(message)
        if not self.polynomials:
            message = "a quasi-cyclic code needs at least one generator polynomial"
            raise ValueError(message)
        if self.height < 1:
            message = f"height must be positive, got {self.height}"
            raise ValueError(message)
        if len(self.polynomials) % self.height:
            message = (
                f"{len(self.polynomials)} polynomials do not divide into "
                f"{self.height} generator rows"
            )
            raise ValueError(message)
        limit = 1 << self.block
        for index, polynomial in enumerate(self.polynomials):
            if polynomial < 0 or polynomial >= limit:
                message = f"polynomial {index} has a term of degree at least {self.block}"
                raise ValueError(message)

    @property
    def index(self) -> int:
        """Return the quasi-cyclic index, the number of blocks per generator."""
        return len(self.polynomials) // self.height

    @property
    def length(self) -> int:
        """Return the code length ``m * p``."""
        return self.index * self.block

    def code(self) -> LinearCode:
        """Build the spanned linear code, over every shift of every generator."""
        width = self.index
        rows = []
        for generator in range(self.height):
            block_row = self.polynomials[generator * width : (generator + 1) * width]
            for shift in range(self.block):
                row = 0
                for position, polynomial in enumerate(block_row):
                    row |= rotate(polynomial, shift, self.block) << (position * self.block)
                rows.append(row)
        return LinearCode(length=self.length, rows=tuple(rows))

    def describe(self) -> str:
        """Render the construction the way the published tables do."""
        return (
            f"QC index {self.index}, block {self.block}, height {self.height}, "
            f"polynomials {[format(p, '#0' + str(self.block + 2) + 'b') for p in self.polynomials]}"
        )


def polynomial_from_degrees(degrees: tuple[int, ...]) -> int:
    """Return the bitmask of a polynomial given the degrees of its terms."""
    value = 0
    for degree in degrees:
        value |= 1 << degree
    return value
